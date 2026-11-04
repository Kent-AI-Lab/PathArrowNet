"""
Directed GNN link predictor on the TRAIN KO graph (the "graph network" rows of the protocol comparison, Fig. 2). Forward pass and inference only.

x(v) = ID(v)*seen + GO(v) (bag of GO terms) or + sum_f Linear(LN(feature_f(v))) (frozen GoBERT / ESM2 KO features);
2 directed message-passing layers (gcn_* configs: undirected GCN):
    h' = LN(h + drop(relu(W_self h + W_out mean_{v in out(u)} h_v + W_in mean_{w in in(u)} h_w)))
Score s(u, v) = <P_src h_u, P_dst h_v> + b_dst(h_v) + b_src(h_u).   Task A (u -> ?): s(u, v).  Task C: s(l, v) + s(v, r).
At inference all train edges are used for message passing and every train KO keeps its ID embedding.
The released runs (config.json + best.pt) were trained on the train split (training code is
not part of this release). Checkpoints are re-scored by benchmark_v2/dump_gnn.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


DENSE = {  # config -> frozen feature tables from kegg_features/ko_features.pt
    "gnn_id+gobert_noBP": ["gobert_noBP_mean"],
    "gnn_id+gobert_all": ["gobert_all_mean"],
    "gnn_id+esm": ["esm2_mean"],
    "gnn_id+gobert_noBP+esm": ["gobert_noBP_mean", "esm2_mean"],
}
# gcn_* = same features, but an undirected GCN encoder (symmetric normalisation, no in/out distinction)
GCN_CONFIGS = ["gcn_id", "gcn_id+gobert_noBP+esm"]


class DirGNN(nn.Module):
    def __init__(self, V, d, layers, dropout, seen, go_sets=None, n_go=0, id_dropout=0.25, dense=None,
                 undirected=False, background=None):
        super().__init__()
        self.undirected = undirected
        # optional external background graph (undirected KO pairs, e.g. STRING experiments): used as an extra,
        # separately weighted message channel in every layer, identical at train and test time
        if background is not None:
            self.register_buffer("bg_src", background[:, 0].contiguous())
            self.register_buffer("bg_dst", background[:, 1].contiguous())
        self.use_bg = background is not None
        dense = dense or []
        self.n_dense = len(dense)
        self.dense_proj = nn.ModuleList()
        for i, (X, has) in enumerate(dense):
            self.register_buffer(f"dense_X{i}", X)
            self.register_buffer(f"dense_has{i}", has.float())
            self.dense_proj.append(nn.Sequential(nn.LayerNorm(X.shape[1]), nn.Linear(X.shape[1], d)))
        self.V, self.use_go, self.id_dropout = V, go_sets is not None, id_dropout
        self.id_emb = nn.Embedding(V, d)
        nn.init.normal_(self.id_emb.weight, std=0.1)
        self.register_buffer("seen", seen)
        if self.use_go:
            flat, offs = [], []
            for v in range(V):
                offs.append(len(flat))
                flat.extend(go_sets.get(v, []))
            self.register_buffer("go_flat", torch.tensor(flat, dtype=torch.long))
            self.register_buffer("go_offs", torch.tensor(offs, dtype=torch.long))
            self.register_buffer("has_go", torch.tensor([bool(go_sets.get(v)) for v in range(V)]))
            self.go_bag = nn.EmbeddingBag(n_go, d, mode="mean")
            nn.init.normal_(self.go_bag.weight, std=0.1)
            self.go_lin = nn.Linear(d, d)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            layer = {"self": nn.Linear(d, d), "out": nn.Linear(d, d, bias=False),
                     "inn": nn.Linear(d, d, bias=False), "ln": nn.LayerNorm(d)}
            if background is not None:
                layer["bg"] = nn.Linear(d, d, bias=False)
            self.layers.append(nn.ModuleDict(layer))
        self.drop = nn.Dropout(dropout)
        self.p_src, self.p_dst = nn.Linear(d, d), nn.Linear(d, d)
        self.b_src, self.b_dst = nn.Linear(d, 1), nn.Linear(d, 1)

    def features(self):
        keep = self.seen.float()
        if self.training and (self.use_go or self.n_dense) and self.id_dropout > 0:
            keep = keep * (torch.rand(self.V, device=keep.device) >= self.id_dropout).float()
        x = self.id_emb.weight * keep[:, None]
        if self.use_go:
            x = x + self.go_lin(self.go_bag(self.go_flat, self.go_offs) * self.has_go[:, None].float())
        for i, proj in enumerate(self.dense_proj):
            x = x + proj(getattr(self, f"dense_X{i}")) * getattr(self, f"dense_has{i}")[:, None]
        return x

    def encode(self, src, dst):
        h = self.features()
        V = self.V
        deg_out = torch.zeros(V, device=h.device).index_add_(0, src, torch.ones_like(src, dtype=h.dtype)).clamp(min=1)
        deg_in = torch.zeros(V, device=h.device).index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype)).clamp(min=1)
        if self.undirected:  # plain GCN: symmetric normalisation over the undirected neighbourhood
            a, b = torch.cat([src, dst]), torch.cat([dst, src])
            deg = torch.zeros(V, device=h.device).index_add_(0, a, torch.ones_like(a, dtype=h.dtype)).clamp(min=1)
            norm = (deg[a] * deg[b]).rsqrt()[:, None]
            for L in self.layers:
                m = torch.zeros_like(h).index_add_(0, a, h[b] * norm)
                h = L["ln"](h + self.drop(F.relu(L["self"](h) + L["out"](m))))
            return h
        if self.use_bg:
            deg_bg = torch.zeros(V, device=h.device).index_add_(
                0, self.bg_src, torch.ones_like(self.bg_src, dtype=h.dtype)).clamp(min=1)
        for L in self.layers:
            m_out = torch.zeros_like(h).index_add_(0, src, h[dst]) / deg_out[:, None]   # successors of u
            m_in = torch.zeros_like(h).index_add_(0, dst, h[src]) / deg_in[:, None]     # predecessors of u
            z = L["self"](h) + L["out"](m_out) + L["inn"](m_in)
            if self.use_bg:
                z = z + L["bg"](torch.zeros_like(h).index_add_(0, self.bg_src, h[self.bg_dst]) / deg_bg[:, None])
            h = L["ln"](h + self.drop(F.relu(z)))
        return h

    def score_targets(self, h, u):        # [B, V]: s(u, v) for all v
        return self.p_src(h[u]) @ self.p_dst(h).T + self.b_dst(h)[:, 0][None, :]

    def score_sources(self, h, v):        # [B, V]: s(w, v) for all w
        return self.p_dst(h[v]) @ self.p_src(h).T + self.b_src(h)[:, 0][None, :]

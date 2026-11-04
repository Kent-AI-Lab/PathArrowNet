"""
PathArrowNet: query-conditioned path GNN (NBFNet-style message passing, as in BioPathNet) over the training KEGG edges of a
split and the STRING experimental background, with frozen ESM2 protein features. Forward pass and inference only.

Model
-----
For an anchor a and query type q (tail / head), node states start as the boundary condition
H0[a] = e_q + P(x_a), H0[v != a] = 0, and are updated for T layers with query-dependent relation vectors:
    m_v = AGG_{(w, r, v)} H[w] * R_q[r]      (distmult messages; PNA-style mean/max/sum-log-degree)
    H'  = H + drop(relu(LN(W [H, m_v, x_v])))
x_v = sum_f Linear(LN(feature_f(v))) are frozen protein features (zero if missing).
Score s_q(a, v) = MLP([H_T[v], x_v, e_q]) + <A_q x_a, B_q x_v> (the bilinear feature term also scores KOs that are
unreachable from a in the training graph, e.g. KOs absent from training).
Task A (u -> ?): log_softmax s_tail(u, .) over all 11,620 KOs.

Message-passing graph at inference: training-owned KO edges of the split (relation 0: u -> v, relation 1: inverse) and the
undirected STRING experimental >= 150 background in four score bins (relations 2-5).

Released configuration (paper): ESM2 features, STRING experimental >= 150 in 4 score bins, d = 32, 6 layers, PNA
aggregation, bilinear term; five seeds per strict split, seed-averaged by averaging row z-scores
(benchmark_v2/improve/calib_ensemble.py --fixed 0,0,0). The KGML relation-level weights of the primary model only change
which training edges are sampled and the soft targets during training; inference is identical. Training code is not part
of this release.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import data as D

TAIL, HEAD = 0, 1


class KONBF(nn.Module):
    """PathArrowNet (class name kept from the training code so that checkpoints load unchanged)."""
    def __init__(self, V, d, layers, dropout, dense, n_rel, bilinear=True, agg="pna"):
        super().__init__()
        self.V, self.d, self.n_rel, self.agg = V, d, n_rel, agg
        self.n_dense = len(dense)
        self.dense_proj = nn.ModuleList()
        for i, (X, has) in enumerate(dense):
            self.register_buffer(f"dense_X{i}", X)
            self.register_buffer(f"dense_has{i}", has.float())
            self.dense_proj.append(nn.Sequential(nn.LayerNorm(X.shape[1]), nn.Linear(X.shape[1], d)))
        self.q_emb = nn.Embedding(2, d)
        self.anchor_proj = nn.Linear(d, d)
        n_agg = {"pna": 3, "sum": 1, "mean": 1}[agg]
        self.rel = nn.ModuleList([nn.Linear(d, n_rel * d) for _ in range(layers)])   # query-dependent relations
        self.upd = nn.ModuleList([nn.Linear(d * (1 + n_agg) + d, d) for _ in range(layers)])
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.ReLU(), nn.Linear(2 * d, 1))
        self.bilinear = bilinear
        if bilinear:
            self.bl_a = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(2)])
            self.bl_b = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])

    def node_x(self):
        x = None
        for i, proj in enumerate(self.dense_proj):
            t = proj(getattr(self, f"dense_X{i}")) * getattr(self, f"dense_has{i}")[:, None]
            x = t if x is None else x + t
        if x is None:
            x = torch.zeros(self.V, self.d, device=self.q_emb.weight.device)
        return x

    def prep(self, fwd_src=None, fwd_dst=None):
        """node inputs (protein features); the edge arguments are accepted for interface compatibility."""
        return self.node_x(), None

    def forward(self, anchors, qtype, src, dst, rel, prep):
        """anchors [B]; qtype int; graph edges src -> dst with relation ids rel. Returns scores [B, V]."""
        B, V, d = len(anchors), self.V, self.d
        x, _ = prep
        q = self.q_emb.weight[qtype]                                      # [d]
        boundary = torch.zeros(B, V, d, device=x.device)
        boundary[torch.arange(B, device=x.device), anchors] = q[None, :] + self.anchor_proj(x[anchors])
        h = boundary
        deg = torch.zeros(V, device=x.device).index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        degc = deg.clamp(min=1)
        logd = torch.log1p(deg)[None, :, None]
        logd = logd / logd.mean()
        xb = x[None].expand(B, -1, -1)
        for L in range(len(self.upd)):
            rv = self.rel[L](q).view(self.n_rel, d)                        # [R, d]
            msg = h[:, src] * rv[rel][None]                                # [B, E, d]
            s = torch.zeros_like(h).index_add_(1, dst, msg)
            aggs = []
            if self.agg in ("pna", "mean"):
                aggs.append(s / degc[None, :, None])
            if self.agg == "sum":
                aggs.append(s)
            if self.agg == "pna":
                mx = torch.full_like(h, -1e4).scatter_reduce(1, dst[None, :, None].expand(B, -1, d), msg,
                                                            "amax", include_self=True)
                mx = torch.where(deg[None, :, None] > 0, mx, torch.zeros_like(mx))
                aggs += [mx, s / degc[None, :, None] * logd]
            z = self.upd[L](torch.cat([h] + aggs + [xb], -1))
            h = h + self.drop(F.relu(self.ln[L](z)))
        qb = q[None, None, :].expand(B, V, -1)
        sc = self.mlp(torch.cat([h, xb, qb], -1))[..., 0]
        if self.bilinear:
            sc = sc + self.bl_a[qtype](x[anchors]) @ self.bl_b[qtype](x).T
        return sc


def build_graph(edges, background, device, bg_rel=None):
    """relation 0: u -> v (train edge), 1: inverse, 2: STRING background (both directions).
    bg_rel (optional, per background row): relation id of each background edge (score bins)."""
    e = torch.tensor(edges, dtype=torch.long)
    src = [e[:, 0], e[:, 1]]
    dst = [e[:, 1], e[:, 0]]
    rel = [torch.zeros(len(e), dtype=torch.long), torch.ones(len(e), dtype=torch.long)]
    if background is not None:
        src.append(background[:, 0])
        dst.append(background[:, 1])
        rel.append(bg_rel if bg_rel is not None else torch.full((len(background),), 2, dtype=torch.long))
    return torch.cat(src).to(device), torch.cat(dst).to(device), torch.cat(rel).to(device), len(e)


@torch.no_grad()
def anchor_scores(model, anchors, qtype, graph, bs):
    """Eval-mode log-softmax scores over all KOs for each unique anchor -> (dict anchor -> row, tensor [n, V]).
    Token ids 0..2 (special tokens) get -inf."""
    model.eval()
    uniq = sorted(set(anchors))
    rows = []
    pr = model.prep(*graph[3])
    for i in range(0, len(uniq), bs):
        a = torch.tensor(uniq[i:i + bs], device=graph[0].device)
        rows.append(model(a, qtype, *graph[:3], pr).float())
    S = torch.cat(rows)
    S[:, :3] = float("-inf")
    return {u: i for i, u in enumerate(uniq)}, torch.log_softmax(S, 1)


@torch.no_grad()
def evaluate(model, graph, queries, cand_mask, bs):
    """Filtered ranks per task ('A': u -> ?, 'C': l -> ? -> r) in eval mode; also per-query sampled AUC for task A
    if the queries carry negatives ('neg')."""
    model.eval()
    out = {}
    for task, qs in queries.items():
        if task == "A":
            idx, S = anchor_scores(model, [q["source"] for q in qs], TAIL, graph, bs)
        else:
            il, SL_ = anchor_scores(model, [q["left"] for q in qs], TAIL, graph, bs)
            ir, SR = anchor_scores(model, [q["right"] for q in qs], HEAD, graph, bs)
        ranks = []
        for i in range(0, len(qs), 4096):
            chunk = qs[i:i + 4096]
            if task == "A":
                sc = S[torch.tensor([idx[q["source"]] for q in chunk], device=S.device)]
            else:
                sc = SL_[torch.tensor([il[q["left"]] for q in chunk], device=SL_.device)] + \
                     SR[torch.tensor([ir[q["right"]] for q in chunk], device=SR.device)]
            sc = torch.nan_to_num(sc, neginf=-1e30)
            rk = D.gpu_ranks(sc, chunk, cand_mask)
            if task == "A" and "neg" in chunk[0]:
                t = torch.tensor([q["target"] for q in chunk], device=sc.device)
                ng = torch.tensor([q["neg"] for q in chunk], device=sc.device)
                sp, sn = sc.gather(1, t[:, None]), sc.gather(1, ng)
                out.setdefault("A_auc", []).extend(((sp > sn).float() + 0.5 * (sp == sn).float()).mean(1).tolist())
            ranks.extend(rk)
        out[task] = ranks
    return out


# ----------------------------------------------------------------------------------------- loading a released run
def n_relations(cfg):
    if not cfg.get("background"):
        return 2
    return 6 if cfg.get("bg_bins4") else 5 if cfg.get("bg_bins") else 3


def dense_from_state(state):
    """Frozen feature tables stored in the checkpoint (buffers dense_X<i>, dense_has<i>)."""
    out, i = [], 0
    while f"dense_X{i}" in state:
        out.append((state[f"dense_X{i}"], state[f"dense_has{i}"]))
        i += 1
    return out


def resolve_split_dir(cfg, split_dir=None):
    """Split directory of a run: explicit argument, else config 'split_dir' re-rooted to this release
    (the part from 'kegg_dataset/' or 'kegg_splits_robustness/' or 'kegg_leaky_splits/' on), else splits_ko."""
    if split_dir:
        return Path(split_dir)
    p = cfg.get("split_dir")
    if not p:
        return D.SPLIT
    p = Path(p)
    for anchor in ("kegg_dataset", "kegg_splits_robustness", "kegg_leaky_splits", "expansion", "benchmark_v2"):
        if anchor in p.parts:
            return D.ROOT.joinpath(*p.parts[p.parts.index(anchor):])
    return p


def load_run(run_dir, device="cpu", split_dir=None, ckpt="best.pt"):
    """Rebuild a PathArrowNet run for inference from <run_dir>/config.json and <run_dir>/<ckpt>.
    Returns (model in eval mode, graph tuple (src, dst, rel, (fwd_src, fwd_dst)), vocab, V, cfg).
    The graph is the train-owned edges of the run's split (leakage asserts enforced) + the configured background."""
    from kegg_pipeline import evaluation as E
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    if cfg.get("hybrid"):
        raise SystemExit("hybrid PathArrowNet (with the DirGNN branch) is not supported by this release")
    if cfg.get("train_on_val"):
        raise SystemExit("train-on-val runs are not supported by this release (graph would include val edges)")
    vocab, V = D.load_vocab(cfg.get("full_candidates", False))
    sd = resolve_split_dir(cfg, split_dir)
    tasks = cfg.get("tasks", "AC")
    qd = {}
    for sp in ("val", "test"):
        qd[sp] = {t: E.load_queries(sd / f"{sp}_{t}_{'next_node' if t == 'A' else 'masked'}.csv.gz", t)
                  for t in tasks if (sd / f"{sp}_{t}_{'next_node' if t == 'A' else 'masked'}.csv.gz").exists()}
    edges = D.load_train_edges(vocab, qd["val"], qd["test"], sd, cfg.get("allow_leaky", False))
    background, bg_rel = None, None
    if cfg.get("background"):
        bins = "bins4" if cfg.get("bg_bins4") else "bins3" if cfg.get("bg_bins") else None
        background, bg_rel = D.load_background(cfg["background"], vocab, bins)
    src, dst, rel, _ = build_graph(edges, background, device, bg_rel)
    state = torch.load(run_dir / ckpt, map_location=device)
    dense = dense_from_state(state)
    model = KONBF(V, cfg["d"], cfg["layers"], cfg["dropout"], dense, n_relations(cfg),
                  bilinear=not cfg.get("no_bilinear", False), agg=cfg.get("agg", "pna")).to(device)
    model.load_state_dict(state)
    model.eval()
    E_t = torch.tensor(edges, device=device)
    return model, (src, dst, rel, (E_t[:, 0], E_t[:, 1])), vocab, V, cfg


PathArrowNet = KONBF

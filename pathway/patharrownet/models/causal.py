"""
GPT-style causal Transformer for task S (prefix of KOs -> next KO), strict protocol, full candidate set.
Forward pass and inference only.

Token inputs: frozen ESM2 / GoBERT(noBP) features, KEGG-encoder vectors of the same split (expansion/repr/runs/<setting>/
<variant>_s<seed>/emb.pt, see patharrownet.models.kegg_encoder), cached description embeddings, plus learned ID embeddings
for KOs seen in train. Score of candidate v for prefix c: <W h_last(c), x_v> + b(x_v).
Released runs: expansion/task_s/runs, expansion/ctxlen/runs (config.json + best.pt). Training code is not part of this
release.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark_v2"))

EXP = ROOT / "expansion"
MAXLEN = 5


def load_input(name, setting, V, repr_seed):
    """returns (X [V, D], has [V]) for one input source."""
    if name in ("esm2_mean", "gobert_noBP_mean", "gobert_all_mean"):
        kf = torch.load(ROOT / "kegg_features" / "ko_features_full.pt")["features"][name]
        return kf["X"].float(), kf["has"].float()
    if name.startswith("repr:"):  # repr:<variant>  -> KEGG encoder of THIS split
        var = name.split(":", 1)[1]
        e = torch.load(EXP / "repr" / "runs" / setting / f"{var}_s{repr_seed}" / "emb.pt")
        assert e["config"]["setting"] == setting, "encoder must come from the same split"
        if var.endswith("_shuf"):
            raise ValueError(name)
        return e["H"].float(), torch.ones(V)
    if name.startswith("reprshuf:"):  # control: rows permuted among KOs with features
        var = name.split(":", 1)[1]
        e = torch.load(EXP / "repr" / "runs" / setting / f"{var}_s{repr_seed}" / "emb.pt")
        H = e["H"].float().clone(); idx = torch.where(e["has_feat"])[0]
        g = torch.Generator().manual_seed(12345)
        H[idx] = H[idx[torch.randperm(len(idx), generator=g)]]
        return H, torch.ones(V)
    raise ValueError(name)


class Causal(nn.Module):
    def __init__(self, V, inputs, seen, d=128, layers=2, heads=4, dropout=0.2, use_id=True, id_dropout=0.25):
        super().__init__()
        self.V, self.use_id, self.id_dropout = V, use_id, id_dropout
        self.register_buffer("seen", seen.float())
        self.id_emb = nn.Embedding(V, d)
        self.proj = nn.ModuleList()
        for i, (X, has) in enumerate(inputs):
            self.register_buffer(f"X{i}", X); self.register_buffer(f"h{i}", has)
            self.proj.append(nn.Sequential(nn.LayerNorm(X.shape[1]), nn.Linear(X.shape[1], d)))
        self.pos = nn.Embedding(MAXLEN, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, d)
        self.bias = nn.Linear(d, 1)

    def table(self):
        """token input table [V, d] (eval mode: ID embeddings of every KO seen in train)."""
        x = 0
        if self.use_id:
            keep = self.seen
            if self.training and self.id_dropout > 0 and len(self.proj):
                keep = keep * (torch.rand(self.V, device=keep.device) >= self.id_dropout).float()
            x = self.id_emb.weight * keep[:, None]
        for i, p in enumerate(self.proj):
            x = x + p(getattr(self, f"X{i}")) * getattr(self, f"h{i}")[:, None]
        return x

    def forward(self, ctx, lens, x):
        B, L = ctx.shape
        e = x[ctx] + self.pos.weight[:L][None]
        pad = torch.arange(L, device=ctx.device)[None] >= lens[:, None]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=ctx.device), 1)
        h = self.enc(e, mask=causal, src_key_padding_mask=pad)
        h = self.ln(h[torch.arange(B, device=ctx.device), lens - 1])
        return self.out(h) @ x.T + self.bias(x)[:, 0][None]


def batchify(ctxs, device, last_only=False):
    if last_only:
        ctxs = [c[-1:] for c in ctxs]
    L = max(len(c) for c in ctxs)
    arr = torch.zeros(len(ctxs), L, dtype=torch.long)
    for i, c in enumerate(ctxs):
        arr[i, :len(c)] = torch.tensor(c)
    return arr.to(device), torch.tensor([len(c) for c in ctxs], device=device)


@torch.no_grad()
def score(model, qs, device, last_only, bs=1024):
    """Eval mode. [n_queries, V] scores for scorelib queries (task A: context = [source])."""
    model.eval(); x = model.table(); out = []
    for i in range(0, len(qs), bs):
        c, l = batchify([q.get("context", [q["source"]]) for q in qs[i:i + bs]], device, last_only)
        out.append(model(c, l, x).float().cpu())
    return torch.cat(out).numpy()


def macro(rk, qs):
    w = np.array([q["weight"] for q in qs])
    return float((w / rk).sum() / w.sum())

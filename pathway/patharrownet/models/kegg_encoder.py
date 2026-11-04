#!/usr/bin/env python3
"""
KEGG encoder: query-independent KO representations h(v) (used by the comparison in the perturbation analysis).
Forward pass and embedding export only.

Encoder   x(v) = MLP([LN(ESM2(v)) * has_esm, LN(GoBERT_noBP(v)) * has_go])       (frozen pretrained features)
          L layers of relational mean aggregation over the TRAIN KEGG next-KO graph of one strict split
          (relations: forward u->v, reverse v->u; optional STRING-exp background), residual + LayerNorm; d = 128.
TwoTower  score_tail(u, v) = <Q h(u), K h(v)> + b(h(v)); the head direction has its own Q', K'.
The released encoders were trained on the train split of their strict split (training code is not
part of this release).

Export (eval mode, full train graph of the encoder's split; same format the analysis scripts read):

    cd $KEGG_HOME/pathway
    python -m patharrownet.models.kegg_encoder --run-dir expansion/repr/runs/strict_v1_full/kegg_enc_s0 --out emb.pt
    -> {"H": [V, d], "has_feat": [V] bool, "seen": [V] bool, "config": {...}}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "benchmark_v2"))
import scorelib as SL  # noqa: E402


def load_train_edges(setting):
    tr = SL.rd(SL.SPLIT_DIRS[setting][0] / "train_A_next_node.csv.gz")
    return sorted({(int(r["context_token_id"]), int(r["target_token_id"])) for r in tr})


def heldout_edges(setting):
    out = set()
    for sp in ("val", "test"):
        for r in SL.rd(SL.SPLIT_DIRS[setting][0] / f"{sp}_A_next_node.csv.gz"):
            out.add((int(r["context_token_id"]), int(r["target_token_id"])))
    return out


def load_bg(V, voc):
    """STRING experimental >=150 background, 4 bins, undirected (as used by the best model)."""
    import csv
    p = ROOT / "kegg_dataset_cache/string/ko_background_exp150_full.tsv"
    out = []
    with p.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            a, b = r["ko_a"], r["ko_b"]
            s = float(r["max_experimental_score"])
            if a in voc and b in voc:
                rel = 0 if s < 400 else 1 if s < 700 else 2 if s < 900 else 3
                out.append((voc[a], voc[b], rel))
    return out


class Encoder(nn.Module):
    def __init__(self, feats, n_rel, d=128, layers=2, dropout=0.1):
        super().__init__()
        self.n_feat = len(feats)
        for i, (X, has) in enumerate(feats):
            self.register_buffer(f"X{i}", X.float())
            self.register_buffer(f"h{i}", has.float())
        self.inp = nn.ModuleList([nn.Sequential(nn.LayerNorm(X.shape[1]), nn.Linear(X.shape[1], d)) for X, _ in feats])
        self.inmlp = nn.Sequential(nn.GELU(), nn.Linear(d, d))
        self.rel = nn.ModuleList([nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_rel)]) for _ in range(layers)])
        self.self_ = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.n_rel, self.d = n_rel, d

    def forward(self, src, dst, rel, V):
        x = 0
        for i, p in enumerate(self.inp):
            x = x + p(getattr(self, f"X{i}")) * getattr(self, f"h{i}")[:, None]
        h = self.inmlp(x)
        deg = [torch.zeros(V, device=h.device).index_add_(0, dst[rel == r], torch.ones(int((rel == r).sum()), device=h.device)).clamp(min=1)
               for r in range(self.n_rel)]
        for l in range(len(self.rel)):
            m = self.self_[l](h)
            for r in range(self.n_rel):
                k = rel == r
                agg = torch.zeros_like(h).index_add_(0, dst[k], self.rel[l][r](h)[src[k]])
                m = m + agg / deg[r][:, None]
            h = self.ln[l](h + self.drop(F.gelu(m)))
        return h


class TwoTower(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])
        self.k = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])
        self.b = nn.ModuleList([nn.Linear(d, 1) for _ in range(2)])

    def score(self, H, anchors, direction):
        """direction 0: anchor -> candidate (successors), 1: candidate -> anchor (predecessors)."""
        return self.q[direction](H[anchors]) @ self.k[direction](H).T + self.b[direction](H)[:, 0][None]


def build_graph(edges, bg, device):
    s = [u for u, v in edges] + [v for u, v in edges]
    t = [v for u, v in edges] + [u for u, v in edges]
    r = [0] * len(edges) + [1] * len(edges)
    for a, b, k in bg:  # undirected background: both directions, relation 2 + bin
        s += [a, b]; t += [b, a]; r += [2 + k, 2 + k]
    return (torch.tensor(s, device=device), torch.tensor(t, device=device), torch.tensor(r, device=device))


def feats_from_state(state):
    out, i = [], 0
    while f"X{i}" in state:
        out.append((state[f"X{i}"], state[f"h{i}"]))
        i += 1
    return out


def load_encoder(ckpt, device="cpu", dropout=0.1):
    """(Encoder, TwoTower) in eval mode from a best.pt {'enc': state, 'tt': state}; d, layers and the number of relations
    are read from the state dict, the frozen feature tables are stored in it as buffers."""
    ck = torch.load(ckpt, map_location=device)
    st = ck["enc"]
    feats = feats_from_state(st)
    layers = len({k.split(".")[1] for k in st if k.startswith("self_.")})
    n_rel = len({k.split(".")[2] for k in st if k.startswith("rel.0.")})
    d = st["self_.0.weight"].shape[0]
    enc = Encoder(feats, n_rel, d, layers, dropout).to(device)
    enc.load_state_dict(st)
    tt = TwoTower(d).to(device)
    tt.load_state_dict(ck["tt"])
    enc.eval(); tt.eval()
    return enc, tt


@torch.no_grad()
def embed(run_dir, setting, device="cpu"):
    """Eval-mode embeddings H [V, d] of an encoder run on the full train graph of `setting`."""
    enc, _ = load_encoder(Path(run_dir) / "best.pt", device)
    voc, V = SL.vocab(setting)
    edges = load_train_edges(setting)
    assert not (set(edges) & heldout_edges(setting)), "LEAKAGE: held-out edge in train edge file"
    bg = load_bg(V, voc) if enc.n_rel > 2 else []
    H = enc(*build_graph(edges, bg, device), V)
    seen = torch.zeros(V, dtype=torch.bool); seen[[x for e in edges for x in e]] = True
    has = torch.stack([getattr(enc, f"h{i}") for i in range(enc.n_feat)]).amax(0).bool().cpu()
    return {"H": H.cpu(), "has_feat": has, "seen": seen,
            "config": {"setting": setting, "d": enc.d, "n_rel": enc.n_rel, "run_dir": str(run_dir)},
            "vocab_file": SL.VOCAB_FILES.get(setting)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--setting", default=None, help="default: the run directory's parent name")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    setting = a.setting or Path(a.run_dir).resolve().parent.name
    e = embed(a.run_dir, setting)
    torch.save(e, a.out)
    print(json.dumps({"out": a.out, "shape": list(e["H"].shape), **e["config"]}))


if __name__ == "__main__":
    main()

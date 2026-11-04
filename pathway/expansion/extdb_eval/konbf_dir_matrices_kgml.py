#!/usr/bin/env python3
"""Tail and head score matrices of the primary PathArrowNet (KGML relation-level weights; checkpoints
xp_kgmlw+exp150bin4+allneg+kl_s42..46) from kegg_pairscore.nbf_matrix. The untrained control (konbf_randinit_tail.npy) does
not depend on training and is not recomputed.
    .venv/bin/python expansion/extdb_eval/konbf_dir_matrices_kgml.py
    -> expansion/extdb_eval/cache/konbf_{tail,head}_kgml.npy"""
import csv, sys
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE.parent / "downstream"))
from patharrownet.models import patharrownet as TN  # noqa: E402
import kegg_pairscore as KP  # noqa: E402

RUN = "xp_kgmlw+exp150bin4+allneg+kl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_orig = TN.anchor_scores
with (ROOT / "kegg_dataset/token_vocab_ko_full.csv").open() as f:
    V = sum(1 for _ in csv.DictReader(f))


def only(direction):
    def f(model, anchors, qtype, graph, bs):
        if qtype == direction:
            return _orig(model, anchors, qtype, graph, bs)
        u = sorted(set(anchors))
        return {a: i for i, a in enumerate(u)}, torch.zeros(len(u), V, device=graph[0].device)
    return f


for name, d in (("konbf_tail_kgml", TN.TAIL), ("konbf_head_kgml", TN.HEAD)):
    out = HERE / "cache" / f"{name}.npy"
    if out.exists():
        print("exists", out, flush=True); continue
    TN.anchor_scores = only(d)
    M = KP.nbf_matrix(list(range(3, V)), run=RUN, dev=DEV)
    np.save(out, M.cpu().numpy().astype(np.float32)); print("saved", name, tuple(M.shape), flush=True)
    del M
    if DEV == "cuda":
        torch.cuda.empty_cache()
TN.anchor_scores = _orig
t = np.load(HERE / "cache/konbf_tail_kgml.npy", mmap_mode="r"); h = np.load(HERE / "cache/konbf_head_kgml.npy", mmap_mode="r")
c = np.load(ROOT / "expansion/downstream/cache/konbf_full_matrix_kgml.npy", mmap_mode="r")
err = max(float(np.abs(t[i:i + 1000] + h[i:i + 1000] - c[i:i + 1000]).max()) for i in range(0, V, 1000))
print("max |tail + head - full kgml matrix| =", err, flush=True)

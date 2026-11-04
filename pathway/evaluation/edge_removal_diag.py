#!/usr/bin/env python3
"""
Edge-removal diagnostic (Suppl. Sec. S11), inference only on CPU.
For training anchors a of a split with training successors Y_a, every y in Y_a is ranked among all KOs (the other members of
Y_a and a itself filtered, mean-tie ranks) under three graphs:
  P  the training graph unchanged;
  R  all edges a -> y, y in Y_a, and their inverse copies removed (the answer-set removal used in training);
  S  only the scored edge a -> y and its inverse removed (batch-edge removal; computed for the first --anchors-s anchors, 30 in the paper).
Reported per run: mean reciprocal rank of the training successors under P, R and S.

    python evaluation/edge_removal_diag.py --run "<label>=<run dir>" [--run ...] --anchors 150 --anchors-s 30 \
        --out results/evaluation/edge_removal_diag.json
Run directories are trained PathArrowNet runs (best.pt + config.json), e.g. checkpoints/patharrownet/strict_v1_full/s42.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import PW

sys.path.insert(0, str(PW))
from patharrownet.models import patharrownet as KN  # noqa: E402


def frank(row, y, excl):
    s = row[y]; m = np.ones(len(row), bool); m[:3] = False; m[list(excl)] = False; m[y] = False
    return 1 + (row[m] > s).sum() + 0.5 * (row[m] == s).sum()


@torch.no_grad()
def score(model, a, src, dst, rel, pr):
    return model(torch.tensor([a]), 0, src, dst, rel, pr)[0].numpy()


def diagnose(run_dir, n_anchors, n_anchors_s, seed=0):
    model, graph, vocab, V, cfg = KN.load_run(Path(run_dir), "cpu")
    src, dst, rel = graph[:3]; fs, fd = graph[3]
    Y = {}
    for u, v in zip(fs.tolist(), fd.tolist()):
        Y.setdefault(u, set()).add(v)
    anchors = sorted(Y); random.Random(seed).shuffle(anchors); anchors = anchors[:n_anchors]
    pr = model.prep(*graph[3]); res = {"P": [], "R": [], "S": []}
    for i, a in enumerate(anchors):
        ys = sorted(Y[a]); excl = set(ys) | {a}; yt = torch.tensor(ys)
        rowP = score(model, a, src, dst, rel, pr)
        kill = ((rel == 0) & (src == a) & torch.isin(dst, yt)) | ((rel == 1) & (dst == a) & torch.isin(src, yt))
        rowR = score(model, a, src[~kill], dst[~kill], rel[~kill], pr)
        for y in ys:
            res["P"].append(1 / frank(rowP, y, excl - {y})); res["R"].append(1 / frank(rowR, y, excl - {y}))
        if i < n_anchors_s:
            for y in ys:
                k1 = ((rel == 0) & (src == a) & (dst == y)) | ((rel == 1) & (dst == a) & (src == y))
                rowS = score(model, a, src[~k1], dst[~k1], rel[~k1], pr)
                res["S"].append(1 / frank(rowS, y, excl - {y}))
    return {k: float(np.mean(v)) for k, v in res.items()} | {"n_pairs": len(res["P"]), "n_pairs_S": len(res["S"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="<label>=<run dir>; repeat for several runs")
    ap.add_argument("--anchors", type=int, default=150)
    ap.add_argument("--anchors-s", type=int, default=30)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="results/evaluation/edge_removal_diag.json")
    a = ap.parse_args(); torch.set_num_threads(a.threads); out = {}
    for spec in a.run:
        label, run_dir = spec.split("=", 1); t0 = time.time()
        out[label] = diagnose(run_dir, a.anchors, a.anchors_s)
        print(f"{label}: P {out[label]['P']:.3f}  R {out[label]['R']:.3f}  S {out[label]['S']:.3f}  "
              f"pairs {out[label]['n_pairs']}  ({time.time() - t0:.0f}s)", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Score matrices of the non-learned baselines of evaluate_ko_baselines.py (same code, same definitions) for the
benchmark_v2 evaluator. All are anchor models (the score depends on the current KO only).

  popularity      train in-degree of the candidate
  cf              neighbourhood collaborative filtering (Jaccard of successor sets) on the train graph
  bigram          interpolated first-order Markov model on train path segments (B windows)
  adjacency       number of train 2-step paths u -> x -> v
  go_profile      GO-profile similarity, all aspects (MF+BP+CC)          -- uses external GO annotations
  go_profile_no_BP GO-profile similarity, MF+CC only                       -- uses external GO annotations

    .venv/bin/python benchmark_v2/dump_baselines.py --setting strict_v1
"""

import argparse
import sys
from pathlib import Path

import numpy as np

B2 = Path(__file__).resolve().parent
ROOT = B2.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(B2))
import scorelib as SL  # noqa: E402
import evaluate_ko_baselines as EB  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="strict_v1")
    a = ap.parse_args()
    voc, V = SL.vocab(a.setting)
    cands = list(range(3, V))
    T = EB.Train(V, SL.SPLIT_DIRS[a.setting][0])
    anchors = SL.anchors_needed(a.setting)
    methods = {"popularity": None, "cf": None, "bigram": None, "adjacency": None}
    scorers = {}
    for name in ("go_profile", "go_profile_no_BP"):
        use_kegg, aspects, drop_iea = EB.GO_VARIANTS[name]
        scorers[name] = EB.Scorers(T, EB.load_ko_go(voc, use_kegg, aspects, drop_iea), cands)
    base = scorers["go_profile_no_BP"]
    for m in list(methods) + list(scorers):
        f = getattr(scorers[m], "A_go_profile") if m in scorers else getattr(base, f"A_{m}")
        sc = np.array([f({"source": u}) for u in anchors], dtype=np.float64)
        out = SL.SCORES / a.setting / f"baseline_{m}" / "s0"
        for split in ("val", "test"):
            SL.save_tail(out / f"tail_{split}.npz", anchors, sc, np.float64)
        q = SL.load_A(a.setting, "test")
        rk, _ = SL.ranks(SL.query_scores(out, "test", "A", q), q, V)
        print(f"{a.setting} {m}: test A MRR micro {np.mean(1 / rk):.4f}", flush=True)


if __name__ == "__main__":
    main()

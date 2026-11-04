#!/usr/bin/env python3
"""
Recovery of the held-out successor set of branching sources (Sec. 4.5, Suppl. Fig. S3): seed-averaged PathArrowNet against
reproduced BioPathNet (its seeds averaged by row z-score); mean over the five splits and pooled pathway-macro differences.

    python evaluation/branch_recall.py           -> results/evaluation/branch_recall.json
"""
from collections import defaultdict

import numpy as np

import branch_lib as BE
from common import CS, PAN_SA, REP, STS, SPLIT, SL, write

KS = ["R@10", "R@50", "R@100", "All@10", "All@100", "RPrec"]
SUBS = {"all": lambda r: True, "branching": lambda r: r["n_children"] >= 2}


def main():
    out, pool, agg = {}, defaultdict(dict), defaultdict(lambda: defaultdict(list))
    for st in STS:
        _, V = SL.vocab(st); qs = SL.load_queries(st, "test", "A")
        P = {m: BE.per_parent(BE.method_scores(st, m, qs), qs, V) for m in (PAN_SA, REP)}
        out[SPLIT[st]] = {}
        for m, rows in P.items():
            for sub, keep in SUBS.items():
                rs = [r for r in rows if keep(r)]
                d = {k: float(np.mean([r[k] for r in rs])) for k in KS}; d["n_parents"] = len(rs)
                out[SPLIT[st]][f"{m}|{sub}"] = d
                for k in KS:
                    agg[f"{m}|{sub}"][k].append(d[k])
        for sub, keep in SUBS.items():
            idx = [i for i, r in enumerate(P[PAN_SA]) if keep(r)]
            ql = BE.parent_queries(P[PAN_SA], idx)
            for k in KS:
                pool[f"{sub}|{k}|PathArrowNet-reproduced BioPathNet"][st] = (np.array([P[PAN_SA][i][k] for i in idx]), np.array([P[REP][i][k] for i in idx]), ql)
    out["mean_over_splits"] = {g: {k: float(np.mean(v)) for k, v in d.items()} for g, d in agg.items()}
    out["pooled"] = {k: CS.pooled(v) for k, v in pool.items()}
    for k, v in out["pooled"].items():
        if k.startswith("branching"):
            print(k, f"{v['est']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]", flush=True)
    write("branch_recall.json", out)


if __name__ == "__main__":
    main()

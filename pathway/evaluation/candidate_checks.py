#!/usr/bin/env python3
"""
Suppl. Sec. S11: (i) candidate KOs without a human gene (Suppl. Table S10c): no test target is one of them, and removing
them from the candidate set leaves the pooled MRR of every Table 1 method unchanged; (ii) test queries whose target KO has
no training edge (unseen targets): pooled PathArrowNet - reproduced BioPathNet differences.

    python evaluation/candidate_checks.py        -> results/evaluation/candidate_checks.json
"""
import csv

import numpy as np

from common import CS, PAN, PAN_SA, PW, REP, REP_SA, SL, STS, Q, rr, runs, write
from main_results import TABLE1


def human_gene_counts():
    with (PW / "kegg_dataset" / "ko_mapping.csv").open() as f:
        return {r["ko_id"]: int(r["n_human_genes"] or 0) for r in csv.DictReader(f)}


def rr_without(st, m, drop):
    """per-query RR (mean over runs) with the KOs in `drop` removed from the candidate set"""
    _, V = SL.vocab(st)
    qs = Q(st)
    out = []
    for r in runs(st, m):
        S = SL.query_scores(r, "test", "A", qs).astype(np.float64)
        S[:, drop] = -np.inf  # no target is in `drop` (asserted in main)
        out.append(1 / SL.ranks(S, qs, V)[0])
    return np.mean(out, 0)


def level(R, keep=None):
    d = {}
    for st in STS:
        k = np.ones(len(Q(st)), bool) if keep is None else keep[st]
        d[st] = (R[st][k], np.zeros(int(k.sum())), [q for q, x in zip(Q(st), k) if x])
    return CS.pooled(d)["est"]


def diff(a, b, keep):
    return CS.pooled({st: (a[st][keep[st]], b[st][keep[st]], [q for q, x in zip(Q(st), keep[st]) if x]) for st in STS})


def main():
    nh = human_gene_counts()
    out = {"no_human_gene": {"methods": {}}, "unseen_targets": {}}
    drop, n_tgt = {}, 0
    for st in STS:
        voc, V = SL.vocab(st)
        inv = {i: t for t, i in voc.items()}
        drop[st] = np.array([i for i in range(3, V) if nh.get(inv[i], 0) == 0])
        n_tgt += sum(nh.get(inv[q["target"]], 0) == 0 for q in Q(st))
    out["no_human_gene"]["n_candidates"] = int(len(drop[STS[0]]))
    out["no_human_gene"]["n_test_targets_without_human_gene"] = int(n_tgt)
    assert n_tgt == 0
    for lab, m in TABLE1:
        a = level({st: rr(st, m) for st in STS})
        b = level({st: rr_without(st, m, drop[st]) for st in STS})
        out["no_human_gene"]["methods"][lab] = {"MRR": a, "MRR_without": b, "change": b - a}
        print(f"{lab:48s} {a:.5f} -> {b:.5f}", flush=True)
    mx = max(abs(v["change"]) for v in out["no_human_gene"]["methods"].values())
    out["no_human_gene"]["max_abs_change"] = mx
    out["no_human_gene"]["below_0.0001"] = int(mx < 1e-4)

    unseen = {st: np.array([not q["seen"] for q in Q(st)]) for st in STS}
    n_u, n = sum(int(u.sum()) for u in unseen.values()), sum(len(Q(st)) for st in STS)
    out["unseen_targets"] = {"n_queries": n_u, "n_all": n, "share": n_u / n,
                             "PathArrowNet - reproduced BioPathNet (seed-averaged)":
                                 diff({st: rr(st, PAN_SA) for st in STS}, {st: rr(st, REP_SA) for st in STS}, unseen),
                             "PathArrowNet - reproduced BioPathNet (single models)":
                                 diff({st: rr(st, PAN) for st in STS}, {st: rr(st, REP) for st in STS}, unseen)}
    write("candidate_checks.json", out)


if __name__ == "__main__":
    main()

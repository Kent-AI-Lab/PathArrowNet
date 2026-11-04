#!/usr/bin/env python3
"""
Compare a score dump written by patharrownet.predict with a reference dump (same benchmark_v2 format: anchors, scores) and,
optionally, the per-query ranks with a run's ranks_<split>_A.csv.gz.

    python -m patharrownet.compare_scores --new <out>/tail_test.npz --ref <scores dir>/tail_test.npz \
        [--new-ranks <out>/ranks_test_A.tsv --ref-ranks <run dir>/ranks_test_A.csv.gz] [--json result.json]

Reported on the anchors present in both dumps: max |new - ref| over finite entries (entries <= -1e29 are the masked
special tokens), the share of anchors with identical top-10 / identical full argsort order, and, for the ranks, the
number of queries with identical filtered rank and the max |rank difference|. With --setting/--split the filtered
task-A ranks implied by both dumps are computed and compared as well.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json

import numpy as np


def read_ranks(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        rd = csv.DictReader(f, delimiter="\t" if str(path).endswith(".tsv") else ",")
        return {int(r["query_index"]): float(r["rank"]) for r in rd}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new-ranks", default=None)
    ap.add_argument("--ref-ranks", default=None)
    ap.add_argument("--setting", default=None, help="with --split: also compare filtered task-A ranks (scorelib)")
    ap.add_argument("--split", default=None, choices=["val", "test"])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    n, r = np.load(a.new), np.load(a.ref)
    pn = {int(x): i for i, x in enumerate(n["anchors"])}
    pr = {int(x): i for i, x in enumerate(r["anchors"])}
    common = sorted(set(pn) & set(pr))
    A = n["scores"][[pn[x] for x in common]].astype(np.float64)
    B = r["scores"][[pr[x] for x in common]].astype(np.float64)
    ma, mb = np.isfinite(A) & (A > -1e29), np.isfinite(B) & (B > -1e29)
    assert (ma == mb).all(), "masked entries differ"
    fin = ma
    with np.errstate(invalid="ignore"):
        diff = np.abs(A - B)[fin]
    top_same = sum(np.array_equal(np.argsort(-a_, kind="stable")[:10], np.argsort(-b_, kind="stable")[:10]) for a_, b_ in zip(A, B))
    order_same = sum(np.array_equal(np.argsort(-a_, kind="stable"), np.argsort(-b_, kind="stable")) for a_, b_ in zip(A, B))
    res = {"n_anchors_compared": len(common), "n_entries_compared": int(fin.sum()), "max_abs_diff": float(diff.max()),
           "mean_abs_diff": float(diff.mean()), "anchors_identical_top10": int(top_same),
           "anchors_identical_full_order": int(order_same)}
    if a.setting and a.split:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark_v2"))
        import scorelib as SL
        _, V = SL.vocab(a.setting)
        keep = set(common)
        qs = [q for q in SL.load_A(a.setting, a.split) if q["source"] in keep]
        rn = SL.ranks(n["scores"][[pn[q["source"]] for q in qs]], qs, V)[0]
        rr = SL.ranks(r["scores"][[pr[q["source"]] for q in qs]], qs, V)[0]
        d = np.abs(rn - rr)
        res.update({"n_queries_ranked": len(qs), "queries_identical_filtered_rank": int((d == 0).sum()),
                    "max_abs_filtered_rank_diff": float(d.max()), "micro_MRR_new_scores": float(np.mean(1 / rn)),
                    "micro_MRR_ref_scores": float(np.mean(1 / rr))})
    if a.new_ranks and a.ref_ranks:
        rn, rr = read_ranks(a.new_ranks), read_ranks(a.ref_ranks)
        qi = sorted(set(rn) & set(rr))
        d = np.array([abs(rn[i] - rr[i]) for i in qi])
        res.update({"n_queries_compared": len(qi), "queries_identical_rank": int((d == 0).sum()),
                    "max_abs_rank_diff": float(d.max()) if len(d) else None,
                    "micro_MRR_new": float(np.mean([1 / rn[i] for i in qi])),
                    "micro_MRR_ref": float(np.mean([1 / rr[i] for i in qi]))})
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

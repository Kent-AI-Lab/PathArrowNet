#!/usr/bin/env python3
"""
Relation-level KGML weighting (Sec. 4.4, Suppl. Table S19): PathArrowNet with and without the weights, five strict splits,
all test task-A queries and two strata of test relations: those produced by an arrow between two single-KO boxes (smallest KO
cartesian factor 1 in some map) and those that exist only through box expansion (factor > 1 in every map). Single models:
per-query RR averaged over the five seeds; seed-averaged: ranks of the mean row z-score of the five seeds. Pooled pathway-macro
levels and differences (10,000 cluster resamples).

    python evaluation/relation_weight_strata.py  -> results/evaluation/relation_weight_strata.json
"""
import csv
from collections import defaultdict

import numpy as np

from common import CS, PAN, PAN0, PW, STS, SL, ZS, write

SEEDS = [f"s{s}" for s in range(42, 47)]


def box_factor():
    """smallest KO cartesian factor of each KO edge per map (directional gene -> gene KGML relations)"""
    csv.field_size_limit(10 ** 9); KD = PW / "kegg_dataset"; kos = {}
    with (KD / "node_occurrences.csv").open() as f:
        for r in csv.DictReader(f):
            if r["entry_type"] == "gene":
                s = set((r["derived_ko_ids_from_hsa_genes"] or r["ko_ids"] or "").replace(";", " ").split())
                kos[(r["pathway_id"], r["entry_id"])] = {("ko:" + k if not k.startswith("ko:") else k) for k in s}
    factor = {}
    with (KD / "kgml_relations.csv").open() as f:
        for r in csv.DictReader(f):
            if r["whether_directional"] != "1" or r["source_entry_type"] != "gene" or r["target_entry_type"] != "gene":
                continue
            p = r["pathway_id"]; S = kos.get((p, r["source_entry"]), set()); T = kos.get((p, r["target_entry"]), set())
            for a in S:
                for b in T:
                    if a != b:
                        k = (p, a, b); factor[k] = min(factor.get(k, 10 ** 9), len(S) * len(T))
    edge_pw = {}
    with (KD / "splits_ko" / "edge_split.csv").open() as f:
        for r in csv.DictReader(f):
            edge_pw[("ko:" + r["source_ko"], "ko:" + r["target_ko"])] = r["pathways"].split(";") if r["pathways"] else []
    return factor, edge_pw


def main():
    factor, edge_pw = box_factor()
    pool = defaultdict(dict); counts = defaultdict(int)
    for st in STS:
        voc, V = SL.vocab(st); inv = {i: t for t, i in voc.items()}
        qa = SL.load_queries(st, "test", "A")
        fs = []
        for q in qa:
            a, b = inv[q["source"]], inv[q["target"]]
            a = a if a.startswith("ko:") else "ko:" + a; b = b if b.startswith("ko:") else "ko:" + b
            vals = [factor[(p, a, b)] for p in edge_pw.get((a, b), []) if (p, a, b) in factor]
            fs.append(min(vals) if vals else 0)
        fs = np.array(fs)
        strata = {"all": fs >= 0, "single_box": fs == 1, "expanded_only": fs > 1}
        rr = {}
        for arm, m in (("weighted", PAN), ("unweighted", PAN0)):
            Zs = [SL.query_scores(SL.SCORES / st / m / s, "test", "A", qa).astype(np.float64) for s in SEEDS]
            rr[(arm, "single")] = (1 / np.array([SL.ranks(z, qa, V)[0] for z in Zs])).mean(0)
            rr[(arm, "seedavg")] = 1 / np.asarray(SL.ranks(np.mean([ZS(z) for z in Zs], 0), qa, V)[0], float)
        for sn, sm in strata.items():
            idx = np.where(sm)[0]; qi = [qa[i] for i in idx]; counts[sn] += len(idx)
            for est in ("single", "seedavg"):
                for arm in ("weighted", "unweighted"):
                    pool[f"level|{arm}|{est}|{sn}"][st] = (rr[(arm, est)][idx], np.zeros(len(idx)), qi)
                pool[f"diff|weighted-unweighted|{est}|{sn}"][st] = (rr[("weighted", est)][idx], rr[("unweighted", est)][idx], qi)
    out = {"n_queries": dict(counts), "pooled": {k: CS.pooled(v) for k, v in pool.items()}}
    for k, v in out["pooled"].items():
        print(f"{k:45s} {v['est']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]", flush=True)
    write("relation_weight_strata.json", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Performance by STRING evidence between source and target (Sec. 4.3, Suppl. Table S11), five strict splits.
Strata are defined on the split's training KEGG edges plus the STRING experimental (>= 150) background:
    direct          the pair is a STRING background edge
    direct_human    direct, and the direct human experimental sub-score of the pair is >= 150
    direct_transfer direct, but only orthology-transferred evidence passes the threshold
    no_direct       not direct;  indirect: graph distance <= 5;  unreachable: distance > 5 (BFS cap 5)
Single models are per-query RRs averaged over seeds; *_seedavg dumps are single matrices. Per split: common_stats.boot on the
stratum; pooled: common_stats.pooled on the stratum subsets of the five splits.

    python evaluation/evidence_strata.py         -> results/evaluation/evidence_strata.json
"""
import csv
from collections import defaultdict, deque

import numpy as np

from common import ADA, ADA_SA, CS, PAN, PAN_SA, PW, REP, REP_SA, STRING_FILE, STS, SPLIT, SL, boot, write

METHODS = [REP, ADA, REP_SA, ADA_SA, PAN, PAN_SA, "baseline_bg_neighbor"]
DIFFS = [  # (new, reference, tag)
    (PAN_SA, REP_SA, "PathArrowNet - reproduced BioPathNet (seed-averaged)"),
    (PAN_SA, ADA_SA, "PathArrowNet - BioPathNet with our adaptations (seed-averaged)"),
    (PAN, REP, "PathArrowNet - reproduced BioPathNet (single models)"),
    (PAN, ADA, "PathArrowNet - BioPathNet with our adaptations (single models)"),
    (PAN_SA, REP, "PathArrowNet (seed-averaged) - reproduced BioPathNet (per-seed mean)"),
    (PAN_SA, ADA, "PathArrowNet (seed-averaged) - BioPathNet with our adaptations (per-seed mean)"),
]
STRATA = ["all", "direct", "direct_human", "direct_transfer", "no_direct", "indirect", "unreachable"]


def rr(st, m, q, V):
    runs = sorted(p for p in (SL.SCORES / st / m).iterdir() if p.is_dir())
    return np.mean([1 / SL.ranks(SL.query_scores(r, "test", "A", q).astype(np.float64), q, V)[0] for r in runs], 0)


def load_bg(voc):
    bg = set()
    for line in STRING_FILE.read_text().splitlines()[1:]:
        a, b = line.split("\t")[:2]; bg.add((voc[a], voc[b]))
    return bg


def load_prov():
    d = {}
    with (PW / "kegg_dataset_cache/string/ko_pair_experimental_provenance.tsv").open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            d[(r["ko_a"], r["ko_b"])] = (int(r["max_exp_direct"]), int(r["max_exp_transferred"]), int(r["max_experimental_score"]))
    return d


def strata_masks(st, q, voc, tok, prov):
    tr = [(int(r["context_token_id"]), int(r["target_token_id"])) for r in SL.rd(SL.SPLIT_DIRS[st][0] / "train_A_next_node.csv.gz")]
    bg = load_bg(voc)
    G = defaultdict(set)
    for a, b in list(tr) + list(bg):
        G[a].add(b); G[b].add(a)

    def dist(s, t, cap=5):
        seen, dq = {s}, deque([(s, 0)])
        while dq:
            x, d = dq.popleft()
            if d >= cap:
                continue
            for y in G[x]:
                if y == t:
                    return d + 1
                if y not in seen:
                    seen.add(y); dq.append((y, d + 1))
        return 99
    D = np.array([dist(z["source"], z["target"]) for z in q])
    direct = np.array([((z["source"], z["target"]) in bg) or ((z["target"], z["source"]) in bg) for z in q])
    human = np.zeros(len(q), bool)
    for i, z in enumerate(q):
        if direct[i]:
            a, b = tok[z["source"]].replace("ko:", ""), tok[z["target"]].replace("ko:", "")
            p = prov.get((a, b)) or prov.get((b, a))
            if p is not None:
                human[i] = p[0] >= 150
    return {"direct": direct, "indirect": (~direct) & (D <= 5), "unreachable": (~direct) & (D > 5), "no_direct": ~direct,
            "direct_human": direct & human, "direct_transfer": direct & ~human, "all": np.ones(len(q), bool)}


def main():
    prov = load_prov()
    res, pool = {}, {s: {t: {} for _, _, t in DIFFS} for s in STRATA}
    levels = {s: {m: [] for m in METHODS} for s in STRATA}
    for st in STS:
        voc, V = SL.vocab(st); tok = {i: t for t, i in voc.items()}
        q = SL.load_queries(st, "test", "A")
        R = {m: rr(st, m, q, V) for m in METHODS}
        masks = strata_masks(st, q, voc, tok, prov)
        res[SPLIT[st]] = {}
        for s in STRATA:
            m = masks[s]; idx = np.where(m)[0]
            d = {"n_queries": int(m.sum()), "n_pathways": len({q[i]["cluster"] for i in idx}), "mrr": {k: float(R[k][m].mean()) for k in METHODS}}
            for a, b, t in DIFFS:
                d[t] = boot(R[a], R[b], q, mask=m)
                pool[s][t][st] = (R[a][idx], R[b][idx], [q[i] for i in idx])
            for k in METHODS:
                levels[s][k].append(R[k][m])
            res[SPLIT[st]][s] = d
    res["pooled"] = {}
    for s in STRATA:
        d = {"n_queries": int(sum(len(v) for v in levels[s][METHODS[0]])),
             "mrr_query_level_over_five_splits": {k: float(np.concatenate(levels[s][k]).mean()) for k in METHODS}}
        for _, _, t in DIFFS:
            d[t] = CS.pooled(pool[s][t])
        res["pooled"][s] = d
        print(s, d["n_queries"], " ".join(f"{t[:40]} {d[t]['est']:+.4f} [{d[t]['lo']:+.4f}, {d[t]['hi']:+.4f}]" for _, _, t in DIFFS[:2]), flush=True)
    write("evidence_strata.json", res)


if __name__ == "__main__":
    main()

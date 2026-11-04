#!/usr/bin/env python3
"""
Test MRR by biology (Suppl. Table S3): strict split 1, task A, grouped by (a) the KGML relation subtypes that produced the
test KO edge (from gene_edges.csv mapped to KO pairs; an edge can carry several subtypes) and (b) the KEGG BRITE class of the
query's first pathway. Per-query RRs are averaged over seeds; the difference seed-averaged PathArrowNet minus reproduced
BioPathNet (per-seed mean) gets a 95% pathway-cluster bootstrap interval (2,000 resamples, one generator for all groups in
the order below). Groups with fewer than 20 queries are omitted.

    python evaluation/error_by_type.py           -> results/evaluation/error_by_type.json
"""
import collections
import csv
import gzip
import itertools

import numpy as np

from common import PAN_SA, PW, REP, SL, write

SETTING = "strict_v1_full"
METHODS = {REP: "reproduced BioPathNet", "baseline_bg_neighbor": "STRING neighbor", PAN_SA: "PathArrowNet"}
FLAGS = {
    "activation": lambda s: "activation" in s, "inhibition": lambda s: "inhibition" in s,
    "activation only (no inhibition/PTM/expression)": lambda s: s <= {"activation", "indirect effect"} and "activation" in s,
    "inhibition only (no activation/PTM/expression)": lambda s: s <= {"inhibition", "indirect effect"} and "inhibition" in s,
    "phosphorylation / dephosphorylation": lambda s: bool(s & {"phosphorylation", "dephosphorylation"}),
    "ubiquitination / other PTM": lambda s: bool(s & {"ubiquitination", "glycosylation", "methylation"}),
    "expression / repression (transcriptional)": lambda s: bool(s & {"expression", "repression"}),
    "indirect effect": lambda s: "indirect effect" in s,
    "state change": lambda s: "state change" in s,
}


def main():
    _, V = SL.vocab(SETTING)
    Q = SL.load_queries(SETTING, "test", "A")
    raw = list(csv.DictReader(gzip.open(SL.SPLIT_DIRS[SETTING][0] / "test_A_next_node.csv.gz", "rt")))
    assert len(raw) == len(Q)
    RR = {}
    for m in METHODS:
        runs = sorted(p for p in (SL.SCORES / SETTING / m).iterdir() if p.is_dir())
        RR[m] = np.mean([1 / SL.ranks(SL.query_scores(r, "test", "A", Q).astype(np.float64), Q, V)[0] for r in runs], 0)

    g2ko = collections.defaultdict(set)
    for r in csv.DictReader(open(PW / "kegg_dataset" / "ko_mapping.csv")):
        for g in r["human_genes_all"].split(";"):
            if g.strip():
                g2ko[g.strip()].add(r["ko_id"])
    sub = collections.defaultdict(set)
    for r in csv.DictReader(open(PW / "kegg_dataset" / "gene_edges.csv")):
        for a, b in itertools.product(g2ko.get(r["source_gene"], ()), g2ko.get(r["target_gene"], ())):
            sub[(a, b)] |= set(r["relation_subtype"].split(";"))
    edge_sub = [sub.get((q["context_token"].replace("ko:", ""), q["target_token"].replace("ko:", "")), set()) for q in raw]

    cl = np.array([q["cluster"] for q in Q])
    rng = np.random.default_rng(0)

    def ci(idx):
        idx = np.asarray(idx); c = cl[idx]; u = np.unique(c); pos = {k: idx[c == k] for k in u}
        d0 = RR[PAN_SA][idx].mean() - RR[REP][idx].mean(); ds = []
        for _ in range(2000):
            s = np.concatenate([pos[k] for k in rng.choice(u, len(u))])
            ds.append(RR[PAN_SA][s].mean() - RR[REP][s].mean())
        return float(d0), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))

    def table(groups):
        rows = {}
        for name, idx in groups:
            idx = np.asarray(idx)
            if len(idx) < 20:
                continue
            d, lo, hi = ci(idx)
            rows[name] = {"n_queries": len(idx), "n_pathways": len(set(cl[idx])), **{lab: float(RR[m][idx].mean()) for m, lab in METHODS.items()},
                          "PathArrowNet_minus_reproduced_BioPathNet": {"est": d, "lo": lo, "hi": hi}}
        return rows

    out = {"relation_subtype": table([("all test edges", np.arange(len(Q)))] + [(k, [i for i, s in enumerate(edge_sub) if s and f(s)]) for k, f in FLAGS.items()])}
    pc = {}
    for r in csv.DictReader(open(PW / "kegg_dataset" / "node_occurrences.csv")):
        pc[r["pathway_id"]] = (r["pathway_class_a"], r["pathway_class_b"])
    ca = [pc[q["pathway_ids"].split(";")[0]][0] for q in raw]; cb = [pc[q["pathway_ids"].split(";")[0]] for q in raw]
    groups_a = [(k, [i for i, x in enumerate(ca) if x == k]) for k in sorted(set(ca), key=lambda k: -ca.count(k))]
    groups_b = [(f"{k[0]} / {k[1]}", [i for i, x in enumerate(cb) if x == k]) for k in sorted(set(cb), key=lambda k: -cb.count(k))]
    out["pathway_class"] = table(groups_a + groups_b)
    write("error_by_type.json", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Does cross-pathway duplication (the main KEGG leakage source) occur in other pathway databases?

Input: Pathway Commons v12 extended SIF per data source (kegg_dataset_cache/pathwaycommons/PathwayCommons12.<src>.hgnc.txt.gz),
columns PARTICIPANT_A, INTERACTION_TYPE, PARTICIPANT_B, ..., PATHWAY_NAMES. Directed gene->gene relations only:
  regulatory  = controls-state-change-of, controls-expression-of, controls-phosphorylation-of, controls-transport-of
  catalysis   = catalysis-precedes (consecutive enzymes in metabolic / reaction chains)
For every source and relation family:
  unique directed gene edges, occurrences (edge x pathway), share of unique edges in >1 pathway,
  leakage under two common protocols (seed 0, 10 repetitions, mean):
    random split of edge occurrences 80/10/10 -> share of test occurrences whose edge also occurs in train
    random pathway holdout 80/10/10 (by pathway) -> share of test-pathway edges that also occur in a train pathway
  ortholog expansion: unique gene edges / unique KO edges (genes mapped to KO through KEGG link/ko/hsa)
    .venv/bin/python benchmark_v2/generalization/pc_duplication.py
"""
import csv
import gzip
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PC = ROOT / "kegg_dataset_cache/pathwaycommons"
SOURCES = ["reactome", "pid", "panther", "netpath", "pathbank", "humancyc", "inoh", "kegg"]
FAM = {"regulatory": {"controls-state-change-of", "controls-expression-of", "controls-phosphorylation-of",
                      "controls-transport-of"},
       "catalysis": {"catalysis-precedes"}}


def symbol_to_ko():
    sym2gene = {}
    for line in (ROOT / "kegg_dataset_cache/kegg_rest/list_hsa.tsv").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 4 or not parts[0].startswith("hsa:"):
            continue
        g = parts[0][4:]
        for s in parts[3].split(";")[0].split(","):
            sym2gene.setdefault(s.strip(), g)
    g2k = defaultdict(set)
    for line in (ROOT / "kegg_dataset_cache/kegg_rest/link_ko_hsa.tsv").read_text().splitlines():
        x, y = line.split()
        g2k[x[4:]].add(y[3:])
    return {s: g2k[g] for s, g in sym2gene.items() if g in g2k}


def load(src, types):
    occ = defaultdict(set)   # edge -> pathways
    with gzip.open(PC / f"PathwayCommons12.{src}.hgnc.txt.gz", "rt") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if len(row) < 6:
                break                      # second section (participant table) starts after an empty line
            a, t, b, pws = row[0], row[1], row[2], row[5]
            if t not in types or a == b or not pws:
                continue
            for p in pws.split(";"):
                if p:
                    occ[(a, b)].add(p)
    return occ


def leakage(occ, reps=10):
    items = [(e, p) for e, ps in occ.items() for p in ps]
    pws = sorted({p for ps in occ.values() for p in ps})
    r_occ, r_pw = [], []
    for s in range(reps):
        rng = random.Random(s)
        it = items[:]
        rng.shuffle(it)
        n = len(it)
        train = {e for e, _ in it[:int(0.8 * n)]}
        test = it[int(0.9 * n):]
        r_occ.append(sum(e in train for e, _ in test) / max(1, len(test)))
        pw = pws[:]
        rng.shuffle(pw)
        m = len(pw)
        trp, tep = set(pw[:int(0.8 * m)]), set(pw[int(0.9 * m):])
        tr_e = {e for e, ps in occ.items() if ps & trp}
        te_e = [e for e, ps in occ.items() if ps & tep]
        r_pw.append(sum(e in tr_e for e in te_e) / max(1, len(te_e)))
    return sum(r_occ) / reps, sum(r_pw) / reps


def main():
    s2k = symbol_to_ko()
    out = []
    for fam, types in FAM.items():
        for src in SOURCES:
            occ = load(src, types)
            if len(occ) < 50:
                continue
            n_pw = len({p for ps in occ.values() for p in ps})
            multi = sum(len(ps) > 1 for ps in occ.values()) / len(occ)
            n_occ = sum(len(ps) for ps in occ.values())
            lo, lp = leakage(occ)
            ko = {(x, y) for a, b in occ for x in s2k.get(a, ()) for y in s2k.get(b, ()) if x != y}
            mapped = [e for e in occ if s2k.get(e[0]) and s2k.get(e[1])]
            row = {"family": fam, "source": src, "pathways": n_pw, "unique_gene_edges": len(occ), "occurrences": n_occ,
                   "share_edges_in_gt1_pathway": round(multi, 3),
                   "random_occurrence_split_test_in_train": round(lo, 3),
                   "random_pathway_holdout_test_in_train": round(lp, 3),
                   "gene_edges_mapped_to_KO": len(mapped), "unique_KO_edges": len(ko),
                   "gene_per_KO_edge": round(len(mapped) / max(1, len(ko)), 2)}
            out.append(row)
            print(row, flush=True)
    p = ROOT / "benchmark_v2/generalization/pc_duplication.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print("wrote", p)


if __name__ == "__main__":
    main()

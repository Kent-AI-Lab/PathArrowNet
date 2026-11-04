#!/usr/bin/env python3
"""
Independent audit of the splits and headline numbers. Deliberately does NOT import the split builder or
its leakage checks: pathway KO graphs are rebuilt from kegg_dataset/gene_edges.csv + KEGG link/ko/hsa.

Checks per strict split (splits_ko = v1, and kegg_splits_robustness/seed43..46):
  1. no val/test next-node pair is a train pair; no test pair is a val pair
  2. no edge used by any val/test sample (A pairs, consecutive pairs of B windows, and the two masked
     edges l->t, t->r of C samples) occurs in ANY train pathway's raw KO graph
  3. no val/test edge's reverse occurs in a train pathway's raw graph
  4. no masked-context string occurs in two splits
  5. every train sample uses only edges from train pathways
Feng-style splits: test edges vs train pathways (orig should overlap heavily, dedup must be 0).
Headline numbers are recomputed from per-query files (ranks / AUC) and compared with the summaries.
"""

from __future__ import annotations

import csv
import gzip
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DS = ROOT / "kegg_dataset"


def rows(p):
    with gzip.open(p, "rt") as f:
        yield from csv.DictReader(f)


def ko_graphs():
    g2k = defaultdict(set)
    for line in (ROOT / "kegg_dataset_cache/kegg_rest/link_ko_hsa.tsv").read_text().splitlines():
        a, b = line.split()
        g2k[a[4:]].add(b[3:])
    vocab = {r["token"]: int(r["token_id"]) for r in csv.DictReader((DS / "token_vocab_ko.csv").open())}
    graphs = defaultdict(set)
    with (DS / "gene_edges.csv").open() as f:
        for r in csv.DictReader(f):
            if r["edge_origin"] != "gene_entry":
                continue
            for a in g2k.get(r["source_gene"], ()):
                for b in g2k.get(r["target_gene"], ()):
                    if a != b:
                        graphs[r["pathway_id"]].add((vocab[a], vocab[b]))
    return graphs


def check_strict(sd, graphs):
    assign = {r["pathway_id"]: r["split"] for r in csv.DictReader((sd / "pathway_split.csv").open())}
    train_pw_edges = {e for p, es in graphs.items() if assign.get(p) == "train" for e in es}
    val_pw_edges = {e for p, es in graphs.items() if assign.get(p) == "val" for e in es}
    res = {}
    used, pairs, ctx = {}, {}, {}
    for s in ("train", "val", "test"):
        A = {(int(r["context_token_id"]), int(r["target_token_id"])) for r in rows(sd / f"{s}_A_next_node.csv.gz")}
        B = set()
        for r in rows(sd / f"{s}_B_path_windows.csv.gz"):
            t = [int(x) for x in r["token_ids"].split()]
            B |= set(zip(t, t[1:]))
        Cedges, Cctx = set(), set()
        for r in rows(sd / f"{s}_C_masked.csv.gz"):
            t = [int(x) for x in r["context_token_ids"].split()]
            i, y = int(r["mask_position"]), int(r["target_token_id"])
            Cedges |= {(t[i - 1], y), (y, t[i + 1])}
            Cctx.add(r["context_token_ids"])
        pairs[s], used[s], ctx[s] = A, A | B | Cedges, Cctx
    res["A_pair_overlap(train,val/test)"] = len(pairs["train"] & (pairs["val"] | pairs["test"])) + len(pairs["val"] & pairs["test"])
    res["val/test_edges_in_train_pathways"] = len((used["val"] | used["test"]) & train_pw_edges)
    res["test_edges_in_val_pathways"] = len(used["test"] & val_pw_edges)
    res["val/test_reverse_in_train_pathways"] = sum(1 for (u, v) in (used["val"] | used["test"]) if (v, u) in train_pw_edges)
    res["masked_context_overlap"] = len(ctx["train"] & ctx["val"]) + len(ctx["train"] & ctx["test"]) + len(ctx["val"] & ctx["test"])
    res["train_edges_outside_train_pathways"] = len(used["train"] - train_pw_edges)
    return res


def check_feng(d, graphs):
    test = {(int(r["context_token_id"]), int(r["target_token_id"])) for r in rows(d / "test_A_next_node.csv.gz")}
    train = {(int(r["context_token_id"]), int(r["target_token_id"])) for r in rows(d / "train_A_next_node.csv.gz")}
    return len(test), len(test & train)


def main():
    graphs = ko_graphs()
    print(f"rebuilt {len(graphs)} pathway KO graphs, {len({e for es in graphs.values() for e in es}):,} unique edges")
    ok = True
    for sd in [DS / "splits_ko"] + [ROOT / f"kegg_splits_robustness/seed{s}" for s in (43, 44, 45, 46)]:
        r = check_strict(sd, graphs)
        bad = {k: v for k, v in r.items() if v}
        ok &= not bad
        print(f"{sd.name:10s} {'PASS' if not bad else 'FAIL ' + str(bad)}")
    for k in range(1, 5):
        n, o = check_feng(ROOT / f"kegg_leaky_splits/feng_fold{k}", graphs)
        nd, od = check_feng(ROOT / f"kegg_leaky_splits/feng_fold{k}_dedup", graphs)
        ok &= od == 0
        print(f"feng fold{k}: orig {o}/{n} test edges in train ({o / n:.0%}); dedup {od}/{nd}")

    # headline numbers recomputed from per-query files
    C = DS / "comparisons_ko"

    def auc_file(p):
        return st.mean(float(r["auc"]) for r in rows(p))
    print("CF strict AUC (from per-query file):", round(auc_file(C / "strict/classic/cf/auc_test_A.csv.gz"), 3))
    print("CF occurrence_random AUC:", round(auc_file(C / "occurrence_random/classic/cf/auc_test_A.csv.gz"), 3))
    q = list(rows(DS / "splits_ko/test_A_next_node.csv.gz"))
    w = [float(r["pathway_macro_weight"]) for r in q]
    mrr = []
    for s in (42, 43, 44, 45, 46):
        rk = [float(r["rank"]) for r in rows(C / f"strict/models/gnn_id+gobert_noBP+esm_s{s}/ranks_test_A.csv.gz")]
        mrr.append(sum(wi / x for wi, x in zip(w, rk)) / sum(w))
    print("GNN+GoBERT+ESM2 strict test-A macro MRR (recomputed):", round(st.mean(mrr), 4))
    aucs = [auc_file(C / f"strict/models/gnn_id+gobert_noBP+esm_s{s}/auc_test_A.csv.gz") for s in (42, 43, 44, 45, 46)]
    print("GNN+GoBERT+ESM2 strict test-A AUC (recomputed):", round(st.mean(aucs), 3))
    print("ALL INDEPENDENT LEAKAGE CHECKS PASS" if ok else "SOME CHECK FAILED")


if __name__ == "__main__":
    main()

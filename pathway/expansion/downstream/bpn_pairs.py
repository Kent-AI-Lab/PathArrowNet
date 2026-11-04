#!/usr/bin/env python3
"""Gene-pair zero-shot scores of the original BioPathNet (cache/bpn_orig_ko_scores.npz from bpn_ko_scores.py) for the
LR / Reactome tasks, in the cache format of kegg_pairscore.gene_pair_scores (symmetric max over KO pairs and both
orientations): cache/<task>_strict_v1_full_bpn_orig_s0.npz. string_baseline.py then scores it with the others.
    .venv/bin/python expansion/downstream/bpn_pairs.py"""
import csv
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
z = np.load(HERE / "cache/bpn_orig_ko_scores.npz"); A, S = z["anchors"], z["S"]
row = {int(a): i for i, a in enumerate(A)}
with (HERE.parents[1] / "kegg_dataset/token_vocab_ko_full.csv").open() as f:
    voc = {r["token"]: int(r["token_id"]) for r in csv.DictReader(f)}
for task, gf in (("lr", "data/ligand_receptor/lr_genes.tsv"), ("reactome", "data/reactome/reactome_genes.tsv")):
    genes = list(csv.DictReader((HERE / gf).open(), delimiter="\t"))
    ks = [[voc[k] for k in (g["kos_in_vocab"] or "").split(";") if k in voc] for g in genes]
    has = np.array([len(k) > 0 for k in ks]); n = len(ks)
    P = np.zeros((n, n), np.float32)
    for i in range(n):
        if not ks[i]:
            continue
        R = S[[row[k] for k in ks[i]]]  # [|KO(i)|, V]
        for j in range(n):
            if ks[j] and j != i:
                P[i, j] = R[:, ks[j]].max()
    P = np.maximum(P, P.T); P[~has] = 0; P[:, ~has] = 0
    np.savez(HERE / f"cache/{task}_strict_v1_full_bpn_orig_s0.npz", P=P, has=has)
    print(task, n, int(has.sum()))

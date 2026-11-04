#!/usr/bin/env python3
"""
KO-level background graph from STRING v12 human, EXPERIMENTS channel only.

The 'database' channel of STRING imports curated pathway databases including KEGG, and Pathway Commons
(BioPathNet's background graph) also integrates KEGG, so both could contain our test edges. We keep only
the experimental channel (BioGRID, IntAct, ... incl. orthology transfer), score >= --min-score.

    python build_string_background.py            -> kegg_dataset_cache/string/ko_background_exp{score}.tsv
                                                    (+ _no_evalpairs variant without any val/test KO pair)
Columns: ko_a, ko_b (unordered, ko_a < ko_b), max experimental score, number of supporting gene pairs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
S = ROOT / "kegg_dataset_cache/string"
DS = ROOT / "kegg_dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=int, default=400)
    a = ap.parse_args()
    ensp2gene = {}
    with gzip.open(S / "9606.protein.aliases.v12.0.txt.gz", "rt") as f:
        next(f)
        for line in f:
            p, alias, src = line.rstrip("\n").split("\t")
            if src == "KEGG_GENEID" and alias.isdigit():
                ensp2gene.setdefault(p, alias)
    g2k = defaultdict(set)
    for line in (ROOT / "kegg_dataset_cache/kegg_rest/link_ko_hsa.tsv").read_text().splitlines():
        x, y = line.split()
        g2k[x[4:]].add(y[3:])
    vocab = {r["token"] for r in csv.DictReader((DS / "token_vocab_ko.csv").open()) if not r["token"].startswith("[")}
    pairs = defaultdict(lambda: [0, 0])
    n_links = n_kept = 0
    with gzip.open(S / "9606.protein.links.detailed.v12.0.txt.gz", "rt") as f:
        next(f)
        for line in f:
            c = line.split()
            n_links += 1
            exp = int(c[6])
            if exp < a.min_score:
                continue
            g1, g2 = ensp2gene.get(c[0]), ensp2gene.get(c[1])
            if not g1 or not g2:
                continue
            n_kept += 1
            for k1 in g2k.get(g1, ()):
                for k2 in g2k.get(g2, ()):
                    if k1 == k2 or k1 not in vocab or k2 not in vocab:
                        continue
                    key = tuple(sorted((k1, k2)))
                    pairs[key][0] = max(pairs[key][0], exp)
                    pairs[key][1] += 1
    # overlap with our split edges (as unordered pairs)
    split_pairs = {}
    for s in ("train", "val", "test"):
        with gzip.open(DS / f"splits_ko/{s}_A_next_node.csv.gz", "rt") as f:
            split_pairs[s] = {tuple(sorted((r["context_token"][3:], r["target_token"][3:]))) for r in csv.DictReader(f)}
    bg = set(pairs)
    print(f"STRING links {n_links:,}; experimental >= {a.min_score} with KEGG gene ids: {n_kept:,}")
    print(f"KO background pairs: {len(bg):,}")
    for s in ("train", "val", "test"):
        ov = len(split_pairs[s] & bg)
        print(f"  {s} KO pairs also in background: {ov:,} / {len(split_pairs[s]):,} ({ov / len(split_pairs[s]):.1%})")
    evalpairs = split_pairs["val"] | split_pairs["test"]
    for name, keep in ((f"ko_background_exp{a.min_score}.tsv", bg),
                       (f"ko_background_exp{a.min_score}_no_evalpairs.tsv", bg - evalpairs)):
        with (S / name).open("w") as f:
            f.write("ko_a\tko_b\tmax_experimental_score\tn_gene_pairs\n")
            for k in sorted(keep):
                f.write(f"{k[0]}\t{k[1]}\t{pairs[k][0]}\t{pairs[k][1]}\n")
        print(f"wrote {name}: {len(keep):,} pairs")


if __name__ == "__main__":
    main()

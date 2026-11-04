#!/usr/bin/env python3
"""
Full candidate space for the strict benchmark: every KO that appears anywhere in the 372 human KEGG maps
(11,620 KOs), instead of the 2,870 KOs that take part in a directed KO->KO edge.

Why: the 2,870-KO vocabulary is defined from ALL directed edges, including val/test edges, so every candidate KO
that is absent from the train graph (530) is a val/test node. A model that simply does not push down
non-train KOs then gets a free advantage on unseen targets. With all map KOs as candidates, 8,750 extra KOs that
never have a directed edge compete with them.

    .venv/bin/python benchmark_v2/full/build_full_candidates.py

Writes (originals untouched):
  kegg_dataset/token_vocab_ko_full.csv          ids 0..2872 identical to token_vocab_ko.csv, extra KOs appended (sorted)
  kegg_features/ko_features_full.pt             same features / aggregation as build_protein_features.py --part ko
  kegg_dataset_cache/string/ko_background_exp400_full.tsv   STRING experiments >= 400, over the full vocabulary
and checks that the first 2,873 feature rows and the background restricted to the old vocabulary are identical
to the existing files.
"""

import csv
import gzip
import re
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "kegg_dataset"
FE = ROOT / "kegg_features"
S = ROOT / "kegg_dataset_cache/string"


def rd(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def build_vocab():
    old = rd(DS / "token_vocab_ko.csv")
    assert [int(r["token_id"]) for r in old] == list(range(len(old)))
    ko_any = set()
    for r in rd(DS / "node_occurrences.csv"):
        # fields are ";"-separated
        ko_any |= {k for k in re.split(r"[;\s]+", r["ko_ids"] + ";" + r["derived_ko_ids_from_hsa_genes"]) if k}
    ko_any = {k.replace("ko:", "") for k in ko_any}
    old_tokens = [r["token"] for r in old]
    assert set(old_tokens[3:]) <= ko_any, "old vocabulary must be a subset of the map KOs"
    names = {r["ko_id"]: r["ko_name"] for r in rd(DS / "ko_mapping.csv")}
    extra = sorted(ko_any - set(old_tokens))
    rows = [dict(r) for r in old] + [{"token_id": str(len(old) + i), "token": k, "ko_name": names.get(k, "")}
                                     for i, k in enumerate(extra)]
    with (DS / "token_vocab_ko_full.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["token_id", "token", "ko_name"])
        w.writeheader()
        w.writerows(rows)
    print(f"vocab: {len(old) - 3} old + {len(extra)} extra = {len(rows) - 3} KO tokens (+3 special)")
    return {r["token"]: int(r["token_id"]) for r in rows}, len(old)


def build_features(vocab, n_old):
    gb = torch.load(FE / "gene_gobert.pt", weights_only=False)
    es = torch.load(FE / "gene_esm2.pt", weights_only=False)
    members = {}
    for r in rd(DS / "ko_mapping.csv"):
        if r["ko_id"] in vocab:
            members[vocab[r["ko_id"]]] = [g for g in r["human_genes_in_pathways"].split(";") if g]
    V = len(vocab)
    sources = {"gobert_all_mean": gb["all"]["mean"], "gobert_all_cls": gb["all"]["cls"],
               "gobert_noBP_mean": gb["noBP"]["mean"], "gobert_noBP_cls": gb["noBP"]["cls"],
               "esm2_mean": es["mean"]}
    old = torch.load(FE / "ko_features.pt", weights_only=False)
    feats = {}
    for name, table in sources.items():
        dim = next(iter(table.values())).shape[0]
        X, has, ngen = torch.zeros(V, dim), torch.zeros(V, dtype=torch.bool), torch.zeros(V, dtype=torch.long)
        for k, gs in members.items():
            vs = [table[g] for g in gs if g in table]
            if vs:
                X[k] = torch.stack(vs).mean(0)
                has[k] = True
                ngen[k] = len(vs)
        o = old["features"][name]
        assert torch.equal(has[:n_old], o["has"]) and torch.allclose(X[:n_old], o["X"]), f"{name}: old rows differ"
        feats[name] = {"X": X, "has": has, "n_genes": ngen}
        print(f"  {name}: {int(has[3:].sum())} / {V - 3} KOs with feature "
              f"(old vocab {int(has[3:n_old].sum())} / {n_old - 3}; extra {int(has[n_old:].sum())} / {V - n_old})")
    torch.save({"features": feats, "vocab_size": V, "sources": old["sources"],
                "vocab_file": "kegg_dataset/token_vocab_ko_full.csv"}, FE / "ko_features_full.pt")


def build_background(vocab, n_old, min_score=400):
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
    kos = {k for k in vocab if not k.startswith("[")}
    pairs = defaultdict(lambda: [0, 0])
    with gzip.open(S / "9606.protein.links.detailed.v12.0.txt.gz", "rt") as f:
        next(f)
        for line in f:
            c = line.split()
            exp = int(c[6])
            if exp < min_score:
                continue
            g1, g2 = ensp2gene.get(c[0]), ensp2gene.get(c[1])
            if not g1 or not g2:
                continue
            for k1 in g2k.get(g1, ()):
                for k2 in g2k.get(g2, ()):
                    if k1 == k2 or k1 not in kos or k2 not in kos:
                        continue
                    key = tuple(sorted((k1, k2)))
                    pairs[key][0] = max(pairs[key][0], exp)
                    pairs[key][1] += 1
    out = S / f"ko_background_exp{min_score}_full.tsv"
    with out.open("w") as f:
        f.write("ko_a\tko_b\tmax_experimental_score\tn_gene_pairs\n")
        for k in sorted(pairs):
            f.write(f"{k[0]}\t{k[1]}\t{pairs[k][0]}\t{pairs[k][1]}\n")
    old_kos = {k for k, i in vocab.items() if 3 <= i < n_old}
    restricted = {k: v for k, v in pairs.items() if k[0] in old_kos and k[1] in old_kos}
    with (S / f"ko_background_exp{min_score}.tsv").open() as f:
        next(f)
        old = {(a, b): [int(s), int(n)] for a, b, s, n in (l.split() for l in f)}
    assert restricted == old, "background restricted to the old vocabulary differs from the existing file"
    touched = {x for k in pairs for x in k}
    print(f"background: {len(pairs):,} KO pairs over {len(touched):,} KOs "
          f"(old file {len(old):,} pairs, identical on the old vocabulary)")


def main():
    vocab, n_old = build_vocab()
    build_features(vocab, n_old)
    build_background(vocab, n_old)


if __name__ == "__main__":
    main()

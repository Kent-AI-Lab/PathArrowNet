#!/usr/bin/env python3
"""
Multi-split check of the best model: settings strict_s{43..46}_full = the 4 extra strict splits
(kegg_splits_robustness/seed{43..46}, same build_ko_leakage_free_split.py protocol, other seeds), full 11,620-KO
candidates. For every split this writes
  benchmark_v2/splits/strict_s{seed}/{train,val,test}_S_prefix_next.csv.gz     (same construction as strict_v1)
  biopathnet_repro/BioPathNet/data/KEGG_KO_strict_s{seed}_full_exp150bin4/      (BPN input: split A edges as
      'precedes', STRING experiments >= 150 in 4 bins as background, all 11,620 KOs as entities)
and checks that the token ids of the split files match token_vocab_ko.csv. Settings are registered in scorelib.py.

    .venv/bin/python benchmark_v2/full/build_robustness_settings.py
"""
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())

import csv
import gzip
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B2 = ROOT / "benchmark_v2"
sys.path.insert(0, str(B2))
import build_sequence_task as BS  # noqa: E402

SEEDS = (43, 44, 45, 46)
BPN_DATA = Path(_KH + "/biopathnet_repro/BioPathNet/data")
SRC_BG = BPN_DATA / "KEGG_KO_strict_v1_full_exp150bin4"


def main():
    voc = {r["token"]: int(r["token_id"]) for r in csv.DictReader((ROOT / "kegg_dataset/token_vocab_ko.csv").open())}
    for seed in SEEDS:
        sd = ROOT / "kegg_splits_robustness" / f"seed{seed}"
        name = f"strict_s{seed}"
        # token ids must be those of token_vocab_ko.csv (= first ids of the full vocabulary)
        for s in BS.SPLITS:
            for r in BS.rd(sd / f"{s}_A_next_node.csv.gz"):
                assert voc[r["context_token"][3:]] == int(r["context_token_id"])
                assert voc[r["target_token"][3:]] == int(r["target_token_id"])
        wins = {s: [(tuple(int(x) for x in r["token_ids"].split()), set(r["pathway_ids"].split(";")))
                    for r in BS.rd(sd / f"{s}_B_path_windows.csv.gz")] for s in BS.SPLITS}
        train_edges = {(int(r["context_token_id"]), int(r["target_token_id"]))
                       for r in BS.rd(sd / "train_A_next_node.csv.gz")}
        S = BS.build_S(wins, train_edges, name)
        for s in BS.SPLITS:
            BS.wr(B2 / "splits" / name / f"{s}_S_prefix_next.csv.gz", S[s])
        dst = BPN_DATA / f"KEGG_KO_{name}_full_exp150bin4"
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("train1.txt", "entity_types.txt", "entity_names.txt"):
            shutil.copy(SRC_BG / f, dst / f)
        n = {}
        for ours, theirs in (("train", "train2"), ("val", "valid"), ("test", "test")):
            with gzip.open(sd / f"{ours}_A_next_node.csv.gz", "rt") as f:
                rows = [(r["context_token"][3:], r["target_token"][3:]) for r in csv.DictReader(f)]
            (dst / f"{theirs}.txt").write_text("".join(f"{u}\tprecedes\t{v}\n" for u, v in rows))
            n[theirs] = len(rows)
        print(f"{name}: BPN data {dst.name} {n}", flush=True)


if __name__ == "__main__":
    main()

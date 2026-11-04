#!/usr/bin/env python3
"""
Convert our leakage-controlled KEGG KO split (pathway/kegg_dataset/splits_ko, v1, task A edges) into
BioPathNet's input format:  <src KO> \t precedes \t <dst KO>
    train2.txt / valid.txt / test.txt = our train / val / test next-KO edges (unchanged)
    train1.txt = empty (no background graph; include_factgraph: no)
    entity_types.txt / entity_names.txt = all 2,870 KO tokens of token_vocab_ko.csv (type "KO")

    python leakage/make_kegg_benchmark_data.py  -> BioPathNet/data/KEGG_KO_strict_v1/
"""
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())

import csv
import gzip
from pathlib import Path

SPLIT = Path(_KH + "/pathway/kegg_dataset/splits_ko")
VOCAB = Path(_KH + "/pathway/kegg_dataset/token_vocab_ko.csv")
DST = Path(__file__).resolve().parent.parent / "BioPathNet/data/KEGG_KO_strict_v1"


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for ours, theirs in (("train", "train2"), ("val", "valid"), ("test", "test")):
        with gzip.open(SPLIT / f"{ours}_A_next_node.csv.gz", "rt") as f:
            rows = [(r["context_token"][3:], r["target_token"][3:]) for r in csv.DictReader(f)]
        (DST / f"{theirs}.txt").write_text("".join(f"{u}\tprecedes\t{v}\n" for u, v in rows))
        print(theirs, len(rows))
    (DST / "train1.txt").write_text("")
    toks = [r for r in csv.DictReader(VOCAB.open()) if not r["token"].startswith("[")]
    (DST / "entity_types.txt").write_text("".join(f"{r['token']}\tKO\n" for r in toks))
    (DST / "entity_names.txt").write_text("".join(f"{r['token']}\t{r['token']}\n" for r in toks))
    print("entities", len(toks))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Sequence-context next-node task (S) and a naive random-segment reference split.

    .venv/bin/python benchmark_v2/build_sequence_task.py

S (prefix -> next KO)
    From every path segment (simple path window, L = 3..6) of a split, every prefix of length >= 2 gives one
    sample: context = (k_1 .. k_m), target = k_{m+1}, m = 2..L-1. Samples are deduplicated within the split.
    Prefix length 1 is task A (u -> v) and is kept in the A files. Together, A + S = "given the current KO or
    the sequence of previous KOs, predict the next KO".
    valid_targets_in_split  = every target observed after the same prefix in the same split (branching:
                              one prefix, several correct next KOs, all kept as separate samples)
    known_targets_in_train  = train successors of the LAST context KO (filtered ranking, as in task A)

strict_v1   : segments of kegg_dataset/splits_ko/{split}_B_path_windows (pathway-disjoint + edge ownership),
              so S inherits the strict split. Output: benchmark_v2/splits/strict_v1/{split}_S_prefix_next.csv.gz
random_segment (Setting A, reference only): every segment of the FULL pathway graphs
              (kegg_dataset/samples/ko_B_path_windows.csv.gz) is assigned to train/val/test at random (80/10/10,
              seed 42) -- the "normal random split" of generated segments. Segments of one pathway, their edges and
              their prefixes land in several splits. A files = consecutive pairs of the segments of each split,
              deduplicated within split (an edge can be in train and test). Output: benchmark_v2/splits/random_segment/
"""

from __future__ import annotations

import csv
import gzip
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from build_ko_leakage_free_split import add_macro_weight  # noqa: E402

DS = ROOT / "kegg_dataset"
SPLIT = DS / "splits_ko"
OUT = Path(__file__).resolve().parent / "splits"
SPLITS = ("train", "val", "test")


def rd(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wr(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def build_S(windows_by_split, train_edges, split_name_prefix):
    """windows_by_split: split -> list of (tuple(token ids), set(pathway ids))."""
    train_tokens = {x for e in train_edges for x in e}
    train_out = defaultdict(set)
    for u, v in train_edges:
        train_out[u].add(v)
    out = {}
    for s in SPLITS:
        S_ = defaultdict(set)
        for w, ps in windows_by_split[s]:
            for m in range(2, len(w)):
                S_[(w[:m], w[m])] |= ps
        vt = defaultdict(set)
        for (ctx, t) in S_:
            vt[ctx].add(t)
        rows = [{"split": s, "length": len(ctx), "context_token_ids": " ".join(map(str, ctx)),
                 "target_token_id": t, "n_pathways": len(ps), "pathway_ids": ";".join(sorted(ps)),
                 "target_seen_in_train": int(t in train_tokens),
                 "context_seen_in_train": int(all(x in train_tokens for x in ctx)),
                 "valid_targets_in_split": " ".join(map(str, sorted(vt[ctx]))),
                 "known_targets_in_train": " ".join(map(str, sorted(train_out.get(ctx[-1], ())))) if s != "train" else ""}
                for (ctx, t), ps in sorted(S_.items())]
        add_macro_weight(rows)
        out[s] = rows
        print(f"{split_name_prefix} {s}: S samples {len(rows):,}; prefixes with >1 valid next KO: "
              f"{sum(1 for c in vt if len(vt[c]) > 1):,} of {len(vt):,}", flush=True)
    return out


def strict():
    wins, train_edges = {}, set()
    for s in SPLITS:
        wins[s] = [(tuple(int(x) for x in r["token_ids"].split()), set(r["pathway_ids"].split(";")))
                   for r in rd(SPLIT / f"{s}_B_path_windows.csv.gz")]
    train_edges = {(int(r["context_token_id"]), int(r["target_token_id"])) for r in rd(SPLIT / "train_A_next_node.csv.gz")}
    S = build_S(wins, train_edges, "strict_v1")
    for s in SPLITS:
        wr(OUT / "strict_v1" / f"{s}_S_prefix_next.csv.gz", S[s])


def random_segment():
    rows = rd(DS / "samples" / "ko_B_path_windows.csv.gz")
    segs = defaultdict(set)
    for r in rows:
        segs[tuple(int(x) for x in r["token_ids"].split())].add(r["pathway_id"])
    keys = sorted(segs)
    random.Random("42|random_segment").shuffle(keys)
    n = len(keys)
    cut = {"train": keys[:int(0.8 * n)], "val": keys[int(0.8 * n):int(0.9 * n)], "test": keys[int(0.9 * n):]}
    wins = {s: [(k, segs[k]) for k in cut[s]] for s in SPLITS}
    # A: consecutive pairs of the segments of each split, dedup within split
    A = {}
    for s in SPLITS:
        d = defaultdict(set)
        for w, ps in wins[s]:
            for e in zip(w, w[1:]):
                d[e] |= ps
        A[s] = d
    train_edges = set(A["train"])
    train_tokens = {x for e in train_edges for x in e}
    train_out = defaultdict(set)
    for u, v in train_edges:
        train_out[u].add(v)
    tok = {int(r["token_id"]): r["token"] for r in csv.DictReader((DS / "token_vocab_ko.csv").open())}
    for s in SPLITS:
        out_s = defaultdict(set)
        for u, v in A[s]:
            out_s[u].add(v)
        arows = [{"split": s, "context_token_id": u, "target_token_id": v, "context_token": tok[u], "target_token": tok[v],
                  "n_pathways": len(ps), "pathway_ids": ";".join(sorted(ps)),
                  "context_seen_in_train": int(u in train_tokens), "target_seen_in_train": int(v in train_tokens),
                  "valid_targets_in_split": " ".join(map(str, sorted(out_s[u]))),
                  "known_targets_in_train": " ".join(map(str, sorted(train_out.get(u, set()) - {v}))) if s != "train" else ""}
                 for (u, v), ps in sorted(A[s].items())]
        # NOTE: known_targets_in_train excludes the query target itself, otherwise a test edge that is also a
        # train edge (possible in this naive split) would filter out its own answer.
        add_macro_weight(arows)
        wr(OUT / "random_segment" / f"{s}_A_next_node.csv.gz", arows)
        brows = [{"split": s, "length": len(w), "token_ids": " ".join(map(str, w)), "n_pathways": len(ps),
                  "pathway_ids": ";".join(sorted(ps))} for w, ps in wins[s]]
        add_macro_weight(brows)
        wr(OUT / "random_segment" / f"{s}_B_path_windows.csv.gz", brows)
        print(f"random_segment {s}: segments {len(brows):,}, A edges {len(arows):,}", flush=True)
    S = build_S(wins, train_edges, "random_segment")
    for s in SPLITS:
        wr(OUT / "random_segment" / f"{s}_S_prefix_next.csv.gz", S[s])


if __name__ == "__main__":
    strict()
    random_segment()

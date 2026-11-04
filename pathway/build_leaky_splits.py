#!/usr/bin/env python3
"""
INTENTIONALLY LEAKY control splits (task A only), to show how common random-edge protocols
inflate link-prediction scores compared with the leakage-controlled split in kegg_dataset/splits_ko.

    python build_leaky_splits.py      -> kegg_leaky_splits/{occurrence_random,unique_random}/

occurrence_random  random 80/10/10 split of KO edge *occurrences* (pathway, u, v). The same KO edge
                   drawn in several pathways can land in train and test at the same time.
unique_random      random 80/10/10 split of *unique* KO edges of the union graph (all pathways merged).
                   No duplicate edges, but reverse edges and the surrounding pathway structure of every
                   test edge remain in train.

Same KO vocabulary and file formats as splits_ko (train/val/test _A_next_node.csv.gz), so the same
evaluation code runs on them. Never use these splits to report model quality.
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_ko_leakage_free_split import add_macro_weight, load_ko_graphs  # noqa: E402
from kegg_pipeline.common import OUT, log, read_csv, write_csv  # noqa: E402

LEAKY = Path(__file__).resolve().parent / "kegg_leaky_splits"
SPLITS = ("train", "val", "test")
RATIOS = (0.8, 0.1, 0.1)
SEED = 42


def cut(items, key):
    rng = random.Random(f"{SEED}|{key}")
    items = sorted(items)
    rng.shuffle(items)
    n = len(items)
    a, b = int(RATIOS[0] * n), int((RATIOS[0] + RATIOS[1]) * n)
    return {"train": items[:a], "val": items[a:b], "test": items[b:]}


def write_split(name, occ_by_split, vocab, note):
    d = LEAKY / name
    d.mkdir(parents=True, exist_ok=True)
    # occ_by_split[s] = list of (pathway, u, v)
    edges = {s: defaultdict(set) for s in SPLITS}
    for s in SPLITS:
        for p, u, v in occ_by_split[s]:
            edges[s][(u, v)].add(p)
    train_out = defaultdict(set)
    for u, v in edges["train"]:
        train_out[u].add(v)
    train_tokens = {x for e in edges["train"] for x in e}
    stats = {}
    for s in SPLITS:
        out_s = defaultdict(set)
        for u, v in edges[s]:
            out_s[u].add(v)
        rows = []
        for (u, v), ps in sorted(edges[s].items()):
            rows.append({"split": s, "context_token_id": vocab[u], "target_token_id": vocab[v],
                         "context_token": f"ko:{u}", "target_token": f"ko:{v}", "n_pathways": len(ps),
                         "pathway_ids": ";".join(sorted(ps)), "context_seen_in_train": int(u in train_tokens),
                         "target_seen_in_train": int(v in train_tokens),
                         "valid_targets_in_split": " ".join(str(vocab[x]) for x in sorted(out_s[u])),
                         "known_targets_in_train": " ".join(str(vocab[x]) for x in sorted(train_out.get(u, ())))
                         if s != "train" else ""})
        add_macro_weight(rows)
        write_csv(d / f"{s}_A_next_node.csv.gz", rows, list(rows[0].keys()))
        if s != "train":
            te = set(edges[s])
            stats[s] = {"queries": len(rows),
                        "edge_also_in_train": sum(1 for e in te if e in edges["train"]),
                        "reverse_edge_in_train": sum(1 for u, v in te if (v, u) in edges["train"]),
                        "target_unseen_in_train": sum(1 for r in rows if not r["target_seen_in_train"])}
    lines = [f"# LEAKY control split: {name}", "", "**Intentionally leaky. For comparison only; never report "
             "model quality on it.**", "", note, "",
             "| split | queries | test edge also in train | reverse edge in train | target unseen in train |",
             "|---|---:|---:|---:|---:|"]
    for s, st in stats.items():
        lines.append(f"| {s} | {st['queries']:,} | {st['edge_also_in_train']:,} "
                     f"({st['edge_also_in_train'] / st['queries']:.1%}) | {st['reverse_edge_in_train']:,} | "
                     f"{st['target_unseen_in_train']:,} |")
    (d / "README.md").write_text("\n".join(lines) + "\n")
    log(f"{name}: " + "; ".join(f"{s} {st}" for s, st in stats.items()))


def main():
    graphs = load_ko_graphs()
    vocab = {r["token"]: int(r["token_id"]) for r in read_csv(OUT / "token_vocab_ko.csv")}
    occ = [(p, u, v) for p, es in graphs.items() for u, v in es]
    write_split("occurrence_random", cut(occ, "occ"), vocab,
                "Random 80/10/10 split of KO edge occurrences (pathway, u, v); a KO edge drawn in several "
                "pathways can be in train and test at once.")
    uniq = defaultdict(set)
    for p, u, v in occ:
        uniq[(u, v)].add(p)
    parts = cut(list(uniq), "unique")
    write_split("unique_random", {s: [(p, u, v) for u, v in parts[s] for p in uniq[(u, v)]] for s in SPLITS}, vocab,
                "Random 80/10/10 split of unique KO edges of the union graph; reverse edges and the pathway "
                "neighbourhood of each test edge stay in train.")


if __name__ == "__main__":
    main()

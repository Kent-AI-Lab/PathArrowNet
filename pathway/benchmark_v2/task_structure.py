#!/usr/bin/env python3
"""
One-to-many (branching) and reciprocal-edge statistics of the next-node tasks.

    .venv/bin/python benchmark_v2/task_structure.py   -> benchmark_v2/out/task_structure.csv

For every split and task (A: u -> v; S: prefix -> next), counts contexts, contexts with >1 valid next KO in the
same split, samples belonging to such contexts, and the distribution of the number of valid next KOs.
Also: reciprocal pairs (a->b and b->a) kept / excluded in the strict split, and the random-uniform
expected MRR given the filtered candidate count.
"""

from __future__ import annotations

import csv
import gzip
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "kegg_dataset"
B2 = Path(__file__).resolve().parent
OUT = B2 / "out"
SPLITS = ("train", "val", "test")


def rd(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    rows = []
    for setting, adir, sdir in (("strict_v1", DS / "splits_ko", B2 / "splits/strict_v1"),
                                ("random_segment", B2 / "splits/random_segment", B2 / "splits/random_segment")):
        for s in SPLITS:
            for task, path, key in (("A", adir / f"{s}_A_next_node.csv.gz", "context_token_id"),
                                    ("S", sdir / f"{s}_S_prefix_next.csv.gz", "context_token_ids")):
                if not path.exists():
                    continue
                data = rd(path)
                tg = defaultdict(set)
                for r in data:
                    tg[r[key]].add(r["target_token_id"])
                k = Counter(len(v) for v in tg.values())
                n_multi = sum(1 for v in tg.values() if len(v) > 1)
                rows.append({"setting": setting, "split": s, "task": task, "samples": len(data), "contexts": len(tg),
                             "contexts_with_>1_valid_next": n_multi,
                             "share_contexts_multi": round(n_multi / len(tg), 3),
                             "samples_in_multi_contexts": sum(len(v) for v in tg.values() if len(v) > 1),
                             "share_samples_multi": round(sum(len(v) for v in tg.values() if len(v) > 1) / len(data), 3),
                             "valid_next_per_context_mean": round(statistics.mean(len(v) for v in tg.values()), 2),
                             "valid_next_per_context_median": statistics.median(len(v) for v in tg.values()),
                             "valid_next_per_context_max": max(len(v) for v in tg.values()),
                             "contexts_with_1/2/3-5/6-10/>10": "/".join(str(x) for x in (
                                 k[1], k[2], sum(k[i] for i in range(3, 6)), sum(k[i] for i in range(6, 11)),
                                 sum(c for i, c in k.items() if i > 10)))})
    # reciprocal edges in the strict split
    es = list(csv.DictReader((DS / "splits_ko" / "edge_split.csv").open()))
    own = {(r["source_ko"], r["target_ko"]): r["split"] for r in es}
    rec = Counter()
    for (a, b), s in own.items():
        if (b, a) in own and a < b:
            rec[f"{s}|{own[(b, a)]}"] += 1
    with (OUT / "task_structure.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (OUT / "reciprocal_pairs_strict.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["owner_of_a->b|owner_of_b->a", "pairs"])
        for k, v in sorted(rec.items()):
            w.writerow([k, v])
    for r in rows:
        print(r)
    print("reciprocal pairs by owner of each direction:", dict(rec))


if __name__ == "__main__":
    main()

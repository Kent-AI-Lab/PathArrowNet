#!/usr/bin/env python3
"""
Leakage-controlled re-split of BioPathNet's KEGG gene-function data (PC_KEGG_0602).

Unit of assignment = (KO group, pathway), where a KO group is a connected set of genes sharing a KEGG
Orthology ID (paralogs). All (gene, KEGGPathway, pathway) triples of one unit go to the same split, so no
test triple can be answered by copying the pathway of a same-KO paralog from train. Units are shuffled
(seed 42) and filled into train/valid/test to match the original triple counts (22,702 / 3,227 / 6,438).
The background graph (train1), entity files and everything else are copied unchanged.

    python leakage/make_kogroup_split.py  -> BioPathNet/data/PC_KEGG_0602_kogroup/
"""

from __future__ import annotations

import random
import shutil
from collections import defaultdict
from pathlib import Path

from baselines import kegg_maps, trip

SRC = Path(__file__).resolve().parent.parent / "BioPathNet/data/PC_KEGG_0602"
DST = SRC.parent / "PC_KEGG_0602_kogroup"


def main():
    tr, va, te = trip(SRC / "train2.txt"), trip(SRC / "valid.txt"), trip(SRC / "test.txt")
    alltrip = tr + va + te
    target = {"train2": len(tr), "valid": len(va), "test": len(te)}
    sym2id, g2k = kegg_maps()
    # union-find over genes sharing any KO
    genes = sorted({h for h, _, _ in alltrip})
    par = {g: g for g in genes}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    by_ko = defaultdict(list)
    for g in genes:
        for k in g2k.get(sym2id.get(g, ""), ()):
            by_ko[k].append(g)
    for ms in by_ko.values():
        for g in ms[1:]:
            par[find(g)] = find(ms[0])
    units = defaultdict(list)
    for h, r, t in alltrip:
        units[(find(h), t)].append((h, r, t))
    keys = sorted(units)
    random.Random("42|kogroup").shuffle(keys)
    out = {"train2": [], "valid": [], "test": []}
    order = ["test", "valid", "train2"]          # fill the small splits first, rest to train
    k = 0
    for s in order[:2]:
        while len(out[s]) < target[s] and k < len(keys):
            out[s] += units[keys[k]]
            k += 1
    for key in keys[k:]:
        out["train2"] += units[key]
    # every valid/test node must also occur in train1 + train2 (true for the original split): for such a gene,
    # move its smallest unit from valid/test into train (moving whole units keeps the split leakage-free)
    brg_nodes = {x for h, _, t in trip(SRC / "train1.txt") for x in (h, t)}
    unit_of = {}
    for key, rows in units.items():
        for r in rows:
            unit_of[r] = key
    changed = True
    while changed:
        changed = False
        train_nodes = brg_nodes | {x for h, _, t in out["train2"] for x in (h, t)}
        for s in ("valid", "test"):
            for h, r, t in list(out[s]):
                if h in train_nodes and t in train_nodes:
                    continue
                node = h if h not in train_nodes else t
                cand = sorted({unit_of[x] for s2 in ("valid", "test") for x in out[s2] if node in (x[0], x[2])},
                              key=lambda kk: (len(units[kk]), kk))
                mv = set(units[cand[0]])
                for s2 in ("valid", "test"):
                    out[s2] = [x for x in out[s2] if x not in mv]
                out["train2"] += units[cand[0]]
                changed = True
                break
            if changed:
                break
    DST.mkdir(exist_ok=True)
    for f in ("train1.txt", "entity_types.txt", "entity_names.txt"):
        shutil.copy(SRC / f, DST / f)
    for s, rows in out.items():
        (DST / f"{s}.txt").write_text("".join("\t".join(x) + "\n" for x in rows))
    # checks
    train_pw = defaultdict(set)
    for h, _, t in out["train2"]:
        train_pw[find(h)].add(t)
    leak = sum(1 for h, _, t in out["test"] if t in train_pw[find(h)])
    train_nodes = {x for h, _, t in trip(DST / "train1.txt") for x in (h, t)} | {x for h, _, t in out["train2"] for x in (h, t)}
    missing = {x for s in ("valid", "test") for h, _, t in out[s] for x in (h, t)} - train_nodes
    print({s: len(v) for s, v in out.items()}, "| units", len(keys), "| KO groups with >1 gene",
          sum(1 for g in genes if find(g) == g and sum(1 for x in genes if find(x) == g) > 1) if len(genes) < 0 else "-")
    print("test triples whose (KO group, pathway) occurs in train:", leak)
    print("valid/test nodes absent from train1+train2:", len(missing), sorted(missing)[:10])


if __name__ == "__main__":
    main()

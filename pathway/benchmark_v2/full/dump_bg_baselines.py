#!/usr/bin/env python3
"""Non-learned baselines that use the STRING background graph (same file as the background of the learned models).
  bg_neighbor     score(u, v) = max experimental STRING score of the KO pair (0 if not a background neighbour),
                  ties broken by train in-degree of v (popularity)
  bg_cf           neighbourhood CF on train edges + background: sum over background neighbours w of u of
                  [w -> v train edge] (u's partners' known successors), + bg_neighbor
    .venv/bin/python benchmark_v2/full/dump_bg_baselines.py --setting strict_v1_full
"""
import argparse, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
B2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(B2))
import scorelib as SL

ap = argparse.ArgumentParser(); ap.add_argument("--setting", default="strict_v1_full"); a = ap.parse_args()
voc, V = SL.vocab(a.setting)
bgf = SL.ROOT / "kegg_dataset_cache/string" / ("ko_background_exp400_full.tsv" if a.setting.endswith("_full") else "ko_background_exp400.tsv")
W = defaultdict(dict)
for l in bgf.read_text().splitlines()[1:]:
    x, y, s, _ = l.split("\t"); x, y, s = voc[x], voc[y], int(s) / 1000
    W[x][y] = W[y][x] = s
tr = [(int(r["context_token_id"]), int(r["target_token_id"])) for r in SL.rd(SL.SPLIT_DIRS[a.setting][0] / "train_A_next_node.csv.gz")]
succ = defaultdict(set); indeg = np.zeros(V)
for u, v in tr: succ[u].add(v); indeg[v] += 1
pop = np.log1p(indeg) / (np.log1p(indeg).max() + 1) * 1e-3
anchors = SL.anchors_needed(a.setting)
for name in ("bg_neighbor", "bg_cf"):
    sc = np.zeros((len(anchors), V))
    for i, u in enumerate(anchors):
        for w, s in W[u].items():
            sc[i, w] += s
            if name == "bg_cf":
                for v in succ[w]: sc[i, v] += 0.5 * s
        sc[i] += pop
    out = SL.SCORES / a.setting / f"baseline_{name}" / "s0"
    for split in ("val", "test"):
        SL.save_tail(out / f"tail_{split}.npz", anchors, sc, np.float64)
    q = SL.load_A(a.setting, "test")
    rk, _ = SL.ranks(SL.query_scores(out, "test", "A", q), q, V)
    print(f"{a.setting} {name}: test A MRR {np.mean(1 / rk):.4f}", flush=True)

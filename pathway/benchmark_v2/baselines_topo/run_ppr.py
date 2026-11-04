#!/usr/bin/env python3
"""B1 baseline_ppr_string      personalised PageRank from the query KO on the STRING background graph only
   B2 baseline_ppr_string+kegg  same walk on STRING (undirected) + directed KEGG training edges (optionally + reverse
                                train edges with a lower weight)
Score(u, c) = stationary PPR mass on c (0 if unreachable). Hyper-parameters (STRING weight score/binary, restart alpha,
w_kegg, w_rev) are chosen on validation task-A MRR only; train edges are the only KEGG input.
    CUDA_VISIBLE_DEVICES="" .venv/bin/python benchmark_v2/baselines_topo/run_ppr.py --setting strict_v1_full
-> benchmark_v2/scores/<setting>/baseline_ppr_string/s0/{tail_val,tail_test}.npz + source.json (grid incl.)
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C, scorelib as SL

ap = argparse.ArgumentParser(); ap.add_argument("--setting", default="strict_v1_full"); ap.add_argument("--iters", type=int, default=50)
a = ap.parse_args(); st = a.setting
voc, V = SL.vocab(st); se = C.load_string(voc); tr = C.load_train(st); anchors = SL.anchors_needed(st)
ALPHAS = (0.15, 0.3, 0.5)
grids = {
    "baseline_ppr_string": [dict(weight=w, alpha=al, w_kegg=0.0, w_rev=0.0) for w in ("score", "binary") for al in ALPHAS],
    "baseline_ppr_string+kegg": [dict(weight=w, alpha=al, w_kegg=wk, w_rev=wr * wk) for w in ("score", "binary") for al in ALPHAS
                                 for wk in (1.0, 3.0, 10.0) for wr in (0.0, 1 / 3)],
}
for name, grid in grids.items():
    best, log = None, []
    for cfg in grid:
        t = time.time()
        g = C.walk_graph(V, se, tr, cfg["weight"], cfg["w_kegg"], cfg["w_rev"]); P, out = g.transition()
        S = C.ppr_batched(P, out, None, anchors, cfg["alpha"], a.iters)
        m = C.val_mrr(st, V, anchors, S)
        log.append({**cfg, "val_MRR": m})
        print(f"{st} {name} {cfg} val A MRR {m:.4f} ({time.time() - t:.1f}s)", flush=True)
        if best is None or m > best[0]:
            best = (m, cfg, S)
    m, cfg, S = best
    outd = SL.SCORES / st / name / "s0"
    for sp in ("val", "test"):
        SL.save_tail(outd / f"tail_{sp}.npz", anchors, S, np.float32)
    test = C.val_mrr(st, V, anchors, S, "test")
    (outd / "source.json").write_text(json.dumps({"method": name, "setting": st, "script": "benchmark_v2/baselines_topo/run_ppr.py",
        "string_file": str(C.STRING_FILE.relative_to(SL.ROOT)), "train_edges": str((SL.SPLIT_DIRS[st][0] / "train_A_next_node.csv.gz").relative_to(SL.ROOT)),
        "iters": a.iters, "dangling": "restart to anchor", "score": "PPR mass (0 if unreachable)",
        "selected": cfg, "selected_val_MRR": m, "test_A_MRR": test, "val_grid": log}, indent=1))
    print(f"==> {st} {name}: selected {cfg} val {m:.4f} test A MRR {test:.4f}", flush=True)

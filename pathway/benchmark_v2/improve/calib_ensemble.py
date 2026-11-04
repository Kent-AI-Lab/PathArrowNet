#!/usr/bin/env python3
"""Calibrate-then-mix ensemble (val-tuned). Per model m: c_m = z( z(seed-avg scores of m) + beta_m * off ), beta_m chosen on
val (A+S mean micro MRR); then score = w * c_a + (1 - w) * c_b, w chosen on val. Also reports reciprocal-rank fusion.
off = [candidate KO has no train edge]. Writes scores/<setting>/<name>/s0/seq_*.npz.
    .venv/bin/python benchmark_v2/improve/calib_ensemble.py --a M1 --b M2 --name NAME"""
import argparse, json, sys
from pathlib import Path
import numpy as np
B2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(B2))
import scorelib as SL
from ensemble import z

ap = argparse.ArgumentParser()
ap.add_argument("--setting", default="strict_v1_full"); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--fixed", default=None, help="w,beta_a,beta_b fixed (no val selection; for models trained on train+val)")
a = ap.parse_args()
_, V = SL.vocab(a.setting)
Q = {(sp, t): SL.load_queries(a.setting, sp, t) for sp in ("val", "test") for t in ("A", "S")}
tr = set()
for r in SL.rd(SL.SPLIT_DIRS[a.setting][0] / "train_A_next_node.csv.gz"): tr |= {int(r["context_token_id"]), int(r["target_token_id"])}
off = np.zeros(V); off[3:] = 1; off[sorted(tr)] = 0
mrr = lambda m, k: float(np.mean(1 / SL.ranks(m, Q[k], V)[0]))
val = lambda M: np.mean([mrr(M[("val", t)], ("val", t)) for t in ("A", "S")])
def seedavg(m):
    runs = sorted(p for p in (SL.SCORES / a.setting / m).iterdir() if p.is_dir())
    return {k: np.mean([z(SL.query_scores(r, *k, qs)) for r in runs], 0) for k, qs in Q.items()}, len(runs)
C, info = {}, {}
for key, m in (("a", a.a), ("b", a.b)):
    Z, n = seedavg(m)
    best = (float(a.fixed.split(",")[1 if key == "a" else 2]) if a.fixed else
            max((-4, -3, -2, -1.5, -1, -0.5, 0), key=lambda b: val({k: Z[k] + b * off for k in Q})))
    C[key] = {k: z(Z[k] + best * off) for k in Q}
    info[key] = {"method": m, "seeds": n, "beta": best}
    print(f"{m} ({n} seeds): beta {best}; val {val(C[key]):.4f} test A {mrr(C[key][('test','A')], ('test','A')):.4f} S {mrr(C[key][('test','S')], ('test','S')):.4f}")
grid = {w: val({k: w * C["a"][k] + (1 - w) * C["b"][k] for k in Q}) for w in [round(x, 2) for x in np.arange(0, 1.001, 0.05)]}
w = float(a.fixed.split(",")[0]) if a.fixed else max(grid, key=grid.get)
grid.setdefault(w, val({k: w * C["a"][k] + (1 - w) * C["b"][k] for k in Q}))
E = {k: w * C["a"][k] + (1 - w) * C["b"][k] for k in Q}
print(f"mix w(a)={w}: val {grid[w]:.4f} test A {mrr(E[('test','A')], ('test','A')):.4f} S {mrr(E[('test','S')], ('test','S')):.4f}")
def rrf(M, k0=60):
    r = (-M).argsort(1).argsort(1) + 1
    return 1.0 / (k0 + r)
R = {k: rrf(C["a"][k]) + rrf(C["b"][k]) for k in Q}
print(f"RRF: val {val(R):.4f} test A {mrr(R[('test','A')], ('test','A')):.4f} S {mrr(R[('test','S')], ('test','S')):.4f}")
out = SL.SCORES / a.setting / a.name / "s0"; out.mkdir(parents=True, exist_ok=True)
for k in Q: np.savez(out / f"seq_{k[0]}_{k[1]}.npz", scores=E[k].astype(np.float32))
(out / "source.json").write_text(json.dumps({**info, "w_a": w, "val_grid": grid}, indent=1))

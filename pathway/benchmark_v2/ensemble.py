#!/usr/bin/env python3
"""
Score-level combination of two stored models (same strict split, same candidates), weight chosen on VAL only.

    .venv/bin/python benchmark_v2/ensemble.py --a biopathnet+bg --b gnn_id+gobert_noBP+esm --name ens_bpnbg+gnn_gobert_esm

Per query row, each model's scores are standardised over the 2,870 candidates (z-score; -inf/unknown KOs -> row min)
and combined as w * z_a + (1 - w) * z_b. Seeds are paired by order (seed i of a with seed i of b). w is chosen from
{0, 0.1, ..., 1} by mean val micro MRR over tasks A and S (averaged over the seed pairs), then applied to test.
Writes seq_{val,test}_{A,S}.npz under scores/strict_v1/<name>/s<k>/ so eval_all.py scores it like any model.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

B2 = Path(__file__).resolve().parent
sys.path.insert(0, str(B2))
import scorelib as SL  # noqa: E402


def z(m):
    m = m.astype(np.float64).copy()
    fin = np.isfinite(m) & (m > -1e29)
    lo = np.where(fin, m, np.inf).min(1, keepdims=True)
    m = np.where(fin, m, lo)
    m[:, :3] = lo[:, 0][:, None]
    return (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    _, V = SL.vocab()
    sa = sorted((SL.SCORES / "strict_v1" / a.a).iterdir())
    sb = sorted((SL.SCORES / "strict_v1" / a.b).iterdir())
    n = min(len(sa), len(sb))
    Q = {(sp, t): SL.load_queries("strict_v1", sp, t) for sp in ("val", "test") for t in ("A", "S")}
    Z = {}
    for i in range(n):
        for k, qs in Q.items():
            Z[(i, k)] = (z(SL.query_scores(sa[i], *k, qs)), z(SL.query_scores(sb[i], *k, qs)))
    grid = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
    val = {}
    for w in grid:
        vals = []
        for i in range(n):
            for t in ("A", "S"):
                za, zb = Z[(i, ("val", t))]
                rk, _ = SL.ranks(w * za + (1 - w) * zb, Q[("val", t)], V)
                vals.append(np.mean(1 / rk))
        val[w] = float(np.mean(vals))
    w = max(grid, key=lambda x: val[x])
    for i in range(n):
        out = SL.SCORES / "strict_v1" / a.name / f"s{i}"
        out.mkdir(parents=True, exist_ok=True)
        for k in Q:
            za, zb = Z[(i, k)]
            np.savez(out / f"seq_{k[0]}_{k[1]}.npz", scores=(w * za + (1 - w) * zb).astype(np.float32))
        (out / "source.json").write_text(json.dumps({"a": str(sa[i]), "b": str(sb[i]), "w_a": w, "val_grid": val}))
    print(f"{a.name}: w({a.a}) = {w} chosen on val; val grid {val}")


if __name__ == "__main__":
    main()

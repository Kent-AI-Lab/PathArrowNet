#!/usr/bin/env python3
"""Per-gene expression change for every single-gene perturbation in Adamson 2016 (CRISPRi, K562) and Norman 2019
(CRISPRa, K562; single perturbations only, 'X+ctrl' / 'ctrl+X'), from the public GEARS-processed files
(Harvard Dataverse 6154417 / 6154020, log1p-normalised, ~5,000 HVGs). Downloaded independently of any other project.

For each perturbation p and measured gene g: delta = mean(p) - mean(ctrl), Welch t, and the two-sided p-value (normal approximation).
Writes data/perturb/stats_<dataset>.npz (genes, perts, n_cells, delta, t, pval, ctrl_mean)."""
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

D = Path(__file__).resolve().parent / "data/perturb"


def col(g, k):
    """old-style anndata column (with __categories) -> numpy array of str"""
    v = g[k][()]
    if "__categories" in g and k in g["__categories"]:
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g["__categories"][k][()]])
        return cats[v]
    return np.array([c.decode() if isinstance(c, bytes) else c for c in v])


def main(name):
    f = h5py.File(D / "raw" / name / "perturb_processed.h5ad", "r")
    data, ind, ptr = f["X/data"][()].astype(np.float64), f["X/indices"][()], f["X/indptr"][()]
    G = int(f["X"].attrs["shape"][1])

    def moments(rows):  # mean and population variance over a row subset of the CSR matrix
        sel = np.concatenate([np.arange(ptr[r], ptr[r + 1]) for r in rows])
        s1 = np.bincount(ind[sel], data[sel], G); s2 = np.bincount(ind[sel], data[sel] ** 2, G)
        return s1 / len(rows), s2 / len(rows) - (s1 / len(rows)) ** 2
    cond = col(f["obs"], "condition"); genes = col(f["var"], "gene_name")
    single = {}
    for c in np.unique(cond):
        parts = c.split("+")
        if c == "ctrl":
            continue
        g = [p for p in parts if p != "ctrl"]
        if len(g) == 1:
            single.setdefault(g[0], []).append(c)
    ctrl = np.where(cond == "ctrl")[0]
    mc, vc = moments(ctrl)
    perts = sorted(single); n = []; dl, tt, pv = [], [], []
    for p in perts:
        idx = np.where(np.isin(cond, single[p]))[0]; mp, vp = moments(idx)
        se = np.sqrt(vp * len(idx) / max(1, len(idx) - 1) / len(idx) + vc * len(ctrl) / (len(ctrl) - 1) / len(ctrl)) + 1e-12
        t = (mp - mc) / se
        # normal approximation to the Welch t (>= ~100 cells per perturbation, ~10^4 controls)
        dl.append(mp - mc); tt.append(t); pv.append(torch.special.erfc(torch.tensor(np.abs(t)) / 2 ** 0.5).numpy()); n.append(len(idx))
    np.savez_compressed(D / f"stats_{name}.npz", genes=genes, perts=np.array(perts), n_cells=np.array(n), n_ctrl=len(ctrl),
                        delta=np.array(dl, np.float32), t=np.array(tt, np.float32), pval=np.array(pv), ctrl_mean=mc)
    print(name, "cells", len(cond), G, "ctrl", len(ctrl), "single perts", len(perts), "median cells", int(np.median(n)))


if __name__ == "__main__":
    for nm in sys.argv[1:] or ["adamson", "norman"]:
        main(nm)

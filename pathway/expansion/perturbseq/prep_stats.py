#!/usr/bin/env python3
"""Per-gene expression change for every single-gene perturbation in the GEARS-processed Perturb-seq files
(Harvard Dataverse; log1p-normalised expression, HVGs only):
    adamson                   Adamson 2016, CRISPRi, K562            (file 6154417)
    norman                    Norman 2019,  CRISPRa, K562            (file 6154020; single perturbations only)
    replogle_k562_essential   Replogle 2022, CRISPRi, K562, essential-gene screen (file 7458695)
    replogle_rpe1_essential   Replogle 2022, CRISPRi, hTERT-RPE1, essential-gene screen (file 7458694)
Raw files: expansion/downstream/data/perturb/raw/<name>/perturb_processed.h5ad (sha256 in ../raw_checksums.sha256).

Two control schemes:
    pooled   delta = mean(pert) - mean(all control cells)   (same as downstream/perturb_prep.py)
    gem      every cell is centred on the mean of the control cells of its own 10x lane (gem group = barcode suffix),
             then pert vs control residuals; removes lane / batch shifts. Lanes without >= MIN_CTRL controls use the
             pooled control mean. This is the main scheme.
Statistic: Welch t, two-sided p from the normal approximation (perturbations have >= ~20 cells, controls thousands).

    .venv/bin/python expansion/perturbseq/prep_stats.py [names]
    -> expansion/perturbseq/data/stats_<name>.npz
"""
import sys
from pathlib import Path

import h5py
import numpy as np
from torch.special import erfc as _erfc
import torch


def erfc(x):
    return _erfc(torch.as_tensor(np.asarray(x, np.float64))).numpy()

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "expansion/downstream/data/perturb/raw"
OUT = Path(__file__).resolve().parent / "data"
MIN_CTRL = 20
META = {"adamson": ("K562", "CRISPRi"), "norman": ("K562", "CRISPRa"),
        "replogle_k562_essential": ("K562", "CRISPRi"), "replogle_rpe1_essential": ("hTERT-RPE1", "CRISPRi")}
# Nadig 2025 (GEO GSE264667), converted to the same layout by nadig_to_processed.py; only run when named explicitly
META_EXTRA = {"nadig_hepg2_essential": ("HepG2", "CRISPRi"), "nadig_jurkat_essential": ("Jurkat", "CRISPRi")}


def col(f, grp, k):
    """obs / var column as numpy str array; handles old (__categories) and new (categories/codes) anndata layouts"""
    g = f[grp]
    if isinstance(g[k], h5py.Group):
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g[k]["categories"][()]])
        return cats[g[k]["codes"][()]]
    v = g[k][()]
    if "__categories" in g and k in g["__categories"]:
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g["__categories"][k][()]])
        return cats[v]
    return np.array([c.decode() if isinstance(c, bytes) else c for c in v])


def welch(a_mean, a_var, na, b_mean, b_var, nb):
    se = np.sqrt(a_var / max(1, na - 1) + b_var / max(1, nb - 1)) + 1e-12  # population var -> unbiased var / n
    t = (a_mean - b_mean) / se
    return t, erfc(np.abs(t) / np.sqrt(2))


def main(name):
    f = h5py.File(RAW / name / "perturb_processed.h5ad", "r")
    n, G = (int(x) for x in f["X"].attrs["shape"])
    X = np.zeros((n, G), np.float32)
    data, ind, ptr = f["X/data"][()], f["X/indices"][()], f["X/indptr"][()]
    X[np.repeat(np.arange(n), np.diff(ptr)), ind] = data
    cond = col(f, "obs", "condition"); genes = col(f, "var", "gene_name")
    gid = col(f, "var", "gene_id") if "gene_id" in f["var"] else np.array([""] * G)
    gem = np.array([b.decode().rsplit("-", 1)[-1] for b in f["obs/cell_barcode"][()]])
    single = {}
    for c in np.unique(cond):
        g = [p for p in c.split("+") if p != "ctrl"]
        if c != "ctrl" and len(g) == 1:
            single.setdefault(g[0], []).append(c)
    is_ctrl = cond == "ctrl"
    pooled_ctrl = X[is_ctrl].mean(0)
    # gem-group-centred residuals
    R = X.copy(); n_fallback = 0
    for g in np.unique(gem):
        rows = gem == g; c = rows & is_ctrl
        if c.sum() >= MIN_CTRL:
            R[rows] -= X[c].mean(0)
        else:
            R[rows] -= pooled_ctrl; n_fallback += int(rows.sum())
    perts = sorted(single)
    out = {k: [] for k in ("delta", "t", "pval", "delta_pooled", "t_pooled", "pval_pooled")}
    ncell, ngem = [], []
    for sch, M in (("", R), ("_pooled", X)):
        C = M[is_ctrl]; mc, vc, nc = C.mean(0), C.var(0), len(C)
        for p in perts:
            P = M[np.isin(cond, single[p])]
            t, pv = welch(P.mean(0), P.var(0), len(P), mc, vc, nc)
            out["delta" + sch].append(P.mean(0) - mc); out["t" + sch].append(t); out["pval" + sch].append(pv)
            if sch == "":
                ncell.append(len(P)); ngem.append(len(np.unique(gem[np.isin(cond, single[p])])))
    cl, mod = {**META, **META_EXTRA}[name]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"stats_{name}.npz", genes=genes, gene_ids=gid, perts=np.array(perts),
                        n_cells=np.array(ncell), n_gem=np.array(ngem), n_ctrl=int(is_ctrl.sum()), n_cells_total=n,
                        n_gem_total=len(np.unique(gem)), n_fallback_cells=n_fallback, cell_line=cl, modality=mod,
                        ctrl_mean=pooled_ctrl, frac_expr_ctrl=(X[is_ctrl] > 0).mean(0),
                        **{k: np.array(v, np.float32 if not k.startswith("pval") else np.float64) for k, v in out.items()})
    print(name, cl, mod, "cells", n, "genes", G, "ctrl", int(is_ctrl.sum()), "gem groups", len(np.unique(gem)),
          "fallback cells", n_fallback, "single perts", len(perts), "cells/pert median", int(np.median(ncell)),
          "min", int(np.min(ncell)), flush=True)


if __name__ == "__main__":
    for nm in sys.argv[1:] or list(META):
        main(nm)

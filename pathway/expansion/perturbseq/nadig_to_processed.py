#!/usr/bin/env python3
"""Nadig 2025 (TRADE; Nat Genet, doi 10.1038/s41588-025-02169-3) HepG2 / Jurkat CRISPRi essential-gene Perturb-seq,
GEO GSE264667 `GSE264667_<line>_raw_singlecell_01.h5ad` (UMI counts of the authors' filtered cells and expressed
genes; dense X; obs has gem_group, gene, sgID_AB, ...), converted to the layout of the GEARS-processed Replogle files so
that prep_stats.py runs on it unchanged:

  - cells: all cells of the file (the authors' cell filter); condition = '<gene>+ctrl', non-targeting -> 'ctrl';
    guides / transcripts (P1, P2, P1P2) of the same gene pooled, as in the GEARS Replogle files
  - perturbations with < MIN_CELLS cells dropped (15 = the smallest perturbation in the GEARS Replogle K562 file)
  - expression: normalize_total to the median total count, log1p (scanpy defaults, as in the GEARS pipeline)
  - genes: 5,000 highly variable genes, scanpy 'seurat' flavour (dispersion z-scored within 20 mean bins), as in GEARS
    (the GEARS files carry var['dispersions_norm']). Re-implemented in numpy because scanpy / scipy are not installed.
  - cell_barcode keeps the '-<gem_group>' suffix used by prep_stats.py for lane-matched controls (checked against
    obs['gem_group']).
Known difference: GEARS also removed perturbations of genes outside its gene graph (Replogle K562: 1,092 of ~2,000);
that filter is not applied here.

    .venv/bin/python -I expansion/perturbseq/nadig_to_processed.py hepg2|jurkat
    -> expansion/downstream/data/perturb/raw/nadig_<line>_essential/perturb_processed.h5ad (+ conversion.json)
"""
import json, sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prep_stats import col  # noqa: E402

RAW = HERE.parent / "downstream/data/perturb/raw"
MIN_CELLS, N_HVG, N_BINS, CHUNK = 15, 5000, 20, 8192


def hvg_seurat(mean, var, n_top):
    """scanpy highly_variable_genes(flavor='seurat', n_top_genes) on normalised (not log) mean / variance"""
    mean = np.where(mean == 0, 1e-12, mean)
    disp = np.log(np.where(var / mean == 0, np.nan, var / mean))
    lmean = np.log1p(mean)
    edges = np.linspace(lmean.min(), lmean.max(), N_BINS + 1)  # pd.cut(bins=20) equivalent
    b = np.clip(np.digitize(lmean, edges[1:-1]), 0, N_BINS - 1)
    dn = np.full_like(disp, np.nan)
    for k in range(N_BINS):
        m = (b == k) & np.isfinite(disp)
        if m.sum() == 0:
            continue
        sd = disp[m].std(ddof=1) if m.sum() > 1 else np.nan
        dn[m] = (disp[m] - disp[m].mean()) / sd if sd and np.isfinite(sd) and sd > 0 else 0.0  # one-gene bins -> 0
    dn = np.where(np.isfinite(dn), dn, -np.inf)
    return np.sort(np.argsort(-dn, kind="stable")[:n_top]), dn


def main(line):
    src = RAW / "nadig_geo" / f"GSE264667_{line}_raw_singlecell_01.h5ad"
    out = RAW / f"nadig_{line}_essential"; out.mkdir(exist_ok=True)
    f = h5py.File(src, "r")
    n, G = f["X"].shape
    gene = col(f, "obs", "gene"); bc = col(f, "obs", "cell_barcode"); gg = col(f, "obs", "gem_group")
    assert all(b.rsplit("-", 1)[-1] == str(g) for b, g in zip(bc, gg)), "barcode suffix != gem_group"
    gname = col(f, "var", "gene_name"); gid = col(f, "var", "gene_id")
    u, c = np.unique(gene, return_counts=True)
    keep_p = {p for p, k in zip(u, c) if p == "non-targeting" or k >= MIN_CELLS}
    keep = np.array([g in keep_p for g in gene])
    rows = np.where(keep)[0]
    # pass 1: totals, then normalised mean / var per gene
    tot = np.zeros(n)
    for s in range(0, n, CHUNK):
        tot[s:s + CHUNK] = f["X"][s:s + CHUNK].sum(1)
    target = float(np.median(tot[rows]))
    s1 = np.zeros(G); s2 = np.zeros(G)
    for s in range(0, n, CHUNK):
        idx = rows[(rows >= s) & (rows < s + CHUNK)]
        if len(idx) == 0:
            continue
        x = f["X"][s:s + CHUNK][idx - s].astype(np.float64) * (target / tot[idx])[:, None]
        s1 += x.sum(0); s2 += (x * x).sum(0)
    nk = len(rows); mean = s1 / nk; var = (s2 - nk * mean ** 2) / (nk - 1)
    hv, dn = hvg_seurat(mean, var, N_HVG)
    # pass 2: log1p-normalised HVG matrix as CSR
    data, ind, ptr = [], [], [0]
    for s in range(0, n, CHUNK):
        idx = rows[(rows >= s) & (rows < s + CHUNK)]
        if len(idx) == 0:
            continue
        x = np.log1p(f["X"][s:s + CHUNK][idx - s][:, hv].astype(np.float64) * (target / tot[idx])[:, None]).astype(np.float32)
        r, cc = np.nonzero(x)
        data.append(x[r, cc]); ind.append(cc.astype(np.int32))
        ptr.extend((np.cumsum(np.bincount(r, minlength=len(idx))) + ptr[-1]).tolist())
    data = np.concatenate(data); ind = np.concatenate(ind); ptr = np.array(ptr, np.int64)
    cond = np.array(["ctrl" if g == "non-targeting" else f"{g}+ctrl" for g in gene[rows]])
    with h5py.File(out / "perturb_processed.h5ad", "w") as o:
        X = o.create_group("X"); X.attrs["encoding-type"] = "csr_matrix"; X.attrs["encoding-version"] = "0.1.0"
        X.attrs["shape"] = np.array([nk, len(hv)])
        X.create_dataset("data", data=data, compression="gzip"); X.create_dataset("indices", data=ind, compression="gzip")
        X.create_dataset("indptr", data=ptr)
        ob = o.create_group("obs"); va = o.create_group("var")
        ob.create_dataset("condition", data=cond.astype("S")); ob.create_dataset("cell_barcode", data=bc[rows].astype("S"))
        ob.create_dataset("gem_group", data=gg[rows].astype(np.int64))
        va.create_dataset("gene_name", data=gname[hv].astype("S")); va.create_dataset("gene_id", data=gid[hv].astype("S"))
        va.create_dataset("dispersions_norm", data=dn[hv])
    info = {"source": src.name, "cells_in_file": int(n), "genes_in_file": int(G), "cells_kept": int(nk),
            "perturbations_in_file": int(len(u) - ("non-targeting" in u)),
            "perturbations_kept": int(len(keep_p) - 1), "control_cells": int((gene == "non-targeting").sum()),
            "min_cells": MIN_CELLS, "normalize_total_target": target, "n_hvg": int(len(hv)),
            "gem_groups": int(len(np.unique(gg[rows]))), "nnz": int(len(data))}
    (out / "conversion.json").write_text(json.dumps(info, indent=1))
    print(line, json.dumps(info), flush=True)


if __name__ == "__main__":
    for ln in sys.argv[1:]:
        main(ln)

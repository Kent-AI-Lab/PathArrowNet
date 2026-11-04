#!/usr/bin/env python3
"""VCC 2025 (Arc Virtual Cell Challenge; H1 hESC, CRISPRi, 10x Flex) -> the layout of the GEARS-processed Replogle files,
so that prep_stats.py / build_relations.py / eval_replication.py / recal run on it unchanged.

Inputs (downstream/data/perturb/raw/vcc2025/): adata_Training.h5ad, adata_Validation.h5ad, adata_Test.h5ad
(integer UMI counts, CSR; obs target_gene / guide_id / batch; var index = gene symbol).
  - the three files are pooled (300 perturbations); PathArrowNet was never trained on any of them
  - non-targeting cells: kept once per cell barcode (each file ships the control pool); duplicates are counted
  - condition '<gene>+ctrl', non-targeting -> 'ctrl'; guides of the same gene pooled
  - perturbations with < MIN_CELLS cells dropped (as nadig_to_processed.py)
  - normalize_total to the median total count of the kept cells, log1p; 5,000 HVGs with the same numpy re-implementation
    of scanpy 'seurat' (nadig_to_processed.hvg_seurat)
  - cell_barcode keeps the '-<batch>' suffix, used by prep_stats.py as the lane (gem group)

    .venv/bin/python -I expansion/perturbseq/vcc2025/vcc_to_processed.py
    -> downstream/data/perturb/raw/vcc2025_h1/perturb_processed.h5ad (+ conversion.json)
"""
import json, sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from nadig_to_processed import hvg_seurat, MIN_CELLS, N_HVG  # noqa: E402

RAW = HERE.parents[1] / "downstream/data/perturb/raw"
SRC = RAW / "vcc2025"
OUT = RAW / "vcc2025_h1"
FILES = ["adata_Training.h5ad", "adata_Validation.h5ad", "adata_Test.h5ad"]
CHUNK = 20000


def cat(f, k):
    g = f["obs"][k]
    cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g["categories"][()]])
    return cats[g["codes"][()]]


def rows_iter(f, rows):
    """yield (row_ids, indptr-local, indices, data) for the selected rows of a CSR h5ad, in chunks of consecutive rows"""
    ptr = f["X/indptr"][()]
    for s in range(0, len(rows), CHUNK):
        r = rows[s:s + CHUNK]
        lo, hi = ptr[r[0]], ptr[r[-1] + 1]
        ind = f["X/indices"][lo:hi]; dat = f["X/data"][lo:hi].astype(np.float64)
        # rows are sorted; build the local pointer for the selected (possibly non-contiguous) rows
        starts = ptr[r] - lo; ends = ptr[r + 1] - lo
        take = np.concatenate([np.arange(a, b) for a, b in zip(starts, ends)]) if len(r) else np.zeros(0, int)
        lp = np.concatenate([[0], np.cumsum(ends - starts)])
        yield r, lp, ind[take], dat[take]


def main():
    OUT.mkdir(exist_ok=True)
    fh = [h5py.File(SRC / n, "r") for n in FILES]
    genes = [np.array([g.decode() for g in f["var/_index"][()]]) for f in fh]
    assert all((g == genes[0]).all() for g in genes), "gene order differs between files"
    gname = genes[0]; G = len(gname)
    sel, info_files, seen_ctrl, n_dup = [], {}, set(), 0
    for n, f in zip(FILES, fh):
        tg = cat(f, "target_gene"); bt = cat(f, "batch"); bc = np.array([b.decode() for b in f["obs/_index"][()]])
        assert all(b.rsplit("-", 1)[-1] == x for b, x in zip(bc, bt)), "barcode suffix != batch"
        keep = []
        for i in range(len(tg)):
            if tg[i] == "non-targeting":
                if bc[i] in seen_ctrl:
                    n_dup += 1; continue
                seen_ctrl.add(bc[i])
            keep.append(i)
        keep = np.array(keep)
        sel.append((keep, tg[keep], bc[keep], bt[keep]))
        info_files[n] = {"cells": int(len(tg)), "ctrl": int((tg == "non-targeting").sum()),
                         "perturbations": int(len(set(tg)) - ("non-targeting" in tg)), "kept": int(len(keep)),
                         "batches": int(len(set(bt)))}
        print(n, info_files[n], flush=True)
    allg = np.concatenate([s[1] for s in sel])
    u, c = np.unique(allg, return_counts=True)
    keep_p = {p for p, k in zip(u, c) if p == "non-targeting" or k >= MIN_CELLS}
    sel = [(r[np.isin(t, list(keep_p))], t[np.isin(t, list(keep_p))], b[np.isin(t, list(keep_p))], x[np.isin(t, list(keep_p))])
           for r, t, b, x in sel]
    # pass 1: totals, then normalised mean / var per gene
    tots = []
    for f, (r, *_rest) in zip(fh, sel):
        ptr = f["X/indptr"][()]; t = np.zeros(len(r))
        for s in range(0, len(r), CHUNK):
            rr = r[s:s + CHUNK]; lo, hi = ptr[rr[0]], ptr[rr[-1] + 1]
            d = f["X/data"][lo:hi].astype(np.float64); cs = np.concatenate([[0], np.cumsum(d)])
            t[s:s + CHUNK] = cs[ptr[rr + 1] - lo] - cs[ptr[rr] - lo]
        tots.append(t)
    target = float(np.median(np.concatenate(tots)))
    s1 = np.zeros(G); s2 = np.zeros(G); nk = 0
    for f, (r, *_rest), t in zip(fh, sel, tots):
        off = 0
        for rr, lp, ind, dat in rows_iter(f, r):
            sc = np.repeat(target / t[off:off + len(rr)], np.diff(lp)); x = dat * sc
            s1 += np.bincount(ind, x, G); s2 += np.bincount(ind, x * x, G); off += len(rr)
        nk += len(r)
    mean = s1 / nk; var = (s2 - nk * mean ** 2) / (nk - 1)
    hv, dn = hvg_seurat(mean, var, N_HVG)
    cmap = np.full(G, -1, np.int64); cmap[hv] = np.arange(len(hv))
    # pass 2: log1p-normalised HVG CSR
    data, inds, ptr_out = [], [], [0]
    for f, (r, *_rest), t in zip(fh, sel, tots):
        off = 0
        for rr, lp, ind, dat in rows_iter(f, r):
            rowid = np.repeat(np.arange(len(rr)), np.diff(lp)); m = cmap[ind] >= 0
            x = np.log1p(dat[m] * (target / t[off:off + len(rr)])[rowid[m]]).astype(np.float32)
            cc = cmap[ind[m]].astype(np.int32); rw = rowid[m]
            o = np.lexsort((cc, rw)); data.append(x[o]); inds.append(cc[o])
            ptr_out.extend((np.cumsum(np.bincount(rw, minlength=len(rr))) + ptr_out[-1]).tolist()); off += len(rr)
    data = np.concatenate(data); inds = np.concatenate(inds); ptr_out = np.array(ptr_out, np.int64)
    tg = np.concatenate([s[1] for s in sel]); bc = np.concatenate([s[2] for s in sel]); bt = np.concatenate([s[3] for s in sel])
    cond = np.array(["ctrl" if g == "non-targeting" else f"{g}+ctrl" for g in tg])
    _, gem = np.unique(bt, return_inverse=True)
    with h5py.File(OUT / "perturb_processed.h5ad", "w") as o:
        X = o.create_group("X"); X.attrs["encoding-type"] = "csr_matrix"; X.attrs["encoding-version"] = "0.1.0"
        X.attrs["shape"] = np.array([nk, len(hv)])
        X.create_dataset("data", data=data, compression="gzip"); X.create_dataset("indices", data=inds, compression="gzip")
        X.create_dataset("indptr", data=ptr_out)
        ob = o.create_group("obs"); va = o.create_group("var")
        ob.create_dataset("condition", data=cond.astype("S")); ob.create_dataset("cell_barcode", data=bc.astype("S"))
        ob.create_dataset("gem_group", data=gem.astype(np.int64))
        va.create_dataset("gene_name", data=gname[hv].astype("S")); va.create_dataset("gene_id", data=np.array([b""] * len(hv)))
        va.create_dataset("dispersions_norm", data=dn[hv])
    info = {"sources": FILES, "files": info_files, "duplicate_control_barcodes_dropped": n_dup,
            "cells_kept": int(nk), "genes_in_file": int(G), "perturbations_kept": int(len(keep_p) - 1),
            "control_cells": int((tg == "non-targeting").sum()), "min_cells": MIN_CELLS,
            "normalize_total_target": target, "n_hvg": int(len(hv)), "batches": int(len(set(bt))), "nnz": int(len(data))}
    (OUT / "conversion.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()

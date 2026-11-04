#!/usr/bin/env python3
"""Original BioPathNet (biopathnet+bg: official NBFNet config + STRING background, no extra node features; strict split 1,
5 seeds, each run's own selected checkpoint) as a zero-shot KO-pair scorer for the downstream controls.

For every anchor KO u (all KOs of the LR / Reactome / perturbation gene lists) the model scores (u, precedes, v) for all
KOs v with its train-only fact graph; rows are log-softmax normalised over candidates (the analogue of PathArrowNet's
log p(v | u ->)) and averaged over seeds. Unknown KOs (not BioPathNet entities) get the row minimum.
Writes cache/bpn_orig_ko_scores.npz (anchors, S [n_anchor, V] float32). BioPathNet venv (model loading: biopathnet_repro/leakage/bpn_infer.py).

    biopathnet_repro/.venv/bin/python expansion/downstream/bpn_ko_scores.py [--all]
"""
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(_KH + "/biopathnet_repro")
BPN = ROOT / "BioPathNet"
PATHWAY = Path(_KH + "/pathway")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(BPN)); sys.path.insert(0, str(PATHWAY / "benchmark_v2"))
import scorelib as SL  # noqa: E402

SETTING, METHOD = "strict_v1_full", "biopathnet+bg"
GENE_FILES = ["data/ligand_receptor/lr_genes.tsv", "data/reactome/reactome_genes.tsv"]


def anchors(voc):
    s = set()
    for f in GENE_FILES:
        for r in csv.DictReader((HERE / f).open(), delimiter="\t"):
            s |= {voc[k] for k in (r["kos_in_vocab"] or "").split(";") if k in voc}
    for d in ("adamson", "norman"):  # perturbation genes (perturbed + measured)
        z = np.load(HERE / f"data/perturb/stats_{d}.npz", allow_pickle=True)
        names = {str(x) for x in z["genes"]} | {str(x) for x in z["perts"]}
        for r in csv.DictReader((HERE / "data/common/gene_map.tsv").open(), delimiter="\t"):
            if r["symbol"] in names:
                s |= {voc[k] for k in (r["kos_in_vocab"] or "").split(";") if k in voc}
    return sorted(s)


def main():
    voc, V = SL.vocab(SETTING)
    all_kos = "--all" in sys.argv  # every KO as anchor (multi-step network evaluation)
    A = list(range(3, V)) if all_kos else anchors(voc)
    print("anchors", len(A), flush=True)
    sys.path.insert(0, str(ROOT / "leakage"))
    import bpn_infer as BI  # dataset / task loading without any training component
    util = BI.setup()
    tot, n = np.zeros((len(A), V), np.float64), 0
    runs = sorted(p for p in (SL.SCORES / SETTING / METHOD).iterdir() if p.is_dir())
    info = []
    for run in runs:
        src = json.loads((run / "source.json").read_text()); wd = src["working_dir"]
        ck = re.findall(r"Load checkpoint from (model_epoch_\d+\.pth)", (Path(wd) / "log.txt").read_text())[-1]
        cfg_path = Path(src["config"])
        cfg, dataset, task, _ = BI.load(cfg_path, BI.data_path(cfg_path, util), None, util)
        BI.load_checkpoint(task, Path(wd) / ck)
        dev = next(task.parameters()).device
        ent = list(dataset.entity_vocab); rel = dataset.inv_relation_vocab["precedes"]
        ours2bpn = {voc[t]: i for i, t in enumerate(ent) if t in voc}
        bpn_ids = torch.tensor([ours2bpn.get(o, -1) for o in range(V)], device=dev); has = bpn_ids >= 0
        N = len(ent); all_idx = torch.arange(N, device=dev)
        rows = []
        with torch.no_grad():
            for i in range(0, len(A), 32):
                ch = A[i:i + 32]
                idx = torch.tensor([ours2bpn.get(x, 0) for x in ch], device=dev)
                r_index = torch.full((len(ch), N), rel, device=dev, dtype=torch.long)
                pred = task.model(task.fact_graph, idx.unsqueeze(-1).expand(-1, N), all_idx.unsqueeze(0).expand(len(ch), -1), r_index)
                full = torch.full((len(ch), V), float("nan"), device=dev, dtype=torch.float64)
                full[:, has] = torch.log_softmax(pred[:, bpn_ids[has]].double(), 1)
                full = torch.where(torch.isnan(full), full.nan_to_num(nan=1e9).amin(1, keepdim=True), full)
                full[torch.tensor([x not in ours2bpn for x in ch], device=dev)] = 0.0  # anchor unknown to BioPathNet
                rows.append(full.cpu().numpy())
        tot += np.concatenate(rows); n += 1
        info.append({"run": run.name, "working_dir": wd, "checkpoint": ck})
        print(run.name, ck, "done", flush=True)
        del task
    np.savez(HERE / ("cache/bpn_orig_ko_scores_all.npz" if all_kos else "cache/bpn_orig_ko_scores.npz"), anchors=np.array(A), S=(tot / n).astype(np.float32),
             info=json.dumps(info))
    print("saved", flush=True)


if __name__ == "__main__":
    main()

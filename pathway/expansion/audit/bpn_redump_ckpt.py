#!/usr/bin/env python3
"""
Re-dump BioPathNet (bpn_keep1hop+exp150bin4+esm+gobert_noBP) scores from a GIVEN checkpoint of an existing run, for the
two runs whose checkpoint selection changes once test edges are removed from the valid filter
(bpn_valid_filter_fix.py: strict_v1_full s1028 epoch 14 -> 12, strict_s44_full s1025 epoch 10 -> 2).
Nothing is retrained; the run's own node_features.pt, config and train-only fact graph are reused (loading via
biopathnet_repro/leakage/bpn_infer.py). As a check, the
originally selected checkpoint is dumped too and must reproduce the stored benchmark_v2 scores (test A MRR).

    biopathnet_repro/.venv/bin/python expansion/audit/bpn_redump_ckpt.py <setting> <seed> <ckpt> [<ckpt> ...]
    -> expansion/audit/corrected_scores/<setting>/bpn_keep1hop+exp150bin4+esm+gobert_noBP/s<seed>__<ckpt>/tail_{val,test}.npz
"""
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(_KH + "/biopathnet_repro")
PATHWAY = Path(_KH + "/pathway")
AUD = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "leakage")); sys.path.insert(0, str(PATHWAY / "benchmark_v2")); sys.path.insert(0, str(PATHWAY))
import bpn_infer as BI  # noqa: E402  (dataset / task loading without any training component)
import scorelib as SL  # noqa: E402

METHOD = "bpn_keep1hop+exp150bin4+esm+gobert_noBP"


def main(setting, seed, ckpts):
    src = json.loads((SL.SCORES / setting / METHOD / f"s{seed}" / "source.json").read_text())
    wd, cfg_path = Path(src["working_dir"]), Path(src["config"])
    util = BI.setup()
    cfg, dataset, task, _ = BI.load(cfg_path, BI.data_path(cfg_path, util), wd / "node_features.pt", util)
    voc, V = SL.vocab(setting)
    anchors = SL.anchors_needed(setting)
    q = SL.load_A(setting, "test")
    out_all = {}
    for ck in ckpts:
        BI.load_checkpoint(task, wd / ck)
        sc = BI.tail_scores(task, dataset, setting, anchors, 32)
        out = AUD / "corrected_scores" / setting / METHOD / f"s{seed}__{ck.replace('.pth', '')}"
        for split in ("val", "test"):
            SL.save_tail(out / f"tail_{split}.npz", anchors, sc)
        rk, _ = SL.ranks(SL.query_scores(out, "test", "A", q), q, V)
        out_all[ck] = float(np.mean(1 / rk))
        print(setting, seed, ck, "test A MRR %.4f" % out_all[ck], flush=True)
    rk0, _ = SL.ranks(SL.query_scores(SL.SCORES / setting / METHOD / f"s{seed}", "test", "A", q), q, V)
    out_all["stored_benchmark_v2"] = float(np.mean(1 / rk0))
    print("stored", out_all["stored_benchmark_v2"], flush=True)
    (AUD / "results_extra" / f"bpn_redump_{setting}_s{seed}.json").write_text(json.dumps(out_all, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:])

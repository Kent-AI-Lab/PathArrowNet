#!/usr/bin/env python3
"""Perturb-seq replication (eval_replication.py, unchanged) with the relation-weighted primary model (KGML weighting)
as "konbf": ko_level() is wrapped so that M["konbf"] = expansion/downstream/cache/konbf_full_matrix_kgml.npy
(5 checkpoints xp_kgmlw+exp150bin4+allneg+kl_s42..46, tail + head^T); every other method (konbf_randinit, kegg_enc,
bpn_orig, string_exp, kegg_direct, random, responder, string_degree) reads the same inputs as before. Datasets are run in
the same groups and order as the original files (rng = default_rng(0) per run, consumed in the same order):
    orig4   replogle_k562_essential replogle_rpe1_essential adamson norman -> results/replication_gem_kgml.json
    nadig   nadig_hepg2_essential nadig_jurkat_essential                   -> results/replication_gem_nadig_kgml.json
    vcc2025 vcc2025_h1                                                      -> results/replication_gem_vcc2025_kgml.json
    .venv/bin/python expansion/perturbseq/eval_replication_kgml.py orig4|nadig|vcc2025 [gem|pooled]"""
import os, sys
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_replication as ER  # noqa: E402

GROUPS = {"orig4": ("_kgml", ["replogle_k562_essential", "replogle_rpe1_essential", "adamson", "norman"]),
          "nadig": ("_nadig_kgml", ["nadig_hepg2_essential", "nadig_jurkat_essential"]),
          "vcc2025": ("_vcc2025_kgml", ["vcc2025_h1"])}
_orig_ko_level = ER.ko_level


def ko_level(voc):
    M = _orig_ko_level(voc)
    M["konbf"] = torch.tensor(np.load(ER.DS / "cache/konbf_full_matrix_kgml.npy"))
    return M


if __name__ == "__main__":
    group = sys.argv[1]; scheme = sys.argv[2] if len(sys.argv) > 2 else "gem"
    tag, dsets = GROUPS[group]
    out = HERE / "results" / f"replication_{scheme}{tag}.json"
    assert not out.exists(), f"{out} exists (eval_replication appends to an existing file)"
    ER.ko_level = ko_level
    os.environ["REPL_TAG"] = tag
    sys.argv = ["eval_replication.py", scheme] + dsets
    ER.main()

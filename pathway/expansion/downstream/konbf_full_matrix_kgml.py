#!/usr/bin/env python3
"""Full KO x KO score matrix of the primary PathArrowNet (KGML relation-level weights; runs kegg_dataset/nbf_ko_full/
xp_kgmlw+exp150bin4+allneg+kl_s42..46) from kegg_pairscore.nbf_matrix. Every KO is an anchor. -> cache/konbf_full_matrix_kgml.npy (float32 [V, V]).
    .venv/bin/python expansion/downstream/konbf_full_matrix_kgml.py"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kegg_pairscore as KP  # noqa: E402
import csv
RUN = "xp_kgmlw+exp150bin4+allneg+kl"
with (KP.ROOT / "kegg_dataset/token_vocab_ko_full.csv").open() as f:
    V = sum(1 for _ in csv.DictReader(f))
out = HERE / "cache/konbf_full_matrix_kgml.npy"
assert not out.exists(), out
import torch
M = KP.nbf_matrix(list(range(3, V)), run=RUN, dev="cuda" if torch.cuda.is_available() else "cpu").cpu().numpy()
np.save(out, M.astype(np.float32))
print("saved", out, M.shape, float(M[3:, 3:].min()), float(M[3:, 3:].max()))

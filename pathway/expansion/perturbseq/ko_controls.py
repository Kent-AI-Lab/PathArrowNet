#!/usr/bin/env python3
"""Full KO x KO matrices for the two GPU-based control methods of the perturbation evaluation, for every KO (the
Adamson / Norman cache only has rows for their genes):
    konbf_randinit   PathArrowNet architecture / features / STRING graph with untrained (seeded) weights, 5 seeds
    kegg_enc         KEGG-pretrained two-tower encoder, seed 0
Uses expansion/downstream/kegg_pairscore.py unchanged.
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python expansion/perturbseq/ko_controls.py
    -> expansion/perturbseq/cache/{konbf_randinit,kegg_enc}_full.npy"""
import csv, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "downstream"))
import kegg_pairscore as KP  # noqa: E402
with (KP.ROOT / "kegg_dataset/token_vocab_ko_full.csv").open() as f:
    V = sum(1 for _ in csv.DictReader(f))
(HERE / "cache").mkdir(exist_ok=True)
S, _ = KP.ko_matrix("kegg_enc", 0, "strict_v1_full", "cuda")
np.save(HERE / "cache/kegg_enc_full.npy", S.cpu().numpy().astype(np.float32)); print("kegg_enc", tuple(S.shape), flush=True)
M = KP.nbf_matrix(list(range(3, V)), dev="cuda", randinit=True).cpu().numpy()
np.save(HERE / "cache/konbf_randinit_full.npy", M.astype(np.float32)); print("randinit", M.shape, flush=True)

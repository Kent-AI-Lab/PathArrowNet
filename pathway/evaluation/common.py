"""
Shared helpers for the paper evaluations. Every script in this folder reads stored score dumps and writes one JSON file to
pathway/results/evaluation/; no model is trained or run here.

Score dumps live in benchmark_v2/scores/<setting>/<method>/s<seed>/ (see benchmark_v2/scorelib.py):
    tail_{val,test}.npz        anchors [n], scores [n, V]   (anchor models: the score depends only on the source KO)
    seq_{val,test}_{A,S}.npz   scores [n_queries, V]        (one row per query; written by seed averaging)
PathArrowNet dumps are written by `python -m patharrownet.predict --dump-benchmark`; seed-averaged dumps by
benchmark_v2/improve/calib_ensemble.py --fixed 0,0,0 (PathArrowNet) or --fixed 1,0,0 (BioPathNet), which averages the row
z-scores of the seeds (no tuning, no mixing of models).

Conventions (paper, Sec. 3.3 and Suppl. S3)
    single model   per-query reciprocal ranks averaged over seeds (= mean of the per-seed MRR)
    seed-averaged  one score matrix: the mean of the row z-scores of the seeds
    per split      common_stats.boot: 2,000 resamples of test pathway clusters, query weights
    pooled         common_stats.pooled: five strict splits, every test pathway weighted equally, 10,000 resamples of
                   connected pathway clusters
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

PW = Path(__file__).resolve().parents[1]                         # .../pathway
sys.path.insert(0, str(PW / "benchmark_v2"))
sys.path.insert(0, str(PW / "expansion"))
import scorelib as SL  # noqa: E402
import common_stats as CS  # noqa: E402
from ensemble import z as ZS  # noqa: E402  (row z-score; masked -1e30 entries are set to the row minimum first)

OUT = Path(os.environ.get("PATHARROWNET_RESULTS", PW / "results" / "evaluation"))
STS = ["strict_v1_full", "strict_s43_full", "strict_s44_full", "strict_s45_full", "strict_s46_full"]
SPLIT = {st: f"split {i + 1}" for i, st in enumerate(STS)}

# method names of the score dumps
PAN, PAN_SA = "clean_nbf_kgml", "clean_nbf_kgml_seedavg"        # PathArrowNet (primary: relation-level KGML weights)
PAN0, PAN0_SA = "clean_nbf_kl", "clean_nbf_kl_seedavg"          # PathArrowNet without relation-level weights
REP, REP_SA = "biopathnet+bg", "clean_bpn_pub_seedavg"          # reproduced BioPathNet (official code and inputs + STRING)
ADA, ADA_SA = "clean_bpn_adapt", "clean_bpn_adapt_seedavg"      # BioPathNet with our adaptations (ESM2 features, 4 STRING bins)
RANKER = "baseline_topo_lr"                                     # network feature ranker, STRING + KEGG training-graph features
REV = "clean_nbf_kgml_revanchors"                               # PathArrowNet dumps that also score the val/test targets
STRING_FILE = PW / "kegg_dataset_cache" / "string" / "ko_background_exp150_full.tsv"

_Q: dict = {}


def Q(st, split="test", task="A"):
    k = (st, split, task)
    if k not in _Q:
        _Q[k] = SL.load_queries(st, split, task)
    return _Q[k]


def runs(st, m, seeds=None):
    d = SL.SCORES / st / m
    if seeds is not None:
        return [d / s for s in seeds]
    return sorted(p for p in d.iterdir() if p.is_dir())


def ranks_all(st, m, split="test", task="A", seeds=None):
    """filtered mean-tie ranks [n_runs, n_queries]"""
    _, V = SL.vocab(st)
    qs = Q(st, split, task)
    return np.array([SL.ranks(SL.query_scores(r, split, task, qs).astype(np.float64), qs, V)[0] for r in runs(st, m, seeds)], dtype=float)


def rr(st, m, split="test", task="A", seeds=None):
    """per-query reciprocal rank averaged over the runs of a method (single model) or of its one seed-averaged dump"""
    return (1 / ranks_all(st, m, split, task, seeds)).mean(0)


def seedavg_rr(st, m, seeds, split="test", task="A"):
    """per-query RR of the seed average built on the fly from a subset of seeds (mean of row z-scores)"""
    _, V = SL.vocab(st)
    qs = Q(st, split, task)
    Z = np.mean([ZS(SL.query_scores(r, split, task, qs)) for r in runs(st, m, seeds)], 0)
    return 1 / np.asarray(SL.ranks(Z, qs, V)[0], dtype=float)


def macro_S(st, R):
    """task-S pathway-macro MRR from ranks [n_runs, n_queries] (mean over runs)"""
    w = np.array([q["weight"] for q in Q(st, "test", "S")])
    return float(np.mean([(w / x).sum() / w.sum() for x in R]))


def boot(a, b, qs, **kw):
    e, lo, hi = CS.boot(a, b, qs, **kw)
    return {"est": e, "lo": lo, "hi": hi}


def sig(d):
    return bool(d["lo"] > 0 or d["hi"] < 0)


def string_pairs(voc):
    """directed set of KO-id pairs with a STRING experimental (>= 150) edge, both directions"""
    S = set()
    for line in STRING_FILE.read_text().splitlines()[1:]:
        a, b = line.split("\t")[:2]
        S.add((voc[a], voc[b])); S.add((voc[b], voc[a]))
    return S


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(obj, indent=1, default=float))
    print("wrote", p, flush=True)
    return p

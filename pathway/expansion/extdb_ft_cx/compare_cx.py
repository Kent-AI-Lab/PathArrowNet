#!/usr/bin/env python3
"""Original against complex-aware SIGNOR split (CPU). Reads predictions only; missing runs are reported, never filled.
  diag   original-split FT test edges stratified by whether their complex-aware component contains a train edge of the
         ORIGINAL split ("sibling in train") -- uses only the existing extdb_ft predictions
  q1     the test edges shared by both splits: original FT - complex-aware FT (link MRR), paired bootstrap over source KOs
  q4     original split: FT-3000 - FT-1500 on the same test edges (link MRR, sign AUROC); selected steps; KEGG test MRR
  strata complex-aware FT link MRR by protein_only / expanded
    .venv/bin/python expansion/extdb_ft_cx/compare_cx.py
"""
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())
import csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

X = Path(_KH + "/pathway/expansion")
sys.path.insert(0, str(X / "extdb_ft"))
import summarize as S  # noqa: E402  (fast_auroc, ci_p)

SEEDS = (42, 43, 44)
ORIG, CX, LONG = X / "extdb_ft/runs", X / "extdb_ft_cx/main/runs", X / "extdb_ft_cx/long/runs"


def rows(p):
    R = list(csv.DictReader(open(p), delimiter="\t"))
    return [r for r in R if r["split"] == "test"], R


def link(run_dir, init="ckpt"):
    f = [run_dir / f"signor_{init}_s{s}/pred_link_test.npz" for s in SEEDS]
    if not all(x.exists() for x in f):
        return None
    return np.mean([1 / np.load(x)["rank"] for x in f], 0)


def sign_p(run_dir):
    f = [run_dir / f"signor_ckpt_s{s}/pred_sign_test.npz" for s in SEEDS]
    if not all(x.exists() for x in f):
        return None
    return np.mean([1 / (1 + np.exp(-np.load(x)["sign_logit"])) for x in f], 0)


def boot_src(src, fn, B=2000):
    u = np.unique(src); idx = {c: np.where(src == c)[0] for c in u}; rng = np.random.default_rng(0)
    return np.array([fn(np.concatenate([idx[c] for c in rng.choice(u, len(u))])) for _ in range(B)])


out = {}
to, _ = rows(X / "extdb_ft/cache/signor.tsv")
tc, allc = rows(X / "extdb_ft_cx/main/cache/signor.tsv")
comp = {(r["x"], r["y"]): r["cx_comp"] for r in allc}
orig_train_comps = {r["cx_comp"] for r in allc if r["split_orig"] == "train"}
ro = link(ORIG)
sib = np.array([comp[(r["x"], r["y"])] in orig_train_comps for r in to])
po = np.array([r["protein_only"] == "1" for r in to])
out["diag_original_split_ft"] = {
    k: {"n": int(m.sum()), "mrr": float(ro[m].mean())} for k, m in
    (("all", np.ones(len(to), bool)), ("sibling_in_train", sib), ("no_sibling_in_train", ~sib),
     ("protein_only", po), ("expanded_only", ~po), ("protein_only_no_sibling", po & ~sib), ("expanded_no_sibling", ~po & ~sib))}

rc = link(CX)
if rc is not None:
    po_c = np.array([r["protein_only"] == "1" for r in tc])
    out["cx_ft_strata"] = {k: {"n": int(m.sum()), "mrr": float(rc[m].mean())} for k, m in
                           (("all", np.ones(len(tc), bool)), ("protein_only", po_c), ("expanded_only", ~po_c))}
    io = {(r["x"], r["y"]): i for i, r in enumerate(to)}
    shared = [(io[(r["x"], r["y"])], j) for j, r in enumerate(tc) if (r["x"], r["y"]) in io]
    a = np.array([i for i, _ in shared]); b = np.array([j for _, j in shared])
    d = ro[a] - rc[b]; src = np.array([to[i]["x"] for i in a])
    out["q1_shared_test_edges"] = {"n": len(shared), "orig_ft_mrr": float(ro[a].mean()), "cx_ft_mrr": float(rc[b].mean()),
                                   **S.ci_p(d.mean(), boot_src(src, lambda ix: d[ix].mean()))}
else:
    out["q1_shared_test_edges"] = "missing"

rl = link(LONG)
if rl is not None:
    src = np.array([r["x"] for r in to]); d = rl - ro
    q4 = {"link_mrr_1500": float(ro.mean()), "link_mrr_3000": float(rl.mean()),
          "link": S.ci_p(d.mean(), boot_src(src, lambda ix: d[ix].mean()))}
    y = np.array([int(r["sign"]) if r["sign"] != "" else -1 for r in to]); k = y >= 0
    pa, pb = sign_p(LONG), sign_p(ORIG)
    est = S.fast_auroc(y[k], pa[k]) - S.fast_auroc(y[k], pb[k])

    def f(ix):
        ix = ix[k[ix]]
        return S.fast_auroc(y[ix], pa[ix]) - S.fast_auroc(y[ix], pb[ix])
    q4["sign_auroc_1500"] = S.fast_auroc(y[k], pb[k]); q4["sign_auroc_3000"] = S.fast_auroc(y[k], pa[k])
    q4["sign"] = S.ci_p(est, boot_src(src, f))
    q4["runs"] = {}
    for s in SEEDS:
        jl = json.loads((LONG / f"signor_ckpt_s{s}/result.json").read_text())
        jo = json.loads((ORIG / f"signor_ckpt_s{s}/result.json").read_text())
        q4["runs"][s] = {"best_step_3000": jl["best_step"], "best_step_1500": jo["best_step"],
                         "steps_run_3000": jl["log"][-1]["step"],
                         "kegg_after_link_3000": jl["final"]["link"]["kegg_test_A_micro_mrr"],
                         "kegg_after_link_1500": jo["final"]["link"]["kegg_test_A_micro_mrr"]}
    out["q4_step_cap"] = q4
else:
    out["q4_step_cap"] = "missing"
(X / "extdb_ft_cx/results/compare.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

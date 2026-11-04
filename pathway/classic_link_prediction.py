#!/usr/bin/env python3
"""
Classic link-prediction methods on a KO split, scored with the shared protocol (filtered MRR/Hits via
kegg_pipeline.evaluation semantics, plus tail-corruption AUC with shared negatives).

    .venv/bin/python classic_link_prediction.py --split-dir kegg_dataset/splits_ko --out kegg_dataset/comparisons_ko/strict
    .venv/bin/python classic_link_prediction.py --split-dir kegg_leaky_splits/unique_random --tasks A --out ...

Methods (fitted on the split's train edges only; no tuning)
    popularity, cf, go_profile_no_BP   the non-learned baselines of evaluate_ko_baselines.py
    adamic_adar          directed 2-hop: sum_x [u->x][x->v] / log(deg x)      (deg = in + out, min 2)
    resource_allocation  directed 2-hop: sum_x [u->x][x->v] / deg x
    katz                 sum_{k=1..4} beta^k (A^k)_uv, beta = 0.05
    svd                  rank-64 truncation of the exact SVD of the adjacency matrix, score = (U S V^T)_uv
    (the node2vec baseline of the paper is a trained embedding; its training code is not part of this release)
Task C (l -> ? -> r) uses M[l, v] + M[v, r] for every matrix method M.
All methods add eps * popularity as an explicit tie-breaker (as in the baselines).
SEAL-type subgraph methods are not included: ranking 2,870 candidates for ~15k queries would require
~43M subgraph extractions.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import evaluate_ko_baselines as EB  # noqa: E402
from kegg_pipeline import evaluation as E  # noqa: E402
from patharrownet.data import gpu_ranks, load_vocab  # noqa: E402



def matrices(T, V, device):
    A = torch.zeros(V, V, device=device)
    for u, v in T.edges:
        A[u, v] = 1.0
    deg = A.sum(0) + A.sum(1)
    out = {}
    w = 1 / torch.log(deg.clamp(min=2))
    out["adamic_adar"] = (A * w[None, :]) @ A
    out["resource_allocation"] = (A * (1 / deg.clamp(min=1))[None, :]) @ A
    beta, P, K = 0.05, torch.eye(V, device=device), torch.zeros(V, V, device=device)
    for k in range(1, 5):
        P = P @ A
        K += beta ** k * P
    out["katz"] = K
    U, S, Vh = torch.linalg.svd(A.double())  # exact SVD (deterministic; randomized svd_lowrank was not)
    out["svd"] = ((U[:, :64] * S[:64]) @ Vh[:64]).float()
    return out


def eval_matrix(M, pop, queries, cand_mask, device):
    """M: [V, V] score matrix (source x target). Returns ranks per task and AUCs for A."""
    res = {}
    for task, qs in queries.items():
        ranks, aucs = [], []
        for i in range(0, len(qs), 4096):
            ch = qs[i:i + 4096]
            if task == "A":
                sc = M[torch.tensor([q["source"] for q in ch], device=device)]
            else:
                l = torch.tensor([q["left"] for q in ch], device=device)
                r = torch.tensor([q["right"] for q in ch], device=device)
                sc = M[l] + M[:, r].T
            sc = sc + EB.EPS * pop[task][None, :]
            ranks += gpu_ranks(sc, ch, cand_mask)
            if task == "A":
                t = torch.tensor([q["target"] for q in ch], device=device)
                ng = torch.tensor([q["neg"] for q in ch], device=device)
                sp, sn = sc.gather(1, t[:, None]), sc.gather(1, ng)
                aucs += ((sp > sn).float() + 0.5 * (sp == sn).float()).mean(1).tolist()
        res[task] = ranks
        if task == "A":
            res["A_auc"] = aucs
    return res


def eval_scorer(fn_by_task, queries, cands):
    res = {}
    for task, qs in queries.items():
        ranks, aucs = [], []
        for q in qs:
            s = fn_by_task[task](q)
            ranks.append(E.filtered_rank(s, q["target"], cands, q["exclude"]))
            if task == "A":
                aucs.append(E.auc_query(s, q))
        res[task] = ranks
        if task == "A":
            res["A_auc"] = aucs
    return res


def save(out, name, split_q, results):
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    m = {}
    for split, qd in split_q.items():
        r = results[split]
        for task, qs in qd.items():
            m[f"{split}_{task}"] = E.aggregate(qs, [E.per_query(x) for x in r[task]])
        if "A_auc" in r:
            m[f"{split}_A_auc"] = E.aggregate_auc(qd["A"], r["A_auc"])
            import csv as _csv
            import gzip as _gz
            with _gz.open(d / f"auc_{split}_A.csv.gz", "wt", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["query_index", "auc"])
                w.writerows(enumerate(r["A_auc"]))
    (d / "metrics.json").write_text(json.dumps(m, indent=2))
    ta = m.get("test_A_auc", {}).get("AUC_micro", float("nan"))
    print(f"{name}: test A MRR macro {next(x['MRR'] for x in m['test_A'] if x['subset'] == 'all' and x['weighting'] == 'macro'):.4f}"
          f"  AUC {ta:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="AC")
    a = ap.parse_args()
    sd, out = Path(a.split_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, V = load_vocab()
    cands = list(range(3, V))
    cand_mask = torch.zeros(V, dtype=torch.bool, device=device)
    cand_mask[3:] = True
    all_edges = E.load_all_ko_edges(vocab)
    split_q = {}
    for split in ("val", "test"):
        split_q[split] = {t: E.load_queries(sd / f"{split}_{t}_{'next_node' if t == 'A' else 'masked'}.csv.gz", t)
                          for t in a.tasks}
        E.attach_negatives(split_q[split]["A"], all_edges, cands, tag=f"{sd.name}|{split}")
    T = EB.Train(V, sd)
    pop = {t: torch.tensor(T.pop[t], device=device) for t in "AC"}

    # non-learned baselines (same scorers as evaluate_ko_baselines.py)
    ko_go = EB.load_ko_go(vocab, *EB.GO_VARIANTS["go_profile_no_BP"])
    Sc = EB.Scorers(T, ko_go, cands)
    for name in ("popularity", "cf", "go_profile_no_BP"):
        meth = "go_profile" if name == "go_profile_no_BP" else name
        fns = {t: getattr(Sc, f"{t}_{meth}") for t in a.tasks}
        save(out, name, split_q, {s: eval_scorer(fns, qd, cands) for s, qd in split_q.items()})

    for name, M in matrices(T, V, device).items():
        save(out, name, split_q, {s: eval_matrix(M, pop, qd, cand_mask, device) for s, qd in split_q.items()})
        del M


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluate every stored score matrix under benchmark_v2/scores/<setting>/<method>/s<seed>/ with ONE protocol.

    .venv/bin/python benchmark_v2/eval_all.py [--settings strict_v1,unique_random,random_segment] [--methods m1,m2]

Per (setting, method, seed, split, task, subset):
  MRR (micro: every query equal), MRR_macro (every held-out pathway equal), Hits@1/3/5/10, mean rank, median rank
  task A only (one AUC value per query, averaged over queries):
    AUC_uniform_s{42..46}  10 uniform random negatives per query, 5 negative seeds (seed 42 = earlier protocol)
    AUC_same_pathway       10 negatives from the query pathway's own KOs that are not successors of u (mean of 5 seeds)
    AUC_two_hop            10 negatives two steps downstream of u (u -> x -> w), not successors of u (mean of 5 seeds)
    AUC_degree_matched     10 negatives drawn proportional to train in-degree (hubs) (mean of 5 seeds)
    AUC_all_candidates     target vs every filtered candidate (= 1 - (rank-1)/(n_candidates-1))
Negatives are generated once per (setting, split, strategy, seed) and shared by every model.
Writes out/results_runs.csv (per seed) and out/results_summary.csv (mean and sd over seeds), and
out/negative_sampling_stats.csv.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

B2 = Path(__file__).resolve().parent
sys.path.insert(0, str(B2))
import scorelib as SL  # noqa: E402

OUT = B2 / "out"
NEG_SEEDS = (42, 43, 44, 45, 46)
HARD = ("same_pathway", "two_hop", "degree_matched")
TAG = {"strict_v1": "splits_ko", "unique_random": "unique_random", "random_segment": "random_segment",
       "strict_v1_full": "splits_ko_full", "strict_v1_full_tv": "splits_ko_full",
       **{f"strict_s{s}_full": f"seed{s}_full" for s in (43, 44, 45, 46)}}


def pad(negs):
    L = max(1, max(len(n) for n in negs))
    arr = np.full((len(negs), L), -1, dtype=np.int64)
    for i, n in enumerate(negs):
        arr[i, :len(n)] = n
    return arr


def auc_vec(sc, t, neg):
    """sc [n, V], t [n], neg [n, L] (-1 = padding). Per-query AUC, NaN if no negatives."""
    m = neg >= 0
    s_pos = sc[np.arange(len(t)), t][:, None]
    s_neg = np.take_along_axis(sc, np.where(m, neg, 0), axis=1)
    v = ((s_pos > s_neg) + 0.5 * (s_pos == s_neg)) * m
    cnt = m.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, v.sum(1) / np.maximum(cnt, 1), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="strict_v1,unique_random,random_segment")
    ap.add_argument("--methods", default="")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--tag", default="", help="suffix of the output files")
    a = ap.parse_args()
    unseen_mask = None
    rows, negstats = [], []
    samplers = {}
    for setting in a.settings.split(","):
        base = SL.SCORES / setting
        if not base.exists():
            continue
        voc, V = SL.vocab(setting)
        if V not in samplers:
            samplers[V] = SL.NegativeSampler(V, setting)
        sampler = samplers[V]
        tasks = ["A"] + (["S"] if SL.SPLIT_DIRS[setting][1] is not None else [])
        qcache, negcache = {}, {}
        for split in a.splits.split(","):
            for task in tasks:
                qcache[(split, task)] = SL.load_queries(setting, split, task)
            qs = qcache[(split, "A")]
            tag = f"{TAG[setting]}|{split}"
            for seed in NEG_SEEDS:
                negcache[(split, "uniform", seed)] = pad(sampler.sample(qs, "uniform", seed, tag))
                for h in HARD:
                    negcache[(split, h, seed)] = pad(sampler.sample(qs, h, seed, tag))
            # negative-set statistics (model independent)
            train_tokens = set()
            for r in SL.rd(SL.SPLIT_DIRS[setting][0] / "train_A_next_node.csv.gz"):
                train_tokens |= {int(r["context_token_id"]), int(r["target_token_id"])}
            for strat in ("uniform",) + HARD:
                ng = negcache[(split, strat, 42)]
                valid = ng >= 0
                same_pw = 0
                for i, q in enumerate(qs):
                    nodes = set().union(*(sampler.pw_nodes.get(p, set()) for p in q["pathway_ids"].split(";")))
                    same_pw += sum(1 for x in ng[i] if x >= 0 and x in nodes)
                negstats.append({"setting": setting, "split": split, "strategy": strat,
                                 "queries": len(qs), "queries_with_>=1_negative": int((valid.sum(1) > 0).sum()),
                                 "queries_with_10_negatives": int((valid.sum(1) >= 10).sum()),
                                 "mean_negatives_per_query": round(float(valid.sum(1).mean()), 2),
                                 "share_neg_unseen_in_train": round(float(np.mean([x not in train_tokens for x in ng[valid]])), 3),
                                 "share_neg_in_query_pathway": round(same_pw / max(1, int(valid.sum())), 3),
                                 "share_neg_known_edge_of_u": 0.0})
        methods = sorted(p.name for p in base.iterdir() if p.is_dir())
        if a.methods:
            methods = [m for m in methods if m in a.methods.split(",")]
        for m in methods:
            for sd in sorted((base / m).iterdir()):
                if not sd.is_dir():
                    continue
                for split in a.splits.split(","):
                    for task in tasks:
                        qs = qcache[(split, task)]
                        try:
                            sc = SL.query_scores(sd, split, task, qs).astype(np.float64)
                        except FileNotFoundError:
                            continue
                        rk, nc = SL.ranks(sc, qs, V)
                        extra = {}
                        if task == "A":
                            t = np.array([q["target"] for q in qs])
                            for seed in NEG_SEEDS:
                                extra[f"AUC_uniform_s{seed}"] = float(np.nanmean(auc_vec(sc, t, negcache[(split, 'uniform', seed)])))
                            u = [extra[f"AUC_uniform_s{s}"] for s in NEG_SEEDS]
                            extra["AUC_uniform_mean5"] = statistics.mean(u)
                            extra["AUC_uniform_sd5"] = statistics.stdev(u)
                            for h in HARD:
                                vals = [float(np.nanmean(auc_vec(sc, t, negcache[(split, h, s)]))) for s in NEG_SEEDS]
                                extra[f"AUC_{h}"] = statistics.mean(vals)
                                extra[f"AUC_{h}_sd5"] = statistics.stdev(vals)
                            extra["AUC_all_candidates"] = float(np.mean(1 - (rk - 1) / (nc - 1)))
                        for subset in ("all", "seen", "unseen"):
                            mt = SL.metrics(qs, rk, subset)
                            if mt is None:
                                continue
                            rows.append({"setting": setting, "method": m, "seed": sd.name, "split": split, "task": task,
                                         "subset": subset, **mt, **(extra if subset == "all" else {})})
                print(f"{setting} {m}: {len(list((base / m).iterdir()))} seeds", flush=True)
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with (OUT / f"results_runs{a.tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    # summary over seeds
    grp = defaultdict(list)
    for r in rows:
        grp[(r["setting"], r["method"], r["split"], r["task"], r["subset"])].append(r)
    summ = []
    num = [c for c in cols if c not in ("setting", "method", "seed", "split", "task", "subset")]
    for k, rs in grp.items():
        d = dict(zip(("setting", "method", "split", "task", "subset"), k))
        d["n_seeds"] = len(rs)
        for c in num:
            xs = [r[c] for r in rs if r.get(c) not in (None, "")]
            if xs:
                d[c] = statistics.mean(xs)
                d[f"{c}_sd"] = statistics.stdev(xs) if len(xs) > 1 else 0.0
        summ.append(d)
    scol = []
    for r in summ:
        for k in r:
            if k not in scol:
                scol.append(k)
    with (OUT / f"results_summary{a.tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scol)
        w.writeheader()
        w.writerows(summ)
    if negstats:
        with (OUT / f"negative_sampling_stats{a.tag}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(negstats[0].keys()))
            w.writeheader()
            w.writerows(negstats)


if __name__ == "__main__":
    main()

"""Shared evaluation protocol for KO-level next-node (A) and masked-node (C) tasks.

Every model (baselines now, trained models later) must be scored through this module so that
numbers are comparable.

Protocol
--------
* Candidates: all KO tokens of token_vocab_ko.csv (special tokens excluded).
* Filtered ranking: for each query the other correct answers in the same split
  (`valid_targets_in_split`), the answers already known from train (`known_targets_in_train`, A only),
  and tokens already present in the query (the source KO for A, context KOs for C; paths are simple)
  are removed from the candidate list. The true target is always kept.
* Ties are resolved by the mean rank: rank = 1 + #(higher) + #(equal)/2, so constant scores are not
  rewarded by lucky ordering.
* Metrics: MRR, Hits@1/3/10, mean rank; micro (every query weight 1) and pathway-macro
  (`pathway_macro_weight`, every held-out pathway equal total weight), overall and split into
  targets seen / unseen in train. 95% CI for the macro MRR by a pathway-cluster bootstrap
  (cluster = first pathway of the query).
"""

from __future__ import annotations

import csv
import gzip
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HITS = (1, 3, 10)


def ids(s: str) -> List[int]:
    return [int(x) for x in s.split()] if s else []


def load_queries(path: Path, task: str) -> List[dict]:
    """task 'A' or 'C'. Returns dicts with target, exclude set, weight, seen flag, cluster, ..."""
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = int(r["target_token_id"])
            q = {"target": t, "weight": float(r["pathway_macro_weight"]),
                 "seen": r["target_seen_in_train"] == "1",
                 "cluster": r["pathway_ids"].split(";")[0], "pathway_ids": r["pathway_ids"]}
            excl = set(ids(r["valid_targets_in_split"]))
            if task == "A":
                u = int(r["context_token_id"])
                q["source"] = u
                excl |= set(ids(r.get("known_targets_in_train", "")))
                excl.add(u)
            else:
                ctx = ids(r["context_token_ids"])
                pos = int(r["mask_position"])
                q.update(context=ctx, pos=pos, left=ctx[pos - 1] if pos > 0 else None,
                         right=ctx[pos + 1] if pos + 1 < len(ctx) else None)
                excl |= {x for i, x in enumerate(ctx) if i != pos}
            excl.discard(t)
            q["exclude"] = excl
            out.append(q)
    return out


def filtered_rank(scores: Sequence[float], target: int, candidates: Sequence[int], exclude: set) -> float:
    s = scores[target]
    gt = eq = 0
    for c in candidates:
        if c == target or c in exclude:
            continue
        v = scores[c]
        if v > s:
            gt += 1
        elif v == s:
            eq += 1
    return 1 + gt + eq / 2


def n_candidates(q: dict, candidates: Sequence[int]) -> int:
    return len(candidates) - sum(1 for c in q["exclude"] if c in candidates_set_cache(candidates))


_cand_cache: Dict[int, set] = {}


def candidates_set_cache(candidates):
    k = id(candidates)
    if k not in _cand_cache:
        _cand_cache[k] = set(candidates)
    return _cand_cache[k]


def random_expected(q: dict, candidates) -> dict:
    """Exact expectation of a uniformly random ranking over the filtered candidates."""
    n = n_candidates(q, candidates)
    return {"rr": sum(1 / k for k in range(1, n + 1)) / n, "rank": (n + 1) / 2,
            **{f"h{k}": min(k, n) / n for k in HITS}}


def per_query(rank: float) -> dict:
    return {"rr": 1 / rank, "rank": rank, **{f"h{k}": float(rank <= k) for k in HITS}}


def aggregate(queries: List[dict], results: List[dict], n_boot: int = 1000, seed: int = 42) -> List[dict]:
    rows = []
    for subset, keep in (("all", lambda q: True), ("target_seen_in_train", lambda q: q["seen"]),
                         ("target_unseen_in_train", lambda q: not q["seen"])):
        idx = [i for i, q in enumerate(queries) if keep(q)]
        if not idx:
            continue
        for weighting in ("micro", "macro"):
            w = [1.0 if weighting == "micro" else queries[i]["weight"] for i in idx]
            W = sum(w)
            row = {"subset": subset, "weighting": weighting, "n_queries": len(idx),
                   "n_pathways": len({queries[i]["cluster"] for i in idx})}
            for m in ("rr", "rank") + tuple(f"h{k}" for k in HITS):
                row[m] = sum(wi * results[i][m] for wi, i in zip(w, idx)) / W
            if weighting == "macro":
                by = defaultdict(lambda: [0.0, 0.0])
                for wi, i in zip(w, idx):
                    by[queries[i]["cluster"]][0] += wi * results[i]["rr"]
                    by[queries[i]["cluster"]][1] += wi
                keys = sorted(by)
                rng = random.Random(f"{seed}|boot")
                stats = []
                for _ in range(n_boot):
                    smp = [keys[rng.randrange(len(keys))] for _ in keys]
                    num = sum(by[k][0] for k in smp)
                    den = sum(by[k][1] for k in smp)
                    stats.append(num / den)
                stats.sort()
                row["rr_ci95_low"] = stats[int(0.025 * n_boot)]
                row["rr_ci95_high"] = stats[int(0.975 * n_boot) - 1]
            rows.append(row)
    for r in rows:
        r["MRR"] = r.pop("rr")
        r["mean_rank"] = r.pop("rank")
        for k in HITS:
            r[f"Hits@{k}"] = r.pop(f"h{k}")
    return rows


# ----------------------------------------------------------------------------- AUC (link prediction style)
def attach_negatives(queries: List[dict], all_edges: set, candidates: Sequence[int], n_neg: int = 10,
                     seed: int = 42, tag: str = "") -> None:
    """Tail corruption for task A: for query (u -> t) draw n_neg candidates v with (u, v) not an edge of
    the FULL KO graph (all splits), v != u, v != t. Deterministic for a given (seed, tag, query order),
    so every method is scored on the same negatives."""
    rng = random.Random(f"{seed}|neg|{tag}")
    for q in queries:
        u, negs = q["source"], []
        while len(negs) < n_neg:
            v = candidates[rng.randrange(len(candidates))]
            if v != u and v != q["target"] and (u, v) not in all_edges:
                negs.append(v)
        q["neg"] = negs


def auc_query(scores: Sequence[float], q: dict) -> float:
    s = scores[q["target"]]
    return sum(1.0 if s > scores[n] else 0.5 if s == scores[n] else 0.0 for n in q["neg"]) / len(q["neg"])


def aggregate_auc(queries: List[dict], aucs: List[float]) -> dict:
    W = sum(q["weight"] for q in queries)
    return {"AUC_micro": sum(aucs) / len(aucs),
            "AUC_macro": sum(q["weight"] * a for q, a in zip(queries, aucs)) / W}


def load_all_ko_edges(vocab: Dict[str, int]) -> set:
    """Every KO edge of every pathway (from splits_ko/edge_split.csv, incl. excluded ones)."""
    path = Path(__file__).resolve().parent.parent / "kegg_dataset" / "splits_ko" / "edge_split.csv"
    with path.open() as f:
        return {(vocab[r["source_ko"]], vocab[r["target_ko"]]) for r in csv.DictReader(f)}


# ----------------------------------------------------------------------------- cluster bootstrap (any per-query value)
def _cluster_sums(queries, values, weighting):
    by = defaultdict(lambda: [0.0, 0.0])
    for q, v in zip(queries, values):
        w = 1.0 if weighting == "micro" else q["weight"]
        by[q["cluster"]][0] += w * v
        by[q["cluster"]][1] += w
    return by


def cluster_bootstrap(queries, values, weighting="micro", n_boot=2000, seed=42, tag="boot"):
    """Estimate and 95% CI of the (micro or pathway-macro) mean of a per-query value, resampling
    pathway clusters (cluster = first pathway of the query, as in aggregate())."""
    by = _cluster_sums(queries, values, weighting)
    keys = sorted(by)
    est = sum(by[k][0] for k in keys) / sum(by[k][1] for k in keys)
    rng = random.Random(f"{seed}|{tag}")
    st = []
    for _ in range(n_boot):
        smp = [keys[rng.randrange(len(keys))] for _ in keys]
        st.append(sum(by[k][0] for k in smp) / sum(by[k][1] for k in smp))
    st.sort()
    return est, st[int(0.025 * n_boot)], st[int(0.975 * n_boot) - 1], sum(1 for x in st if x <= 0) / n_boot


def paired_cluster_bootstrap(queries, values_a, values_b, weighting="micro", n_boot=2000, seed=42):
    """CI of mean(a - b) over the same queries (paired), resampling pathway clusters.
    Returns (delta, ci_low, ci_high, bootstrap fraction <= 0)."""
    return cluster_bootstrap(queries, [a - b for a, b in zip(values_a, values_b)], weighting, n_boot, seed, "paired")

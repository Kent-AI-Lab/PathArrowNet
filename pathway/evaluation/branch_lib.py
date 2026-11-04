"""
Multi-answer (branch) evaluation helpers. Per test parent u (source of >= 1 test task-A query), its children C(u) are all
test targets of u. Candidates are all KOs except those removed by the usual filter (union of the queries' filter sets, minus
C(u)), so the children of a parent are not filtered against each other. Per parent:
    R@K       |top-K ∩ C(u)| / |C(u)|
    All@K     all children of u in the top K
    Any@K     at least one child in the top K
    RPrec     precision at K = |C(u)|
"""
from collections import defaultdict

import numpy as np

from common import SL

KS = [1, 3, 5, 10, 20, 50, 100]


def zr(m):
    return (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-9)


def method_scores(st, m, qs):
    """mean row z-score over the runs of a method (one run for a seed-averaged dump)"""
    runs = sorted(p for p in (SL.SCORES / st / m).iterdir() if p.is_dir())
    return np.mean([zr(np.nan_to_num(SL.query_scores(r, "test", "A", qs).astype(np.float64), neginf=-1e9)) for r in runs], 0)


def per_parent(S, qs, V):
    par = defaultdict(list)
    for i, q in enumerate(qs):
        par[q["source"]].append(i)
    out = []
    for u, idx in par.items():
        C = {qs[i]["target"] for i in idx}
        excl = set().union(*[qs[i]["exclude"] for i in idx]) - C
        s = S[idx[0]].copy()
        s[:3] = -np.inf; s[list(excl)] = -np.inf; s[u] = -np.inf
        order = np.argsort(-s, kind="stable")
        pos = {int(v): r for r, v in enumerate(order[: max(KS + [len(C)])])}
        hits = sorted(pos.get(c, 10 ** 9) for c in C)
        rec = {k: sum(h < k for h in hits) / len(C) for k in KS}
        out.append({"parent": int(u), "n_children": len(C), "cluster": qs[idx[0]]["cluster"], "pathway_ids": qs[idx[0]]["pathway_ids"],
                    **{f"R@{k}": rec[k] for k in KS}, **{f"All@{k}": float(rec[k] == 1.0) for k in KS},
                    **{f"Any@{k}": float(rec[k] > 0) for k in KS}, "RPrec": sum(h < len(C) for h in hits) / len(C)})
    return out


def parent_queries(rows, idx):
    """query-like dicts for common_stats (one value per parent, clustered by its pathway)"""
    return [{"cluster": rows[i]["cluster"], "pathway_ids": rows[i]["pathway_ids"], "weight": 1.0} for i in idx]

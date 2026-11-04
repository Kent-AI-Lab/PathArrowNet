"""Directed-graph statistics and bounded training-sample construction.

Strategies
----------
A  next-node:     every unique directed edge (u -> v) of a pathway graph.
B  path windows:  simple directed paths with exactly L nodes (L = 3..6).
                  Cycles cannot be traversed because paths never revisit a node.
                  Per pathway and per L, at most `cap` paths are kept:
                    * if the number of length-L *walks* (an upper bound on
                      simple paths, computed exactly by DP) is <= enum_limit,
                      all simple paths are enumerated and, if more than `cap`,
                      `cap` are drawn uniformly with a seeded RNG;
                    * otherwise paths are drawn uniformly over walks
                      (walk-count weighted), non-simple walks are rejected and
                      duplicates discarded, until `cap` unique paths or the
                      attempt budget is exhausted.  The total number of simple
                      paths is then only *estimated* (walks x acceptance rate).
C  masked node:   for every B path, each interior position i (0 < i < L-1) is
                  masked:  A -> B -> [MASK] -> D, target = C.
"""

from __future__ import annotations

import bisect
import random
import statistics
from collections import defaultdict
from itertools import accumulate
from typing import Dict, Iterable, List, Sequence, Set, Tuple

Edge = Tuple[str, str]


# --------------------------------------------------------------------------- graph stats
def tarjan_scc(nodes: Iterable[str], adj: Dict[str, List[str]]) -> List[List[str]]:
    """Iterative Tarjan (no recursion limit issues)."""
    index, low, onstack, stack, comps = {}, {}, set(), [], []
    counter = 0
    for root in nodes:
        if root in index:
            continue
        work = [(root, 0)]
        while work:
            v, i = work.pop()
            if i == 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                onstack.add(v)
            nbrs = adj.get(v, [])
            recursed = False
            while i < len(nbrs):
                w = nbrs[i]
                i += 1
                if w not in index:
                    work.append((v, i))
                    work.append((w, 0))
                    recursed = True
                    break
                if w in onstack:
                    low[v] = min(low[v], index[w])
            if recursed:
                continue
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    onstack.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                comps.append(comp)
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
    return comps


def graph_stats(edges: Set[Edge]) -> dict:
    nodes = sorted({u for u, _ in edges} | {v for _, v in edges})
    out, inn = defaultdict(set), defaultdict(set)
    for u, v in edges:
        out[u].add(v)
        inn[v].add(u)
    adj = {u: sorted(vs) for u, vs in out.items()}
    comps = tarjan_scc(nodes, adj)
    cyclic = [c for c in comps if len(c) > 1 or (c[0], c[0]) in edges]
    # weakly connected components
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    wcc = len({find(n) for n in nodes})
    return {
        "nodes": len(nodes), "edges": len(edges),
        "sources": sum(1 for n in nodes if not inn[n]),
        "sinks": sum(1 for n in nodes if not out[n]),
        "branch_nodes": sum(1 for n in nodes if len(out[n]) > 1),
        "merge_nodes": sum(1 for n in nodes if len(inn[n]) > 1),
        "max_out_degree": max((len(out[n]) for n in nodes), default=0),
        "max_in_degree": max((len(inn[n]) for n in nodes), default=0),
        "scc_count": len(comps),
        "cyclic_scc_count": len(cyclic),
        "nodes_in_cycles": sum(len(c) for c in cyclic),
        "largest_scc": max((len(c) for c in comps), default=0),
        "weak_components": wcc,
    }


# --------------------------------------------------------------------------- path windows
def _walk_counts(adj: Dict[str, List[str]], nodes: List[str], L: int) -> List[Dict[str, int]]:
    """W[k][v] = number of walks with k+1 nodes starting at v (k = 0..L-1)."""
    W = [{v: 1 for v in nodes}]
    for _ in range(1, L):
        prev = W[-1]
        W.append({v: sum(prev[w] for w in adj.get(v, ())) for v in nodes})
    return W


def _enumerate_simple(adj, nodes, L, limit_hint):
    res = []
    for s in nodes:
        stack = [(s, [s])]
        while stack:
            v, path = stack.pop()
            if len(path) == L:
                res.append(tuple(path))
                continue
            for w in adj.get(v, ()):
                if w not in path:
                    stack.append((w, path + [w]))
    return res


def sample_paths(edges: Set[Edge], L: int, cap: int, seed_key: str,
                 enum_limit: int = 300_000, attempt_factor: int = 30) -> dict:
    nodes = sorted({u for u, _ in edges} | {v for _, v in edges})
    adj = defaultdict(list)
    for u, v in sorted(edges):
        if u != v:
            adj[u].append(v)
    rng = random.Random(seed_key)
    W = _walk_counts(adj, nodes, L)
    total_walks = sum(W[L - 1].values())
    info = {"L": L, "walks": total_walks}
    if total_walks == 0:
        return dict(info, paths=[], mode="none", total_simple=0, total_simple_is_exact=True)
    if total_walks <= enum_limit:
        allp = _enumerate_simple(adj, nodes, L, total_walks)
        n = len(allp)
        paths = allp if n <= cap else rng.sample(allp, cap)
        return dict(info, paths=sorted(paths), mode="exhaustive" if n <= cap else "exhaustive_then_subsampled",
                    total_simple=n, total_simple_is_exact=True)
    # walk-weighted uniform sampling with rejection of non-simple walks
    starts = [v for v in nodes if W[L - 1][v] > 0]
    start_cum = list(accumulate(W[L - 1][v] for v in starts))
    cum = [dict() for _ in range(L)]  # cum[k][v]: cumulative weights over adj[v] using W[k]
    got, attempts, accepted = set(), 0, 0
    max_attempts = attempt_factor * cap

    def pick(seq, cw):
        return seq[bisect.bisect_right(cw, rng.random() * cw[-1])]
    while len(got) < cap and attempts < max_attempts:
        attempts += 1
        v = pick(starts, start_cum)
        path = [v]
        for k in range(L - 2, -1, -1):
            nb = adj[v]
            cw = cum[k].get(v)
            if cw is None:
                cw = cum[k][v] = list(accumulate(W[k][w] for w in nb))
            v = pick(nb, cw)
            path.append(v)
        if len(set(path)) == L:
            accepted += 1
            got.add(tuple(path))
    est = int(round(total_walks * accepted / attempts)) if attempts else 0
    return dict(info, paths=sorted(got), mode="walk_sampled", total_simple=est,
                total_simple_is_exact=False, attempts=attempts, accepted=accepted)


def masked_from_path(path: Sequence[str]) -> List[Tuple[tuple, int, str]]:
    """[(masked_context, mask_position, target)] for interior positions."""
    out = []
    for i in range(1, len(path) - 1):
        ctx = tuple(p if j != i else "[MASK]" for j, p in enumerate(path))
        out.append((ctx, i, path[i]))
    return out


# --------------------------------------------------------------------------- concentration
def concentration(per_pathway: Dict[str, int], total_pathways: int) -> dict:
    vals = sorted(per_pathway.values(), reverse=True)
    vals = [v for v in vals if v > 0]
    tot = sum(vals)

    def top(k):
        return sum(vals[:k]) / tot if tot else 0.0

    def gini(xs):
        xs = sorted(xs)
        n = len(xs)
        if n == 0 or sum(xs) == 0:
            return 0.0
        cum = sum((i + 1) * x for i, x in enumerate(xs))
        return (2 * cum) / (n * sum(xs)) - (n + 1) / n
    ranked = sorted(per_pathway.items(), key=lambda kv: -kv[1])
    return {
        "total_samples": tot,
        "pathways_contributing": len(vals),
        "pathways_total": total_pathways,
        "mean_per_contributing_pathway": (tot / len(vals)) if vals else 0,
        "median_per_contributing_pathway": statistics.median(vals) if vals else 0,
        "median_per_pathway_all372": statistics.median(vals + [0] * (total_pathways - len(vals))),
        "max_from_one_pathway": vals[0] if vals else 0,
        "max_pathway": ranked[0][0] if ranked and ranked[0][1] else "",
        "top1_share": top(1), "top5_share": top(5), "top10_share": top(10),
        "gini_contributing": gini(vals),
        "top5_pathways": ";".join(f"{p}:{c}" for p, c in ranked[:5] if c),
    }

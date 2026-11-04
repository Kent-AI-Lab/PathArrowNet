"""Shared statistics: per-split paired bootstrap and the overlap-aware pooled five-split estimate.

pooled(per_split, macro_weights=False):
  per_split = {setting: (rr_new [n], rr_ref [n], qs)}.
  1. per split s and test pathway p (= q["cluster"]): d_ps = mean_q(RR_new - RR_ref)   (with macro_weights=True the
     mean is weighted by q["weight"], i.e. the S-macro convention)
  2. d_p = mean of d_ps over the splits where p is a test pathway; estimate = mean_p d_p (pathway-macro)
  3. bootstrap (B = 10000) over connected components of pathways linked by multi-pathway queries in any split.
"""
from collections import defaultdict

import numpy as np

B = 10000


class UF:
    def __init__(s): s.p = {}
    def f(s, x):
        s.p.setdefault(x, x)
        while s.p[x] != x:
            s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def u(s, a, b): s.p[s.f(a)] = s.f(b)


def pooled(per_split, macro_weights=False, seed=20261012):
    uf = UF(); dps = defaultdict(list)
    for st, (a, b, qs) in per_split.items():
        for q in qs:
            ps = q["pathway_ids"].split(";")
            for x in ps:
                uf.f(x)
            for x, y in zip(ps, ps[1:]):
                uf.u(x, y)
        d = np.asarray(a) - np.asarray(b); cl = np.array([q["cluster"] for q in qs])
        w = np.array([q["weight"] for q in qs]) if macro_weights else np.ones(len(qs))
        for p in np.unique(cl):
            m = cl == p; dps[p].append(float((w[m] * d[m]).sum() / w[m].sum()))
    ps = sorted(dps); dp = np.array([np.mean(dps[p]) for p in ps])
    g = np.array([uf.f(p) for p in ps]); ug, inv = np.unique(g, return_inverse=True)
    s1 = np.bincount(inv, weights=dp, minlength=len(ug)); c1 = np.bincount(inv, minlength=len(ug)).astype(float)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(ug), (B, len(ug)))
    bs = s1[idx].sum(1) / c1[idx].sum(1)
    p = max(1 / B, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return {"est": float(dp.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "p": float(min(1.0, p)), "n_pathways": len(ps), "n_clusters": int(len(ug))}


def boot(rr_new, rr_ref, qs, macro=False, mask=None, n=2000, seed=0):
    """Per-split pathway-cluster bootstrap of mean (or S-macro weighted mean) RR difference."""
    m = np.ones(len(qs), bool) if mask is None else mask
    d = (np.asarray(rr_new) - np.asarray(rr_ref))[m]
    cl = np.array([q["cluster"] for q in qs])[m]
    w = np.array([q["weight"] for q in qs])[m] if macro else np.ones(m.sum())
    u = np.unique(cl); idx = {c: np.where(cl == c)[0] for c in u}; rng = np.random.default_rng(seed); bs = []
    for _ in range(n):
        ii = np.concatenate([idx[c] for c in rng.choice(u, len(u))])
        bs.append((w[ii] * d[ii]).sum() / w[ii].sum())
    return float((w * d).sum() / w.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

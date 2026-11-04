#!/usr/bin/env python3
"""
Training-scheme comparisons of PathArrowNet without relation-level weights (Sec. 4.4-4.5, Suppl. Table S12). Only score
dumps of the trained variants are read.
(a) Set-valued (KL) against single-answer cross-entropy (CE), single models, seed-matched (split 1: s42-s46, splits 2-5:
    s42-s44): branch metrics per test parent (branch_lib; each seed's rows z-scored, metrics averaged over seeds) and task-A
    MRR on queries whose source has more than five held-out successors; pooled pathway-macro differences KL - CE.
(b) One difference at a time on split 1 (five seeds each): task-A MRR on queries whose source has more than five held-out
    successors, paired against PathArrowNet (per-query RR averaged over seeds; 2,000 pathway resamples), and R-precision of
    the held-out successor sets of those sources. Variants:
        set-valued target, batch edges removed only          clean_nbf_kl_keepedges
        single answer, answer sets removed, sibling masking  clean_xp_ce_answeredges
        single answer, batch edges removed, no masking       clean_xp_ce_nofilter
    and, over all queries, the collapse of the single-answer scheme when the full answer sets are removed (against the
    single-answer reference clean_nbf_ce).

    python evaluation/training_scheme.py         -> results/evaluation/training_scheme.json
"""
from collections import defaultdict

import numpy as np

import branch_lib as BE
from common import CS, PAN0, STS, SPLIT, SL, ZS, boot, write

KEYS = ["R@10", "R@100", "All@10", "RPrec"]
SUBS = {"branching": lambda r: r["n_children"] >= 2, "gt5": lambda r: r["n_children"] > 5, "all": lambda r: True}
CE = "clean_nbf_ce"
DECOUPLE = {"set-valued; batch edges only": "clean_nbf_kl_keepedges", "single answer; answer sets; masking": "clean_xp_ce_answeredges",
            "single answer; batch edges; no masking": "clean_xp_ce_nofilter"}


def per_parent_mean(st, m, seeds, qs, V):
    rows = None
    for sd in seeds:
        P = BE.per_parent(ZS(SL.query_scores(SL.SCORES / st / m / sd, "test", "A", qs)), qs, V)
        if rows is None:
            rows = [{**r, **{k: r[k] / len(seeds) for k in KEYS}} for r in P]
        else:
            for r0, r in zip(rows, P):
                assert r0["parent"] == r["parent"]
                for k in KEYS:
                    r0[k] += r[k] / len(seeds)
    return rows


def rr_seeds(st, m, seeds, qs, V):
    return np.mean([1 / SL.ranks(SL.query_scores(SL.SCORES / st / m / sd, "test", "A", qs).astype(np.float64), qs, V)[0] for sd in seeds], 0)


def gt5_mask(qs):
    n = defaultdict(int)
    for q in qs:
        n[q["source"]] += 1
    return np.array([n[q["source"]] > 5 for q in qs])


def main():
    out, pools = {"kl_vs_ce": {}}, defaultdict(dict)
    for st in STS:
        _, V = SL.vocab(st); qs = SL.load_queries(st, "test", "A")
        seeds = [f"s{s}" for s in (range(42, 47) if st == STS[0] else range(42, 45))]
        Pk, Pc = per_parent_mean(st, PAN0, seeds, qs, V), per_parent_mean(st, CE, seeds, qs, V)
        d = {"seeds": seeds}
        for sub, keep in SUBS.items():
            idx = [i for i, r in enumerate(Pk) if keep(r)]; ql = BE.parent_queries(Pk, idx)
            d[sub] = {"n_parents": len(idx)}
            for k in KEYS:
                a = np.array([Pk[i][k] for i in idx]); b = np.array([Pc[i][k] for i in idx])
                d[sub][k] = {"kl": float(a.mean()), "ce": float(b.mean()), "kl_minus_ce": boot(a, b, ql)}
                pools[f"{sub}|{k}"][st] = (a, b, ql)
        mask = gt5_mask(qs); rk, rc = rr_seeds(st, PAN0, seeds, qs, V), rr_seeds(st, CE, seeds, qs, V)
        d["taskA_gt5"] = {"n_queries": int(mask.sum()), "kl": float(rk[mask].mean()), "ce": float(rc[mask].mean()), "kl_minus_ce": boot(rk, rc, qs, mask=mask)}
        idx = np.where(mask)[0]; pools["taskA|gt5"][st] = (rk[idx], rc[idx], [qs[i] for i in idx])
        out["kl_vs_ce"][SPLIT[st]] = d
    out["kl_vs_ce"]["pooled"] = {k: CS.pooled(v) for k, v in pools.items()}
    out["kl_vs_ce"]["mean_over_splits"] = {f"{sub}|{k}": {e: float(np.mean([out["kl_vs_ce"][SPLIT[st]][sub][k][e] for st in STS])) for e in ("kl", "ce")}
                                           for sub in SUBS for k in KEYS}
    for k in ("taskA|gt5", "gt5|R@10", "gt5|RPrec"):
        r = out["kl_vs_ce"]["pooled"][k]; print(f"KL - CE {k}: {r['est']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]", flush=True)

    # (b) decoupling, split 1
    st = STS[0]; _, V = SL.vocab(st); qs = SL.load_queries(st, "test", "A"); seeds = [f"s{s}" for s in range(42, 47)]
    mask = gt5_mask(qs); ref = rr_seeds(st, PAN0, seeds, qs, V)
    gt5_parents = lambda P: [r for r in P if r["n_children"] > 5]
    dec = {"PathArrowNet (set-valued; answer sets)": {"RPrec_gt5": float(np.mean([r["RPrec"] for r in gt5_parents(per_parent_mean(st, PAN0, seeds, qs, V))]))}}
    for lab, m in DECOUPLE.items():
        a = rr_seeds(st, m, seeds, qs, V)
        dec[lab] = {"gt5_minus_PathArrowNet": boot(a, ref, qs, mask=mask), "n_queries_gt5": int(mask.sum()),
                    "RPrec_gt5": float(np.mean([r["RPrec"] for r in gt5_parents(per_parent_mean(st, m, seeds, qs, V))])),
                    "n_sources_gt5": len(gt5_parents(per_parent_mean(st, m, seeds, qs, V))), "test_MRR_all": float(a.mean())}
    a, c = rr_seeds(st, "clean_xp_ce_answeredges", seeds, qs, V), rr_seeds(st, CE, seeds, qs, V)
    dec["single answer, answer sets removed: all queries vs single-answer reference"] = {"test_MRR": float(a.mean()), "minus_reference": boot(a, c, qs),
                                                                                         "per_seed_test_MRR": [float((1 / SL.ranks(SL.query_scores(SL.SCORES / st / "clean_xp_ce_answeredges" / sd, "test", "A", qs).astype(np.float64), qs, V)[0]).mean()) for sd in seeds]}
    out["decoupling_split1"] = dec
    write("training_scheme.json", out)


if __name__ == "__main__":
    main()

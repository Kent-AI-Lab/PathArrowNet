#!/usr/bin/env python3
"""
Ranking among supported candidates (Sec. 4.3, Suppl. Table S15). For test task-A queries whose target is a direct STRING
experimental (>= 150) neighbor of the source, the candidates are the source's STRING neighbors minus the usual filter set
(other answers of the query, training successors, context); the target's mean-tie rank within that set is compared between
methods and the STRING rule (maximal experimental score of the pair). 'bin' restricts the candidates to the score bin of the
true edge (150-399 / 400-699 / 700-899 / >= 900); the paper reports sets with at least two candidates.

    python evaluation/conditional_ranking.py     -> results/evaluation/conditional_ranking.json
"""
from collections import defaultdict

import numpy as np

from common import ADA, CS, PAN_SA, REP, STRING_FILE, STS, SPLIT, SL, boot, runs, write

BINS = [(150, 399), (400, 699), (700, 899), (900, 10 ** 6)]
METHODS = [PAN_SA, ADA, REP]


def string_scores(voc):
    S = {}
    for line in STRING_FILE.read_text().splitlines()[1:]:
        a, b, s, _ = line.split("\t"); S[(voc[a], voc[b])] = int(s); S[(voc[b], voc[a])] = int(s)
    return S


def binof(s):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= s <= hi:
            return i
    return -1


def rank_in_set(scores_row, cand, target):
    s = scores_row[cand]; t = scores_row[target]
    return 1 + float((s > t).sum()) + float((s == t).sum() - 1) / 2  # the target itself is in cand


def main():
    out, pool = {}, defaultdict(dict)
    for st in STS:
        voc, V = SL.vocab(st); SS = string_scores(voc)
        nb = defaultdict(set)
        for (a, b) in SS:
            nb[a].add(b)
        q = SL.load_queries(st, "test", "A")
        sel = [i for i, x in enumerate(q) if x["target"] in nb[x["source"]]]
        qs = [q[i] for i in sel]
        cands, cands_bin = [], []
        for x in qs:
            c = (nb[x["source"]] - set(x["exclude"]) - {x["source"]}) | {x["target"]}
            c = sorted(v for v in c if v >= 3); cands.append(np.array(c))
            b = binof(SS[(x["source"], x["target"])])
            cands_bin.append(np.array([v for v in c if binof(SS[(x["source"], v)]) == b]))
        R = {}
        for m in METHODS:
            rr = {"all": np.zeros(len(qs)), "bin": np.zeros(len(qs))}; h1 = {"all": np.zeros(len(qs)), "bin": np.zeros(len(qs))}
            rs = runs(st, m)
            for r in rs:
                M = SL.query_scores(r, "test", "A", q).astype(np.float64)
                for j, i in enumerate(sel):
                    for var, C in (("all", cands[j]), ("bin", cands_bin[j])):
                        rk = rank_in_set(M[i], C, q[i]["target"]); rr[var][j] += 1 / rk / len(rs); h1[var][j] += float(rk <= 1) / len(rs)
            R[m] = {"rr": rr, "h1": h1}
        rr = {"all": np.zeros(len(qs)), "bin": np.zeros(len(qs))}; h1 = {"all": np.zeros(len(qs)), "bin": np.zeros(len(qs))}
        for j, x in enumerate(qs):
            for var, C in (("all", cands[j]), ("bin", cands_bin[j])):
                row = np.full(V, -np.inf); row[C] = [SS[(x["source"], v)] for v in C]
                rk = rank_in_set(row, C, x["target"]); rr[var][j] = 1 / rk; h1[var][j] = float(rk <= 1)
        R["string_rule"] = {"rr": rr, "h1": h1}
        sizes = {"all": np.array([len(c) for c in cands]), "bin": np.array([len(c) for c in cands_bin])}
        d = {"n_queries": len(qs)}
        for var in ("all", "bin"):
            d[var] = {}
            for sub, mask in (("all_queries", np.ones(len(qs), bool)), ("set_size_ge2", sizes[var] >= 2)):
                e = {"n": int(mask.sum()), "mrr": {m: float(R[m]["rr"][var][mask].mean()) for m in R}, "hits1": {m: float(R[m]["h1"][var][mask].mean()) for m in R}}
                idx = np.where(mask)[0]
                for m in METHODS:
                    for ref in ("string_rule", ADA):
                        if m != ref:
                            e[f"{m}-{ref}"] = boot(R[m]["rr"][var], R[ref]["rr"][var], qs, mask=mask)
                            pool[f"{var}|{sub}|{m}-{ref}"][st] = (R[m]["rr"][var][idx], R[ref]["rr"][var][idx], [qs[i] for i in idx])
                for m in R:  # query-level means over the concatenated splits (Table S15 levels)
                    pool[f"levels|{var}|{sub}|{m}|mrr"].setdefault(st, R[m]["rr"][var][idx])
                    pool[f"levels|{var}|{sub}|{m}|hits1"].setdefault(st, R[m]["h1"][var][idx])
                d[var][sub] = e
        out[SPLIT[st]] = d
    out["pooled"] = {k: CS.pooled(v) for k, v in pool.items() if not k.startswith("levels")}
    out["concatenated_levels"] = {k[len("levels|"):]: float(np.concatenate(list(v.values())).mean()) for k, v in pool.items() if k.startswith("levels")}
    out["n_queries"] = {k[len("levels|"):]: int(sum(len(x) for x in v.values())) for k, v in pool.items() if k.startswith("levels") and k.endswith(f"{PAN_SA}|mrr")}
    for k in ("all|set_size_ge2", "bin|set_size_ge2"):
        for ref in ("string_rule", ADA):
            r = out["pooled"][f"{k}|{PAN_SA}-{ref}"]
            print(f"{k} PathArrowNet - {ref}: {r['est']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]", flush=True)
    write("conditional_ranking.json", out)


if __name__ == "__main__":
    main()

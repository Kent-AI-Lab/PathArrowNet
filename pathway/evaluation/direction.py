#!/usr/bin/env python3
"""
Edge direction beyond the undirected background (Sec. 4.3, Suppl. Table S16).
For every test task-A edge a -> b whose reverse b -> a is not a KEGG edge in any map, the indicator is
1[z_a(b) > z_b(a)] (ties 1/2), where z_x is the row z-score of the tail scores of anchor x. Seed-averaged: z rows averaged over
the five seeds; per-seed: the indicator averaged over seeds. STRING-supported pairs have a direct STRING experimental edge,
which carries no direction. Needs the reverse-anchor dumps of PathArrowNet (benchmark_v2/scores/<setting>/clean_nbf_kgml_revanchors/,
written by `python -m patharrownet.predict --dump-benchmark --with-targets`).
Node-role rule (no training): the degree-share rule orients a pair by the training out-degree share out/(out+in) of its
endpoints (ties 1/2).

    python evaluation/direction.py               -> results/evaluation/direction.json
"""
import csv
import gzip
from collections import defaultdict

import numpy as np

from common import CS, PW, REV, STS, SPLIT, SL, ZS, string_pairs, write

SEEDS = [f"s{s}" for s in range(42, 47)]
_KE = None


def kegg_edges():
    """every directed KEGG KO edge of the dataset (all maps, all splits)"""
    global _KE
    if _KE is None:
        _KE = set()
        with (PW / "kegg_dataset/splits_ko/edge_split.csv").open() as f:
            for r in csv.DictReader(f):
                _KE.add((r["source_ko"], r["target_ko"]))
    return _KE


def dir_edges(st, split):
    voc, V = SL.vocab(st); tok = {i: t.replace("ko:", "") for t, i in voc.items()}; SP = string_pairs(voc)
    keep = [q for q in SL.load_queries(st, split, "A") if (tok[q["target"]], tok[q["source"]]) not in kegg_edges()]
    return keep, np.array([(q["source"], q["target"]) in SP for q in keep])


def load_rows(paths):
    rows = []
    for p in paths:
        d = np.load(p / "tail_test.npz"); anc = [int(a) for a in d["anchors"]]
        rows.append(({a: i for i, a in enumerate(anc)}, ZS(d["scores"].astype(np.float64))))
    return rows


def dir_indicators(rows, qs):
    n = len(qs); f_avg = np.zeros(n); f_seed = np.zeros(n); cache = {}

    def zavg(a):
        if a not in cache:
            cache[a] = np.mean([Z[pos[a]] for pos, Z in rows], 0)
        return cache[a]
    for j, q in enumerate(qs):
        a, b = q["source"], q["target"]; za, zb = zavg(a), zavg(b)
        f_avg[j] = float(za[b] > zb[a]) + 0.5 * float(za[b] == zb[a])
        for pos, Z in rows:
            x, y = Z[pos[a]][b], Z[pos[b]][a]; f_seed[j] += (float(x > y) + 0.5 * float(x == y)) / len(rows)
    return f_avg, f_seed


def degree_rule(st, qs):
    with gzip.open(SL.SPLIT_DIRS[st][0] / "train_A_next_node.csv.gz", "rt") as f:
        tr = sorted({(int(r["context_token_id"]), int(r["target_token_id"])) for r in csv.DictReader(f)})
    outd, ind = defaultdict(int), defaultdict(int)
    for u, v in tr:
        outd[u] += 1; ind[v] += 1
    share = lambda x: outd[x] / (outd[x] + ind[x]) if outd[x] + ind[x] else 0.5
    f_ = np.array([share(q["source"]) for q in qs], float); r_ = np.array([share(q["target"]) for q in qs], float)
    return (f_ > r_) + 0.5 * (f_ == r_)


def main():
    out, pool, poolS, poolR, poolR1 = {}, {}, {}, {}, {}
    for st in STS:
        qd, is_s = dir_edges(st, "test")
        sa, ps = dir_indicators(load_rows([SL.SCORES / st / REV / s for s in SEEDS]), qd)
        r1 = degree_rule(st, qd)
        d = {"n_edges": len(qd), "n_string": int(is_s.sum())}
        for sub, m in (("all", np.ones(len(qd), bool)), ("string", is_s)):
            ii = np.where(m)[0]; e, lo, hi = CS.boot(sa[ii], np.full(len(ii), 0.5), [qd[i] for i in ii])
            d[sub] = {"share": float(sa[ii].mean()), "lo": 0.5 + lo, "hi": 0.5 + hi, "share_per_seed": float(ps[ii].mean())}
        ii = np.where(is_s)[0]
        d["degree_rule_string_share"] = float(r1[ii].mean())
        d["model_minus_rule_string_query"] = float(sa[ii].mean() - r1[ii].mean())
        out[SPLIT[st]] = d
        pool[st] = (sa, np.full(len(qd), 0.5), qd); poolS[st] = (sa[ii], np.full(len(ii), 0.5), [qd[i] for i in ii])
        poolR[st] = (sa[ii], r1[ii], [qd[i] for i in ii]); poolR1[st] = (r1[ii], np.full(len(ii), 0.5), [qd[i] for i in ii])
        print(SPLIT[st], d, flush=True)
    out["pooled_share_minus_half_all"] = CS.pooled(pool)
    out["pooled_share_minus_half_string"] = CS.pooled(poolS)
    out["pooled_model_minus_degree_rule_string"] = CS.pooled(poolR)
    out["pooled_degree_rule_minus_half_string"] = CS.pooled(poolR1)
    out["query_level_share_all"] = float(np.concatenate([v[0] for v in pool.values()]).mean())
    out["query_level_share_string"] = float(np.concatenate([v[0] for v in poolS.values()]).mean())
    out["query_level_degree_rule_string"] = float(np.concatenate([v[0] for v in poolR1.values()]).mean())
    out["query_level_model_minus_rule_string"] = float(np.concatenate([v[0] - v[1] for v in poolR.values()]).mean())
    out["n_edges"] = int(sum(len(v[0]) for v in pool.values())); out["n_string"] = int(sum(len(v[0]) for v in poolS.values()))
    write("direction.json", out)


if __name__ == "__main__":
    main()

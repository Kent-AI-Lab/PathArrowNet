#!/usr/bin/env python3
"""
Non-learned baselines for the KO-level split, scored with kegg_pipeline/evaluation.py.

All statistics are fitted on the TRAIN split only; nothing is tuned (fixed constants), so val and
test are both clean reports. Every baseline adds a tiny popularity term (eps * popularity) as an
explicit, documented tie-breaker.

Baselines
---------
random      exact expectation of a uniform random ranking
popularity  A: train in-degree of the candidate;  C: how often the candidate is a masked target in train
bigram      transition model from train path windows, interpolated with the unigram (lambda=0.9)
            A: P(v|u);  C: log P(v|left) + log P(right|v)
adjacency   train-graph structure.  A: number of 2-hop paths u->x->v;
            C: [left->v] + [v->right] + 2*[left->v][v->right]
cf          neighbourhood collaborative filtering on the train graph (Jaccard similarity).
            A: sum_u' J(out(u), out(u')) [u'->v];   C: same for left plus in-neighbour version for right
go_profile* GO similarity (no graph edges of the query needed): GO profile of the query's train
            neighbours (targets of u / of left, sources of right) plus the query KOs themselves at
            half weight, matched against each candidate's GO set (inverted index, 1/sqrt(|GO_v|)).
            KO GO = GO of member human genes (gene_go_terms_collapsed) U KEGG KO DBLINKS GO.
            Ablations probe feature leakage (GO may encode pathway membership):
              go_profile_no_BP   MF + CC only (Biological Process terms like "MAPK cascade" removed)
              go_profile_curated gene-level non-IEA annotations only, no KEGG DBLINKS GO

Outputs: kegg_dataset/baselines_ko/  results.csv, results.md, leakage_probes.csv,
         ranks_{split}_{task}.csv.gz (per-query ranks of every baseline, for paired comparisons)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kegg_pipeline import evaluation as E  # noqa: E402
from kegg_pipeline.common import OUT, log, read_csv, write_csv  # noqa: E402

SPLIT = OUT / "splits_ko"
RES = OUT / "baselines_ko"
EPS = 1e-6
LAMBDA = 0.9


class Train:
    def __init__(self, vocab_n, split_dir=None):
        split_dir = split_dir or SPLIT
        self.V = vocab_n
        self.out, self.inn = defaultdict(set), defaultdict(set)
        with gzip.open(split_dir / "train_A_next_node.csv.gz", "rt") as f:
            for r in csv.DictReader(f):
                u, v = int(r["context_token_id"]), int(r["target_token_id"])
                self.out[u].add(v)
                self.inn[v].add(u)
        self.edges = {(u, v) for u, vs in self.out.items() for v in vs}
        self.big, self.uni = Counter(), Counter()
        if (split_dir / "train_B_path_windows.csv.gz").exists():
            with gzip.open(split_dir / "train_B_path_windows.csv.gz", "rt") as f:
                for r in csv.DictReader(f):
                    p = E.ids(r["token_ids"])
                    self.uni.update(p)
                    self.big.update(zip(p, p[1:]))
        self.src_tot = Counter()
        for (a, _), c in self.big.items():
            self.src_tot[a] += c
        self.uni_tot = sum(self.uni.values())
        self.mask_freq = Counter()
        if (split_dir / "train_C_masked.csv.gz").exists():
            with gzip.open(split_dir / "train_C_masked.csv.gz", "rt") as f:
                for r in csv.DictReader(f):
                    self.mask_freq[int(r["target_token_id"])] += 1
        indeg = [len(self.inn.get(i, ())) for i in range(vocab_n)]
        mf = [self.mask_freq.get(i, 0) for i in range(vocab_n)]
        self.pop = {"A": self._norm(indeg), "C": self._norm(mf)}

    @staticmethod
    def _norm(xs):
        m = max(xs) or 1
        return [x / m for x in xs]

    def p_uni(self, v):
        return (self.uni.get(v, 0) + 1) / (self.uni_tot + self.V)

    def p_big(self, a, b):
        tot = self.src_tot.get(a, 0)
        ml = self.big.get((a, b), 0) / tot if tot else 0.0
        return LAMBDA * ml + (1 - LAMBDA) * self.p_uni(b) if tot else self.p_uni(b)


GO_VARIANTS = {
    # name: (use KEGG KO DBLINKS GO, allowed aspects, drop electronic-only (IEA) gene-GO pairs)
    "go_profile": (True, {"MF", "BP", "CC"}, False),
    "go_profile_no_BP": (True, {"MF", "CC"}, False),
    "go_profile_curated": (False, {"MF", "BP", "CC"}, True),
}


def load_ko_go(vocab, use_kegg, aspects, drop_iea):
    """KO GO set = GO of member human genes (+ KEGG KO DBLINKS GO), filtered by aspect/evidence."""
    from kegg_pipeline.annotations import parse_obo
    obo = parse_obo()
    gene_go = defaultdict(set)
    for r in read_csv(OUT / "gene_go_terms_collapsed.csv"):
        if r["go_aspect"] in aspects and not (drop_iea and r["electronic_only"] == "1"):
            gene_go[r["kegg_gene_id"][4:]].add(r["go_id"])
    ko_go = {}
    for r in read_csv(OUT / "ko_mapping.csv"):
        k = r["ko_id"]
        if k not in vocab:
            continue
        s = set()
        if use_kegg:
            s = {x for x in r["ko_go_ids_kegg_dblinks"].split(";") if x and obo.get(x, {}).get("aspect") in aspects}
        for g in r["human_genes_in_pathways"].split(";"):
            if g:
                s |= gene_go.get(g, set())
        ko_go[vocab[k]] = s
    return ko_go


class Scorers:
    def __init__(self, T: Train, ko_go, candidates):
        self.T, self.go, self.cands = T, ko_go, candidates
        self.go_inv = defaultdict(list)
        for k, gs in ko_go.items():
            for g in gs:
                self.go_inv[g].append(k)
        self.go_norm = {k: 1 / math.sqrt(len(gs)) for k, gs in ko_go.items() if gs}

    def dense(self, task, sparse=None):
        base = [EPS * x for x in self.T.pop[task]]
        if sparse:
            for k, v in sparse.items():
                base[k] += v
        return base

    # ---------------- A: u -> ?
    def A_popularity(self, q):
        return self.dense("A")

    def A_bigram(self, q):
        u = q["source"]
        s = self.dense("A")
        for c in self.cands:
            s[c] += math.log(self.T.p_big(u, c))
        return s

    def A_adjacency(self, q):
        sp = Counter()
        for x in self.T.out.get(q["source"], ()):
            for v in self.T.out.get(x, ()):
                sp[v] += 1
        return self.dense("A", sp)

    def _cf(self, nbrs_of_query, index_nb, forward):
        """sum over similar nodes n' of J(N(q), N(n')) * [n' links candidate]."""
        sp = Counter()
        if not nbrs_of_query:
            return sp
        sims = Counter()
        for x in nbrs_of_query:
            for n2 in index_nb.get(x, ()):
                sims[n2] += 1
        for n2, inter in sims.items():
            nb2 = (self.T.out if forward else self.T.inn).get(n2, set())
            j = inter / (len(nbrs_of_query) + len(nb2) - inter)
            for v in nb2:
                sp[v] += j
        return sp

    def A_cf(self, q):
        u = q["source"]
        sp = self._cf(self.T.out.get(u, set()), self.T.inn, True)
        sp.pop(u, None)
        return self.dense("A", sp)

    def _go_profile(self, members, selves):
        prof = Counter()
        for m in members:
            for g in self.go.get(m, ()):
                prof[g] += 1 / max(1, len(members))
        for m in selves:
            for g in self.go.get(m, ()):
                prof[g] += 0.5
        sp = Counter()
        for g, w in prof.items():
            for k in self.go_inv.get(g, ()):
                sp[k] += w
        return {k: v * self.go_norm.get(k, 0) for k, v in sp.items()}

    def A_go_profile(self, q):
        u = q["source"]
        return self.dense("A", self._go_profile(self.T.out.get(u, set()), [u]))

    # ---------------- C: left -> ? -> right
    def C_popularity(self, q):
        return self.dense("C")

    def C_bigram(self, q):
        l, r = q["left"], q["right"]
        s = self.dense("C")
        for c in self.cands:
            s[c] += math.log(self.T.p_big(l, c)) + math.log(self.T.p_big(c, r))
        return s

    def C_adjacency(self, q):
        l, r = q["left"], q["right"]
        a = self.T.out.get(l, set())
        b = self.T.inn.get(r, set())
        sp = Counter()
        for v in a:
            sp[v] += 1
        for v in b:
            sp[v] += 1
        for v in a & b:
            sp[v] += 2
        return self.dense("C", sp)

    def C_cf(self, q):
        l, r = q["left"], q["right"]
        sp = self._cf(self.T.out.get(l, set()), self.T.inn, True)
        sp2 = self._cf(self.T.inn.get(r, set()), self.T.out, False)
        for k, v in sp2.items():
            sp[k] += v
        return self.dense("C", sp)

    def C_go_profile(self, q):
        l, r = q["left"], q["right"]
        members = list(self.T.out.get(l, set())) + list(self.T.inn.get(r, set()))
        return self.dense("C", self._go_profile(members, [l, r]))


BASELINES = ["random", "popularity", "bigram", "adjacency", "cf"] + list(GO_VARIANTS)


def leakage_probes(split, task, queries, T):
    out = {"split": split, "task": task, "n_queries": len(queries)}
    if task == "A":
        out["target_edge_in_train"] = sum(1 for q in queries if (q["source"], q["target"]) in T.edges)
        out["reverse_edge_in_train"] = sum(1 for q in queries if (q["target"], q["source"]) in T.edges)
        out["source_has_train_out_edges"] = sum(1 for q in queries if T.out.get(q["source"]))
    else:
        out["left_target_edge_in_train"] = sum(1 for q in queries if (q["left"], q["target"]) in T.edges)
        out["target_right_edge_in_train"] = sum(1 for q in queries if (q["target"], q["right"]) in T.edges)
        out["both_edges_in_train"] = sum(1 for q in queries if (q["left"], q["target"]) in T.edges
                                         and (q["target"], q["right"]) in T.edges)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    vocab_rows = read_csv(OUT / "token_vocab_ko.csv")
    vocab = {r["token"]: int(r["token_id"]) for r in vocab_rows}
    V = len(vocab_rows)
    cands = [int(r["token_id"]) for r in vocab_rows if not r["token"].startswith("[")]
    t0 = time.time()
    T = Train(V)
    S = None
    go_scorers = {}
    for name, (use_kegg, aspects, drop_iea) in GO_VARIANTS.items():
        kg = load_ko_go(vocab, use_kegg, aspects, drop_iea)
        go_scorers[name] = Scorers(T, kg, cands)
        S = S or go_scorers[name]
        log(f"{name}: {sum(1 for k in cands if kg.get(k))}/{len(cands)} KOs with GO, "
            f"mean {sum(len(kg.get(k, ())) for k in cands) / len(cands):.1f} terms")
    log(f"train fitted in {time.time() - t0:.0f}s: {len(T.edges):,} edges, {len(T.big):,} bigrams")

    results, probes = [], []
    for split in a.splits.split(","):
        for task in ("A", "C"):
            fn = SPLIT / f"{split}_{task}_{'next_node' if task == 'A' else 'masked'}.csv.gz"
            qs = E.load_queries(fn, task)
            probes.append(leakage_probes(split, task, qs, T))
            per = {}
            for b in BASELINES:
                t1 = time.time()
                if b == "random":
                    per[b] = [E.random_expected(q, cands) for q in qs]
                else:
                    f = (getattr(go_scorers[b], f"{task}_go_profile") if b in go_scorers
                         else getattr(S, f"{task}_{b}"))
                    per[b] = [E.per_query(E.filtered_rank(f(q), q["target"], cands, q["exclude"])) for q in qs]
                for row in E.aggregate(qs, per[b], n_boot=a.n_boot):
                    results.append({"split": split, "task": task, "baseline": b, **row})
                log(f"{split} {task} {b}: {time.time() - t1:.0f}s")
            rank_rows = [{"query_index": i, "pathway_ids": q["pathway_ids"], "target_token_id": q["target"],
                          "target_seen_in_train": int(q["seen"]), "pathway_macro_weight": q["weight"],
                          **{f"rank_{b}": round(per[b][i]["rank"], 2) for b in BASELINES}}
                         for i, q in enumerate(qs)]
            write_csv(RES / f"ranks_{split}_{task}.csv.gz", rank_rows, list(rank_rows[0].keys()))
    fields = ["split", "task", "baseline", "subset", "weighting", "n_queries", "n_pathways", "MRR",
              "rr_ci95_low", "rr_ci95_high", "Hits@1", "Hits@3", "Hits@10", "mean_rank"]
    write_csv(RES / "results.csv", results, fields)
    pf = sorted({k for p in probes for k in p}, key=lambda k: (k not in ("split", "task", "n_queries"), k))
    write_csv(RES / "leakage_probes.csv", probes, pf)
    write_md(results, probes, len(cands))
    print((RES / "results.md").read_text())


def write_md(results, probes, n_cands):
    def get(split, task, b, subset, w):
        for r in results:
            if (r["split"], r["task"], r["baseline"], r["subset"], r["weighting"]) == (split, task, b, subset, w):
                return r

    lines = [f"""# KO-level baselines (dataset v1)

Produced by `evaluate_ko_baselines.py`; protocol in `kegg_pipeline/evaluation.py`.
Candidates: {n_cands:,} KOs, filtered ranking (other correct answers in the split, train-known answers
for A, and KOs already in the query are removed), ties by mean rank. Statistics are fitted on train
only; no tuning, so val and test are both clean. Macro = every held-out pathway has equal weight;
CI = 95% pathway-cluster bootstrap of the macro MRR.
"""]
    for task, title in (("A", "Task A: next KO (u -> ?)"), ("C", "Task C: masked KO (... left [MASK] right ...)")):
        for split in ("test", "val"):
            r0 = get(split, task, "random", "all", "micro")
            if not r0:
                continue
            nseen = get(split, task, "random", "target_seen_in_train", "micro")
            nun = get(split, task, "random", "target_unseen_in_train", "micro")
            lines.append(f"## {title}: {split}\n\n{r0['n_queries']:,} queries from {r0['n_pathways']} pathways "
                         f"(target seen in train: {nseen['n_queries'] if nseen else 0:,}; unseen: {nun['n_queries'] if nun else 0:,}).\n")
            lines.append("| baseline | MRR micro | MRR macro [95% CI] | Hits@1 | Hits@10 macro | MRR seen | MRR unseen |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for b in BASELINES:
                mi, ma = get(split, task, b, "all", "micro"), get(split, task, b, "all", "macro")
                se = get(split, task, b, "target_seen_in_train", "micro")
                un = get(split, task, b, "target_unseen_in_train", "micro")
                lines.append(f"| {b} | {mi['MRR']:.4f} | {ma['MRR']:.4f} [{ma['rr_ci95_low']:.4f}, {ma['rr_ci95_high']:.4f}] | "
                             f"{mi['Hits@1']:.4f} | {ma['Hits@10']:.4f} | {se['MRR'] if se else float('nan'):.4f} | "
                             f"{un['MRR'] if un else float('nan'):.4f} |")
            lines.append("")
    lines.append("## Leakage probes (edge-level; target edges must be 0)\n")
    keys = sorted({k for p in probes for k in p} - {"split", "task"})
    lines.append("| split | task | " + " | ".join(keys) + " |")
    lines.append("|---|---|" + "---:|" * len(keys))
    for p in probes:
        lines.append(f"| {p['split']} | {p['task']} | " + " | ".join(str(p.get(k, '-')) for k in keys) + " |")
    (RES / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

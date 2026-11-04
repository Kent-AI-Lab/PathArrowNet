"""
Unified query loading, score storage and metrics for the KO next-node benchmark (benchmark_v2).

Every model is scored the same way:
  1. The model writes a score for EVERY KO candidate (2,870 tokens; ids 3..2872 of token_vocab_ko.csv).
       anchor models (score depends only on the current KO u):  scores/<setting>/<method>/s<seed>/tail_<split>.npz
                                                               anchors [n], scores [n, V] float32
       sequence models (score depends on the whole prefix):     .../seq_<split>_<task>.npz  scores [n_queries, V]
  2. This module turns scores into filtered ranks and metrics.

Filtered ranking (identical to kegg_pipeline/evaluation.py, which is used for all earlier results):
  candidates = all KO tokens; removed from the candidate list for one query = the other correct next KOs of the same
  context in the same split (branching), the next KOs of the current KO already known from train, and the KOs
  already in the context (paths are simple). The true target is always kept. rank = 1 + #(higher) + #(equal)/2.
MRR = mean over queries of 1/rank (micro = every query weight 1; macro = every held-out pathway the same total
weight). Hits@k = share of queries with rank <= k.

AUC (sampled): per query, P(score(target) > score(negative)) over n negatives (ties 0.5), averaged over queries.
"""

from __future__ import annotations

import csv
import gzip
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kegg_pipeline import evaluation as E  # noqa: E402

DS = ROOT / "kegg_dataset"
B2 = Path(__file__).resolve().parent
SCORES = B2 / "scores"
SPLIT_DIRS = {
    "strict_v1": (DS / "splits_ko", B2 / "splits" / "strict_v1"),
    "unique_random": (ROOT / "kegg_leaky_splits" / "unique_random", None),
    "random_segment": (B2 / "splits" / "random_segment", B2 / "splits" / "random_segment"),
}
# strict_v1_full: same split and queries as strict_v1, but the candidates are all 11,620 KOs of the 372 human maps
# (token_vocab_ko_full.csv; ids 0..2872 identical to token_vocab_ko.csv). See full/build_full_candidates.py.
SPLIT_DIRS["strict_v1_full"] = SPLIT_DIRS["strict_v1"]
# strict_v1_full_tv: final models trained on train + val edges, evaluated on test only (val successors also filtered)
SPLIT_DIRS["strict_v1_full_tv"] = SPLIT_DIRS["strict_v1"]
VOCAB_FILES = {"strict_v1_full": "token_vocab_ko_full.csv", "strict_v1_full_tv": "token_vocab_ko_full.csv"}
# strict_s{43..46}_full: the 4 extra strict splits (kegg_splits_robustness/), full candidates (full/build_robustness_settings.py)
for _s in (43, 44, 45, 46):
    SPLIT_DIRS[f"strict_s{_s}_full"] = (ROOT / "kegg_splits_robustness" / f"seed{_s}", B2 / "splits" / f"strict_s{_s}")
    VOCAB_FILES[f"strict_s{_s}_full"] = "token_vocab_ko_full.csv"
# Leaky control settings (common random protocols): task A only, full candidates.
# Never used for model selection on the strict benchmark; see full/build_leaky_settings.py.
for _l in ("unique_random", "occurrence_random"):
    SPLIT_DIRS[f"{_l}_full"] = (ROOT / "kegg_leaky_splits" / _l, None)
    VOCAB_FILES[f"{_l}_full"] = "token_vocab_ko_full.csv"
HITS = (1, 3, 5, 10)


def vocab(setting=None):
    with (DS / VOCAB_FILES.get(setting, "token_vocab_ko.csv")).open() as f:
        v = {r["token"]: int(r["token_id"]) for r in csv.DictReader(f)}
    return v, len(v)


def rd(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_A(setting, split):
    return E.load_queries(SPLIT_DIRS[setting][0] / f"{split}_A_next_node.csv.gz", "A")


def load_S(setting, split):
    d = SPLIT_DIRS[setting][1]
    out = []
    for r in rd(d / f"{split}_S_prefix_next.csv.gz"):
        ctx = [int(x) for x in r["context_token_ids"].split()]
        t = int(r["target_token_id"])
        excl = set(E.ids(r["valid_targets_in_split"])) | set(E.ids(r["known_targets_in_train"])) | set(ctx)
        excl.discard(t)
        out.append({"context": ctx, "source": ctx[-1], "target": t, "exclude": excl,
                    "weight": float(r["pathway_macro_weight"]), "seen": r["target_seen_in_train"] == "1",
                    "cluster": r["pathway_ids"].split(";")[0], "pathway_ids": r["pathway_ids"]})
    return out


def load_queries(setting, split, task):
    qs = load_A(setting, split) if task == "A" else load_S(setting, split)
    if setting.endswith("_tv") and split == "test":
        qs = _add_val_knowledge(setting, qs)
    return qs


def _add_val_knowledge(setting, qs):
    """*_tv settings (final model trained on train + val): at test time the val edges are known as well, so the next
    KOs of the current KO known from val are filtered like the train ones, and 'seen' means seen in train or val."""
    train_dir = SPLIT_DIRS[setting][0]
    succ, nodes = defaultdict(set), set()
    for sp in ("train", "val"):
        for r in rd(train_dir / f"{sp}_A_next_node.csv.gz"):
            u, v = int(r["context_token_id"]), int(r["target_token_id"])
            nodes |= {u, v}
            if sp == "val":
                succ[u].add(v)
    out = []
    for q in qs:
        q = dict(q)
        q["exclude"] = (set(q["exclude"]) | succ[q["source"]]) - {q["target"]}
        q["seen"] = q["target"] in nodes
        out.append(q)
    return out


def anchors_needed(setting, splits=("val", "test")):
    s = set()
    for sp in splits:
        s |= {q["source"] for q in load_A(setting, sp)}
        if SPLIT_DIRS[setting][1] is not None:
            s |= {q["source"] for q in load_S(setting, sp)}
    return sorted(s)


def save_tail(path, anchors, scores, dtype=np.float32):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, anchors=np.asarray(anchors, dtype=np.int64), scores=np.asarray(scores, dtype=dtype))


def query_scores(run_dir, split, task, queries):
    """[n_queries, V] score matrix for the queries of (split, task) from whatever the run stored."""
    seq = run_dir / f"seq_{split}_{task}.npz"
    if seq.exists():
        m = np.load(seq)["scores"]
        assert m.shape[0] == len(queries), f"{seq}: {m.shape[0]} rows vs {len(queries)} queries"
        return m
    z = np.load(run_dir / f"tail_{split}.npz")
    pos = {int(a): i for i, a in enumerate(z["anchors"])}
    return z["scores"][[pos[q["source"]] for q in queries]]


def ranks(scores, queries, V):
    """Filtered mean-tie ranks, vectorised; semantics identical to evaluation.filtered_rank."""
    n = len(queries)
    valid = np.zeros((n, V), dtype=bool)
    valid[:, 3:] = True
    for i, q in enumerate(queries):
        if q["exclude"]:
            valid[i, list(q["exclude"])] = False
        valid[i, q["target"]] = False
    t = np.array([q["target"] for q in queries])
    s = scores[np.arange(n), t][:, None]
    gt = ((scores > s) & valid).sum(1)
    eq = ((scores == s) & valid).sum(1)
    ncand = valid.sum(1) + 1
    return 1 + gt + eq / 2, ncand


def metrics(queries, rk, subset="all"):
    keep = {"all": lambda q: True, "seen": lambda q: q["seen"], "unseen": lambda q: not q["seen"]}[subset]
    idx = [i for i, q in enumerate(queries) if keep(q)]
    if not idx:
        return None
    out = {"n_queries": len(idx), "n_pathways": len({queries[i]["cluster"] for i in idx})}
    r = np.array([rk[i] for i in idx])
    w = np.array([queries[i]["weight"] for i in idx])
    out["MRR"] = float((1 / r).mean())
    out["MRR_macro"] = float((w / r).sum() / w.sum())
    for k in HITS:
        out[f"Hits@{k}"] = float((r <= k).mean())
    out["mean_rank"] = float(r.mean())
    out["median_rank"] = float(np.median(r))
    out["1/MRR"] = 1 / out["MRR"]
    return out


# ----------------------------------------------------------------------------- negatives for AUC
class NegativeSampler:
    """All strategies exclude u, the target, and every KO v with (u, v) an edge of the FULL KO graph (all pathways,
    all splits, incl. reverse-excluded edges), so a known correct next KO is never used as a negative.

    uniform         uniform over the 2,870 KO candidates (the protocol of every earlier AUC number; seed 42 with tag
                    '<split dir>|<split>' reproduces kegg_pipeline.evaluation.attach_negatives exactly)
    same_pathway    KOs that are nodes of the query's own pathway(s) (full pathway KO graph) but not successors of u:
                    'in the same pathway, not directly downstream'
    two_hop         KOs reachable as u -> x -> w in the full KO graph (any pathway) that are not successors of u:
                    'indirectly downstream' (graph-context negatives)
    degree_matched  sampled with probability proportional to train in-degree + 1 (popular hub targets), so a model
                    cannot win by ranking hubs above rare KOs
    """

    def __init__(self, V, setting=None):
        self.V = V
        voc, V_voc = vocab(setting)
        assert V_voc == V, (V_voc, V)
        self.all_edges = E.load_all_ko_edges(voc)
        self.succ = defaultdict(set)
        for u, v in self.all_edges:
            self.succ[u].add(v)
        sys.path.insert(0, str(ROOT))
        from build_ko_leakage_free_split import load_ko_graphs
        self.pw_nodes = {p: {voc[x] for e in es for x in e} for p, es in load_ko_graphs().items()}
        indeg = defaultdict(int)
        for r in rd(DS / "splits_ko" / "train_A_next_node.csv.gz"):
            indeg[int(r["target_token_id"])] += 1
        self.cands = list(range(3, V))
        self.deg_w = [indeg[c] + 1 for c in self.cands]

    def pool(self, q, strategy):
        u, t = q["source"], q["target"]
        bad = self.succ[u] | {u, t}
        if strategy == "same_pathway":
            nodes = set().union(*(self.pw_nodes.get(p, set()) for p in q["pathway_ids"].split(";")))
            return sorted(nodes - bad)
        if strategy == "two_hop":
            return sorted({w for x in self.succ[u] for w in self.succ[x]} - bad)
        raise ValueError(strategy)

    def sample(self, queries, strategy, seed, tag, n_neg=10):
        """Returns list (per query) of negative ids (may be shorter than n_neg for pool strategies, [] = skip)."""
        if strategy == "uniform":
            rng = random.Random(f"{seed}|neg|{tag}")
            out = []
            for q in queries:
                u, negs = q["source"], []
                while len(negs) < n_neg:
                    v = self.cands[rng.randrange(len(self.cands))]
                    if v != u and v != q["target"] and (u, v) not in self.all_edges:
                        negs.append(v)
                out.append(negs)
            return out
        rng = random.Random(f"{seed}|{strategy}|{tag}")
        out = []
        for q in queries:
            if strategy == "degree_matched":
                bad = self.succ[q["source"]] | {q["source"], q["target"]}
                negs = []
                while len(negs) < n_neg:
                    v = rng.choices(self.cands, weights=self.deg_w, k=1)[0]
                    if v not in bad:
                        negs.append(v)
                out.append(negs)
            else:
                p = self.pool(q, strategy)
                out.append(rng.sample(p, min(n_neg, len(p))) if p else [])
        return out


def auc(scores, queries, negs):
    vals = []
    for i, (q, ng) in enumerate(zip(queries, negs)):
        if not ng:
            vals.append(None)
            continue
        s = scores[i, q["target"]]
        sn = scores[i, ng]
        vals.append(float(((s > sn) + 0.5 * (s == sn)).mean()))
    return vals


def auc_all_candidates(rk, ncand):
    """AUC of the target against EVERY filtered candidate = 1 - (rank - 1) / (n_candidates - 1)."""
    return [1 - (r - 1) / (n - 1) for r, n in zip(rk, ncand)]


def mean_valid(vals, queries=None, macro=False):
    pairs = [(v, q["weight"] if (macro and queries) else 1.0) for v, q in zip(vals, queries or [{}] * len(vals)) if v is not None]
    return sum(v * w for v, w in pairs) / sum(w for _, w in pairs), len(pairs)


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)

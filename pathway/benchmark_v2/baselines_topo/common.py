"""Shared graph utilities for the training-light network baselines (benchmark_v2/baselines_topo).

Graphs (node ids = token ids of token_vocab_ko_full.csv, V = 11,623 incl. 3 special ids that the evaluation never
ranks):
  STRING   kegg_dataset_cache/string/ko_background_exp150_full.tsv, undirected, weight = max_experimental_score/1000
           (or 1 for the binary variant)
  KEGG     train_A_next_node.csv.gz of the setting, directed (context -> target). NEVER val/test edges.

Everything is batched over "columns" (one column = one query anchor, or one training edge with leave-one-out
removal of that edge) with sparse x dense products on CPU torch.

Leave-one-out (LOO): for a training edge a->c the features of (a, every candidate) are computed on the graph without
the KEGG edge a->c (and, when reverse train edges are used in the walk graph, without c->a). This is done exactly
with per-column rank-one corrections of the propagation step (see ppr_batched / bfs_batched), so the positive
edge never leaks into its own features.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore", category=UserWarning)

import gzip
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

B2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(B2))
import scorelib as SL  # noqa: E402

torch.set_num_threads(8)
STRING_FILE = SL.ROOT / "kegg_dataset_cache/string/ko_background_exp150_full.tsv"
GENE_FILE = SL.ROOT / "kegg_dataset_cache/kegg_rest/link_ko_hsa.tsv"


def load_string(voc):
    """list of (i, j, score/1000) for the undirected STRING KO pairs (each pair once)."""
    out = []
    with STRING_FILE.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out.append((voc[r["ko_a"]], voc[r["ko_b"]], int(r["max_experimental_score"]) / 1000))
    return out


def load_train(setting):
    """directed KEGG training edges (u, v), unique."""
    tr = [(int(r["context_token_id"]), int(r["target_token_id"]))
          for r in SL.rd(SL.SPLIT_DIRS[setting][0] / "train_A_next_node.csv.gz")]
    assert len(tr) == len(set(tr))
    return tr


def gene_counts(voc, V):
    cnt = Counter()
    with GENE_FILE.open() as f:
        for line in f:
            hsa, ko = line.rstrip("\n").split("\t")
            cnt[ko[3:] if ko.startswith("ko:") else ko] += 1
    g = np.zeros(V, dtype=np.float32)
    for k, c in cnt.items():
        if k in voc:
            g[voc[k]] = c
    return g


class Graph:
    """Weighted directed multigraph as COO arrays (src, dst, w); builds column-normalised sparse propagation matrices."""

    def __init__(self, V):
        self.V = V
        self.src, self.dst, self.w = [], [], []

    def add(self, src, dst, w):
        self.src.append(np.asarray(src, dtype=np.int64)); self.dst.append(np.asarray(dst, dtype=np.int64))
        self.w.append(np.asarray(w, dtype=np.float32))
        return self

    def arrays(self):
        if not self.src:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
        return np.concatenate(self.src), np.concatenate(self.dst), np.concatenate(self.w)

    def out_weight(self):
        s, d, w = self.arrays()
        return np.bincount(s, weights=w, minlength=self.V).astype(np.float32)

    def transition(self):
        """sparse CSR P with P[dst, src] = w / out(src)  (so that P @ x propagates mass along edges)."""
        s, d, w = self.arrays()
        out = self.out_weight()
        vals = w / out[s]
        M = torch.sparse_coo_tensor(torch.tensor(np.stack([d, s])), torch.tensor(vals), (self.V, self.V)).coalesce()
        return M.to_sparse_csr(), out

    def binary(self):
        """sparse CSR B with B[dst, src] = 1 if any edge src->dst (for BFS counts)."""
        s, d, _ = self.arrays()
        M = torch.sparse_coo_tensor(torch.tensor(np.stack([d, s])), torch.ones(len(s)), (self.V, self.V)).coalesce()
        M = torch.sparse_coo_tensor(M.indices(), torch.ones(M._nnz()), (self.V, self.V)).coalesce()
        return M.to_sparse_csr()

    def out_lists(self):
        """dict src -> (dst array, w array) (summed over parallel edges)."""
        s, d, w = self.arrays()
        acc = defaultdict(lambda: defaultdict(float))
        for a, b, x in zip(s.tolist(), d.tolist(), w.tolist()):
            acc[a][b] += x
        return {a: (np.array(list(m.keys()), dtype=np.int64), np.array(list(m.values()), dtype=np.float32)) for a, m in acc.items()}


def string_graph(V, string_edges, weight="score"):
    g = Graph(V)
    i = np.array([e[0] for e in string_edges]); j = np.array([e[1] for e in string_edges])
    w = np.array([e[2] for e in string_edges], dtype=np.float32) if weight == "score" else np.ones(len(string_edges), np.float32)
    g.add(i, j, w); g.add(j, i, w)
    return g


def walk_graph(V, string_edges, train_edges, weight="score", w_kegg=0.0, w_rev=0.0):
    g = string_graph(V, string_edges, weight)
    if train_edges and w_kegg > 0:
        u = np.array([e[0] for e in train_edges]); v = np.array([e[1] for e in train_edges])
        g.add(u, v, np.full(len(u), w_kegg, np.float32))
        if w_rev > 0:
            g.add(v, u, np.full(len(u), w_rev, np.float32))
    return g


def _loo_tables(removals, out_lists, out):
    """removals: list over columns of list of (node, target, w_removed) edges deleted from the walk graph of that column.
    Removals of the same column are grouped by source node (several edges of one source are deleted together, so the
    column of P' is A[:,node] minus all removed targets, normalised by out - sum(w)). Returns flattened index tables for
    the exact correction  P'x - Px = x[node] * (A[:,node] (1/out' - 1/out) - sum_t w_t e_t / out')  (ppr_batched)."""
    g_node, g_col, g_out, g_outp = [], [], [], []
    t_tgt, t_col, t_w, t_grp = [], [], [], []
    ex_rows, ex_cols, ex_w, ex_grp = [], [], [], []
    for k, rem in enumerate(removals):
        by_src = defaultdict(list)
        for (node, tgt, w) in rem:
            by_src[node].append((tgt, w))
        for node, lst in by_src.items():
            gi = len(g_node); wsum = sum(w for _, w in lst)
            g_node.append(node); g_col.append(k); g_out.append(out[node]); g_outp.append(max(out[node] - wsum, 0.0))
            for tgt, w in lst:
                t_tgt.append(tgt); t_col.append(k); t_w.append(w); t_grp.append(gi)
            dsts, ws = out_lists[node]
            ex_rows.append(dsts); ex_cols.append(np.full(len(dsts), k, np.int64)); ex_w.append(ws); ex_grp.append(np.full(len(dsts), gi, np.int64))
    if not g_node:
        return None
    T = lambda a, dt: torch.tensor(np.asarray(a), dtype=dt)
    return dict(node=T(g_node, torch.long), col=T(g_col, torch.long), out=T(g_out, torch.float32), outp=T(g_outp, torch.float32),
                t_tgt=T(t_tgt, torch.long), t_col=T(t_col, torch.long), t_w=T(t_w, torch.float32), t_grp=T(t_grp, torch.long),
                ex_rows=T(np.concatenate(ex_rows), torch.long), ex_cols=T(np.concatenate(ex_cols), torch.long),
                ex_w=T(np.concatenate(ex_w), torch.float32), ex_grp=T(np.concatenate(ex_grp), torch.long))


def ppr_batched(P, out, out_lists, anchors, alpha, iters=50, removals=None, chunk=1024, dangling="restart"):
    """Personalised PageRank with restart prob alpha from every anchor (columns). Returns [len(anchors), V] float32.
    P: CSR transition (P[dst,src] = w/out(src)); out: out-weights; removals (optional): per column list of
    (node, target, w) edges to delete from the walk graph for that column only (leave-one-out / leave-pathway-out).
    Dangling mass (nodes without out-edges) is returned to the anchor ('restart'), so every column sums to 1."""
    V = P.shape[0]; n = len(anchors); res = np.zeros((n, V), dtype=np.float32)
    for s in range(0, n, chunk):
        idx = list(range(s, min(n, s + chunk))); K = len(idx)
        anc = torch.tensor([anchors[i] for i in idx]); col = torch.arange(K)
        E = torch.zeros(V, K); E[anc, col] = 1.0
        X = E.clone()
        tab = _loo_tables([removals[i] for i in idx], out_lists, out) if removals is not None else None
        if tab is not None:
            inv_outp = torch.where(tab["outp"] > 0, 1.0 / tab["outp"].clamp(min=1e-12), torch.zeros_like(tab["outp"]))
            dcoef = inv_outp - 1.0 / tab["out"]
            tcoef = tab["t_w"] * inv_outp[tab["t_grp"]]
        for _ in range(iters):
            Y = torch.sparse.mm(P, X)
            if tab is not None:
                xa = X[tab["node"], tab["col"]]
                coef = xa * dcoef
                Y.index_put_((tab["ex_rows"], tab["ex_cols"]), coef[tab["ex_grp"]] * tab["ex_w"], accumulate=True)
                Y.index_put_((tab["t_tgt"], tab["t_col"]), -xa[tab["t_grp"]] * tcoef, accumulate=True)
            if dangling == "restart":
                Y[anc, col] += 1.0 - Y.sum(0)
            X = alpha * E + (1 - alpha) * Y
        res[s:s + K] = X.clamp(min=0).T.numpy()          # correction round-off can leave ~-1e-7
    return res


def bfs_batched(Bm, anchors, max_d=5, removals=None, string_adj=None, chunk=1024):
    """Shortest-path lengths from every anchor to all nodes, capped: dist in {0..max_d}, unreachable (> max_d) = max_d + 1.
    Bm: CSR binary B[dst,src]. removals: per column list of (node, target) train edges absent for that column; the edge
    is only removed when string_adj (dense bool [V,V]) says no STRING edge backs it."""
    V = Bm.shape[0]; n = len(anchors); res = np.full((n, V), max_d + 1, dtype=np.int8)
    for s in range(0, n, chunk):
        idx = list(range(s, min(n, s + chunk))); K = len(idx)
        anc = torch.tensor([anchors[i] for i in idx]); col = torch.arange(K)
        F = torch.zeros(V, K); F[anc, col] = 1.0
        visited = F.bool(); dist = torch.full((V, K), max_d + 1, dtype=torch.int8); dist[anc, col] = 0
        r_node = r_tgt = r_col = None
        if removals is not None:
            rn, rt, rc = [], [], []
            for k, rem in enumerate(removals):
                for (node, tgt) in rem:
                    if string_adj is None or not string_adj[node, tgt]:
                        rn.append(node); rt.append(tgt); rc.append(k)
            if rn:
                r_node, r_tgt, r_col = (torch.tensor(x) for x in (rn, rt, rc))
        for d in range(1, max_d + 1):
            C = torch.sparse.mm(Bm, F)
            if r_node is not None:
                C.index_put_((r_tgt, r_col), -F[r_node, r_col], accumulate=True)
            new = (C > 0.5) & ~visited
            dist[new] = d; visited |= new; F = new.float()
            if not new.any():
                break
        res[s:s + K] = dist.T.numpy()
    return res


def val_mrr(setting, V, anchors, S, split="val"):
    """task-A micro MRR of an [anchors, V] score matrix on a split (used for hyper-parameter selection on val)."""
    q = SL.load_A(setting, split)
    pos = {int(a): i for i, a in enumerate(anchors)}
    m = S[[pos[x["source"]] for x in q]].astype(np.float64)
    rk, _ = SL.ranks(m, q, V)
    return float(np.mean(1 / rk))

#!/usr/bin/env python3
"""Zero-shot transfer of KEGG-trained PathArrowNet to SIGNOR / WikiPathways (inference only; no model is trained or tuned).

Inputs
  cache/edges.tsv.gz (build_edges.py). Positives are NOVEL directed KO edges (neither orientation in KEGG
  edge_split.csv, so no KEGG train / val / test edge is a positive). KEGG strict-split TEST edges are scored by the same
  code as a positive control (kegg_test).
Scores (directional KO x KO, row u = source, column v = target; all from existing caches, KEGG train graph only)
  konbf            PathArrowNet, 5 seeds: log p(v | u ->) + log p(u | -> v)  (the score of DOWNSTREAM_TASK_1 / perturbation work)
  konbf_tail       PathArrowNet, 5 seeds: log p(v | u ->) only (the Task-A score)            [konbf_dir_matrices.py]
  randinit         same architecture / features / STRING graph, untrained weights, 5 seeds, tail + head
  randinit_tail    same, tail only
  bpn_orig         original BioPathNet (+STRING background), 5 seeds, tail log-softmax (downstream/bpn_ko_scores.py)
  string_direct    STRING-exp >= 150 KO-pair score (symmetric; the background graph PathArrowNet sees)
  string_aa        Adamic-Adar on the STRING-exp150 KO graph (symmetric)
  degree           next-node: KEGG train in-degree of the target (+1e-3 x STRING degree); edges: product of degrees
  random           uniform scores (seed 0)
(a) next-node ranking: one query per source KO x of a database (global over its pathways). Candidates = all 11,620 KOs.
    Filtered: x itself, every other external target of x in the same database, every KEGG target of x (any split).
    Rank = 1 + #higher + 0.5 x #ties (mean tie rank, fair to sparse baselines). One sample per novel edge (Task-A micro).
(b) edge AUROC: each novel edge (x, y) gets 5 target corruptions (x, y') with y' drawn proportional to in-degree in the
    database's directed KO graph and 5 source corruptions (x', y) proportional to out-degree (degree-matched, as the
    LR / Reactome negatives remove the popularity shortcut); a corruption is rejected if it is an edge of the database or
    of KEGG in either orientation, or a self pair. Negatives follow their parent positive's strata. Seed 0.
(c) direction AUROC: novel edges whose reverse is not in the database: P(S[x, y] > S[y, x]) + 0.5 P(tie).
Strata: all novel; clean (no STRING-exp150, no Reactome step, no KGML relation); noSTRING (no STRING-exp150);
  train_nodes (both KOs in the KEGG train graph); WikiPathways independent / kegg_adapted.
Statistics: paired bootstrap over source KOs (2000 for (a)/(c), 1000 for (b)): konbf minus each control, 95% CI.
    .venv/bin/python expansion/extdb_eval/eval_zeroshot_kgml.py   -> results/zeroshot_kgml.json
"""
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())
import collections, csv, gzip, json, sys, time
from pathlib import Path
import numpy as np
import torch

P = Path(_KH) / 'pathway'; X = P / 'expansion/extdb_eval'
t0 = time.time()
voc = {}
with (P / 'kegg_dataset/token_vocab_ko_full.csv').open() as f:
    for r in csv.DictReader(f): voc[r['token']] = int(r['token_id'])
V = len(voc)

# ---------------------------------------------------------------- reference graphs
kegg_dir, kegg_split = set(), {}
with open(P / 'kegg_dataset/splits_ko/edge_split.csv') as f:
    for r in csv.DictReader(f):
        if r['source_ko'] in voc and r['target_ko'] in voc:
            e = (voc[r['source_ko']], voc[r['target_ko']]); kegg_dir.add(e); kegg_split[e] = r['split']
kegg_out = collections.defaultdict(set)
for a, b in kegg_dir: kegg_out[a].add(b)
indeg_train = np.zeros(V); outdeg_train = np.zeros(V)
for (a, b), s in kegg_split.items():
    if s == 'train': indeg_train[b] += 1; outdeg_train[a] += 1
train_nodes = {k for e, s in kegg_split.items() if s == 'train' for k in e}
W = np.zeros((V, V), np.float32); sdeg = np.zeros(V)
with open(P / 'kegg_dataset_cache/string/ko_background_exp150_full.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        a, b = voc[r['ko_a']], voc[r['ko_b']]; W[a, b] = W[b, a] = float(r['max_experimental_score'])
A = (W > 0).astype(np.float32); sdeg = A.sum(1)
w_aa = np.where(sdeg > 0, 1 / np.log(sdeg + 1 + 1e-9), 0).astype(np.float32)
At = torch.from_numpy(A)
AA = (At * torch.from_numpy(w_aa)[None]) @ At  # dense CPU matmul; Adamic-Adar: sum_z 1/log(deg z + 1)
AA = AA.numpy(); np.fill_diagonal(AA, 0)
print(f'reference graphs {time.time() - t0:.0f}s', flush=True)

# ---------------------------------------------------------------- score matrices
def ld(p): return np.load(p, mmap_mode='r')
bz = np.load(P / 'expansion/downstream/cache/bpn_orig_ko_scores_all.npz')
BPN = np.full((V, V), np.nan, np.float32); BPN[bz['anchors']] = bz['S']
BPN[np.isnan(BPN)] = np.nanmin(bz['S'])
rng0 = np.random.default_rng(0)
deg_t = indeg_train + 1e-3 * sdeg
SC = {'konbf': ld(P / 'expansion/downstream/cache/konbf_full_matrix_kgml.npy'),
      'konbf_tail': ld(X / 'cache/konbf_tail_kgml.npy'),
      'randinit': ld(P / 'expansion/perturbseq/cache/konbf_randinit_full.npy'),
      'randinit_tail': ld(X / 'cache/konbf_randinit_tail.npy'),
      'bpn_orig': BPN, 'string_direct': W, 'string_aa': AA,
      'random': rng0.random((V, V), dtype=np.float32)}
ROWFN = {m: (lambda rows, M=M: np.asarray(M[rows], dtype=np.float64)) for m, M in SC.items()}
ROWFN['degree'] = lambda rows: np.repeat(deg_t[None], len(rows), 0)
METHODS = list(ROWFN)
CONTROLS = [m for m in METHODS if m not in ('konbf', 'konbf_tail')]


def pair_scores_fast(m, xs, ys):
    """memmap fancy indexing per row block (sorted by row) to avoid python loops."""
    if m == 'degree':
        return (outdeg_train[xs] + 1e-3 * sdeg[xs]) * deg_t[ys]
    M = SC[m]; out = np.empty(len(xs)); order = np.argsort(xs, kind='stable'); xs_s, ys_s = xs[order], ys[order]
    ur, st = np.unique(xs_s, return_index=True); st = list(st) + [len(xs_s)]
    for i in range(0, len(ur), 512):
        rows = np.asarray(M[ur[i:i + 512]], dtype=np.float64)
        for j in range(i, min(i + 512, len(ur))):
            a, b = st[j], st[j + 1]
            out[order[a:b]] = rows[j - i, ys_s[a:b]]
    return out


# ---------------------------------------------------------------- edge sets
DB = collections.defaultdict(dict)  # db -> (x, y) -> row
with gzip.open(X / 'cache/edges.tsv.gz', 'rt') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        if r['x'] in voc and r['y'] in voc: DB[r['db']][(voc[r['x']], voc[r['y']])] = r
# positive control: KEGG strict-split test edges (same fields)
for e, s in kegg_split.items():
    if s == 'test':
        DB['kegg_test'][e] = {'kegg': 'novel', 'string150': str(int(W[e] > 0)), 'reactome': '0', 'kgml_any': '0',
                              'clean': '0', 'train_nodes': str(int(e[0] in train_nodes and e[1] in train_nodes)), 'wp_class': ''}


def strata(db, r):
    s = {'all': True, 'noSTRING': r['string150'] == '0', 'train_nodes': r['train_nodes'] == '1'}
    if db != 'kegg_test': s['clean'] = r['clean'] == '1'
    if db == 'wikipathways':
        s['wp_independent'] = r['wp_class'] == 'independent'; s['wp_kegg_adapted'] = r['wp_class'] == 'kegg_adapted'
        s['wp_independent_clean'] = s['wp_independent'] and s['clean']
    return s


def boot_diff(vals, src, ref, B=2000, seed=0):
    """vals/ref: per-sample metric; paired bootstrap over source clusters. returns mean diff and 95% CI."""
    u, inv = np.unique(src, return_inverse=True)
    sv = np.bincount(inv, vals, len(u)); sr = np.bincount(inv, ref, len(u)); n = np.bincount(inv, None, len(u))
    rng = np.random.default_rng(seed); ds = np.empty(B)
    for b in range(B):
        w = np.bincount(rng.integers(0, len(u), len(u)), None, len(u))
        ds[b] = (w @ sv - w @ sr) / (w @ n)
    return float(vals.mean() - ref.mean()), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def wauc_setup(s, y):
    o = np.argsort(s, kind='stable'); s_o = s[o]
    # tie groups
    grp = np.concatenate([[0], np.cumsum(s_o[1:] != s_o[:-1])])
    return o, grp, y[o]


def wauc(setup, w):
    """AUROC with per-sample weights w (bootstrap multiplicities), ties counted 0.5."""
    o, grp, y_o = setup; w_o = w[o]
    wn = np.where(y_o == 0, w_o, 0.0); wp = np.where(y_o == 1, w_o, 0.0)
    gn = np.bincount(grp, wn); gp = np.bincount(grp, wp)
    below = np.concatenate([[0], np.cumsum(gn)[:-1]])
    num = (gp * (below + 0.5 * gn)).sum(); den = gp.sum() * gn.sum()
    return num / den if den > 0 else float('nan')


RES = {'meta': {'V': V, 'candidates': V - 3, 'methods': METHODS, 'date': '2026-10-15'}}
for db in ('signor_global', 'signor_pathways', 'wikipathways', 'kegg_test'):
    E = DB[db]; tdb = time.time()
    all_out = collections.defaultdict(set)
    for (x, y) in E: all_out[x].add(y)
    pos = [(x, y) for (x, y), r in E.items() if r['kegg'] == 'novel']
    st = {k: np.array([strata(db, E[e])[k] for e in pos]) for k in strata(db, E[pos[0]])}
    xs = np.array([e[0] for e in pos]); ys = np.array([e[1] for e in pos])
    R = {'n_novel_edges': len(pos), 'n_source_kos': int(len(np.unique(xs))), 'strata_sizes': {k: int(v.sum()) for k, v in st.items()}}
    # ---------------- (a) next-node ranking
    qx = np.unique(xs); by_src = collections.defaultdict(list)
    for i, x in enumerate(xs): by_src[x].append(i)
    by_src = {x: np.array(v) for x, v in by_src.items()}
    RR = {}
    for m in METHODS:
        rr = np.empty(len(pos))
        for i in range(0, len(qx), 256):
            rows = qx[i:i + 256]; S = ROWFN[m](rows)
            for j, x in enumerate(rows):
                sel = by_src[x]; tg = ys[sel]; s_t = S[j, tg].copy()  # true scores before filtering
                row = S[j]; row[:3] = -np.inf; row[list((all_out[x] | kegg_out[x]) | {x})] = -np.inf
                hi = (row[None, :] > s_t[:, None]).sum(1); tie = (row[None, :] == s_t[:, None]).sum(1)
                rr[sel] = 1 / (1 + hi + 0.5 * tie)
        RR[m] = rr
    A_res = {}
    for k, msk in st.items():
        if msk.sum() == 0: continue
        d = {m: {'mrr': float(RR[m][msk].mean()), 'hits10': float((1 / RR[m][msk] <= 10.5).mean()),
                 'hits1': float((1 / RR[m][msk] <= 1.5).mean())} for m in METHODS}
        for main in ('konbf', 'konbf_tail'):
            for c in CONTROLS:
                if c == main: continue
                d[f'{main}_minus_{c}'] = boot_diff(RR[main][msk], xs[msk], RR[c][msk])
        d['n'] = int(msk.sum()); d['n_sources'] = int(len(np.unique(xs[msk])))
        A_res[k] = d
    R['next_node'] = A_res
    print(f'[{db}] next-node done {time.time() - tdb:.0f}s', flush=True)
    # ---------------- (b) edge AUROC with degree-matched corruptions
    und = set(E) | {(b, a) for a, b in E} | kegg_dir | {(b, a) for a, b in kegg_dir}
    indeg = np.bincount([y for _, y in E], minlength=V).astype(float); outdeg = np.bincount([x for x, _ in E], minlength=V).astype(float)
    pin = indeg / indeg.sum(); pout = outdeg / outdeg.sum()
    rng = np.random.default_rng(0)
    nx_, ny_, npar = [], [], []
    for i, (x, y) in enumerate(pos):
        got = 0
        while got < 5:
            for c in rng.choice(V, 20, p=pin):
                if c != x and (x, c) not in und and got < 5: nx_.append(x); ny_.append(c); npar.append(i); got += 1
        got = 0
        while got < 5:
            for c in rng.choice(V, 20, p=pout):
                if c != y and (c, y) not in und and got < 5: nx_.append(c); ny_.append(y); npar.append(i); got += 1
    nx_, ny_, npar = np.array(nx_), np.array(ny_), np.array(npar)
    ax = np.concatenate([xs, nx_]); ay = np.concatenate([ys, ny_]); par = np.concatenate([np.arange(len(pos)), npar])
    lab = np.concatenate([np.ones(len(pos)), np.zeros(len(nx_))])
    PS = {m: pair_scores_fast(m, ax, ay) for m in METHODS}
    B_res = {}
    for k, msk in st.items():
        if msk.sum() == 0: continue
        keep = msk[par]; y_k = lab[keep]; src = xs[par[keep]]
        u, inv = np.unique(src, return_inverse=True)
        setups = {m: wauc_setup(PS[m][keep], y_k) for m in METHODS}
        base = {m: wauc(setups[m], np.ones(len(y_k))) for m in METHODS}
        rngb = np.random.default_rng(1); boots = collections.defaultdict(list)
        for b in range(1000):
            w = np.bincount(rngb.integers(0, len(u), len(u)), None, len(u))[inv].astype(float)
            vals = {m: wauc(setups[m], w) for m in METHODS}
            for main in ('konbf', 'konbf_tail'):
                for c in CONTROLS:
                    boots[f'{main}_minus_{c}'].append(vals[main] - vals[c])
        d = {m: base[m] for m in METHODS}
        for key, v in boots.items():
            a, c = key.split('_minus_')
            d[key] = (base[a] - base[c], float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
        d['n_pos'] = int(msk.sum()); d['n_neg'] = int((keep & (lab == 0)).sum())
        B_res[k] = d
    R['edge_auroc'] = B_res
    print(f'[{db}] edge AUROC done {time.time() - tdb:.0f}s', flush=True)
    # ---------------- (c) direction AUROC
    one = np.array([(y, x) not in E for x, y in pos])
    C_res = {}
    for k, msk in st.items():
        mk = msk & one
        if mk.sum() == 0: continue
        fw = {m: pair_scores_fast(m, xs[mk], ys[mk]) for m in METHODS if m not in ('string_direct', 'string_aa', 'random')}
        bw = {m: pair_scores_fast(m, ys[mk], xs[mk]) for m in fw}
        v = {m: (fw[m] > bw[m]) + 0.5 * (fw[m] == bw[m]) for m in fw}
        d = {m: float(v[m].mean()) for m in v}
        for main in ('konbf', 'konbf_tail'):
            for c in ('randinit', 'randinit_tail', 'bpn_orig', 'degree'):
                d[f'{main}_minus_{c}'] = boot_diff(v[main].astype(float), xs[mk], v[c].astype(float))
            d[f'{main}_minus_chance'] = boot_diff(v[main].astype(float), xs[mk], np.full(mk.sum(), 0.5))
        d['n'] = int(mk.sum()); C_res[k] = d
    R['direction_auroc'] = C_res
    RES[db] = R
    print(f'[{db}] done {time.time() - tdb:.0f}s; novel {len(pos)}', flush=True)
    (X / 'results/zeroshot_kgml.json').write_text(json.dumps(RES, indent=1))
print(f'total {time.time() - t0:.0f}s')

#!/usr/bin/env python3
"""Additional zero-shot analyses of the primary PathArrowNet (inference only; nothing trained or tuned):
  1. relation-type strata (SIGNOR mechanism / sign / DIRECT flag; WikiPathways interaction type; NCI-PID SIF type);
  2. pathway-cluster bootstrap (resample pathways; edge weight = sum over its pathways of the draw count / #pathways)
     for databases with pathway units (signor_pathways, wikipathways, nci_pid), next to the source-KO bootstrap;
  3. NCI-PID (cache/edges_pid.tsv.gz, build_edges_pid.py) as a third source, same three evaluations.
Scores, filtering, corruption sampling (seed 0, identical draw order), tie handling and strata definitions are the same
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())
as in eval_zeroshot_kgml.py, so that the 'all' / 'clean' rows reproduce results/zeroshot_kgml.json (checked in the output).
An edge belongs to every relation class any of its records carries (classes overlap).
    .venv/bin/python expansion/extdb_eval/eval_addendum_kgml.py   -> results/zeroshot_addendum_kgml.json
"""
import collections, csv, gzip, json, time
from pathlib import Path
import numpy as np
import torch

P = Path(_KH) / 'pathway'; X = P / 'expansion/extdb_eval'
t0 = time.time()
voc = {}
with (P / 'kegg_dataset/token_vocab_ko_full.csv').open() as f:
    for r in csv.DictReader(f): voc[r['token']] = int(r['token_id'])
V = len(voc)

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
W = np.zeros((V, V), np.float32)
with open(P / 'kegg_dataset_cache/string/ko_background_exp150_full.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        a, b = voc[r['ko_a']], voc[r['ko_b']]; W[a, b] = W[b, a] = float(r['max_experimental_score'])
A = (W > 0).astype(np.float32); sdeg = A.sum(1)
w_aa = np.where(sdeg > 0, 1 / np.log(sdeg + 1 + 1e-9), 0).astype(np.float32)
At = torch.from_numpy(A)
AA = ((At * torch.from_numpy(w_aa)[None]) @ At).numpy(); np.fill_diagonal(AA, 0)


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
CMP = [('konbf', 'bpn_orig'), ('konbf', 'randinit'), ('konbf', 'string_aa'), ('konbf_tail', 'bpn_orig'), ('konbf', 'degree')]
print(f'setup {time.time() - t0:.0f}s', flush=True)


def pair_scores_fast(m, xs, ys):
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


DB = collections.defaultdict(dict)
for fn in ('edges.tsv.gz', 'edges_pid.tsv.gz'):
    with gzip.open(X / 'cache' / fn, 'rt') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['x'] in voc and r['y'] in voc: DB[r['db']][(voc[r['x']], voc[r['y']])] = r

SIG_MECH = {'binding': 'binding', 'phosphorylation': 'phosphorylation', 'dephosphorylation': 'dephosphorylation',
            'transcriptional regulation': 'transcriptional', 'ubiquitination': 'ubiquitination',
            'polyubiquitination': 'ubiquitination', 'relocalization': 'relocalization'}
WP_CLS = {'mim-binding': 'binding', 'Arrow': 'stimulation', 'mim-stimulation': 'stimulation',
          'mim-necessary-stimulation': 'stimulation', 'TBar': 'inhibition', 'mim-inhibition': 'inhibition',
          'mim-transcription-translation': 'transcription', 'mim-catalysis': 'catalysis_conversion',
          'mim-conversion': 'catalysis_conversion', 'mim-modification': 'catalysis_conversion',
          'mim-cleavage': 'catalysis_conversion', 'mim-covalent-bond': 'catalysis_conversion', 'mim-gap': 'catalysis_conversion'}
PID_CLS = {'controls-state-change-of': 'state_change', 'controls-expression-of': 'expression',
           'controls-phosphorylation-of': 'phosphorylation', 'controls-transport-of': 'transport',
           'catalysis-precedes': 'catalysis_precedes'}


def rel_classes(db, r):
    rels = [x for x in r['rels'].split(';') if x]; out = set()
    if db.startswith('signor'):
        for x in rels:
            m, _, sgn = x.partition('|'); out.add('mech_' + SIG_MECH.get(m, 'other')); out.add('sign_' + (sgn or 'unknown'))
        d = set(r['signor_direct'].split(';')) - {''}
        out.add('direct_t' if 't' in d else 'direct_f_only')
    elif db == 'wikipathways':
        for x in rels:
            base = x.replace('@anchor', ''); out.add('type_' + WP_CLS.get(base, 'other'))
        out.add('anchor_only' if rels and all('@anchor' in x for x in rels) else 'has_node_to_node')
    elif db == 'nci_pid':
        for x in rels: out.add('type_' + PID_CLS.get(x, 'other'))
    return out


def base_strata(db, r):
    return {'all': True, 'clean': r['clean'] == '1'}


def cluster_weights(groups, ng, rng):
    """groups: list of per-edge arrays of pathway indices; returns per-edge weights for one pathway resample."""
    cnt = np.bincount(rng.integers(0, ng, ng), None, ng)
    return np.array([cnt[g].mean() for g in groups])


def boot_src(vals, ref, src, B=2000, seed=0):
    u, inv = np.unique(src, return_inverse=True)
    sv = np.bincount(inv, vals, len(u)); sr = np.bincount(inv, ref, len(u)); n = np.bincount(inv, None, len(u))
    rng = np.random.default_rng(seed); ds = np.empty(B)
    for b in range(B):
        w = np.bincount(rng.integers(0, len(u), len(u)), None, len(u))
        ds[b] = (w @ sv - w @ sr) / (w @ n)
    return [float(vals.mean() - ref.mean()), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]


def boot_pw(vals, ref, WB):
    """WB: [B, n] pathway-resample weights for these samples."""
    d = (WB @ (vals - ref)) / np.maximum(WB.sum(1), 1e-12)
    return [float(vals.mean() - ref.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def wauc_setup(s, y):
    o = np.argsort(s, kind='stable'); s_o = s[o]
    grp = np.concatenate([[0], np.cumsum(s_o[1:] != s_o[:-1])])
    return o, grp, y[o]


def wauc(setup, w):
    o, grp, y_o = setup; w_o = w[o]
    wn = np.where(y_o == 0, w_o, 0.0); wp = np.where(y_o == 1, w_o, 0.0)
    gn = np.bincount(grp, wn); gp = np.bincount(grp, wp)
    below = np.concatenate([[0], np.cumsum(gn)[:-1]])
    num = (gp * (below + 0.5 * gn)).sum(); den = gp.sum() * gn.sum()
    return num / den if den > 0 else float('nan')


PW_DBS = {'signor_pathways', 'wikipathways', 'nci_pid'}
RES = {'meta': {'date': '2026-10-15', 'methods': METHODS, 'comparisons': CMP, 'min_stratum_n': 100}}
for db in ('signor_global', 'signor_pathways', 'wikipathways', 'nci_pid'):
    E = DB[db]; tdb = time.time()
    all_out = collections.defaultdict(set)
    for (x, y) in E: all_out[x].add(y)
    pos = [(x, y) for (x, y), r in E.items() if r['kegg'] == 'novel']
    xs = np.array([e[0] for e in pos]); ys = np.array([e[1] for e in pos])
    rc = [rel_classes(db, E[e]) for e in pos]; classes = sorted(set().union(*rc))
    st = {}
    for bk in ('all', 'clean'):
        bm = np.array([base_strata(db, E[e])[bk] for e in pos])
        st[bk] = bm
        for c in classes:
            st[f'{bk}|{c}'] = bm & np.array([c in s for s in rc])
    st = {k: v for k, v in st.items() if v.sum() >= 100}
    # pathway groups
    if db in PW_DBS:
        units = sorted({u for e in pos for u in E[e]['units'].split(';') if u})
        ui = {u: i for i, u in enumerate(units)}
        groups = [np.array([ui[u] for u in E[e]['units'].split(';') if u]) for e in pos]
        rngp = np.random.default_rng(7)
        WBp = np.stack([cluster_weights(groups, len(units), rngp) for _ in range(2000)])  # [B, n_pos]
    R = {'n_novel_edges': len(pos), 'n_source_kos': int(len(np.unique(xs))), 'strata_sizes': {k: int(v.sum()) for k, v in st.items()},
         'n_pathways': len(units) if db in PW_DBS else None}
    # (a) next-node
    qx = np.unique(xs); by_src = collections.defaultdict(list)
    for i, x in enumerate(xs): by_src[x].append(i)
    by_src = {x: np.array(v) for x, v in by_src.items()}
    RR = {}
    for m in METHODS:
        rr = np.empty(len(pos))
        for i in range(0, len(qx), 256):
            rows = qx[i:i + 256]; S = ROWFN[m](rows)
            for j, x in enumerate(rows):
                sel = by_src[x]; tg = ys[sel]; s_t = S[j, tg].copy()
                row = S[j]; row[:3] = -np.inf; row[list((all_out[x] | kegg_out[x]) | {x})] = -np.inf
                hi = (row[None, :] > s_t[:, None]).sum(1); tie = (row[None, :] == s_t[:, None]).sum(1)
                rr[sel] = 1 / (1 + hi + 0.5 * tie)
        RR[m] = rr
    A_res = {}
    for k, msk in st.items():
        d = {m: float(RR[m][msk].mean()) for m in METHODS}
        d['hits10'] = {m: float((1 / RR[m][msk] <= 10.5).mean()) for m in ('konbf', 'konbf_tail', 'bpn_orig', 'string_direct')}
        for a, c in CMP:
            d[f'{a}-{c}|src'] = boot_src(RR[a][msk], RR[c][msk], xs[msk])
            if db in PW_DBS: d[f'{a}-{c}|pw'] = boot_pw(RR[a][msk], RR[c][msk], WBp[:, msk])
        d['n'] = int(msk.sum()); A_res[k] = d
    R['next_node_mrr'] = A_res
    print(f'[{db}] next-node {time.time() - tdb:.0f}s', flush=True)
    # (b) edge AUROC, corruptions drawn exactly as in eval_zeroshot.py
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
    BM = sorted({m for p in CMP for m in p})
    PS = {m: pair_scores_fast(m, ax, ay) for m in METHODS}
    B_res = {}
    for k, msk in st.items():
        keep = msk[par]; y_k = lab[keep]; pk = par[keep]; src = xs[pk]
        setups = {m: wauc_setup(PS[m][keep], y_k) for m in METHODS}
        d = {m: wauc(setups[m], np.ones(len(y_k))) for m in METHODS}
        u, inv = np.unique(src, return_inverse=True); rngb = np.random.default_rng(1)
        bs = collections.defaultdict(list); bp = collections.defaultdict(list)
        for b in range(1000):
            w = np.bincount(rngb.integers(0, len(u), len(u)), None, len(u))[inv].astype(float)
            v = {m: wauc(setups[m], w) for m in BM}
            for a, c in CMP: bs[(a, c)].append(v[a] - v[c])
            if db in PW_DBS:
                wp = WBp[b][pk]
                v = {m: wauc(setups[m], wp) for m in BM}
                for a, c in CMP: bp[(a, c)].append(v[a] - v[c])
        for a, c in CMP:
            d[f'{a}-{c}|src'] = [d[a] - d[c], float(np.nanpercentile(bs[(a, c)], 2.5)), float(np.nanpercentile(bs[(a, c)], 97.5))]
            if db in PW_DBS:
                d[f'{a}-{c}|pw'] = [d[a] - d[c], float(np.nanpercentile(bp[(a, c)], 2.5)), float(np.nanpercentile(bp[(a, c)], 97.5))]
        d['n_pos'] = int(msk.sum()); B_res[k] = d
    R['edge_auroc'] = B_res
    print(f'[{db}] edge AUROC {time.time() - tdb:.0f}s', flush=True)
    # (c) direction AUROC
    one = np.array([(y, x) not in E for x, y in pos])
    C_res = {}
    for k, msk in st.items():
        mk = msk & one
        if mk.sum() < 100: continue
        fm = ('konbf', 'konbf_tail', 'randinit', 'bpn_orig', 'degree')
        v = {}
        for m in fm:
            fw = pair_scores_fast(m, xs[mk], ys[mk]); bw = pair_scores_fast(m, ys[mk], xs[mk])
            v[m] = ((fw > bw) + 0.5 * (fw == bw)).astype(float)
        d = {m: float(v[m].mean()) for m in v}
        half = np.full(mk.sum(), 0.5)
        d['konbf-chance|src'] = boot_src(v['konbf'], half, xs[mk])
        d['konbf-bpn_orig|src'] = boot_src(v['konbf'], v['bpn_orig'], xs[mk])
        if db in PW_DBS:
            d['konbf-chance|pw'] = boot_pw(v['konbf'], half, WBp[:, mk])
            d['konbf-bpn_orig|pw'] = boot_pw(v['konbf'], v['bpn_orig'], WBp[:, mk])
        d['n'] = int(mk.sum()); C_res[k] = d
    R['direction_auroc'] = C_res
    RES[db] = R
    print(f'[{db}] done {time.time() - tdb:.0f}s; novel {len(pos)}', flush=True)
    (X / 'results/zeroshot_addendum_kgml.json').write_text(json.dumps(RES, indent=1))

# reproduction check against results/zeroshot.json (all / clean rows)
Z = json.loads((X / 'results/zeroshot_kgml.json').read_text()); chk = {}
for db in ('signor_global', 'signor_pathways', 'wikipathways'):
    for k in ('all', 'clean'):
        chk[f'{db}|{k}'] = {'mrr_konbf': [Z[db]['next_node'][k]['konbf']['mrr'], RES[db]['next_node_mrr'][k]['konbf']],
                            'auroc_konbf': [Z[db]['edge_auroc'][k]['konbf'], RES[db]['edge_auroc'][k]['konbf']]}
RES['reproduction_check'] = chk
(X / 'results/zeroshot_addendum_kgml.json').write_text(json.dumps(RES, indent=1))
print(json.dumps(chk, indent=1)); print(f'total {time.time() - t0:.0f}s')

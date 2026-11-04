#!/usr/bin/env python3
"""Complex-aware SIGNOR split (CPU).

Starts from expansion/extdb_ft/cache/signor.tsv (same rows, same filters, same KO-class groups) and merges groups:
every SIGNOR release row whose endpoint is a complex or a protein family is expanded to KO pairs exactly as in
expansion/extdb/survey_extdb.py (signor_recs); all KO-class groups produced by one such row are merged (union-find,
transitive, so groups that share a member pair through different rows end up together). Protein-protein rows add no
merge beyond the existing KO-class grouping. The merged component is then assigned with the original rule:
sha1(component id) % 100 -> 80 / 10 / 10 train / val / test; a component with any non-eligible edge goes to train.
Component id = lexicographically smallest original group key in the component.
Output: expansion/extdb_ft_cx/main/cache/signor.tsv (original columns; split replaced; extra columns cx_comp,
split_orig) and results/split_summary.json.
    .venv/bin/python -I expansion/extdb_ft_cx/build_cx.py
"""
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())
import collections, csv, hashlib, json
from pathlib import Path

P = Path(_KH + '/pathway')
X = P / 'expansion'
OUT = X / 'extdb_ft_cx'

SRC = (X / 'extdb/survey_extdb.py').read_text()
cut = SRC.index('# ------------------------------------------------------------------ OmniPath')
RUN = ("def run(name, recs, raw_nodes, n_pw, extra=None):\n"
       "    r, units, novel, clean = analyze(name, recs, raw_nodes, n_pw, extra)\n"
       "    RESULTS[name] = r; NOVEL[name] = (novel, clean)\n")
assert SRC.count(RUN) == 1
ns = {'__name__': 'survey_part', 'CAPTURE': {}}
exec(compile(SRC[:cut].replace(RUN, "def run(name, recs, raw_nodes, n_pw, extra=None):\n    CAPTURE[name] = recs\n"),
             'survey_extdb.py[part]', 'exec'), ns)
kos, sig_expand, K1, SIG_OK = ns['kos'], ns['sig_expand'], ns['K1'], ns['SIG_OK']

rows = list(csv.DictReader(open(X / 'extdb_ft/cache/signor.tsv'), delimiter='\t'))
pair_group = {(r['x'], r['y']): r['group'] for r in rows}

par = {}


def find(k):
    par.setdefault(k, k)
    while par[k] != k:
        par[k] = par[par[k]]; k = par[k]
    return k


def union(a, b):
    a, b = find(a), find(b)
    if a != b:
        if b < a: a, b = b, a
        par[b] = a                                         # root = smallest key


for r in rows:
    find(r['group'])
n_exp_rows, n_exp_rows_hit = 0, 0
for r in ns['rel_rows']:                                   # same filters as survey_extdb.signor_recs
    g = lambda k: r.get(K1[k], '') or ''
    if g('tax') not in ('9606', '-1', ''): continue
    if g('ta') not in SIG_OK or g('tb') not in SIG_OK: continue
    A, ha = sig_expand(g('ta'), g('ia'), g('da')); B, hb = sig_expand(g('tb'), g('ib'), g('db'))
    if not A or not B or (ha == 'protein' and hb == 'protein'): continue
    n_exp_rows += 1
    gs = set()
    for a, ea in A:
        for b, eb in B:
            if a == b: continue
            for x in kos(ea):
                for y in kos(eb):
                    for k in ((x, y), (y, x)):
                        if k in pair_group: gs.add(pair_group[k])
    gs = sorted(gs)
    if len(gs) > 1: n_exp_rows_hit += 1
    for k in gs[1:]:
        union(gs[0], k)

comp = {r['group']: find(r['group']) for r in rows}
inel = {comp[r['group']] for r in rows if r['eligible'] != '1'}
for r in rows:
    c = comp[r['group']]
    h = int(hashlib.sha1(c.encode()).hexdigest(), 16) % 100
    r['split_orig'] = r['split']; r['cx_comp'] = c
    r['split'] = 'train' if c in inel else 'train' if h < 80 else 'val' if h < 90 else 'test'
tr_src = collections.Counter(r['x'] for r in rows if r['split'] == 'train')
for r in rows:
    r['src_in_train'] = int(tr_src[r['x']] > 0)
with open(OUT / 'main/cache/signor.tsv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t'); w.writeheader(); w.writerows(rows)

# checks: no component / original group / unordered pair in two splits
for key in ('cx_comp', 'group'):
    s = collections.defaultdict(set)
    for r in rows: s[r[key]].add(r['split'])
    assert all(len(v) == 1 for v in s.values()), key
s = collections.defaultdict(set)
for r in rows: s[frozenset((r['x'], r['y']))].add(r['split'])
assert all(len(v) == 1 for v in s.values())

sizes = collections.Counter(comp[r['group']] for r in rows)
big = sizes.most_common(5)
summ = {'rows': len(rows), 'orig_groups': len(set(comp)), 'components': len(sizes),
        'expanded_signor_rows_considered': n_exp_rows, 'expanded_rows_merging_2plus_groups': n_exp_rows_hit,
        'largest_components_edges': [n for _, n in big],
        'largest_component_eligible_only': [c not in inel for c, _ in big],
        'component_size_quantiles_edges': {q: float(__import__('numpy').quantile(list(sizes.values()), q)) for q in (0.5, 0.9, 0.99)}}
for sp in ('train', 'val', 'test'):
    R = [r for r in rows if r['split'] == sp]
    summ[sp] = {'edges': len(R), 'components': len({r['cx_comp'] for r in R}), 'src_kos': len({r['x'] for r in R}),
                'sign_act': sum(r['sign'] == '0' for r in R), 'sign_inh': sum(r['sign'] == '1' for r in R),
                'protein_only': sum(r['protein_only'] == '1' for r in R), 'clean': sum(r['clean'] == '1' for r in R),
                'src_in_train': sum(r['src_in_train'] for r in R),
                'from_orig': dict(collections.Counter(r['split_orig'] for r in R))}
(OUT / 'results/split_summary.json').write_text(json.dumps(summ, indent=1))
print(json.dumps(summ, indent=1))

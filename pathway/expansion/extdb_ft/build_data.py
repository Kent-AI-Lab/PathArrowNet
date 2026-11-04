#!/usr/bin/env python3
"""KO-level fine-tuning datasets and splits for SIGNOR and WikiPathways (CPU).

Input: expansion/extdb_eval/cache/edges.tsv.gz (built by extdb_eval/build_edges.py; KEGG split 1 overlap columns),
SIGNOR protein-only KO edges from the survey parser (expansion/extdb/survey_extdb.py, executed verbatim up to the OmniPath
section, as in extdb_eval/build_edges.py).

Per db (signor = signor_global; wikipathways), one row per directed KO pair x -> y (x != y, both in the KO vocabulary):
  dropped     pair is a KEGG split-1 val / test / excluded edge in either direction; or the two KOs co-occur in one KEGG
              split-1 val or test pathway (co-membership; keeps the post-fine-tuning KEGG test check clean)
  eligible    (may enter val / test) kegg == novel and, for WikiPathways, wp_class == independent
  group       unordered pair of KO classes; KO class = connected component of KOs that share a human gene
              (gene_map kos_in_vocab), so reverse duplicates, repeated records of one interaction and paralogous KO
              projections of one gene pair always fall into the same split
  split       sha1(group) % 100: < 80 train, < 90 val, else test (80/10/10), for groups whose edges are all eligible;
              any group with a non-eligible edge goes to train
  sign        SIGNOR: 0 = only up-regulation records, 1 = only down-regulation records, empty = unsigned / both
              WikiPathways: 0 = only {mim-stimulation, mim-necessary-stimulation, mim-transcription-translation},
              1 = only {mim-inhibition, TBar}; generic 'Arrow' and other heads carry no sign; both -> empty
  flags       clean, string150 (from edges.tsv); protein_only (SIGNOR: edge produced by at least one protein-protein
              record, i.e. not only by complex / family expansion; WikiPathways: 1)
Output expansion/extdb_ft/cache/{signor,wikipathways}.tsv and results/data_summary.json.
Run (from Kegg/pathway): .venv/bin/python -I expansion/extdb_ft/build_data.py
"""
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())
import collections, csv, gzip, hashlib, json
from pathlib import Path

P = Path(_KH + '/pathway')
X = P / 'expansion'
OUT = X / 'extdb_ft'
vocab = {r['token']: int(r['token_id']) for r in csv.DictReader(open(P / 'kegg_dataset/token_vocab_ko_full.csv'))}

# ---- SIGNOR protein-only KO edges (survey parser, unchanged)
SRC = (X / 'extdb/survey_extdb.py').read_text()
cut = SRC.index('# ------------------------------------------------------------------ OmniPath')
RUN = ("def run(name, recs, raw_nodes, n_pw, extra=None):\n"
       "    r, units, novel, clean = analyze(name, recs, raw_nodes, n_pw, extra)\n"
       "    RESULTS[name] = r; NOVEL[name] = (novel, clean)\n")
assert SRC.count(RUN) == 1
ns = {'__name__': 'survey_part', 'CAPTURE': {}}
exec(compile(SRC[:cut].replace(RUN, "def run(name, recs, raw_nodes, n_pw, extra=None):\n    CAPTURE[name] = recs\n"),
             'survey_extdb.py[part]', 'exec'), ns)
kos = ns['kos']
sig_prot = {(x, y) for r in ns['CAPTURE']['signor_global_protein_only'] if r['directed']
            for x in kos(r['ea']) for y in kos(r['eb']) if x != y}

# ---- KO classes (KOs sharing a gene)
par = {}


def find(k):
    par.setdefault(k, k)
    while par[k] != k:
        par[k] = par[par[k]]; k = par[k]
    return k


for r in csv.DictReader(open(X / 'downstream/data/common/gene_map.tsv'), delimiter='\t'):
    ks = [k for k in r['kos_in_vocab'].split(';') if k]
    for k in ks[1:]:
        a, b = find(ks[0]), find(k)
        if a != b: par[a] = b

# ---- KEGG split-1 val/test co-membership
pw_split = {r['pathway_id']: r['split'] for r in csv.DictReader(open(P / 'kegg_dataset/splits_ko/pathway_split.csv'))}
mem = collections.defaultdict(set)
for r in csv.DictReader(open(P / 'kegg_dataset/splits_ko/edge_split.csv')):
    for pw in r['pathways'].split(';'):
        if pw_split.get(pw) in ('val', 'test'):
            mem[pw] |= {r['source_ko'], r['target_ko']}
co = collections.defaultdict(set)
for ks in mem.values():
    for k in ks: co[k] |= ks

POS_WP = {'mim-stimulation', 'mim-necessary-stimulation', 'mim-transcription-translation'}
NEG_WP = {'mim-inhibition', 'TBar'}


def sign_of(db, rels):
    s = set()
    for r in rels.split(';'):
        if db == 'signor':
            t = r.split('|')[-1]; s |= {0} if t == 'up' else {1} if t == 'down' else set()
        else:
            t = r.replace('@anchor', ''); s |= {0} if t in POS_WP else {1} if t in NEG_WP else set()
    return str(next(iter(s))) if len(s) == 1 else ''


summary = {}
for db, src in (('signor', 'signor_global'), ('wikipathways', 'wikipathways')):
    rows, c = [], collections.Counter()
    for r in csv.DictReader(gzip.open(X / 'extdb_eval/cache/edges.tsv.gz', 'rt'), delimiter='\t'):
        if r['db'] != src: continue
        c['input'] += 1
        if r['x'] not in vocab or r['y'] not in vocab or r['x'] == r['y']:
            c['drop_not_in_vocab'] += 1; continue
        if r['kegg'] != 'novel' and not r['kegg'].endswith('_train'):
            c['drop_kegg_val_test_excluded'] += 1; continue
        if r['y'] in co.get(r['x'], ()):
            c['drop_kegg_valtest_comembership'] += 1; continue
        elig = r['kegg'] == 'novel' and (db == 'signor' or r['wp_class'] == 'independent')
        g = '|'.join(sorted((find(r['x']), find(r['y']))))
        rows.append(dict(x=r['x'], y=r['y'], xid=vocab[r['x']], yid=vocab[r['y']], kegg=r['kegg'], eligible=int(elig),
                         group=g, sign=sign_of(db, r['rels']), clean=r['clean'], string150=r['string150'],
                         protein_only=int(db != 'signor' or (r['x'], r['y']) in sig_prot), wp_class=r['wp_class'],
                         rels=r['rels']))
    ginel = {r['group'] for r in rows if not r['eligible']}
    for r in rows:
        h = int(hashlib.sha1(r['group'].encode()).hexdigest(), 16) % 100
        r['split'] = 'train' if r['group'] in ginel else 'train' if h < 80 else 'val' if h < 90 else 'test'
    tr_src = collections.Counter(r['x'] for r in rows if r['split'] == 'train')
    for r in rows:
        r['src_in_train'] = int(tr_src[r['x']] > 0)
    with open(OUT / f'cache/{db}.tsv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t'); w.writeheader(); w.writerows(rows)
    s = {'filter': dict(c), 'kept': len(rows), 'groups': len({r['group'] for r in rows})}
    for sp in ('train', 'val', 'test'):
        R = [r for r in rows if r['split'] == sp]
        s[sp] = {'edges': len(R), 'groups': len({r['group'] for r in R}), 'src_kos': len({r['x'] for r in R}),
                 'sign_act': sum(r['sign'] == '0' for r in R), 'sign_inh': sum(r['sign'] == '1' for r in R),
                 'unsigned': sum(r['sign'] == '' for r in R), 'clean': sum(r['clean'] == '1' for r in R),
                 'protein_only': sum(r['protein_only'] for r in R), 'src_in_train': sum(r['src_in_train'] for r in R),
                 'kegg_train_overlap': sum(r['kegg'] != 'novel' for r in R)}
    # leakage asserts: no group, and no unordered pair, in two splits
    gs = collections.defaultdict(set); ps = collections.defaultdict(set)
    for r in rows:
        gs[r['group']].add(r['split']); ps[frozenset((r['x'], r['y']))].add(r['split'])
    assert all(len(v) == 1 for v in gs.values()) and all(len(v) == 1 for v in ps.values())
    summary[db] = s
    print(db, json.dumps(s), flush=True)
(OUT / 'results/data_summary.json').write_text(json.dumps(summary, indent=1))

#!/usr/bin/env python3
"""KO-level directed edge table of NCI-PID (Pathway Commons v14 extended SIF, data source 'pid') for the zero-shot
addendum (CPU, no model). Same conventions as build_edges.py: the mapping / reference-set part of
expansion/extdb/survey_extdb.py (everything before the WikiPathways section) is executed verbatim, and the PC parsing
rules (directed types DIR_T, CHEBI rows dropped, pathway column as units) are copied from its Pathway Commons section.
Raw file: expansion/extdb/raw/pathwaycommons/pc-hgnc.txt.gz (downloaded 2026-10-15 by the survey; sha256 in
expansion/extdb/raw/SOURCES.tsv). No new download.

Output cache/edges_pid.tsv.gz with the columns of cache/edges.tsv.gz (db = nci_pid; wp_class empty; rels = SIF type).
Run (from Kegg/pathway): .venv/bin/python -I expansion/extdb_eval/build_edges_pid.py
"""
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())
import collections, csv, gzip, json
from pathlib import Path

P = Path(_KH + '/pathway')
SRC = (P / 'expansion/extdb/survey_extdb.py').read_text()
cut = SRC.index('# ------------------------------------------------------------------ WikiPathways GPML')
ns = {'__name__': 'survey_part'}
exec(compile(SRC[:cut], 'survey_extdb.py[part]', 'exec'), ns)
kos, sym_to_e = ns['kos'], ns['sym_to_e']
kegg_dir, kegg_split, string150 = ns['kegg_dir'], ns['kegg_split'], ns['string150']
reactome_und, kgml_any, train_nodes = ns['reactome_und'], ns['kgml_any'], ns['train_nodes']

DIR_T = {'controls-state-change-of', 'controls-expression-of', 'controls-phosphorylation-of', 'controls-transport-of',
         'catalysis-precedes'}
E = collections.defaultdict(lambda: {'units': set(), 'rels': set()})
n_rows = 0
with gzip.open(P / 'expansion/extdb/raw/pathwaycommons/pc-hgnc.txt.gz', 'rt') as f:
    next(f)
    for line in f:
        if not line.strip(): break
        c = line.rstrip('\n').split('\t')
        if len(c) < 4 or c[1] not in DIR_T: continue
        if 'pid' not in c[3].split(';'): continue
        if c[0].upper().startswith('CHEBI:') or c[2].upper().startswith('CHEBI:'): continue
        n_rows += 1
        pws = [p for p in (c[5].split(';') if len(c) > 5 else []) if p] or ['(no pathway)']
        for x in kos(sym_to_e(c[0])):
            for y in kos(sym_to_e(c[2])):
                if x != y:
                    E[(x, y)]['units'].update(pws); E[(x, y)]['rels'].add(c[1])

out = P / 'expansion/extdb_eval/cache/edges_pid.tsv.gz'
c = collections.Counter()
with gzip.open(out, 'wt') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['db', 'x', 'y', 'kegg', 'string150', 'reactome', 'kgml_any', 'clean', 'train_nodes', 'units', 'wp_class', 'rels', 'signor_direct'])
    for (x, y), v in sorted(E.items()):
        if (x, y) in kegg_dir: kg = 'same_' + kegg_split[(x, y)]
        elif (y, x) in kegg_dir: kg = 'reverse_' + kegg_split[(y, x)]
        else: kg = 'novel'
        s = int(frozenset((x, y)) in string150); r = int(frozenset((x, y)) in reactome_und); k = int(frozenset((x, y)) in kgml_any)
        cl = int(kg == 'novel' and not (s or r or k)); tn = int(x in train_nodes and y in train_nodes)
        w.writerow(['nci_pid', x, y, kg, s, r, k, cl, tn, ';'.join(sorted(v['units'])), '', ';'.join(sorted(v['rels'])), ''])
        c['edges'] += 1; c[kg.split('_')[0]] += 1; c['novel_clean'] += cl; c['novel_string150'] += s and kg == 'novel'
stats = {'directed_pid_rows': n_rows, **c}
(P / 'expansion/extdb_eval/results/edges_pid_summary.json').write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))

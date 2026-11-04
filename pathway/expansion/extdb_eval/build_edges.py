#!/usr/bin/env python3
"""KO-level directed edge tables of SIGNOR and WikiPathways for the zero-shot evaluation (CPU, no model).

The parsing and every convention (gene -> KO, KEGG / STRING / Reactome / KGML reference sets, WikiPathways provenance
flags, SIGNOR complex / family expansion) are taken verbatim from expansion/extdb/survey_extdb.py: its source is read
and executed up to (not including) the OmniPath section, with its `run` replaced by a function that keeps the records.
survey_extdb.py itself is not modified.

Output expansion/extdb_eval/cache/edges.tsv.gz, one row per (source db, directed KO pair x -> y):
  db               signor_global | signor_pathways | wikipathways
  x, y             KO ids (x != y)
  kegg             novel | same_<split> | reverse_<split>   (edge_split.csv, either orientation)
  string150        1 if the unordered pair is in the STRING-exp150 background used by PathArrowNet
  reactome         1 if a Reactome next-step pair in either direction
  kgml_any         1 if any KGML gene-gene relation (any type) links the pair
  clean            novel and string150 = reactome = kgml_any = 0     (survey definition)
  train_nodes      1 if x and y both occur in KEGG strict-split TRAIN edges
  units            ';'-joined pathway ids (WikiPathways / SIGNOR pathways) or 'global'
  wp_class         WikiPathways only: independent | kegg_adapted | reactome_derived; an edge is kegg_adapted if every
                   unit carrying it is flagged (KeggConverter / KeggId, a KEGG mention, or > 75 % KEGG edge overlap)
  rels             ';'-joined relation labels;  signor_direct: SIGNOR DIRECT flags seen ('t', 'f')
Run (from Kegg/pathway): .venv/bin/python -I expansion/extdb_eval/build_edges.py
"""
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())
import collections, csv, gzip, json
from pathlib import Path

P = Path(_KH + '/pathway')
SRC = (P / 'expansion/extdb/survey_extdb.py').read_text()
cut = SRC.index('# ------------------------------------------------------------------ OmniPath')
RUN = ("def run(name, recs, raw_nodes, n_pw, extra=None):\n"
       "    r, units, novel, clean = analyze(name, recs, raw_nodes, n_pw, extra)\n"
       "    RESULTS[name] = r; NOVEL[name] = (novel, clean)\n")
assert SRC.count(RUN) == 1
body = SRC[:cut].replace(RUN, "def run(name, recs, raw_nodes, n_pw, extra=None):\n    CAPTURE[name] = recs\n")
ns = {'__name__': 'survey_part', 'CAPTURE': {}}
exec(compile(body, 'survey_extdb.py[part]', 'exec'), ns)
CAP = ns['CAPTURE']; kos = ns['kos']
kegg_dir, kegg_split, string150 = ns['kegg_dir'], ns['kegg_split'], ns['string150']
reactome_und, kgml_any, train_nodes = ns['reactome_und'], ns['kgml_any'], ns['train_nodes']
wp_flags, wp_mention = ns['wp_flags'], ns['wp_mention']
print('captured', {k: len(v) for k, v in CAP.items()}, flush=True)


def project(recs):
    E = collections.defaultdict(lambda: {'units': set(), 'rels': set(), 'direct': set()})
    for r in recs:
        if not r['directed']:
            continue
        for x in kos(r['ea']):
            for y in kos(r['eb']):
                if x != y:
                    e = E[(x, y)]; e['units'].add(r['unit']); e['rels'].add(r['rel'])
                    if r.get('direct'): e['direct'].add(r['direct'])
    return E


# WikiPathways: per-pathway KEGG overlap (either orientation) for the adaptation flag
WP = project(CAP['wikipathways_all'])
pw_edges = collections.defaultdict(set)
for e, v in WP.items():
    for u in v['units']: pw_edges[u].add(e)
pw_over = {u: sum(1 for e in E if e in kegg_dir or (e[1], e[0]) in kegg_dir) / len(E) for u, E in pw_edges.items()}


def wp_unit_class(u):
    kc, rc = wp_flags.get(u, (False, False))
    if rc: return 'reactome_derived'
    if kc or wp_mention.get(u, False) or pw_over.get(u, 0) > 0.75: return 'kegg_adapted'
    return 'independent'


SETS = {'signor_global': project(CAP['signor_global']), 'signor_pathways': project(CAP['signor_pathways']), 'wikipathways': WP}
out = P / 'expansion/extdb_eval/cache/edges.tsv.gz'
stats = {}
with gzip.open(out, 'wt') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['db', 'x', 'y', 'kegg', 'string150', 'reactome', 'kgml_any', 'clean', 'train_nodes', 'units', 'wp_class', 'rels', 'signor_direct'])
    for db, E in SETS.items():
        c = collections.Counter()
        for (x, y), v in sorted(E.items()):
            if (x, y) in kegg_dir: kg = 'same_' + kegg_split[(x, y)]
            elif (y, x) in kegg_dir: kg = 'reverse_' + kegg_split[(y, x)]
            else: kg = 'novel'
            s = int(frozenset((x, y)) in string150); r = int(frozenset((x, y)) in reactome_und); k = int(frozenset((x, y)) in kgml_any)
            cl = int(kg == 'novel' and not (s or r or k))
            tn = int(x in train_nodes and y in train_nodes)
            wc = ''
            if db == 'wikipathways':
                cls = {wp_unit_class(u) for u in v['units']}
                wc = 'independent' if 'independent' in cls else ('kegg_adapted' if 'kegg_adapted' in cls else 'reactome_derived')
            w.writerow([db, x, y, kg, s, r, k, cl, tn, ';'.join(sorted(v['units'])), wc, ';'.join(sorted(v['rels'])), ';'.join(sorted(v['direct']))])
            c['edges'] += 1; c[kg.split('_')[0]] += 1; c['novel_clean'] += cl; c['novel_string150'] += s and kg == 'novel'
            if wc: c[f'novel_wp_{wc}'] += kg == 'novel'
        stats[db] = dict(c)
stats['wikipathways_units_by_class'] = dict(collections.Counter(wp_unit_class(u) for u in pw_edges))
(P / 'expansion/extdb_eval/results/edges_summary.json').write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))

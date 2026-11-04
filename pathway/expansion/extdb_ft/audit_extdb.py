#!/usr/bin/env python3
"""Verified statistics of the external directed relationship datasets (CPU, no model).

Every number is recomputed from the raw files (provenance: expansion/extdb/raw/SOURCES.tsv). Parsing / mapping
conventions are those of expansion/extdb/survey_extdb.py (executed verbatim up to the OmniPath section, its `run`
replaced by a capture, as in extdb_eval/build_edges.py); NCI-PID rows follow extdb_eval/build_edges_pid.py.
KO-level overlap with KEGG uses the KO edge tables of extdb_eval/cache/edges*.tsv.gz (KEGG split 1 =
kegg_dataset/splits_ko/edge_split.csv, the split of the PathArrowNet checkpoints).

Sign conventions (gene-level records):
  SIGNOR        EFFECT up-regulates* -> positive, down-regulates* -> negative, 'form complex' undirected, else unsigned
  WikiPathways  mim-stimulation, mim-necessary-stimulation, mim-transcription-translation -> positive;
                mim-inhibition, TBar -> negative; 'Arrow' -> generic arrow (counted separately, not a sign);
                other arrow heads (binding, conversion, catalysis, cleavage, modification ...) -> unsigned
  NCI-PID (PC SIF) no sign field
Evidence: SIGNOR PMID column (manually curated from literature) and DIRECT flag; NCI-PID INTERACTION_PUBMED_ID;
WikiPathways interaction-level <BiopaxRef> (literature reference attached to the interaction).
Perturbation support (no such field in any source): gene-level directed pair (a -> b) with a perturbed and b measured
in one of the 6 processed Perturb-seq datasets ("tested"), and b `responsive` to a ("supported").
Run (from Kegg/pathway): .venv/bin/python -I expansion/extdb_ft/audit_extdb.py
"""
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())
import collections, csv, gzip, json, re, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

csv.field_size_limit(10 ** 9)
P = Path(_KH + '/pathway')
X = P / 'expansion'
SRC = (X / 'extdb/survey_extdb.py').read_text()
cut = SRC.index('# ------------------------------------------------------------------ OmniPath')
RUN = ("def run(name, recs, raw_nodes, n_pw, extra=None):\n"
       "    r, units, novel, clean = analyze(name, recs, raw_nodes, n_pw, extra)\n"
       "    RESULTS[name] = r; NOVEL[name] = (novel, clean)\n")
assert SRC.count(RUN) == 1
body = SRC[:cut].replace(RUN, "def run(name, recs, raw_nodes, n_pw, extra=None):\n    CAPTURE[name] = (recs, raw_nodes)\n")
ns = {'__name__': 'survey_part', 'CAPTURE': {}}
exec(compile(body, 'survey_extdb.py[part]', 'exec'), ns)
CAP, kos, sym_to_e = ns['CAPTURE'], ns['kos'], ns['sym_to_e']
OUT = {}

POS_WP = {'mim-stimulation', 'mim-necessary-stimulation', 'mim-transcription-translation'}
NEG_WP = {'mim-inhibition', 'TBar'}


def wp_sign(rel):
    r = rel.replace('@anchor', '')
    return 'pos' if r in POS_WP else 'neg' if r in NEG_WP else 'arrow' if r == 'Arrow' else 'unsigned'


def sig_sign(rel):
    s = rel.split('|')[-1]
    return 'pos' if s == 'up' else 'neg' if s == 'down' else 'unsigned'


# ---------------------------------------------------------------- perturbation support (gene level, Entrez)
pert_tested = collections.defaultdict(set)   # entrez a -> set of measured entrez b, any dataset
pert_resp = set()
meas = collections.defaultdict(set); perts = collections.defaultdict(set)
for d in (X / 'perturbseq', X / 'perturbseq/nadig'):
    for r in csv.DictReader((d / 'measured_genes.tsv').open(), delimiter='\t'):
        if r['entrez']: meas[r['dataset']].add(r['entrez'])
    for r in csv.DictReader((d / 'perturbations.tsv').open(), delimiter='\t'):
        if r['entrez']: perts[r['dataset']].add(r['entrez'])
    with gzip.open(d / 'samples_responsive.tsv.gz', 'rt') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['label'] == 'responsive' and r['pert_entrez'] and r['resp_entrez']:
                pert_resp.add((r['pert_entrez'], r['resp_entrez']))
for ds in perts:
    for a in perts[ds]:
        pert_tested[a] |= meas[ds]


def gene_level(name, recs, raw_nodes, sign_fn, raw_count, evidence=None):
    """recs: survey records (gene-level, complexes / families expanded). raw_nodes: raw node id -> entrez or None."""
    r = {'raw_records': raw_count, 'parsed_gene_gene_records': len(recs)}
    r['raw_nodes'] = len(raw_nodes)
    r['nodes_mapped_entrez'] = sum(1 for e in raw_nodes.values() if e)
    r['nodes_mapped_ko'] = sum(1 for e in raw_nodes.values() if kos(e))
    r['node_map_rate_entrez'] = round(r['nodes_mapped_entrez'] / max(1, r['raw_nodes']), 4)
    r['node_map_rate_ko'] = round(r['nodes_mapped_ko'] / max(1, r['raw_nodes']), 4)
    dirr = [x for x in recs if x['directed']]
    r['directed_records'] = len(dirr)
    usable = [x for x in dirr if x['ea'] and x['eb'] and x['ea'] != x['eb']]
    r['usable_directed_records_both_entrez'] = len(usable)
    r['usable_directed_records_both_ko'] = sum(1 for x in usable if kos(x['ea']) and kos(x['eb']))
    r['unique_genes_in_usable'] = len({g for x in usable for g in (x['ea'], x['eb'])})
    pairs = collections.defaultdict(collections.Counter)
    for x in usable:
        pairs[(x['ea'], x['eb'])][sign_fn(x['rel'])] += 1
    r['unique_directed_gene_pairs'] = len(pairs)
    sc = collections.Counter()
    for p, c in pairs.items():
        s = set(c) - {'unsigned', 'arrow'}
        sc['pos_only' if s == {'pos'} else 'neg_only' if s == {'neg'} else 'contradictory_pos_and_neg' if s == {'pos', 'neg'}
           else ('generic_arrow_only' if 'arrow' in c else 'unsigned_only')] += 1
    r['directed_gene_pairs_by_sign'] = dict(sc)
    rc = collections.Counter(sign_fn(x['rel']) for x in usable)
    r['usable_records_by_sign'] = dict(rc)
    r['duplicate_records'] = sum(sum(c.values()) - len(c) for c in pairs.values())  # same pair, same sign label, repeated
    r['pairs_with_reverse_pair_present'] = sum(1 for (a, b) in pairs if (b, a) in pairs)
    tested = [(a, b) for (a, b) in pairs if b in pert_tested.get(a, ())]
    r['perturbation_tested_pairs'] = len(tested)
    r['perturbation_supported_pairs(responsive)'] = sum(1 for p in tested if p in pert_resp)
    if evidence: r.update(evidence)
    return r


def ko_level(db, path=X / 'extdb_eval/cache/edges.tsv.gz'):
    r = collections.Counter(); split = collections.Counter(); sg = collections.Counter()
    nodes = set()
    with gzip.open(path, 'rt') as f:
        for e in csv.DictReader(f, delimiter='\t'):
            if e['db'] != db: continue
            r['ko_directed_edges'] += 1; nodes |= {e['x'], e['y']}
            split[e['kegg']] += 1
            r['novel'] += e['kegg'] == 'novel'; r['novel_clean'] += int(e['clean'])
            r['novel_string150'] += e['kegg'] == 'novel' and e['string150'] == '1'
            fn = wp_sign if db == 'wikipathways' else sig_sign if db.startswith('signor') else (lambda _: 'unsigned')
            s = {fn(x) for x in e['rels'].split(';')} - {'unsigned', 'arrow'}
            sg['pos_only' if s == {'pos'} else 'neg_only' if s == {'neg'} else 'contradictory' if len(s) == 2 else 'no_sign'] += 1
            if db == 'wikipathways': r[f"wp_{e['wp_class']}"] += 1
    return {**dict(r), 'ko_nodes': len(nodes), 'kegg_overlap_by_split_1': dict(split), 'ko_edges_by_sign': dict(sg)}


# ---------------------------------------------------------------- SIGNOR (global release + pathway files)
rel_rows, pw_rows = ns['rel_rows'], ns['pw_rows']
srecs, snodes = CAP['signor_global']
pm = [r for r in rel_rows if r['TAX_ID'] in ('9606', '-1', '')]
ev = {'release_rows_all_organisms_all_entity_types': len(rel_rows),
      'rows_human_or_unspecified_tax': len(pm),
      'rows_protein_complex_family_both_ends': sum(1 for r in pm if r['TYPEA'] in ns['SIG_OK'] and r['TYPEB'] in ns['SIG_OK']),
      'rows_with_pmid': sum(1 for r in rel_rows if re.fullmatch(r'\d+', (r['PMID'] or '').strip())),
      'rows_direct_t': sum(1 for r in rel_rows if r['DIRECT'] == 't'),
      'rows_effect_up': sum(1 for r in rel_rows if r['EFFECT'].startswith('up')),
      'rows_effect_down': sum(1 for r in rel_rows if r['EFFECT'].startswith('down')),
      'rows_form_complex': sum(1 for r in rel_rows if r['EFFECT'] == 'form complex'),
      'usable_records_with_direct_t': sum(1 for x in srecs if x['directed'] and x['ea'] and x['eb'] and x['direct'] == 't'),
      'usable_records_from_complex_family_expansion': sum(1 for x in srecs if x['directed'] and x['ea'] and x['eb'] and x['expanded']),
      'version': 'SIGNOR Oct2026 release (Oct2026_release.txt), downloaded 2026-10-15', 'license': 'CC BY 4.0'}
OUT['signor_global'] = {'gene_level': gene_level('signor_global', srecs, snodes, sig_sign, len(rel_rows), ev),
                        'ko_level': ko_level('signor_global')}
precs, pnodes = CAP['signor_pathways']
OUT['signor_pathways'] = {'gene_level': gene_level('signor_pathways', precs, pnodes, sig_sign, len(pw_rows),
                                                   {'n_pathways': len({r['pathway_id'] for r in pw_rows}),
                                                    'version': '148 SIGNOR pathway files, downloaded 2026-10-15'}),
                          'ko_level': ko_level('signor_pathways')}

# ---------------------------------------------------------------- WikiPathways
wrecs, wnodes = CAP['wikipathways_all']
zf = zipfile.ZipFile(P / 'expansion/extdb/raw/wikipathways/wikipathways-20261010-gpml-Homo_sapiens.zip')
NS = '{http://pathvisio.org/GPML/2013a}'
n_inter = n_ref = n_files = 0
for fn in zf.namelist():
    if not fn.endswith('.gpml'): continue
    n_files += 1
    root = ET.fromstring(zf.read(fn).decode('utf-8', 'replace'))
    for it in root.findall(f'{NS}Interaction'):
        n_inter += 1; n_ref += it.find(f'{NS}BiopaxRef') is not None
flags = ns['wp_flags']
OUT['wikipathways'] = {'gene_level': gene_level('wikipathways', wrecs, wnodes, wp_sign, n_inter, {
    'gpml_files': n_files, 'gpml_interactions_with_BiopaxRef(literature)': n_ref,
    'pathways_flag_keggconverter': sum(k for k, _ in flags.values()),
    'pathways_flag_reactome_derived': sum(r for _, r in flags.values()),
    'version': 'wikipathways-20261010-gpml-Homo_sapiens.zip, downloaded 2026-10-15', 'license': 'CC0 1.0'}),
    'ko_level': ko_level('wikipathways')}

# ---------------------------------------------------------------- NCI-PID (Pathway Commons v14 extended SIF)
DIR_T = {'controls-state-change-of', 'controls-expression-of', 'controls-phosphorylation-of', 'controls-transport-of',
         'catalysis-precedes'}
raw_pid = 0; recs = []; nodes = {}; n_pmid = 0; types = collections.Counter()
with gzip.open(P / 'expansion/extdb/raw/pathwaycommons/pc-hgnc.txt.gz', 'rt') as f:
    next(f)
    for line in f:
        if not line.strip(): break
        c = line.rstrip('\n').split('\t')
        if len(c) < 4 or 'pid' not in c[3].split(';'): continue
        raw_pid += 1; types[c[1]] += 1
        if c[1] not in DIR_T: continue
        if c[0].upper().startswith('CHEBI:') or c[2].upper().startswith('CHEBI:'): continue
        ea, eb = sym_to_e(c[0]), sym_to_e(c[2]); nodes[c[0]] = ea; nodes[c[2]] = eb
        n_pmid += bool(len(c) > 4 and c[4].strip())
        recs.append({'ea': ea, 'eb': eb, 'rel': c[1], 'directed': True})
OUT['nci_pid'] = {'gene_level': gene_level('nci_pid', recs, nodes, lambda _: 'unsigned', raw_pid, {
    'pid_rows_by_sif_type': dict(types), 'directed_protein_rows_with_pubmed': n_pmid,
    'version': 'Pathway Commons v14 pc-hgnc.txt.gz (data source pid), downloaded 2026-10-15',
    'note': 'SIF has no activation / inhibition field'}),
    'ko_level': ko_level('nci_pid', X / 'extdb_eval/cache/edges_pid.tsv.gz')}

# ---------------------------------------------------------------- KEGG reference (split 1)
kc = collections.Counter(); ksign = {}
with open(P / 'kegg_dataset/splits_ko/edge_split.csv') as f:
    for r in csv.DictReader(f): kc[r['split']] += 1
with open(X / 'signed/ko_edge_sign.tsv') as f:
    rd = csv.DictReader(f, delimiter='\t'); cols = rd.fieldnames
    sc = collections.Counter(r['sign'] for r in rd)
OUT['kegg_split1'] = {'ko_directed_edges_by_split': dict(kc), 'sign_file_columns': cols, 'sign_counts': dict(sc)}
(X / 'extdb_ft/results/audit.json').write_text(json.dumps(OUT, indent=1))
print(json.dumps(OUT, indent=1))

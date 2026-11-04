"""Ligand-receptor (LR) gene-pair table + KEGG leakage annotation.

Sources (downloaded to data/ligand_receptor/raw/):
  CellPhoneDB v5.0.0 (github ventolab/cellphonedb-data tag v5.0.0) interaction/gene/protein/complex_input.csv
  OmniPath ligrecextra (omnipathdb.org/interactions?datasets=ligrecextra, license=academic)
Outputs (data/ligand_receptor/):
  lr_pairs_gene.tsv   one row per unordered gene pair (Entrez), with source flags, roles and KEGG-overlap annotations
  lr_genes.tsv        gene universe (ligand / receptor role, KO, feature coverage) for negative sampling
  lr_summary.txt
Run: .venv/bin/python -I build_lr_pairs.py
"""
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())
import csv, collections, itertools

P = _KH + '/pathway'
D = f'{P}/expansion/downstream/data'
R = f'{D}/ligand_receptor/raw'

# ---------- gene map ----------
gm = {}
sym2e, uni2e = {}, {}
with open(f'{D}/common/gene_map.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        gm[r['entrez']] = r
        if r['symbol']: sym2e.setdefault(r['symbol'], r['entrez'])
with open(f'{D}/common/raw/hgnc_complete_set.txt', newline='') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        if r['status'] == 'Approved' and r['entrez_id']:
            for u in r['uniprot_ids'].strip('"').split('|'):
                if u: uni2e.setdefault(u, r['entrez_id'])
with open(f'{P}/kegg_dataset/gene_uniprot_mapping.csv') as f:
    for r in csv.DictReader(f):
        for u in (r['uniprot_accessions'] or '').split(';'):
            if u: uni2e.setdefault(u, r['ncbi_gene_id'])

# ---------- CellPhoneDB ----------
cp_gene = {}
with open(f'{R}/cpdb_v5.0.0_gene_input.csv') as f:
    for r in csv.DictReader(f):
        cp_gene.setdefault(r['uniprot'], r['hgnc_symbol'] or r['gene_name'])
cp_prot = {}
with open(f'{R}/cpdb_v5.0.0_protein_input.csv') as f:
    for r in csv.DictReader(f):
        cp_prot[r['uniprot']] = r
cp_cplx = {}
with open(f'{R}/cpdb_v5.0.0_complex_input.csv') as f:
    for r in csv.DictReader(f):
        cp_cplx[r['complex_name']] = [r[f'uniprot_{i}'] for i in range(1, 6) if r.get(f'uniprot_{i}')]

unmapped = collections.Counter()

def u2e(u):
    e = uni2e.get(u)
    if e is None and u in cp_gene: e = sym2e.get(cp_gene[u])
    if e is None: unmapped[u] += 1
    return e

def cp_partner(p):
    us = cp_cplx.get(p, [p])
    return [e for e in (u2e(u) for u in us) if e]

pairs = collections.defaultdict(lambda: dict(sources=set(), classes=set(), directionality=set(),
                                              ordered=set(), omni_sources=set(), from_complex=False))
n_cp = 0
with open(f'{R}/cpdb_v5.0.0_interaction_input.csv') as f:
    for r in csv.DictReader(f):
        n_cp += 1
        A, B = cp_partner(r['partner_a']), cp_partner(r['partner_b'])
        cplx = r['partner_a'] in cp_cplx or r['partner_b'] in cp_cplx
        for a, b in itertools.product(A, B):
            if a == b: continue
            k = tuple(sorted((a, b)))
            d = pairs[k]; d['sources'].add('CellPhoneDB_v5'); d['classes'].add(r['classification'])
            d['directionality'].add(r['directionality']); d['from_complex'] |= cplx
            if r['directionality'] == 'Ligand-Receptor':
                d['ordered'].add((a, b))   # CellPhoneDB convention partner_a ~ ligand side (not guaranteed)

n_om = 0
with open(f'{R}/omnipath_ligrecextra.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        n_om += 1
        def side(acc, sym):
            if acc.startswith('COMPLEX:'):
                return [e for e in (u2e(u) for u in acc[8:].split('_')) if e], True
            e = u2e(acc) or sym2e.get(sym)
            return ([e] if e else []), False
        A, ca = side(r['source'], r['source_genesymbol']); B, cb = side(r['target'], r['target_genesymbol'])
        for a, b in itertools.product(A, B):
            if a == b: continue
            k = tuple(sorted((a, b)))
            d = pairs[k]; d['sources'].add('OmniPath_ligrecextra'); d['from_complex'] |= (ca or cb)
            d['omni_sources'] |= set(r['sources'].split(';')); d['ordered'].add((a, b))

# ---------- KEGG annotations ----------
# gene-level: any KGML relation (directed or not, any type) between the two genes, by pathway
kgml_pairs = collections.defaultdict(set)
with open(f'{P}/kegg_dataset/kgml_relations.csv') as f:
    for r in csv.DictReader(f):
        if r['source_entry_type'] != 'gene' or r['target_entry_type'] != 'gene': continue
        S = [x.split(':')[1] for x in r['source_kegg_ids'].split() if x.startswith('hsa:')]
        T = [x.split(':')[1] for x in r['target_kegg_ids'].split() if x.startswith('hsa:')]
        for a, b in itertools.product(S, T):
            if a != b: kgml_pairs[tuple(sorted((a, b)))].add(r['pathway_id'])
# KO-level strict split edges (directed edges, either orientation)
ko_edge = {}
with open(f'{P}/kegg_dataset/splits_ko/edge_split.csv') as f:
    for r in csv.DictReader(f):
        ko_edge[(r['source_ko'], r['target_ko'])] = r['split']
psplit = {}
with open(f'{P}/kegg_dataset/splits_ko/pathway_split.csv') as f:
    for r in csv.DictReader(f):
        psplit[r['pathway_id']] = r['split']
ko_paths = {}
with open(f'{P}/kegg_dataset/ko_mapping.csv') as f:
    for r in csv.DictReader(f):
        ko_paths[r['ko_id']] = set(filter(None, (r['pathways_as_ortholog_node'] + ';' + r['pathways_via_human_gene_nodes']).split(';')))

def kos(e):
    return [k for k in gm.get(e, {}).get('kos_in_vocab', '').split(';') if k]

order = {'train': 0, 'val': 1, 'test': 2, 'excluded_reverse': 3}
out = []
for (a, b), d in pairs.items():
    ka, kb = kos(a), kos(b)
    sp = {ko_edge[(x, y)] for x in ka for y in kb if (x, y) in ko_edge} | \
         {ko_edge[(y, x)] for x in ka for y in kb if (y, x) in ko_edge}
    ko_split = min(sp, key=order.get) if sp else 'none'
    pa = set().union(*[ko_paths.get(k, set()) for k in ka]) if ka else set()
    pb = set().union(*[ko_paths.get(k, set()) for k in kb]) if kb else set()
    shared = pa & pb
    co_train = any(psplit.get(p) == 'train' for p in shared)
    kg = kgml_pairs.get((a, b), set())
    ga, gb = gm.get(a, {}), gm.get(b, {})
    kegg_derived_src = sorted(s for s in d['omni_sources'] if 'KEGG' in s.upper() or s in ('CellChatDB', 'CellCall'))
    out.append(dict(gene_a=a, gene_b=b, symbol_a=ga.get('symbol', ''), symbol_b=gb.get('symbol', ''),
                    sources=';'.join(sorted(d['sources'])), n_omnipath_resources=len(d['omni_sources']),
                    kegg_derived_resources=';'.join(kegg_derived_src),
                    cpdb_classification=';'.join(sorted(d['classes'])), cpdb_directionality=';'.join(sorted(d['directionality'])),
                    ligand_receptor_orientation=';'.join(f'{x}>{y}' for x, y in sorted(d['ordered'])),
                    from_complex=int(d['from_complex']),
                    kos_a=';'.join(ka), kos_b=';'.join(kb), both_have_ko=int(bool(ka and kb)),
                    both_gene_esm2=int(ga.get('gene_has_esm2') == '1' and gb.get('gene_has_esm2') == '1'),
                    both_gene_gobert=int(ga.get('gene_has_gobert') == '1' and gb.get('gene_has_gobert') == '1'),
                    kgml_any_relation=int(bool(kg)), kgml_pathways=';'.join(sorted(kg)),
                    ko_edge_split=ko_split, same_ko=int(bool(set(ka) & set(kb))),
                    ko_copathway_any=int(bool(shared)), ko_copathway_train=int(co_train)))
cols = list(out[0].keys())
with open(f'{D}/ligand_receptor/lr_pairs_gene.tsv', 'w', newline='') as f:
    w = csv.DictWriter(f, cols, delimiter='\t'); w.writeheader()
    for r in sorted(out, key=lambda r: (int(r['gene_a']), int(r['gene_b']))): w.writerow(r)

# gene universe with roles
role = collections.defaultdict(set)
for (a, b), d in pairs.items():
    for x, y in d['ordered']:
        role[x].add('ligand'); role[y].add('receptor')
    if not d['ordered']:
        role[a].add('other'); role[b].add('other')
deg = collections.Counter(x for k in pairs for x in k)
# KO-grouped gene-disjoint split (genes sharing a vocab KO always land in the same split), 80/10/10, seed 42
import random
parent = {}
def find(x):
    while parent.setdefault(x, x) != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for e in deg:
    for k in kos(e): parent[find('g' + e)] = find('k' + k)
comps = collections.defaultdict(list)
for e in sorted(deg, key=int): comps[find('g' + e)].append(e)
keys = sorted(comps); random.Random(42).shuffle(keys)
gsplit, tot = {}, 0
for k in keys:
    fr = tot / len(deg)
    for e in comps[k]: gsplit[e] = 'train' if fr < 0.8 else ('val' if fr < 0.9 else 'test')
    tot += len(comps[k])
with open(f'{D}/ligand_receptor/lr_genes.tsv', 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['entrez', 'symbol', 'roles', 'degree', 'kos_in_vocab', 'gene_has_esm2', 'gene_has_gobert', 'gene_split'])
    for e in sorted(deg, key=int):
        g = gm.get(e, {})
        w.writerow([e, g.get('symbol', ''), ';'.join(sorted(role[e])), deg[e], g.get('kos_in_vocab', ''),
                    g.get('gene_has_esm2', '0'), g.get('gene_has_gobert', '0'), gsplit[e]])

# summary
C = collections.Counter
s = []
s.append(f'CellPhoneDB v5 interactions: {n_cp}; OmniPath ligrecextra interactions: {n_om}; unmapped UniProt accs: {len(unmapped)}')
s.append(f'unique unordered gene pairs: {len(out)}; genes: {len(deg)}')
s.append('by source: ' + str(C(r["sources"] for r in out)))
cov = [r for r in out if r['both_have_ko']]
s.append(f'both genes have KO in 11,620 vocab: {len(cov)} ({len(cov)/len(out):.1%}); both gene ESM2: {sum(r["both_gene_esm2"] for r in out)}; both GoBERT: {sum(r["both_gene_gobert"] for r in out)}')
s.append('genes with KO in vocab: %d / %d' % (sum(1 for e in deg if gm.get(e, {}).get('kos_in_vocab')), len(deg)))
for name, sub in [('ALL', out), ('both_have_ko', cov), ('CellPhoneDB-only-source', [r for r in out if r['sources'] == 'CellPhoneDB_v5'])]:
    s.append(f'--- {name} (n={len(sub)})')
    s.append('  ko_edge_split: ' + str(C(r['ko_edge_split'] for r in sub)))
    s.append('  kgml_any_relation: %d' % sum(r['kgml_any_relation'] for r in sub))
    s.append('  same_ko: %d' % sum(r['same_ko'] for r in sub))
    s.append('  ko co-pathway any: %d ; co-pathway in a TRAIN pathway: %d' % (sum(r['ko_copathway_any'] for r in sub), sum(r['ko_copathway_train'] for r in sub)))
    clean = [r for r in sub if r['ko_edge_split'] == 'none' and not r['kgml_any_relation'] and not r['ko_copathway_train'] and not r['same_ko']]
    s.append(f'  CLEAN (no KO edge, no KGML relation, not co-member of any train pathway, not same KO): {len(clean)}')
    s.append('  with KEGG-derived resource (CellChatDB/CellCall/KEGG*): %d' % sum(bool(r['kegg_derived_resources']) for r in sub))
hsa04060 = [r for r in out if 'hsa04060' in r['kgml_pathways'].split(';')]
s.append(f'pairs with a KGML relation in hsa04060: {len(hsa04060)}; split of hsa04060: {psplit.get("hsa04060")}')
s.append('KGML pathways most often hit: ' + str(C(p for r in out for p in r['kgml_pathways'].split(';') if p).most_common(10)))
s.append('split of top pathways: ' + str({p: psplit.get(p) for p, _ in C(p for r in out for p in r['kgml_pathways'].split(';') if p).most_common(10)}))
for a_, b_ in [('train', 'train'), ('val', 'val'), ('test', 'test')]:
    sub = [r for r in out if {gsplit[r['gene_a']], gsplit[r['gene_b']]} == {a_}]
    s.append(f'gene-disjoint {a_}-{a_} pairs: {len(sub)}; of which CLEAN: ' + str(sum(1 for r in sub if r['ko_edge_split'] == 'none' and not r['kgml_any_relation'] and not r['ko_copathway_train'] and not r['same_ko'])))
s.append('genes per split: ' + str(C(gsplit.values())))
open(f'{D}/ligand_receptor/lr_summary.txt', 'w').write('\n'.join(s) + '\n')
print('\n'.join(s))

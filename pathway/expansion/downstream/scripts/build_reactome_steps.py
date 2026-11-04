"""Reactome "next reaction step" gene-pair table (downstream task 3: transfer of KEGG pathway-step knowledge).

Source  Reactome release 97 BioPAX level 3, Homo_sapiens.owl (reactome.org/download/current/biopax.zip, CC0).
Pairs   For every PathwayStep s1 --nextStep--> s2 with reactions r1 in s1, r2 in s2 (a Control/Catalysis step process is
        mapped to the reaction it controls), every gene a in actors(r1) and b in actors(r2), a != b, gives the
        directed pair a -> b (same direction as a KEGG next-KO edge). actors(r) = proteins of the controllers
        (catalysts / regulators) of r and of the LEFT-side participants of r (complexes and sets expanded to their
        protein subunits). Reactions with > MAX_ACT actor genes are skipped (large complexes would explode the pair
        count and are not KEGG-like steps).
Map     UniProt -> Entrez via HGNC (+ project gene_uniprot_mapping.csv), exactly as build_lr_pairs.py.
KEGG    leakage annotation identical to the LR table: ko_edge_split (KO pair is a directed KEGG edge of the strict
        split, either orientation), kgml_any_relation (gene-gene KGML relation in any map), same_ko, ko_copathway_train.
        STRING-exp KO background flags as in add_string_flags.py.
Split   KO-grouped gene-disjoint 80/10/10, seed 42 (same procedure as lr_genes.tsv).
Outputs data/reactome/{reactome_pairs_gene.tsv, reactome_genes.tsv, reactome_pairs_gene_stringflags.tsv, summary.txt}
Run: .venv/bin/python -I build_reactome_steps.py
"""
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())
import collections, csv, itertools, random
import xml.etree.ElementTree as ET

P = _KH + '/pathway'
D = f'{P}/expansion/downstream/data'
R = f'{D}/reactome'
MAX_ACT = 10
BP = '{http://www.biopax.org/release/biopax-level3.owl#}'
RDF = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
CONV = {'BiochemicalReaction', 'Transport', 'TransportWithBiochemicalReaction', 'ComplexAssembly', 'Degradation',
        'Conversion', 'MolecularInteraction', 'TemplateReaction'}
CTRL = {'Catalysis', 'Control', 'Modulation', 'TemplateReactionRegulation'}

# ---------- parse BioPAX ----------
obj = {}  # id -> (type, {prop: [values]})
for ev, el in ET.iterparse(f'{R}/raw/Homo_sapiens.owl', events=('end',)):
    if not el.tag.startswith(BP) or (RDF + 'ID') not in el.attrib:
        continue
    t = el.tag[len(BP):]
    props = collections.defaultdict(list)
    for c in el:
        name = c.tag[len(BP):] if c.tag.startswith(BP) else None
        if name is None:
            continue
        v = c.attrib.get(RDF + 'resource')
        props[name].append(v[1:] if v and v.startswith('#') else (v or (c.text or '').strip()))
    keep = {'PathwayStep': ('stepProcess', 'nextStep'), 'Protein': ('entityReference', 'memberPhysicalEntity'),
            'Complex': ('component', 'memberPhysicalEntity'), 'ProteinReference': ('xref',),
            'UnificationXref': ('db', 'id')}
    if t in keep:
        obj[el.attrib[RDF + 'ID']] = (t, {k: props.get(k, []) for k in keep[t]})
    elif t in CONV:
        obj[el.attrib[RDF + 'ID']] = (t, {'left': props.get('left', []), 'right': props.get('right', [])})
    elif t in CTRL:
        obj[el.attrib[RDF + 'ID']] = (t, {'controller': props.get('controller', []), 'controlled': props.get('controlled', [])})
    elif t in ('SmallMolecule', 'Dna', 'Rna', 'PhysicalEntity', 'Pathway'):
        obj[el.attrib[RDF + 'ID']] = (t, {})
    el.clear()

# ---------- UniProt -> Entrez ----------
uni2e = {}
with open(f'{D}/common/raw/hgnc_complete_set.txt', newline='') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        if r['status'] == 'Approved' and r['entrez_id']:
            for u in r['uniprot_ids'].strip('"').split('|'):
                if u: uni2e.setdefault(u, r['entrez_id'])
with open(f'{P}/kegg_dataset/gene_uniprot_mapping.csv') as f:
    for r in csv.DictReader(f):
        for u in (r['uniprot_accessions'] or '').split(';'):
            if u: uni2e.setdefault(u, r['ncbi_gene_id'])


def pref_genes(pr):
    out = set()
    for x in obj.get(pr, (None, {}))[1].get('xref', []):
        t, p = obj.get(x, (None, {}))
        if t == 'UnificationXref' and p['db'] and p['db'][0] == 'UniProt':
            acc = p['id'][0].split('-')[0]
            if acc in uni2e: out.add(uni2e[acc])
    return out


_memo = {}


def genes_of(e, depth=0):
    if e in _memo: return _memo[e]
    t, p = obj.get(e, (None, {}))
    g = set()
    if depth < 12:
        if t == 'Protein':
            for pr in p['entityReference']: g |= pref_genes(pr)
            for m in p['memberPhysicalEntity']: g |= genes_of(m, depth + 1)
        elif t == 'Complex':
            for c in p['component'] + p['memberPhysicalEntity']: g |= genes_of(c, depth + 1)
    _memo[e] = g
    return g


controllers = collections.defaultdict(set)
for i, (t, p) in obj.items():
    if t in CTRL:
        for c in p['controlled']:
            controllers[c] |= set(p['controller'])


def actors(r):
    t, p = obj.get(r, (None, {}))
    if t not in CONV: return set()
    g = set()
    for e in p['left'] + list(controllers.get(r, ())): g |= genes_of(e)
    return g


def step_reactions(s):
    out = set()
    for x in obj[s][1]['stepProcess']:
        t, p = obj.get(x, (None, {}))
        if t in CONV: out.add(x)
        elif t in CTRL: out |= {c for c in p['controlled'] if obj.get(c, (None,))[0] in CONV}
    return out


steps = [i for i, (t, _) in obj.items() if t == 'PathwayStep']
act = {}
n_skip = 0
directed = collections.Counter()
n_rr = set()
for s1 in steps:
    for s2 in obj[s1][1]['nextStep']:
        if s2 not in obj: continue
        for r1 in step_reactions(s1):
            for r2 in step_reactions(s2):
                if r1 == r2: continue
                for r in (r1, r2):
                    if r not in act: act[r] = actors(r)
                if len(act[r1]) > MAX_ACT or len(act[r2]) > MAX_ACT:
                    n_skip += 1; continue
                n_rr.add((r1, r2))
                for a, b in itertools.product(act[r1], act[r2]):
                    if a != b: directed[(a, b)] += 1

# ---------- KEGG annotation (identical to build_lr_pairs.py) ----------
gm = {}
with open(f'{D}/common/gene_map.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): gm[r['entrez']] = r
kgml_pairs = collections.defaultdict(set)
with open(f'{P}/kegg_dataset/kgml_relations.csv') as f:
    for r in csv.DictReader(f):
        if r['source_entry_type'] != 'gene' or r['target_entry_type'] != 'gene': continue
        S = [x.split(':')[1] for x in r['source_kegg_ids'].split() if x.startswith('hsa:')]
        T = [x.split(':')[1] for x in r['target_kegg_ids'].split() if x.startswith('hsa:')]
        for a, b in itertools.product(S, T):
            if a != b: kgml_pairs[tuple(sorted((a, b)))].add(r['pathway_id'])
ko_edge = {}
with open(f'{P}/kegg_dataset/splits_ko/edge_split.csv') as f:
    for r in csv.DictReader(f): ko_edge[(r['source_ko'], r['target_ko'])] = r['split']
psplit = {}
with open(f'{P}/kegg_dataset/splits_ko/pathway_split.csv') as f:
    for r in csv.DictReader(f): psplit[r['pathway_id']] = r['split']
ko_paths = {}
with open(f'{P}/kegg_dataset/ko_mapping.csv') as f:
    for r in csv.DictReader(f):
        ko_paths[r['ko_id']] = set(filter(None, (r['pathways_as_ortholog_node'] + ';' + r['pathways_via_human_gene_nodes']).split(';')))
bg = {}
for name, bit in (('exp150', 1), ('exp400', 2)):
    with open(f'{P}/kegg_dataset_cache/string/ko_background_{name}_full.tsv') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            k = tuple(sorted((r['ko_a'], r['ko_b']))); bg[k] = bg.get(k, 0) | bit


def kos(e):
    return [k for k in gm.get(e, {}).get('kos_in_vocab', '').split(';') if k]


order = {'train': 0, 'val': 1, 'test': 2, 'excluded_reverse': 3}
und = collections.defaultdict(lambda: {'ab': 0, 'ba': 0})
for (a, b), c in directed.items():
    k = tuple(sorted((a, b), key=int)); und[k]['ab' if (a, b) == k else 'ba'] += c
out, sflags = [], []
for (a, b), d in sorted(und.items(), key=lambda x: (int(x[0][0]), int(x[0][1]))):
    ka, kb = kos(a), kos(b)
    sp = {ko_edge[(x, y)] for x in ka for y in kb if (x, y) in ko_edge} | {ko_edge[(y, x)] for x in ka for y in kb if (y, x) in ko_edge}
    pa = set().union(*[ko_paths.get(k, set()) for k in ka]) if ka else set()
    pb = set().union(*[ko_paths.get(k, set()) for k in kb]) if kb else set()
    shared = pa & pb
    kg = kgml_pairs.get((a, b), set())
    sb = 0
    for x in ka:
        for y in kb: sb |= bg.get(tuple(sorted((x, y))), 0)
    out.append(dict(gene_a=a, gene_b=b, symbol_a=gm.get(a, {}).get('symbol', ''), symbol_b=gm.get(b, {}).get('symbol', ''),
                    n_steps_ab=d['ab'], n_steps_ba=d['ba'], kos_a=';'.join(ka), kos_b=';'.join(kb), both_have_ko=int(bool(ka and kb)),
                    kgml_any_relation=int(bool(kg)), ko_edge_split=min(sp, key=order.get) if sp else 'none',
                    same_ko=int(bool(set(ka) & set(kb))), ko_copathway_any=int(bool(shared)),
                    ko_copathway_train=int(any(psplit.get(p) == 'train' for p in shared))))
    sflags.append((a, b, int(sb & 1 > 0), int(sb & 2 > 0)))
with open(f'{R}/reactome_pairs_gene.tsv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]), delimiter='\t'); w.writeheader(); w.writerows(out)
with open(f'{R}/reactome_pairs_gene_stringflags.tsv', 'w') as f:
    f.write('gene_a\tgene_b\tstring_exp150\tstring_exp400\n')
    for row in sflags: f.write('\t'.join(map(str, row)) + '\n')

deg = collections.Counter(x for k in und for x in k)
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
with open(f'{R}/reactome_genes.tsv', 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['entrez', 'symbol', 'roles', 'degree', 'kos_in_vocab', 'gene_has_esm2', 'gene_has_gobert', 'gene_split'])
    for e in sorted(deg, key=int):
        g = gm.get(e, {})
        w.writerow([e, g.get('symbol', ''), 'gene', deg[e], g.get('kos_in_vocab', ''), g.get('gene_has_esm2', '0'),
                    g.get('gene_has_gobert', '0'), gsplit[e]])

C = collections.Counter
clean = [r for r in out if r['ko_edge_split'] == 'none' and not r['kgml_any_relation'] and not r['ko_copathway_train'] and not r['same_ko']]
s = [f'Reactome release 97 BioPAX; PathwayStep objects {len(steps)}; reaction->reaction steps used {len(n_rr)}; skipped (> {MAX_ACT} actors) {n_skip}',
     f'directed gene pairs {len(directed)}; unordered {len(out)}; genes {len(deg)}',
     f'both genes have KO: {sum(r["both_have_ko"] for r in out)}',
     'ko_edge_split: ' + str(C(r['ko_edge_split'] for r in out)),
     f'kgml_any_relation {sum(r["kgml_any_relation"] for r in out)}; same_ko {sum(r["same_ko"] for r in out)}; '
     f'co-pathway(train) {sum(r["ko_copathway_train"] for r in out)}',
     f'CLEAN (no KO edge, no KGML relation, not co-member of a train pathway, not same KO): {len(clean)}; '
     f'with KO on both sides: {sum(r["both_have_ko"] for r in clean)}',
     f'STRING exp150 flagged {sum(x[2] for x in sflags)}',
     'genes per split: ' + str(C(gsplit.values()))]
open(f'{R}/summary.txt', 'w').write('\n'.join(s) + '\n')
print('\n'.join(s))

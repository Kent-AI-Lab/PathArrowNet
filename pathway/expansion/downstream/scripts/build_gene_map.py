"""Build the human gene <-> KO <-> feature availability map shared by all downstream tasks.

Inputs (read-only, existing project files):
  kegg_dataset_cache/kegg_rest/link_ko_hsa.tsv   (KEGG REST link/ko/hsa)
  kegg_dataset/token_vocab_ko_full.csv           (11,620 KO vocab used by the models)
  kegg_features/ko_features_full.pt              (KO-level feature 'has' masks)
  kegg_features/gene_esm2.pt, gene_gobert.pt     (gene-level features keyed by NCBI GeneID)
  downstream/data/common/raw/hgnc_complete_set.txt (HGNC, downloaded)
Output: downstream/data/common/gene_map.tsv
Run: .venv/bin/python -I build_gene_map.py
"""
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())
import csv, collections, os
import torch

P = _KH + '/pathway'
D = f'{P}/expansion/downstream/data/common'

g2ko = collections.defaultdict(set)
with open(f'{P}/kegg_dataset_cache/kegg_rest/link_ko_hsa.tsv') as f:
    for line in f:
        a, b = line.rstrip('\n').split('\t')
        g2ko[a.split(':')[1]].add(b.split(':')[1])

vocab = {}
with open(f'{P}/kegg_dataset/token_vocab_ko_full.csv') as f:
    for r in csv.DictReader(f):
        if r['token'].startswith('K'):
            vocab[r['token']] = int(r['token_id'])
kf = torch.load(f'{P}/kegg_features/ko_features_full.pt', weights_only=False)
ko_has = {name: v['has'].numpy() for name, v in kf['features'].items()}

esm = torch.load(f'{P}/kegg_features/gene_esm2.pt', weights_only=False)['mean']
gob = torch.load(f'{P}/kegg_features/gene_gobert.pt', weights_only=False)['all']['mean']

# genes appearing in KEGG pathway maps (gene_edges / ko_mapping human_genes_in_pathways)
in_path = set()
with open(f'{P}/kegg_dataset/ko_mapping.csv') as f:
    for r in csv.DictReader(f):
        for g in (r['human_genes_in_pathways'] or '').split(';'):
            if g: in_path.add(g)

rows = []
seen = set()
with open(f'{D}/raw/hgnc_complete_set.txt', newline='') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        if r['status'] != 'Approved':
            continue
        e = r['entrez_id'].strip()
        rows.append(dict(entrez=e, symbol=r['symbol'], ensembl=r['ensembl_gene_id'].strip(),
                         locus_group=r['locus_group'], prev_alias=r['prev_symbol'].strip('"').replace('|', ',') + '|' +
                         r['alias_symbol'].strip('"').replace('|', ',')))
        if e: seen.add(e)
# KEGG genes with KO but not in HGNC (rare)
for e in g2ko:
    if e not in seen:
        rows.append(dict(entrez=e, symbol='', ensembl='', locus_group='kegg_only', prev_alias=''))

out = f'{D}/gene_map.tsv'
cols = ['entrez', 'symbol', 'ensembl', 'locus_group', 'kos', 'kos_in_vocab', 'n_kos_in_vocab',
        'ko_has_esm2', 'ko_has_gobert', 'gene_has_esm2', 'gene_has_gobert', 'in_kegg_pathway_maps', 'prev_alias']
st = collections.Counter()
with open(out, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t'); w.writerow(cols)
    for r in rows:
        e = r['entrez']
        kos = sorted(g2ko.get(e, ()))
        kv = [k for k in kos if k in vocab]
        kes = int(any(ko_has['esm2_mean'][vocab[k]] for k in kv))
        kgo = int(any(ko_has['gobert_all_mean'][vocab[k]] for k in kv))
        ge, gg = int(e in esm), int(e in gob)
        ip = int(e in in_path)
        w.writerow([e, r['symbol'], r['ensembl'], r['locus_group'], ';'.join(kos), ';'.join(kv), len(kv),
                    kes, kgo, ge, gg, ip, r['prev_alias']])
        pc = r['locus_group'] == 'protein-coding gene'
        st['rows'] += 1; st['protein_coding'] += pc
        st['has_ko'] += bool(kos); st['has_ko_in_vocab'] += bool(kv)
        st['pc_has_ko_in_vocab'] += pc and bool(kv)
        st['gene_esm2'] += ge; st['gene_gobert'] += gg; st['in_pathway_maps'] += ip
        st['ko_vocab_and_ko_esm2'] += kes
print(dict(st))
print('KO vocab size', len(vocab), 'human genes with any KO', len(g2ko),
      'KOs reached by human genes', len({k for s in g2ko.values() for k in s}),
      'of which in vocab', len({k for s in g2ko.values() for k in s if k in vocab}))

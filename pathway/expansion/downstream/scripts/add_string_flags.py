"""Flag gene pairs whose KO pair is in the STRING-v12 experimental-channel KO background used by the best model
(kegg_dataset_cache/string/ko_background_exp150_full.tsv / exp400_full.tsv).
Outputs:
  data/coexpression_gtex/string_exp_flags.npy  uint8 [G,G]: 1 = KO pair in exp150 background, 2 = in exp400
  data/ligand_receptor/lr_pairs_gene_stringflags.tsv  (gene_a, gene_b, string_exp150, string_exp400)
Run: .venv/bin/python -I add_string_flags.py
"""
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())
import csv, collections
import numpy as np
P = _KH + '/pathway'
D = f'{P}/expansion/downstream/data'
bg = {}
for name, bit in (('exp150', 1), ('exp400', 2)):
    with open(f'{P}/kegg_dataset_cache/string/ko_background_{name}_full.tsv') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            k = tuple(sorted((r['ko_a'], r['ko_b'])))
            bg[k] = bg.get(k, 0) | bit
genes = list(csv.DictReader(open(f'{D}/coexpression_gtex/genes.tsv'), delimiter='\t'))
ko2g = collections.defaultdict(list)
for i, g in enumerate(genes):
    for k in g['kos_in_vocab'].split(';'):
        if k: ko2g[k].append(i)
G = len(genes); F = np.zeros((G, G), np.uint8)
for (a, b), bit in bg.items():
    ia, ib = ko2g.get(a), ko2g.get(b)
    if ia and ib:
        F[np.ix_(ia, ib)] |= bit; F[np.ix_(ib, ia)] |= bit
np.fill_diagonal(F, 0)
np.save(f'{D}/coexpression_gtex/string_exp_flags.npy', F)
iu = np.triu_indices(G, 1)
print('coexp pairs flagged exp150:', int((F[iu] & 1 > 0).sum()), 'exp400:', int((F[iu] & 2 > 0).sum()), 'of', len(iu[0]))

gm = {r['entrez']: r for r in csv.DictReader(open(f'{D}/common/gene_map.tsv'), delimiter='\t')}
rows = list(csv.DictReader(open(f'{D}/ligand_receptor/lr_pairs_gene.tsv'), delimiter='\t'))
cnt = collections.Counter()
with open(f'{D}/ligand_receptor/lr_pairs_gene_stringflags.tsv', 'w') as f:
    f.write('gene_a\tgene_b\tstring_exp150\tstring_exp400\n')
    for r in rows:
        ka = [k for k in gm[r['gene_a']]['kos_in_vocab'].split(';') if k] if r['gene_a'] in gm else []
        kb = [k for k in gm[r['gene_b']]['kos_in_vocab'].split(';') if k] if r['gene_b'] in gm else []
        b = 0
        for x in ka:
            for y in kb: b |= bg.get(tuple(sorted((x, y))), 0)
        f.write(f"{r['gene_a']}\t{r['gene_b']}\t{int(b & 1 > 0)}\t{int(b & 2 > 0)}\n")
        cnt['exp150'] += b & 1 > 0; cnt['exp400'] += b & 2 > 0
print('LR pairs flagged', dict(cnt), 'of', len(rows))

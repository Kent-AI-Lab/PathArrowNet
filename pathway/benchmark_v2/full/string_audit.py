#!/usr/bin/env python3
"""Audit of the STRING background against the strict KEGG split (KO level, unordered pairs).
For train / val / test KO edges and for control pairs: share covered by each STRING channel (max over gene pairs),
experimental score distribution, and coverage by KGML relation subtype (physical vs transcriptional)."""
import csv, gzip, random, sys
from collections import defaultdict, Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "kegg_dataset_cache/string"; DS = ROOT / "kegg_dataset"
e2g = {}
with gzip.open(S / "9606.protein.aliases.v12.0.txt.gz", "rt") as f:
    next(f)
    for l in f:
        p, a, src = l.rstrip("\n").split("\t")
        if src == "KEGG_GENEID" and a.isdigit(): e2g.setdefault(p, a)
g2k = defaultdict(set)
for l in (ROOT / "kegg_dataset_cache/kegg_rest/link_ko_hsa.tsv").read_text().splitlines():
    x, y = l.split(); g2k[x[4:]].add(y[3:])
kos = {r["token"] for r in csv.DictReader(open(DS / "token_vocab_ko_full.csv")) if not r["token"].startswith("[")}
CH = ["neighborhood", "fusion", "cooccurence", "coexpression", "experimental", "database", "textmining", "combined_score"]
best = defaultdict(lambda: [0] * len(CH))
with gzip.open(S / "9606.protein.links.detailed.v12.0.txt.gz", "rt") as f:
    hdr = next(f).split()
    idx = [hdr.index(c) for c in CH]
    for l in f:
        c = l.split()
        g1, g2 = e2g.get(c[0]), e2g.get(c[1])
        if not g1 or not g2: continue
        for k1 in g2k.get(g1, ()):
            for k2 in g2k.get(g2, ()):
                if k1 == k2 or k1 not in kos or k2 not in kos: continue
                b = best[tuple(sorted((k1, k2)))]
                for j, i in enumerate(idx):
                    v = int(c[i])
                    if v > b[j]: b[j] = v
pairs = {}
for s in ("train", "val", "test"):
    with gzip.open(DS / f"splits_ko/{s}_A_next_node.csv.gz", "rt") as f:
        pairs[s] = {tuple(sorted((r["context_token"][3:], r["target_token"][3:]))) for r in csv.DictReader(f)}
# control: random KO pairs where both KOs are nodes of the same test pathway but not a KEGG edge anywhere
allpairs = set()
with open(DS / "splits_ko/edge_split.csv") as f:
    for r in csv.DictReader(f): allpairs.add(tuple(sorted((r["source_ko"], r["target_ko"]))))
sys.path.insert(0, str(ROOT))
from build_ko_leakage_free_split import load_ko_graphs
G = load_ko_graphs()
testpw = {r["pathway_id"] for r in csv.DictReader(open(DS / "splits_ko/pathway_split.csv")) if r["split"] == "test"}
rng = random.Random(0); ctrl = set()
for p in sorted(testpw):
    nodes = sorted({x for e in G.get(p, ()) for x in e})
    for _ in range(60):
        if len(nodes) < 2: break
        a, b = rng.sample(nodes, 2); k = tuple(sorted((a, b)))
        if k not in allpairs: ctrl.add(k)
vocab_list = sorted(kos); rand = set()
while len(rand) < 5000:
    a, b = rng.sample(vocab_list, 2); k = tuple(sorted((a, b)))
    if k not in allpairs: rand.add(k)
def row(name, P):
    out = [f"{name:38s} n={len(P):6d}"]
    for j, ch in enumerate(CH):
        th = 150 if ch != "combined_score" else 400
        out.append(f"{ch[:6]}≥{th}: {sum(best[k][j] >= th for k in P if k in best) / len(P):5.1%}")
    print(" ".join(out))
print("Share of KO pairs with channel score >= 150 (combined >= 400):")
for s in ("train", "val", "test"): row(f"{s} KEGG edges", pairs[s])
row("test-pathway non-edge KO pairs (control)", ctrl)
row("random KO pairs (control)", rand)
ex = [best[k][CH.index("experimental")] for k in pairs["test"] if k in best and best[k][CH.index("experimental")] >= 150]
print("\ntest edges with exp>=150: score bins", Counter(min(v // 100 * 100, 900) for v in ex))
# among test edges covered by experiments, how many are ALSO in the database channel
j_e, j_d = CH.index("experimental"), CH.index("database")
cov = [k for k in pairs["test"] if k in best and best[k][j_e] >= 150]
print(f"test edges exp>=150: {len(cov)}; of these database>0: {sum(best[k][j_d] > 0 for k in cov)}")
# by KGML subtype (KO edge -> subtypes from gene_edges)
sub = defaultdict(set)
for r in csv.DictReader(open(DS / "gene_edges.csv")):
    if r["edge_origin"] != "gene_entry": continue
    for a in g2k.get(r["source_gene"], ()):
        for b in g2k.get(r["target_gene"], ()):
            if a != b:
                for t in (r.get("relation_subtype") or "").split(";"):
                    if t: sub[tuple(sorted((a, b)))].add(t)
print("\nexperimental >= 150 coverage of ALL KEGG edges (train+val+test) by KGML subtype:")
allk = pairs["train"] | pairs["val"] | pairs["test"]
by = defaultdict(list)
for k in allk:
    for t in sub.get(k, {"?"}): by[t].append(k)
for t, P in sorted(by.items(), key=lambda x: -len(x[1])):
    if len(P) >= 100:
        print(f"  {t:22s} n={len(P):6d} exp>=150 {sum(best[k][j_e] >= 150 for k in P if k in best) / len(P):5.1%}  exp>=400 {sum(best[k][j_e] >= 400 for k in P if k in best) / len(P):5.1%}")

# ---- direct vs orthology-transferred experiments (protein.links.full) ----
full = defaultdict(lambda: [0, 0, 0])   # direct, transferred, detailed-experimental (check)
chk = []
with gzip.open(S / "9606.protein.links.full.v12.0.txt.gz", "rt") as f:
    hdr = next(f).split(); i_d, i_t = hdr.index("experiments"), hdr.index("experiments_transferred")
    for l in f:
        c = l.split()
        d, t = int(c[i_d]), int(c[i_t])
        if d == 0 and t == 0: continue
        g1, g2 = e2g.get(c[0]), e2g.get(c[1])
        if not g1 or not g2: continue
        for k1 in g2k.get(g1, ()):
            for k2 in g2k.get(g2, ()):
                if k1 == k2 or k1 not in kos or k2 not in kos: continue
                b = full[tuple(sorted((k1, k2)))]
                b[0] = max(b[0], d); b[1] = max(b[1], t)
j_e = CH.index("experimental")
print("\nexperimental >= 150 split by evidence origin (KO pair max):")
for name, P in (("train", pairs["train"]), ("val", pairs["val"]), ("test", pairs["test"]), ("test-pw non-edge", ctrl), ("random", rand)):
    d = sum(full[k][0] >= 150 for k in P if k in full) / len(P)
    t = sum(full[k][1] >= 150 for k in P if k in full) / len(P)
    to = sum(full[k][1] >= 150 and full[k][0] < 150 for k in P if k in full) / len(P)
    print(f"  {name:18s} direct human >=150 {d:5.1%} | transferred >=150 {t:5.1%} | transferred only {to:5.1%}")
print("\nKEGG edges by KGML subtype: share with experimental >= 150 / >= 400 (detailed file):")
allk = pairs["train"] | pairs["val"] | pairs["test"]
by = defaultdict(list)
for k in allk:
    for t in sub.get(k, {"?"}): by[t].append(k)
for t, P in sorted(by.items(), key=lambda x: -len(x[1])):
    if len(P) >= 100:
        print(f"  {t:20s} n={len(P):6d} exp>=150 {sum(best[k][j_e] >= 150 for k in P if k in best) / len(P):5.1%}  exp>=400 {sum(best[k][j_e] >= 400 for k in P if k in best) / len(P):5.1%}")

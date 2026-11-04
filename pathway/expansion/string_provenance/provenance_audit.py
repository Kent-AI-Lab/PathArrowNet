#!/usr/bin/env python3
"""STRING experimental-evidence provenance audit of the KO background graph (CPU only, numpy only).

Rebuilds the direct / transferred split of the experimental channel at KO level with the SAME 11,620-KO vocabulary
(token_vocab_ko_full.csv) as the exp150 background graph. Writes everything under expansion/string_provenance/:
  ko_pair_experimental_provenance_fullvocab.tsv  corrected provenance table (same columns as the cache file)
  tables/*.csv, audit_results.json, PROVENANCE_AUDIT.md, t21_string_provenance.tex
"""
import csv, gzip, json, random, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "kegg_dataset_cache/string"; DS = ROOT / "kegg_dataset"
OUT = ROOT / "expansion/string_provenance"; TAB = OUT / "tables"; TAB.mkdir(exist_ok=True)
PROV = OUT / "ko_pair_experimental_provenance_fullvocab.tsv"
SPLITS = {"strict_v1_full": DS / "splits_ko", **{f"strict_s{s}_full": ROOT / "kegg_splits_robustness" / f"seed{s}" for s in (43, 44, 45, 46)}}
TH = (150, 400, 700, 900); BINS = ((150, 399), (400, 699), (700, 899), (900, 1000))
R = {}  # json results
csv_tables = {}


def log(*a):
    print(*a, flush=True)


def write_table(name, header, rows):
    csv_tables[name] = (header, rows)
    with (TAB / f"{name}.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


# ---------------------------------------------------------------- mappings
ensp2gene = {}
with gzip.open(S / "9606.protein.aliases.v12.0.txt.gz", "rt") as f:
    next(f)
    for line in f:
        p, alias, src = line.rstrip("\n").split("\t")
        if src == "KEGG_GENEID" and alias.isdigit():
            ensp2gene.setdefault(p, alias)
g2k = defaultdict(set); k2g = defaultdict(set)
for line in (ROOT / "kegg_dataset_cache/kegg_rest/link_ko_hsa.tsv").read_text().splitlines():
    x, y = line.split(); g2k[x[4:]].add(y[3:]); k2g[y[3:]].add(x[4:])
vocab_list = [r["token"] for r in csv.DictReader((DS / "token_vocab_ko_full.csv").open()) if not r["token"].startswith("[")]
vocab = set(vocab_list); assert len(vocab) == 11620, len(vocab)
small_vocab = {r["token"] for r in csv.DictReader((DS / "token_vocab_ko.csv").open()) if not r["token"].startswith("[")}
members = {k: len(k2g.get(k, ())) for k in vocab_list}
ex = {}  # exp150 graph: pair -> (score, n_gene_pairs)
with (S / "ko_background_exp150_full.tsv").open() as f:
    for r in csv.DictReader(f, delimiter="\t"):
        ex[(r["ko_a"], r["ko_b"])] = (int(r["max_experimental_score"]), int(r["n_gene_pairs"]))
log(f"exp150 graph: {len(ex):,} pairs, {len({k for p in ex for k in p}):,} KOs")

# ---------------------------------------------------------------- provenance (full vocab)
P = {}
if PROV.exists() and "--rebuild" not in sys.argv:
    with PROV.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            P[(r["ko_a"], r["ko_b"])] = [int(r["max_exp_direct"]), int(r["max_exp_transferred"]), int(r["n_gene_pairs_direct"]),
                                         int(r["n_gene_pairs_transferred"]), int(r["n_gene_pairs_any"])]
    log(f"loaded provenance {len(P):,} pairs")
else:
    P = defaultdict(lambda: [0, 0, 0, 0, 0]); n_links = n_exp = 0; prot_dir = prot_tr = prot_both = 0
    with gzip.open(S / "9606.protein.links.full.v12.0.txt.gz", "rt") as f:
        hdr = next(f).split(); i_d, i_t = hdr.index("experiments"), hdr.index("experiments_transferred")
        for line in f:
            c = line.split(); n_links += 1
            d, t = int(c[i_d]), int(c[i_t])
            if d == 0 and t == 0:
                continue
            n_exp += 1; prot_dir += d > 0 and t == 0; prot_tr += d == 0 and t > 0; prot_both += d > 0 and t > 0
            g1, g2 = ensp2gene.get(c[0]), ensp2gene.get(c[1])
            if not g1 or not g2:
                continue
            for k1 in g2k.get(g1, ()):
                for k2 in g2k.get(g2, ()):
                    if k1 == k2 or k1 not in vocab or k2 not in vocab:
                        continue
                    r = P[tuple(sorted((k1, k2)))]
                    r[0] = max(r[0], d); r[1] = max(r[1], t); r[2] += d > 0; r[3] += t > 0; r[4] += 1
    P = dict(P)
    R["protein_links"] = {"directed_rows": n_links, "rows_with_experimental_evidence": n_exp, "direct_only": prot_dir,
                          "transferred_only": prot_tr, "both": prot_both}
    with PROV.open("w") as f:
        f.write("ko_a\tko_b\tmax_exp_direct\tmax_exp_transferred\tn_gene_pairs_direct\tn_gene_pairs_transferred\tn_gene_pairs_any\tmax_experimental_score\n")
        for k in sorted(set(P) | set(ex)):
            r = P.get(k, [0, 0, 0, 0, 0]); f.write(f"{k[0]}\t{k[1]}\t{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\t{ex.get(k, (-1,))[0]}\n")
    log(f"rows {n_links:,}; with exp evidence {n_exp:,}; KO pairs with evidence {len(P):,}; wrote {PROV}")
Z = [0, 0, 0, 0, 0]
def prov(k): return P.get(k, Z)

# ---------------------------------------------------------------- 1. composition
log("== 1. composition")
n_dir_any = sum(prov(k)[0] > 0 for k in ex); n_tr_any = sum(prov(k)[1] > 0 for k in ex)
n_none = sum(prov(k)[0] == 0 and prov(k)[1] == 0 for k in ex)
d150_not_in = sum(1 for k, r in P.items() if r[0] >= 150 and k not in ex)
comb_only = sum(1 for k in ex if prov(k)[0] < 150 and prov(k)[1] < 150)
R["composition_overall"] = {"exp150_pairs": len(ex), "direct_gt0": n_dir_any, "transferred_gt0": n_tr_any, "neither_gt0": n_none,
                            "direct_ge150": sum(prov(k)[0] >= 150 for k in ex), "transferred_ge150": sum(prov(k)[1] >= 150 for k in ex),
                            "neither_ge150_but_combined_ge150": comb_only, "direct_ge150_pairs_not_in_exp150_graph": d150_not_in,
                            "ko_pairs_with_any_experimental_evidence": len(P),
                            "cache_file_note": "kegg_dataset_cache/string/ko_pair_experimental_provenance.tsv was built with token_vocab_ko.csv (2,871 KOs); "
                                               "159,588 exp150 pairs had direct=transferred=0 there. This audit uses the 11,620-KO vocabulary."}
rows = []
for T in TH:
    E = [k for k, v in ex.items() if v[0] >= T]
    d = sum(prov(k)[0] >= T and prov(k)[1] < T for k in E); t = sum(prov(k)[1] >= T and prov(k)[0] < T for k in E)
    b = sum(prov(k)[0] >= T and prov(k)[1] >= T for k in E); n = len(E) - d - t - b
    rows.append([T, len(E), d, t, b, n, round(d / len(E), 4), round(t / len(E), 4), round(b / len(E), 4), round(n / len(E), 4), round((d + b) / len(E), 4)])
write_table("t1_composition_by_threshold", ["threshold", "pairs_combined_ge_T", "direct_only_ge_T", "transferred_only_ge_T", "both_ge_T",
                                            "neither_ge_T", "frac_direct_only", "frac_transferred_only", "frac_both", "frac_neither", "frac_direct_ge_T"], rows)
rows = []
for lo, hi in BINS:
    E = [k for k, v in ex.items() if lo <= v[0] <= hi]
    d150 = sum(prov(k)[0] >= 150 for k in E); dlo = sum(prov(k)[0] >= lo for k in E); dgt0 = sum(prov(k)[0] > 0 for k in E)
    tlo = sum(prov(k)[1] >= lo for k in E); none = sum(prov(k)[0] == 0 and prov(k)[1] == 0 for k in E)
    rows.append([f"{lo}-{hi}" if hi < 1000 else f">={lo}", len(E), d150, round(d150 / len(E), 4), dlo, round(dlo / len(E), 4), dgt0, round(dgt0 / len(E), 4),
                 tlo, round(tlo / len(E), 4), none])
write_table("t1b_composition_by_bin", ["bin", "pairs", "direct_ge150", "frac_direct_ge150", "direct_ge_bin_low", "frac_direct_ge_bin_low", "direct_gt0",
                                       "frac_direct_gt0", "transferred_ge_bin_low", "frac_transferred_ge_bin_low", "no_evidence_recorded"], rows)

# ---------------------------------------------------------------- 2. coverage
log("== 2. coverage")
graphs = defaultdict(set)  # pathway -> KO edges (like build_ko_leakage_free_split.load_ko_graphs)
with (DS / "gene_edges.csv").open() as f:
    for r in csv.DictReader(f):
        if r["edge_origin"] != "gene_entry":
            continue
        for a in g2k.get(r["source_gene"], ()):
            for b in g2k.get(r["target_gene"], ()):
                if a != b:
                    graphs[r["pathway_id"]].add((a, b))
allpairs = set()
with (DS / "splits_ko/edge_split.csv").open() as f:
    for r in csv.DictReader(f):
        allpairs.add(tuple(sorted((r["source_ko"], r["target_ko"]))))
rng = random.Random(0); rand = set()
while len(rand) < 20000:
    a, b = rng.sample(vocab_list, 2); k = tuple(sorted((a, b)))
    if k not in allpairs:
        rand.add(k)
rand_h = {k for k in rand if members[k[0]] > 0 and members[k[1]] > 0}


def cov(Pset):
    n = len(Pset); a = sum(k in ex for k in Pset); d = sum(k in ex and prov(k)[0] >= 150 for k in Pset)
    t = a - d; dg = sum(prov(k)[0] > 0 for k in Pset); d_in_any = sum(prov(k)[0] >= 150 for k in Pset)
    return n, a / n, d / n, t / n, dg / n, (d / a if a else float("nan"))


COLS = ["n_pairs", "any_exp150", "direct_ge150", "transferred_only", "direct_gt0", "direct_share_among_covered"]
rows = []; per = defaultdict(lambda: defaultdict(list)); split_sets = {}
for setting, d in SPLITS.items():
    sets = {}
    for s in ("train", "val", "test"):
        with gzip.open(d / f"{s}_A_next_node.csv.gz", "rt") as f:
            sets[s] = {tuple(sorted((r["context_token"][3:], r["target_token"][3:]))) for r in csv.DictReader(f)}
    testpw = {r["pathway_id"] for r in csv.DictReader((d / "pathway_split.csv").open()) if r["split"] == "test"}
    ctrl = set(); ctrl60 = set(); rng = random.Random(0)
    for p in sorted(testpw):
        nodes = sorted({x for e in graphs.get(p, ()) for x in e})
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if (nodes[i], nodes[j]) not in allpairs:
                    ctrl.add((nodes[i], nodes[j]))
        for _ in range(60):  # sampled version, as in benchmark_v2/full/string_audit.py
            if len(nodes) < 2: break
            a, b = rng.sample(nodes, 2); k = tuple(sorted((a, b)))
            if k not in allpairs: ctrl60.add(k)
    sets["test_map_nonedge_all"] = ctrl; sets["test_map_nonedge_sampled60"] = ctrl60; sets["random_vocab"] = rand; sets["random_vocab_both_human"] = rand_h
    split_sets[setting] = sets
    for name, Pset in sets.items():
        c = cov(Pset); rows.append([setting, name, c[0]] + [round(x, 4) for x in c[1:]])
        for col, x in zip(COLS, c): per[name][col].append(x)
for name in per:
    rows.append(["mean_5_splits", name] + [round(float(np.nanmean(per[name][c])), 4) for c in COLS])
write_table("t2_coverage_by_split", ["split", "pair_set"] + COLS, rows)
R["coverage_mean"] = {name: {c: float(np.nanmean(per[name][c])) for c in COLS} for name in per}

# score-bin distribution of covered pairs, main split test vs controls
rows = []
for name in ("train", "val", "test", "test_map_nonedge_all", "random_vocab"):
    Pset = split_sets["strict_v1_full"][name]; inE = [k for k in Pset if k in ex]
    for lo, hi in BINS:
        E = [k for k in inE if lo <= ex[k][0] <= hi]; d = sum(prov(k)[0] >= 150 for k in E)
        rows.append([name, f"{lo}-{hi}" if hi < 1000 else f">={lo}", len(E), round(len(E) / len(Pset), 4), d, round(d / len(E), 4) if E else ""])
write_table("t2b_main_split_bins", ["pair_set", "bin", "pairs_in_bin", "frac_of_set", "direct_ge150", "frac_direct_ge150_in_bin"], rows)

# ---------------------------------------------------------------- 3. KO size bias
log("== 3. KO size")
deg_e = Counter(); deg_d = Counter(); nb = defaultdict(set)
for (a, b), v in ex.items():
    deg_e[a] += 1; deg_e[b] += 1; nb[a].add(b); nb[b].add(a)
for k, r in P.items():
    if r[0] >= 150:
        deg_d[k[0]] += 1; deg_d[k[1]] += 1


def rank(x):
    x = np.asarray(x, float); u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    start = np.cumsum(cnt) - cnt + 1; return (start + (cnt - 1) / 2)[inv]


def spearman(x, y):
    rx, ry = rank(x), rank(y); return float(np.corrcoef(rx, ry)[0, 1])


m = np.array([members[k] for k in vocab_list]); de = np.array([deg_e[k] for k in vocab_list]); dd = np.array([deg_d[k] for k in vocab_list])
h = m > 0
R["ko_size"] = {"n_vocab": len(vocab_list), "n_with_human_genes": int(h.sum()),
                "spearman_members_vs_deg_exp150_all": spearman(m, de), "spearman_members_vs_deg_direct150_all": spearman(m, dd),
                "spearman_members_vs_deg_exp150_humanKOs": spearman(m[h], de[h]), "spearman_members_vs_deg_direct150_humanKOs": spearman(m[h], dd[h]),
                "spearman_members_vs_deg_exp150_deg_gt0": spearman(m[de > 0], de[de > 0])}
rows = []
for lab, lo, hi in (("0", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4-9", 4, 9), (">=10", 10, 10 ** 9)):
    sel = (m >= lo) & (m <= hi)
    if sel.sum() == 0: continue
    q = lambda v: [int(np.percentile(v, p)) for p in (25, 50, 75)]
    rows.append([lab, int(sel.sum()), round(float((de[sel] > 0).mean()), 3)] + q(de[sel]) + [round(float(de[sel].mean()), 1), round(float((dd[sel] > 0).mean()), 3)]
                + q(dd[sel]) + [round(float(dd[sel].mean()), 1)])
write_table("t3_degree_by_member_bin", ["members_bin", "n_kos", "frac_deg_exp150_gt0", "deg_exp150_q25", "deg_exp150_q50", "deg_exp150_q75", "deg_exp150_mean",
                                        "frac_deg_direct150_gt0", "deg_direct150_q25", "deg_direct150_q50", "deg_direct150_q75", "deg_direct150_mean"], rows)

# test queries (main split): member count of true targets vs candidates
with gzip.open(DS / "splits_ko/test_A_next_node.csv.gz", "rt") as f:
    Q = [(r["context_token"][3:], r["target_token"][3:]) for r in csv.DictReader(f)]
tm = np.array([members[t] for _, t in Q]); cm = m
nbm = np.array([members[x] for c, _ in Q for x in nb.get(c, ())])
nbm_direct = np.array([members[x] for c, _ in Q for x in nb.get(c, ()) if prov(tuple(sorted((c, x))))[0] >= 150])
# per query: P(members(target) > members(random candidate)) with 0.5 for ties (AUC-like)
srt = np.sort(cm)
auc = float(np.mean([(np.searchsorted(srt, v, "left") + 0.5 * (np.searchsorted(srt, v, "right") - np.searchsorted(srt, v, "left"))) / len(srt) for v in tm]))
auc_h = float(np.mean([(np.searchsorted(np.sort(cm[h]), v, "left") + 0.5 * (np.searchsorted(np.sort(cm[h]), v, "right") - np.searchsorted(np.sort(cm[h]), v, "left"))) / h.sum() for v in tm]))
per_q_nb = [np.median([members[x] for x in nb.get(c, ())]) for c, _ in Q if nb.get(c)]
def desc(v):
    v = np.asarray(v); return [len(v), int(np.percentile(v, 25)), float(np.median(v)), int(np.percentile(v, 75)), round(float(v.mean()), 2), round(float((v >= 10).mean()), 3), round(float((v == 1).mean()), 3)]
rows = [["test targets (strict split 1 queries)"] + desc(tm), ["all 11,620 candidates"] + desc(cm), ["candidates with human genes"] + desc(cm[h]),
        ["candidates with an exp150 edge to the query context (pooled over queries)"] + desc(nbm),
        ["candidates with a direct>=150 edge to the query context (pooled)"] + desc(nbm_direct),
        ["per-query median members of exp150 neighbours of the context"] + desc(per_q_nb)]
write_table("t3b_target_member_count", ["set", "n", "q25", "median", "q75", "mean", "frac_ge10", "frac_eq1"], rows)
R["ko_size"].update({"auc_target_members_vs_all_candidates": auc, "auc_target_members_vs_human_candidates": auc_h,
                     "n_test_queries": len(Q), "n_queries_with_exp150_neighbours": sum(1 for c, _ in Q if nb.get(c))})
# supporting gene pairs: test edges vs matched non-edges
rows = []
for setting in SPLITS:
    sets = split_sets[setting]
    for name in ("test", "test_map_nonedge_all", "random_vocab"):
        Pset = sets[name]; inE = [k for k in Pset if k in ex]
        ng = np.array([prov(k)[4] for k in inE]); ngd = np.array([prov(k)[2] for k in inE])
        dens = np.array([prov(k)[4] / (members[k[0]] * members[k[1]]) for k in inE])
        ng_all = np.array([prov(k)[4] for k in Pset])
        prod = np.array([members[k[0]] * members[k[1]] for k in inE])
        rows.append([setting, name, len(Pset), len(inE), float(np.median(ng)), round(float(ng.mean()), 2), float(np.median(ngd)), round(float(ngd.mean()), 2),
                     float(np.median(prod)), round(float(np.median(dens)), 3), round(float(dens.mean()), 3), float(np.median(ng_all)), round(float(ng_all.mean()), 2)])
write_table("t3c_gene_pairs_edges_vs_nonedges", ["split", "pair_set", "n_pairs", "n_in_exp150", "ngp_any_median_in_exp150", "ngp_any_mean_in_exp150",
                                                 "ngp_direct_median_in_exp150", "ngp_direct_mean_in_exp150", "members_product_median_in_exp150",
                                                 "ngp_per_member_pair_median", "ngp_per_member_pair_mean", "ngp_any_median_all_pairs", "ngp_any_mean_all_pairs"], rows)
# AUC-type: P(ngp(test edge) > ngp(non-edge)) among exp150-covered pairs, main split
te = np.array([prov(k)[4] for k in split_sets["strict_v1_full"]["test"] if k in ex]); ne = np.sort([prov(k)[4] for k in split_sets["strict_v1_full"]["test_map_nonedge_all"] if k in ex])
R["ko_size"]["auc_ngp_test_edge_vs_testmap_nonedge_in_exp150"] = float(np.mean([(np.searchsorted(ne, v, "left") + 0.5 * (np.searchsorted(ne, v, "right") - np.searchsorted(ne, v, "left"))) / len(ne) for v in te]))
sc_te = np.array([ex[k][0] for k in split_sets["strict_v1_full"]["test"] if k in ex]); sc_ne = np.sort([ex[k][0] for k in split_sets["strict_v1_full"]["test_map_nonedge_all"] if k in ex])
R["ko_size"]["auc_score_test_edge_vs_testmap_nonedge_in_exp150"] = float(np.mean([(np.searchsorted(sc_ne, v, "left") + 0.5 * (np.searchsorted(sc_ne, v, "right") - np.searchsorted(sc_ne, v, "left"))) / len(sc_ne) for v in sc_te]))
R["tables"] = {k: {"header": h_, "rows": r_} for k, (h_, r_) in csv_tables.items()}
(OUT / "audit_results.json").write_text(json.dumps(R, indent=1, default=float))
for name, (hdr, rws) in csv_tables.items():
    log(f"\n### {name}\n| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for r in rws: log("| " + " | ".join(str(x) for x in r) + " |")
log(json.dumps({k: v for k, v in R.items() if k != "tables"}, indent=1, default=float))

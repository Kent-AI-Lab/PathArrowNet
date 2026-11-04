#!/usr/bin/env python3
"""
Dataset statistics for the KO next-node benchmark (dataset v1, strict split v1).

    .venv/bin/python benchmark_v2/dataset_stats.py

Reads only existing files (kegg_dataset/*, kegg_dataset/splits_ko/*); writes benchmark_v2/out/:
  dataset_summary.csv           one row per statistic
  pathway_length_table.csv      one row per pathway (KO nodes, edges, branches, reciprocal pairs, depth, split, samples)
  pathway_length_bins.csv       length-bin counts (all usable pathways and per split)
  task_structure.csv            one-to-many / branching / reciprocal statistics per split and task

"Pathway length" = number of distinct KO nodes that take part in >= 1 directed KO edge of the pathway
(the nodes a next-node model can actually be asked about). Depth = number of nodes on the longest path of
the pathway's SCC-condensed DAG (a cycle counts as one node), a proxy for the longest linear cascade.
"""

from __future__ import annotations

import re
import csv
import gzip
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from build_ko_leakage_free_split import load_ko_graphs  # noqa: E402

DS = ROOT / "kegg_dataset"
SPLIT = DS / "splits_ko"
OUT = Path(__file__).resolve().parent / "out"
SPLITS = ("train", "val", "test")
BINS = [("0 (no directed KO edge)", 0, 0), ("1-4", 1, 4), ("5-10", 5, 10), ("11-20", 11, 20),
        ("21-50", 21, 50), ("51-100", 51, 100), (">100", 101, 10 ** 9)]


def rd(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wr(p, rows):
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pw_all = [l.split("\t") for l in (ROOT / "kegg_feasibility_cache/pathway_list.tsv").read_text().splitlines() if l]
    names = {p: n.replace(" - Homo sapiens (human)", "") for p, n in pw_all}
    gstats = {r["pathway_id"]: r for r in rd(DS / "gene_graph_stats_by_pathway.csv")}
    psplit = {r["pathway_id"]: r for r in rd(SPLIT / "pathway_split.csv")}
    spp = {r["pathway_id"]: r for r in rd(DS / "samples_per_pathway.csv")}
    graphs = load_ko_graphs()
    rel = rd(DS / "kgml_relations.csv")

    rows = []
    for p, _ in pw_all:
        es = graphs.get(p, set())
        g = nx.DiGraph()
        g.add_edges_from(es)
        n = g.number_of_nodes()
        recip = sum(1 for u, v in es if (v, u) in es) // 2
        depth = 0
        if n:
            c = nx.condensation(g)
            depth = nx.dag_longest_path_length(c) + 1
        rows.append({
            "pathway_id": p, "pathway_name": names[p], "class": gstats[p]["pathway_class_a"],
            "all_entries_genes": gstats[p]["gene_nodes"],
            "ko_nodes": n, "ko_edges": len(es),
            "branch_nodes_outdeg_ge2": sum(1 for x in g if g.out_degree(x) >= 2),
            "merge_nodes_indeg_ge2": sum(1 for x in g if g.in_degree(x) >= 2),
            "reciprocal_pairs": recip, "depth_condensed_longest_path": depth,
            "split": psplit[p]["split"] if p in psplit else "unusable",
            "ko_edges_owned_by_own_split": psplit[p]["ko_edges_owned_by_own_split"] if p in psplit else 0,
            "windows_L3_6_full_graph": sum(int(spp[p][f"ko_B_windows_L{L}"]) for L in (3, 4, 5, 6)),
            "masked_samples_full_graph": sum(int(spp[p][f"ko_C_masked_L{L}"]) for L in (3, 4, 5, 6)),
        })
    wr(OUT / "pathway_length_table.csv", rows)

    usable = [r for r in rows if r["ko_edges"] > 0]
    L = [r["ko_nodes"] for r in usable]
    summ = []

    def add(k, v, note=""):
        summ.append({"statistic": k, "value": v, "note": note})
    add("original KEGG human pathways (hsa)", len(rows), "kegg_feasibility_cache/pathway_list.tsv")
    add("pathways with >=1 KGML relation", len({r['pathway_id'] for r in rel}))
    add("usable pathways (>=1 directed KO->KO edge)", len(usable),
        "metabolic maps only have ECrel/compound (non-directional) relations and give no KO->KO edge")
    add("pathways in strict split (train/val/test)", "/".join(str(sum(1 for r in usable if r['split'] == s)) for s in SPLITS))
    for k, f in (("min", min), ("median", statistics.median), ("mean", statistics.mean), ("max", max)):
        add(f"pathway length (KO nodes) {k}", round(f(L), 2))
    add("pathway length (KO nodes) quartiles Q1/Q3", f"{statistics.quantiles(L, n=4)[0]}/{statistics.quantiles(L, n=4)[2]}")
    for lab, lo, hi in BINS[1:]:
        add(f"usable pathways with {lab} KO nodes", sum(1 for x in L if lo <= x <= hi))
    for t in (10, 20, 50, 100):
        add(f"usable pathways with >{t} KO nodes", sum(1 for x in L if x > t))
    E = [r["ko_edges"] for r in usable]
    add("pathway directed KO edges min/median/mean/max", f"{min(E)}/{statistics.median(E)}/{statistics.mean(E):.1f}/{max(E)}")
    D = [r["depth_condensed_longest_path"] for r in usable]
    add("longest cascade (condensed DAG path, nodes) min/median/mean/max",
        f"{min(D)}/{statistics.median(D)}/{statistics.mean(D):.1f}/{max(D)}")
    vocab = rd(DS / "token_vocab_ko.csv")
    kos = {e for es in graphs.values() for x in es for e in x}
    add("unique KO nodes in directed KO graph (= candidate vocabulary)", len(kos),
        f"token_vocab_ko.csv has {len(vocab) - 3} KO tokens + 3 special")
    ko_any = set()
    for r in rd(DS / "node_occurrences.csv"):
        ko_any |= {k for k in re.split(r"[;\s]+", r["ko_ids"] + ";" + r["derived_ko_ids_from_hsa_genes"]) if k}
    ko_any = {k.replace("ko:", "") for k in ko_any}
    add("unique KOs appearing anywhere in the 372 human pathway maps", len(ko_any),
        "includes metabolic enzymes without directed relations; these are never a next-node target")
    union = set().union(*graphs.values())
    add("unique directed KO edges (union over pathways)", len(union))
    add("directed KO edge occurrences (sum over pathways)", sum(len(es) for es in graphs.values()))
    add("edges occurring in >1 pathway (share of unique edges)",
        f"{sum(1 for e in union if sum(1 for es in graphs.values() if e in es) > 1)}")
    recip = {frozenset(e) for e in union if (e[1], e[0]) in union}
    add("reciprocal KO pairs (A->B and B->A both directed edges, union graph)", len(recip),
        f"= {2 * len(recip)} directed edges kept as two separate samples")
    add("reciprocal KO pairs within a single pathway (sum over pathways)", sum(r["reciprocal_pairs"] for r in usable))
    nondir = Counter()
    for r in rel:
        if r["whether_directional"] == "0":
            nondir[r["relation_type"]] += 1
    add("KGML relations total / directional / non-directional",
        f"{len(rel)}/{sum(1 for r in rel if r['whether_directional'] == '1')}/{sum(1 for r in rel if r['whether_directional'] == '0')}",
        "non-directional (binding/association, dissociation, compound, ECrel, missing interaction) are NOT turned into edges")
    gu = nx.DiGraph()
    gu.add_edges_from(union)
    add("branch KOs (out-degree>=2) in union graph", sum(1 for x in gu if gu.out_degree(x) >= 2))
    add("merge KOs (in-degree>=2) in union graph", sum(1 for x in gu if gu.in_degree(x) >= 2))
    add("branch nodes summed over pathways (out-degree>=2 within pathway)", sum(r["branch_nodes_outdeg_ge2"] for r in usable))
    add("out-degree in union graph median/mean/max",
        f"{statistics.median([gu.out_degree(x) for x in gu if gu.out_degree(x)])}/"
        f"{statistics.mean([gu.out_degree(x) for x in gu if gu.out_degree(x)]):.2f}/{max(d for _, d in gu.out_degree())}",
        "over KOs with out-degree >= 1")
    ss = {(r["split"], r["strategy"]): r for r in rd(SPLIT / "split_stats.csv")}
    for s in SPLITS:
        add(f"{s}: A next-node samples (u->v, unique)", ss[(s, "A_next_node")]["unique_samples"])
        b = ss[(s, "B_path_windows")]
        add(f"{s}: B path segments (simple paths L=3/4/5/6)", f"{b['unique_samples']} ({b['L3']}/{b['L4']}/{b['L5']}/{b['L6']})")
        add(f"{s}: C masked-node samples", ss[(s, "C_masked")]["unique_samples"])
    sfile = Path(__file__).resolve().parent / "splits" / "strict_v1"
    for s in SPLITS:
        p = sfile / f"{s}_S_prefix_next.csv.gz"
        if p.exists():
            rr = rd(p)
            add(f"{s}: S prefix->next samples (prefix length 2..5; length 1 = task A)", len(rr),
                "; ".join(f"len{k}={v}" for k, v in sorted(Counter(len(r['context_token_ids'].split()) for r in rr).items())))
    wr(OUT / "dataset_summary.csv", summ)

    bins = []
    for lab, lo, hi in BINS:
        row = {"ko_nodes_bin": lab, "all_372": sum(1 for r in rows if lo <= r["ko_nodes"] <= hi)}
        for s in SPLITS:
            row[s] = sum(1 for r in rows if r["split"] == s and lo <= r["ko_nodes"] <= hi)
        row["share_of_usable"] = round(sum(1 for x in L if lo <= x <= hi) / len(L), 3) if lo > 0 else ""
        row["edges_in_bin_share"] = round(sum(r["ko_edges"] for r in usable if lo <= r["ko_nodes"] <= hi) / sum(E), 3) if lo > 0 else ""
        bins.append(row)
    wr(OUT / "pathway_length_bins.csv", bins)
    for r in summ:
        print(f"{r['statistic']}: {r['value']}" + (f"   [{r['note']}]" if r["note"] else ""))
    print()
    for b in bins:
        print(b)


if __name__ == "__main__":
    main()

"""Final feasibility report (CSV + Markdown), README and terminal FINAL SUMMARY.

Every metric is labelled:
  observed   counted directly from a source record (KGML element, KEGG link/conv line,
             GOA/gene2go annotation line, KO flat-file field)
  derived    produced by a transformation we chose (cartesian gene expansion, graph
             projection, SCC analysis, sample construction, deduplication)
  estimated  extrapolated from random sampling (not exact)
"""

from __future__ import annotations

import csv
import datetime as dt
import json

from .common import CACHE, FAILURE_LOG, OUT, write_csv
from . import annotations as A


def pct(x):
    return f"{100 * x:.1f}%"


def fmt(v):
    if isinstance(v, float):
        if v.is_integer():
            return f"{int(v):,}"
        return f"{v:.4f}" if v < 1 else f"{v:,.1f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def failures_summary():
    if not FAILURE_LOG.exists():
        return 0, {}
    rows = list(csv.DictReader(FAILURE_LOG.open()))
    by = {}
    for r in rows:
        by[r["stage"]] = by.get(r["stage"], 0) + 1
    return len(rows), by

SPLIT_DIR = OUT / "splits_ko"


def load_split():
    """Summary of the KO leakage-controlled split (build_ko_leakage_free_split.py), or None."""
    need = ["split_stats.csv", "leakage_checks.csv", "split_config.csv", "pathway_split.csv"]
    if not all((SPLIT_DIR / f).exists() for f in need):
        return None
    rd = lambda f: list(csv.DictReader((SPLIT_DIR / f).open(encoding="utf-8")))
    return {"stats": {(r["split"], r["strategy"]): r for r in rd("split_stats.csv")},
            "checks": {r["check"]: int(r["value"]) for r in rd("leakage_checks.csv")},
            "config": {r["k"]: r["v"] for r in rd("split_config.csv")},
            "pathways": rd("pathway_split.csv")}


def split_section(sp):
    SPL = ("train", "val", "test")
    st, ck, cf = sp["stats"], sp["checks"], sp["config"]
    cls = sorted({r["pathway_class_a"] for r in sp["pathways"]})
    cnt = {(c, s): sum(1 for r in sp["pathways"] if r["pathway_class_a"] == c and r["split"] == s) for c in cls for s in SPL}
    owned = {s: ck[f"{s}_unique_edges"] for s in SPL}
    tot_owned = sum(owned.values())
    leak_rows = [k for k in ck if k.startswith(("overlap_", "edges_shared_"))]
    md = f"""## 10. KO-level leakage-controlled split

Built by `build_ko_leakage_free_split.py` (outputs in `splits_ko/`; details in `splits_ko/README.md`).
Parameters: seed={cf['seed']}, target owned-edge ratios={cf['ratios']}, min train fraction per KEGG class={cf['min_train_class_frac']},
pathway grouping threshold={cf['group_threshold']}, path cap per pathway per length=min({cf['max_paths_per_pathway']}, {cf['cap_per_edge']} x owned edges).

**Why not a random or plain pathway split.** Section 8 shows that 60% of edge occurrences repeat across
pathways; clustering pathways by KO-edge overlap (coefficient >= 0.5) chains 136 pathways holding ~79% of
edges into one component, so whole-cluster holdout is impossible.

**Scheme.**
1. Near-copy pathways (KO-edge overlap coefficient >= {cf['group_threshold']}) are grouped and assigned together.
2. Groups are moved from train to val/test greedily (seeded, round-robin over KEGG classes) to reach the target
   share of *owned* unique edges, while every KEGG class keeps >= {float(cf['min_train_class_frac']):.0%} of its pathways in train.
3. Each unique KO edge is owned by exactly one split: train if it occurs in any train pathway, else val, else test.
   Val/test edges whose reverse is owned by an earlier split are excluded from all splits.
4. Samples (A/B/C) are rebuilt per split from owned edges only and deduplicated within the split; val/test masked
   samples whose context was already used by an earlier split are removed (e.g. `A [MASK] C` has no visible edge).
5. All cross-split checks are hard assertions: the build aborts if any is non-zero.

### 10a. Assignment

| KEGG class | pathways | train | val | test | train fraction |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(f"| {c} | {sum(cnt[(c, s)] for s in SPL)} | {cnt[(c, 'train')]} | {cnt[(c, 'val')]} | {cnt[(c, 'test')]} | "
               f"{cnt[(c, 'train')] / max(1, sum(cnt[(c, s)] for s in SPL)):.0%} |" for c in cls) + f"""
| **total** | {len(sp['pathways'])} | {sum(cnt[(c, 'train')] for c in cls)} | {sum(cnt[(c, 'val')] for c in cls)} | {sum(cnt[(c, 'test')] for c in cls)} | |

| split | owned unique KO edges | share |
|---|---:|---:|
""" + "\n".join(f"| {s} | {owned[s]:,} | {owned[s] / tot_owned:.1%} |" for s in SPL) + f"""

Edges removed from val/test: {ck['edges_ceded_to_train_from_val_test_pathways']:,} edge occurrences shared with a train pathway (ceded to train);
{ck['val_edges_excluded_reverse']} val and {ck['test_edges_excluded_reverse']} test edges whose reverse belongs to an earlier split (excluded);
{ck['val_masked_dropped_context_seen_in_earlier_split']} val and {ck['test_masked_dropped_context_seen_in_earlier_split']} test masked samples whose context was used by an earlier split (dropped).

### 10b. Samples per split (deduplicated within split)

| split | strategy | unique samples | pathways w/ samples | median / pathway | max 1 pathway | top1 | top5 | top10 | Gini | target unseen in train | ambiguous contexts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {s} | {g} | {int(r['unique_samples']):,} | {r['pathways_with_samples']} | {fmt(float(r['median_per_pathway']))} | "
        f"{int(r['max_from_one_pathway']):,} ({r['max_pathway']}) | {pct(float(r['top1_share']))} | {pct(float(r['top5_share']))} | "
        f"{pct(float(r['top10_share']))} | {float(r['gini']):.3f} | {r['target_unseen_in_train'] or '-'} | "
        f"{pct(float(r['ambiguous_context_frac'])) if r['ambiguous_context_frac'] else '-'} |"
        for g in ("A_next_node", "B_path_windows", "C_masked") for s in SPL for r in [st[(s, g)]]) + """

Path windows per length: """ + "; ".join(
        f"{s}: " + ", ".join(f"L{L}={int(st[(s, 'B_path_windows')][f'L{L}']):,}" for L in (3, 4, 5, 6)) for s in SPL) + f"""

### 10c. Leakage checks (all must be 0; `splits_ko/leakage_checks.csv`)

| check | value |
|---|---:|
""" + "\n".join(f"| {k} | {ck[k]} |" for k in leak_rows) + f"""
| **all cross-split overlaps zero** | **{'yes' if ck['ALL_CROSS_SPLIT_OVERLAPS_ZERO'] else 'NO'}** |

### 10d. Remaining caveats (flagged, not leakage)

* Val/test are small ({st[('val', 'B_path_windows')]['pathways_with_samples']} / {st[('test', 'B_path_windows')]['pathways_with_samples']} pathways yield paths) and concentrated: the largest pathway gives
  {pct(float(st[('val', 'C_masked')]['top1_share']))} of val and {pct(float(st[('test', 'C_masked')]['top1_share']))} of test masked samples. Report pathway-macro averages
  (`pathway_macro_weight`, every pathway equal total weight) next to micro averages.
* Targets never seen in train (`target_seen_in_train=0`): {st[('val', 'C_masked')]['target_unseen_in_train']} val / {st[('test', 'C_masked')]['target_unseen_in_train']} test masked samples;
  a pure token model cannot predict them, report separately. Val/test KOs absent from train: {ck['val_tokens_unseen_in_train']} / {ck['test_tokens_unseen_in_train']}.
* Next-node contexts are single KOs, so one source KO has different correct targets in different splits (standard
  transductive link prediction). Use `known_targets_in_train` for filtered ranking and `valid_targets_in_split`
  for set-based metrics; exact-match accuracy is ill-posed ({pct(float(st[('test', 'A_next_node')]['ambiguous_context_frac']))} of test next-node contexts have >1 answer).
* Train is split-owned edges only: {ck['edges_ceded_to_train_from_val_test_pathways']:,} shared edge occurrences in val/test pathways appear in train through
  other pathways, so val/test measure generalisation to relations never drawn in any train pathway.
"""
    rows = [("split", "Pathways train / val / test",
             " / ".join(str(sum(cnt[(c, s)] for c in cls)) for s in SPL), "derived", "KO-level split, grouped pathway holdout"),
            ("split", "Owned unique KO edges train / val / test", " / ".join(f"{owned[s]:,}" for s in SPL), "derived", "unique-edge ownership"),
            ("split", "Val/test edges excluded as reverse of earlier split",
             f"{ck['val_edges_excluded_reverse']} / {ck['test_edges_excluded_reverse']}", "derived", "")]
    for g in ("A_next_node", "B_path_windows", "C_masked"):
        rows.append(("split", f"{g} unique samples train / val / test",
                     " / ".join(f"{int(st[(s, g)]['unique_samples']):,}" for s in SPL), "derived", "deduplicated within split"))
    rows.append(("split", "All cross-split leakage checks zero", "yes" if ck["ALL_CROSS_SPLIT_OVERLAPS_ZERO"] else "NO",
                 "derived", f"{len(leak_rows)} checks, hard-asserted"))
    return md, rows



def write_report(ctx):
    m = ctx.metrics
    kg, gg, sm, mt = m["kgml"], m["gene_graph"], m["sampling"], m["metabolic"]
    up, go, ko = m.get("uniprot", {}), m.get("go", {}), m.get("ko", {})
    g, k = sm["gene"], sm["ko"]
    nt = kg["node_occurrences_by_type"]
    L = [int(x) for x in ctx.args.window_lengths.split(",")]

    rows = []

    def add(section, metric, value, kind, note=""):
        rows.append({"section": section, "metric": metric, "value": value, "kind": kind, "definition_or_source": note})

    add("core", "Human pathways", kg["pathways_parsed"], "observed", "KEGG list/pathway/hsa; KGML parsed from v2 cache")
    add("core", "All node occurrences (KGML entries)", kg["node_occurrences"], "observed", "all <entry> elements, all types")
    for t in ("gene", "ortholog", "compound", "group", "map"):
        add("core", f"Node occurrences: {t}", nt.get(t, 0), "observed", f"KGML entry type={t}")
    add("core", "Unique human genes", kg["unique_hsa_genes_gene_entries"], "observed", "hsa IDs in gene entries")
    add("core", "Unique KO nodes (ortholog entries)", kg["unique_ko_ortholog_nodes"], "observed", "ko IDs in ortholog entries of hsa maps")
    add("core", "Unique KO IDs incl. KO of pathway genes", ko.get("unique_ko_ids", ""), "derived", "ortholog-entry KOs U link/ko/hsa of pathway genes")
    add("core", "Unique compounds (cpd, any entry)", kg["unique_compounds_cpd"], "observed", "cpd IDs in compound entries; glycans/drugs counted separately")
    add("core", "Unique glycans / drugs in compound entries", f'{kg["unique_glycans"]} / {kg["unique_drugs"]}', "observed", "")
    add("core", "Unique reaction IDs (KGML reaction records)", mt["unique_reaction_ids"], "observed", "rn IDs in <reaction> elements")
    add("core", "KGML relations (all)", kg["relations"], "observed", "")
    add("core", "KGML relations directional", kg["relations_directional"], "derived", "conservative subtype rule identical to v2 audit")
    add("core", "Entry-level gene->gene directional relations", gg["entry_level_gene_relations"], "observed", "gene entry -> gene entry, before paralog expansion")
    add("core", "Gene directed edges (pathway occurrences)", gg["gene_directed_edges_pathway_occurrences_strict"], "derived",
        "unique (src,tgt) per pathway after cartesian expansion of multi-gene entries")
    add("core", "Gene directed edges incl. group(complex) expansion", gg["gene_directed_edges_pathway_occurrences_incl_group"], "derived", "not used for sampling")
    add("core", "Unique gene pairs across pathways", gg["unique_gene_pairs_global"], "derived", "")
    add("core", "Genes present in directed gene graph", gg["unique_genes_in_directed_graph"], "derived", "")
    add("core", "Reaction records", mt["reaction_records"], "observed", "")
    add("core", "Reaction edges: substrate->product (pathway occurrences)", mt["substrate_product_edges_pathway_occurrences"], "derived",
        "reversible reactions contribute both directions")
    add("core", "Gene branch points (pathway occurrences)", gg["gene_branch_points_pathway_occurrences"], "derived", "out-degree>1")
    add("core", "Gene merge points (pathway occurrences)", gg["gene_merge_points_pathway_occurrences"], "derived", "in-degree>1")
    if up:
        add("mapping", "UniProt mapped genes", up["genes_mapped_to_uniprot"], "observed", "KEGG conv/uniprot/hsa (v2 cache)")
        add("mapping", "UniProt coverage", pct(up["uniprot_coverage"]), "derived", "")
        add("mapping", "Genes mapping to >1 UniProt accession", up["genes_one_to_many_uniprot"], "observed", "")
        add("mapping", "Protein sequence available genes (KEGG aaseq)", up["genes_with_kegg_aaseq"], "observed", "KEGG get/<gene>/aaseq")
        add("mapping", "Sequence coverage", pct(up["kegg_aaseq_coverage"]), "derived", "")
        add("mapping", "Sequence request errors remaining", up["genes_aaseq_status_error"], "observed", "retry on rerun")
    if go:
        add("mapping", "GO mapped genes (GOA U gene2go, non-NOT)", go["union_genes_with_go"], "observed", "")
        add("mapping", "GO coverage", pct(go["union_go_coverage"]), "derived", "")
        add("mapping", "GO mapped genes via EBI GOA (UniProt)", go["goa_genes_with_go"], "observed", "goa_human.gaf joined on KEGG->UniProt accessions")
        add("mapping", "GO mapped genes via NCBI gene2go (GeneID)", go.get("gene2go_genes_with_go", 0), "observed", "")
        add("mapping", "GO mapped genes excluding IEA-only", go["union_nonIEA_genes_with_go"], "derived", "")
        add("mapping", "Unique GO terms", go["union_unique_go_terms"], "observed", "")
        add("mapping", "Mean / median GO terms per mapped gene",
            f'{go["union_mean_go_per_mapped_gene"]:.1f} / {go["union_median_go_per_mapped_gene"]}', "derived", "")
        add("mapping", "Gene-GO pairs by aspect", json.dumps(go["union_gene_go_pairs_by_aspect"]), "observed", "")
        add("mapping", "Genes with >=1 term per aspect", json.dumps(go["union_genes_with_aspect"]), "observed", "")
    if ko:
        for key in ("unique_ko_ids", "ko_ids_with_human_gene_mapping", "ko_ids_only_as_ortholog_nodes_no_human_gene",
                    "ko_ids_mapping_one_human_gene", "ko_ids_mapping_multiple_human_genes", "ko_ids_with_go_kegg_dblinks",
                    "ko_ids_with_go_via_member_genes_derived", "ko_entries_fetch_error_or_missing", "pathway_genes_with_ko",
                    "pathway_genes_with_multiple_ko", "pathway_genes_without_ko"):
            add("ko", key, ko[key], "derived" if "derived" in key else "observed", "")
    add("metabolic", "Pathways with reaction records", mt["pathways_with_reaction_records"], "observed", "")
    add("metabolic", "... of which global overview maps (011xx/012xx)", mt["pathways_with_reaction_records"] - mt["pathways_with_reactions_excluding_global_maps"], "observed", "")
    add("metabolic", "Pathways with reactions but no gene directed graph", mt["pathways_with_reactions_but_no_gene_directed_graph"], "derived", "")
    add("metabolic", "Unique C/G IDs in reactions", mt["unique_compound_ids_in_reactions"], "observed", "")
    add("metabolic", "Unique R IDs", mt["unique_reaction_ids"], "observed", "")
    add("metabolic", "Reversible / irreversible reaction records",
        f'{mt["reaction_records_reversible"]} / {mt["reaction_records_irreversible"]}', "observed", "")
    add("metabolic", "Compound branch points (pathway occurrences; excluding global maps)",
        f'{mt["compound_branch_points_pathway_occurrences"]} ({mt["compound_branch_points_non_global"]})', "derived", "")
    for level, st in (("gene", g), ("ko", k)):
        add("samples", f"[{level}] A local next-node samples", st["A_next_node"]["total_samples"], "derived",
            f'unique {st["A_next_node"]["unique_samples"]}')
        for x in L:
            b, c = st[f"B_windows_L{x}"], st[f"C_masked_L{x}"]
            add("samples", f"[{level}] B path windows L={x} (retained)", b["total_samples"], "derived",
                f'unique {b["unique_samples"]}; cap {ctx.args.max_paths_per_pathway}/pathway')
            add("samples", f"[{level}] B simple paths L={x} before cap",
                b["total_simple_paths_all_pathways"], "estimated" if b["total_simple_paths_includes_estimates"] else "derived", "")
            add("samples", f"[{level}] C masked completion L={x}", c["total_samples"], "derived", f'unique {c["unique_samples"]}')
        add("samples", f"[{level}] B path windows all L", st[f"B_windows_maxL{L[-1]}"]["total_samples"], "derived", "")
        add("samples", f"[{level}] C masked completion all L", st[f"C_masked_maxL{L[-1]}"]["total_samples"], "derived", "")
    add("reference", "v2 audit: SCC-DAG source->sink routes (NOT samples)", 15443980, "derived", "combinatorial explosion; superseded")
    sp = load_split()
    split_md = ""
    if sp:
        split_md, srows = split_section(sp)
        for r in srows:
            add(*r)
    write_csv(OUT / "feasibility_report.csv", rows, ["section", "metric", "value", "kind", "definition_or_source"])

    # ---------------------------------------------------------------- markdown
    conc_cols = ["total_samples", "unique_samples", "pathways_contributing", "median_per_contributing_pathway",
                 "median_per_pathway_all372", "max_from_one_pathway", "max_pathway", "top1_share", "top5_share", "top10_share", "gini_contributing"]
    conc_hdr = ["total", "unique", "pathways", "median/pw (contrib.)", "median/pw (all 372)", "max 1 pw", "max pathway",
                "top1", "top5", "top10", "Gini"]

    def conc_table(st):
        out = ["| strategy | " + " | ".join(conc_hdr) + " |", "|---" * (len(conc_hdr) + 1) + "|"]
        order = ["A_next_node"] + [f"B_windows_L{x}" for x in L] + [f"C_masked_L{x}" for x in L] + \
                [f"B_windows_maxL{L[-1]}", f"C_masked_maxL{L[-1]}"]
        for s in order:
            d = st[s]
            vals = []
            for c in conc_cols:
                v = d.get(c, "")
                vals.append(pct(v) if c.startswith("top") and c.endswith("share") else fmt(v))
            out.append(f"| {s} | " + " | ".join(vals) + " |")
        return "\n".join(out)

    def amb_table(st):
        out = ["| strategy | ambiguous contexts (>1 valid target) | samples whose key occurs in >=2 pathways | pathways capped/sampled |",
               "|---|---|---|---|"]
        for s in ["A_next_node"] + [f"C_masked_L{x}" for x in L]:
            d = st[s]
            out.append(f"| {s} | {pct(d.get('ambiguous_context_frac', 0))} | {pct(d.get('multi_pathway_sample_frac', 0))} | "
                       f"{d.get('pathways_capped_or_sampled', '-')} |")
        return "\n".join(out)

    main_rows = [r for r in rows if r["section"] in ("core", "mapping", "samples", "reference")]
    nfail, fby = failures_summary()
    gaf_date = ""
    if A.GOA_GAF.exists():
        import gzip
        with gzip.open(A.GOA_GAF, "rt") as f:
            for line in f:
                if line.startswith("!date-generated") or line.startswith("!Generated"):
                    gaf_date = line.strip("!\n ")
                    break
                if not line.startswith("!"):
                    break

    ngenes = kg["unique_hsa_genes_gene_entries"]
    md = f"""# KEGG human pathway dataset: feasibility report

Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `build_kegg_pathway_dataset.py`
(seed={ctx.args.seed}, max paths per pathway per length={ctx.args.max_paths_per_pathway}, window lengths={ctx.args.window_lengths}).

**Kinds.** *observed* = counted directly from a source record; *derived* = produced by a transformation
chosen here (paralog/cartesian expansion, graph projection, sampling, dedup); *estimated* = extrapolated
from random sampling. Sample counts are **not** independent biological observations.

Sources: KEGG KGML (cached by the v2 audit; KGML creation dates ~2024), KEGG REST lists/links
(fetched {dt.date.today()}), EBI GOA `goa_human.gaf` ({gaf_date or 'date n/a'}), NCBI `gene2go`
(human rows), GO `go-basic.obo`, UniProt REST (human accession table).

## 1. Main table

| section | metric | value | kind | note |
|---|---|---:|---|---|
""" + "\n".join(f"| {r['section']} | {r['metric']} | {fmt(r['value'])} | {r['kind']} | {r['definition_or_source']} |" for r in main_rows) + f"""

## 2. Gene directed graph

* Pathways with a usable directed gene graph: **{gg['pathways_with_gene_directed_graph']} / {kg['pathways_parsed']}**
  (median {gg['median_nodes_per_usable_pathway']} nodes, {gg['median_edges_per_usable_pathway']} edges; max {gg['max_edges_one_pathway']:,} edges in one pathway).
* Sources {gg['gene_sources_total']:,}, sinks {gg['gene_sinks_total']:,}, branch nodes {gg['gene_branch_points_pathway_occurrences']:,},
  merge nodes {gg['gene_merge_points_pathway_occurrences']:,} (all summed over pathway occurrences).
* Cycles: {gg['pathways_with_cycles']} pathways contain {gg['cyclic_sccs_total']} non-trivial SCCs; largest SCC = {gg['largest_scc_any_pathway']} genes.
* Only **{gg['unique_genes_in_directed_graph']:,} of {ngenes:,}** pathway genes ever appear in a directed gene edge.
* {gg['entry_level_gene_relations']:,} entry-level relations become {gg['gene_directed_edges_pathway_occurrences_strict']:,} gene edges
  (x{gg['gene_directed_edges_pathway_occurrences_strict'] / gg['entry_level_gene_relations']:.1f}); {pct(gg['gene_entries_with_multiple_genes_frac'])} of gene entries
  hold >1 paralog and {pct(gg['gene_edge_rows_from_multi_gene_entries_frac'])} of gene edges come from a multi-gene entry.
* Per-pathway details: `gene_graph_stats_by_pathway.csv`.

## 3. Training-sample constructions and pathway concentration

Top-k shares are fractions of *all* samples of that construction contributed by the k largest pathways.
Rows `*_maxL{L[-1]}` pool all lengths 3..{L[-1]}.

### 3a. Gene tokens

{conc_table(g)}

{amb_table(g)}

### 3b. KO tokens (gene edges projected through link/ko/hsa)

{conc_table(k)}

{amb_table(k)}

Simple paths before the cap (gene level): """ + ", ".join(
        f"L={x}: {g[f'B_windows_L{x}']['total_simple_paths_all_pathways']:,}" +
        (" (includes walk-sampling estimates)" if g[f'B_windows_L{x}']['total_simple_paths_includes_estimates'] else "")
        for x in L) + """.
Per-pathway enumeration mode and totals: `path_enumeration_gene.csv`, `path_enumeration_ko.csv`.

"""
    if up:
        md += f"""## 4. UniProt and sequence

* KEGG->UniProt: {up['genes_mapped_to_uniprot']:,}/{up['unique_pathway_genes']:,} ({pct(up['uniprot_coverage'])}); one-to-many: {up['genes_one_to_many_uniprot']:,} genes
  (max {up['max_uniprot_per_gene']} accessions), {up['gene_uniprot_pairs']:,} gene-accession pairs; {up['genes_with_reviewed_accession']:,} genes have a Swiss-Prot accession.
* Unmapped genes by reason: {json.dumps(up['unmapped_genes_by_reason'])}.
* Unmapped genes for which UniProt itself lists the GeneID (recoverable, *not merged*): {up['unmapped_genes_recoverable_via_uniprot_geneid_xref']}.
* Accessions in KEGG conv that are absent from current UniProt: {up['genes_with_accession_not_in_current_uniprot']} genes.
* KEGG amino-acid sequence: {up['genes_with_kegg_aaseq']:,} ({pct(up['kegg_aaseq_coverage'])}); median length {up['aaseq_length_median']}; request errors remaining {up['genes_aaseq_status_error']}.
* KEGG gene types of pathway genes: {json.dumps(up['kegg_gene_type_counts'])}.

"""
    if go:
        md += f"""## 5. GO

| source | genes with >=1 GO | coverage | unique GO terms | mean / median terms per mapped gene |
|---|---:|---:|---:|---:|
""" + "\n".join(
            f"| {lab} | {go[p + '_genes_with_go']:,} | {pct(go[p + '_go_coverage'])} | {go[p + '_unique_go_terms']:,} | "
            f"{go[p + '_mean_go_per_mapped_gene']:.1f} / {go[p + '_median_go_per_mapped_gene']} |"
            for lab, p in (("EBI GOA (via UniProt)", "goa"), ("NCBI gene2go (via GeneID)", "gene2go"),
                           ("union", "union"), ("union, excluding IEA-only pairs", "union_nonIEA"))) + f"""

* Gene-GO pairs by aspect (union): {json.dumps(go['union_gene_go_pairs_by_aspect'])}; genes with >=1 term per aspect: {json.dumps(go['union_genes_with_aspect'])}.
* Directed-graph genes with GO: {go['directed_graph_genes_with_go']:,}/{go['directed_graph_genes']:,}.
* Genes without any GO, by KEGG gene type: {json.dumps(go['genes_without_go_by_type'])}.
* NOT-qualified rows kept but flagged: {go['negated_rows']:,}; rows with obsolete GO IDs: {go['obsolete_term_rows']:,}.

"""
    if ko:
        md += f"""## 6. KO layer

* Unique KO IDs: {ko['unique_ko_ids']:,} ({ko['ko_ids_as_ortholog_nodes']:,} as ortholog nodes, {ko['ko_ids_via_human_genes']:,} via human genes).
* With human gene mapping: {ko['ko_ids_with_human_gene_mapping']:,}; KO-only (no human gene): {ko['ko_ids_only_as_ortholog_nodes_no_human_gene']:,}.
* One human gene (genome-wide): {ko['ko_ids_mapping_one_human_gene']:,}; multiple human genes: {ko['ko_ids_mapping_multiple_human_genes']:,} (max {ko['max_human_genes_per_ko']}).
  Restricted to genes drawn in pathways: one gene {ko['ko_ids_via_genes_one_pathway_gene']:,}, multiple genes {ko['ko_ids_via_genes_multiple_pathway_genes']:,}.
* GO from KEGG KO DBLINKS: {ko['ko_ids_with_go_kegg_dblinks']:,} KOs; GO derivable from member human genes: {ko['ko_ids_with_go_via_member_genes_derived']:,}; KO entries not fetched/failed: {ko['ko_entries_fetch_error_or_missing']}.
* Pathway genes with a KO: {ko['pathway_genes_with_ko']:,}; with >1 KO: {ko['pathway_genes_with_multiple_ko']}; without KO: {ko['pathway_genes_without_ko']}.
* Directed graph size: gene level {gg['unique_genes_in_directed_graph']:,} tokens / {gg['unique_gene_pairs_global']:,} unique edges vs KO level {gg['ko_graph_unique_nodes']:,} tokens / {gg['ko_graph_unique_pairs_global']:,} unique edges.

"""
    md += f"""## 7. Metabolic compound-reaction graph (kept separate)

* Pathways with reaction records: {mt['pathways_with_reaction_records']} ({mt['pathways_with_reactions_but_no_gene_directed_graph']} of them have **no** directed gene graph).
* Unique compound/glycan IDs in reactions: {mt['unique_compound_ids_in_reactions']:,}; unique R IDs: {mt['unique_reaction_ids']:,} ({mt['reaction_ids_only_in_global_maps']} only in global maps).
* Reversible / irreversible records: {mt['reaction_records_reversible']:,} / {mt['reaction_records_irreversible']:,}; global maps are {pct(mt['global_maps_reversible_frac'])} reversible (drawing default, not biology).
* Substrate->product edges: {mt['substrate_product_edges_pathway_occurrences']:,} ({mt['substrate_product_edges_non_global_maps']:,} excluding global maps).
* Compound branch points: {mt['compound_branch_points_pathway_occurrences']:,} ({mt['compound_branch_points_non_global']:,} excluding global maps).
* Per-pathway: `metabolic_graph_stats_by_pathway.csv`; edges: `metabolic_compound_edges.csv`.

## 8. Bias and leakage risks (quantified)

* **Cross-pathway duplication:** {pct(gg['edge_occurrences_whose_pair_is_in_2plus_pathways_frac'])} of gene-edge occurrences are pairs that occur in >=2 pathways;
  {pct(gg['disease_edges_also_in_non_disease_pathways_frac'])} of edge occurrences in disease maps (hsa05xxx) also occur in non-disease maps.
  A random or even pathway-level split leaks unless splits are made on deduplicated edges / connected sub-networks
  (the KO-level split in section 10 addresses this).
* **Paralog cartesian expansion:** one drawn box with n paralogs becomes n x m edges; near-identical samples differ only by a paralog
  and will straddle a random split.
* **Target ambiguity:** {pct(g['A_next_node']['ambiguous_context_frac'])} of gene next-node contexts have >1 valid target; exact-match accuracy is ill-posed,
  rank metrics over the set of valid targets are needed.
* **Pathway concentration:** see section 3 top-k shares; caps flatten the largest pathways only partially.
* **Global overview maps (hsa011xx/012xx)** duplicate reactions from specific maps and should be excluded or split together.
* **Coverage bias:** directed edges exist only for signalling/regulatory maps; metabolic maps contribute zero gene edges.

## 9. Incomplete / unavailable

* Failures logged in `logs/failures.csv`: {nfail} ({json.dumps(fby)}). Failed requests are never counted as "no annotation".
"""
    if up and go and ko:
        md += f"""* UniProt: {up['unique_pathway_genes'] - up['genes_mapped_to_uniprot']} genes unmapped; {sum(v for k_, v in up['unmapped_genes_by_reason'].items() if k_.startswith('non-protein'))}
  are tRNA/rRNA/ncRNA (no protein expected), {up['unmapped_genes_by_reason'].get('protein-coding but no KEGG conv/uniprot link', 0)} protein-coding genes have no KEGG link
  ({up['unmapped_genes_recoverable_via_uniprot_geneid_xref']} of these are listed by UniProt's GeneID xref; reported in
  `uniprot_side_extra_accessions_by_geneid_xref`, not merged).
* Sequence: {up['unique_pathway_genes'] - up['genes_with_kegg_aaseq']} genes without KEGG aaseq (the non-coding genes); UniProt sequences not downloaded
  (only lengths), KEGG sequences are in `gene_sequences_kegg.faa`.
* GO: {go['unique_pathway_genes'] - go['union_genes_with_go']} genes without GO ({json.dumps(go['genes_without_go_by_type'])});
  {go['uniprot_mapped_genes_without_goa']} UniProt-mapped genes have no GOA row. Isoform-level (goa_human_isoform) and RNAcentral (goa_human_rna)
  files were not joined; ncRNA GO comes only from gene2go.
* KO GO: KEGG REST has no KO->GO link operation; GO for KOs comes from the DBLINKS field of each KO flat file.
  {ko['unique_ko_ids'] - ko['ko_ids_with_go_kegg_dblinks']} KOs have no GO DBLINK (entry fetched successfully, field absent).
* Group (complex) semantics are only kept as `group_expanded` edges and not used in samples; `missing interaction`
  and bare `indirect` relations are non-directional by the inherited rule.
* Modules: only KO->module membership (link/module/ko); module internal structure (definitions/reaction order) not parsed.
"""
    md += ("\n" + split_md) if split_md else ""
    (OUT / "feasibility_report.md").write_text(md, encoding="utf-8")
    write_readme(ctx)
    print_final(ctx)


def write_readme(ctx):
    a = ctx.args
    txt = f"""# kegg_dataset

KEGG human pathway modelling dataset built by `../build_kegg_pathway_dataset.py`
(helpers in `../kegg_pipeline/`). Continues the v2 feasibility audit; its caches in
`../kegg_feasibility_cache/` are only read. New downloads are cached in `../kegg_dataset_cache/`.

Rerun: `python build_kegg_pathway_dataset.py` (resumable; cached files and completed API
batches are reused; failed requests keep status `error` and are retried).
Parameters used: seed={a.seed}, --max-paths-per-pathway {a.max_paths_per_pathway}, --window-lengths {a.window_lengths},
--enum-walk-limit {a.enum_walk_limit}, --sleep {a.sleep}.

## Files

| file | content |
|---|---|
| node_occurrences.csv | one row per KGML entry (all types: gene, ortholog, compound, group, map); KEGG IDs split by namespace; group components; reaction IDs; `derived_ko_ids_from_hsa_genes` (via link/ko/hsa) |
| entities.csv | deduplicated identifiers (hsa gene, ko, cpd, gl, dr, path, group composition) with names and occurrence counts |
| kgml_relations.csv | every KGML relation, with types, subtypes, subtype values and `whether_directional` |
| reactions.csv | every KGML reaction record (substrates/products, reversibility, enzyme entry genes/KOs) |
| gene_edges.csv | directed gene->gene edges, dedup per pathway; `edge_origin` = gene_entry (used) or group_expanded (complex members; not used for sampling) |
| gene_graph_stats_by_pathway.csv | nodes/edges/sources/sinks/branch/merge/SCC per pathway, gene and KO level |
| token_vocab_gene.csv, token_vocab_ko.csv | token IDs (0 [PAD], 1 [MASK], 2 [UNK]) |
| samples/{{gene,ko}}_A_next_node.csv.gz | strategy A (one row per directed edge per pathway) |
| samples/{{gene,ko}}_B_path_windows.csv.gz | strategy B simple-path windows, L=3..6 |
| samples/{{gene,ko}}_C_masked.csv.gz | strategy C masked interior node; `path_id` links to B |
| path_enumeration_{{gene,ko}}.csv | per pathway x L: walk upper bound, total simple paths (exact or estimated), retained, mode |
| sampling_strategy_stats.csv | totals, uniques, per-pathway median/max, top-1/5/10 shares, Gini, ambiguity |
| samples_per_pathway.csv | per-pathway sample counts for every construction |
| metabolic_graph_stats_by_pathway.csv, metabolic_compound_edges.csv | compound -> reaction -> compound layer (not mixed with gene graph) |
| gene_uniprot_mapping.csv | one row per pathway gene: accessions, one-to-many flag, Swiss-Prot/TrEMBL split, failure reason, KEGG aaseq status/length |
| gene_uniprot_pairs.csv | long format gene-accession pairs |
| gene_sequences_kegg.faa | KEGG amino-acid sequences of pathway genes |
| gene_go_mapping.csv | every GO annotation row (GOA via UniProt accession, gene2go via GeneID) with aspect, evidence, qualifier, source |
| gene_go_terms_collapsed.csv | positive (non-NOT) unique gene-GO pairs with sources and evidence codes; input for later GO embeddings |
| ko_mapping.csv | KO table: pathways, modules, human genes, KO GO (KEGG DBLINKS), derived GO of member genes |
| feasibility_report.csv / .md | final report |
| splits_ko/ | KO-level leakage-controlled train/val/test split (`../build_ko_leakage_free_split.py`): pathway_split.csv, edge_split.csv, {{train,val,test}}_{{A_next_node,B_path_windows,C_masked}}.csv.gz, split_stats.csv, leakage_checks.csv, split_config.csv, README.md |
| logs/metrics.json, logs/failures.csv, logs/run_*.log | raw metrics, explicit failure log, run logs |

## Key definitions

* Directional relation: any subtype in {{activation, inhibition, expression, repression, indirect effect, state change,
  phosphorylation, dephosphorylation, glycosylation, ubiquitination, methylation}} (identical to v2; `binding/association`,
  `dissociation`, `compound`, `missing interaction` and the rare bare `indirect` label are non-directional).
* Gene edge: gene entry -> gene entry directional relation expanded to all (paralog) gene pairs; self pairs dropped.
* KO edge: gene edge mapped through KEGG link/ko/hsa; KO self pairs dropped.
* Path windows: simple paths (no repeated node, so cycles are never re-entered) with exactly L nodes.
  Exhaustive enumeration when the exact walk count <= enum limit, otherwise walk-count-weighted uniform sampling with
  rejection. At most `cap` paths per pathway and L, subsampled with `random.Random("{a.seed}|level|pathway|L")`.
* Masked samples: every interior position of each retained path.
* Reaction edges are never converted into gene->gene causal edges.

See `feasibility_report.md` for numbers, concentration statistics, incomplete mappings and leakage risks.
"""
    (OUT / "README.md").write_text(txt, encoding="utf-8")


def print_final(ctx):
    m = ctx.metrics
    kg, gg, mt = m["kgml"], m["gene_graph"], m["metabolic"]
    up, go, ko = m.get("uniprot", {}), m.get("go", {}), m.get("ko", {})
    g = m["sampling"]["gene"]
    L = ctx.args.window_lengths.split(",")[-1]
    lines = [
        "================ FINAL SUMMARY ================",
        f"Pathways parsed                   {kg['pathways_parsed']}",
        f"Node occurrences                  {kg['node_occurrences']:,}  {kg['node_occurrences_by_type']}",
        f"Unique genes / KO nodes / cpd     {kg['unique_hsa_genes_gene_entries']:,} / {kg['unique_ko_ortholog_nodes']:,} / {kg['unique_compounds_cpd']:,}",
        f"Unique reactions                  {mt['unique_reaction_ids']:,}",
        f"Pathways with gene digraph        {gg['pathways_with_gene_directed_graph']}",
        f"Gene directed edges (occ.)        {gg['gene_directed_edges_pathway_occurrences_strict']:,}  (unique pairs {gg['unique_gene_pairs_global']:,})",
        f"Gene branch points (occ.)         {gg['gene_branch_points_pathway_occurrences']:,}",
        f"Substrate->product edges (occ.)   {mt['substrate_product_edges_pathway_occurrences']:,}",
    ]
    if up:
        lines += [f"UniProt mapped genes              {up['genes_mapped_to_uniprot']:,} ({pct(up['uniprot_coverage'])})",
                  f"KEGG sequence genes               {up['genes_with_kegg_aaseq']:,} ({pct(up['kegg_aaseq_coverage'])})"]
    if go:
        lines += [f"GO mapped genes (union)           {go['union_genes_with_go']:,} ({pct(go['union_go_coverage'])}); GOA only {go['goa_genes_with_go']:,}",
                  f"Unique GO terms                   {go['union_unique_go_terms']:,}"]
    if ko:
        lines += [f"KO IDs / with human genes / w GO  {ko['unique_ko_ids']:,} / {ko['ko_ids_with_human_gene_mapping']:,} / {ko['ko_ids_with_go_kegg_dblinks']:,}"]
    lines += [
        f"A next-node samples (gene)        {g['A_next_node']['total_samples']:,} (unique {g['A_next_node']['unique_samples']:,}; top10 {pct(g['A_next_node']['top10_share'])})",
        f"B windows L3..{L} (gene)            {g[f'B_windows_maxL{L}']['total_samples']:,} (top10 {pct(g[f'B_windows_maxL{L}']['top10_share'])})",
        f"C masked L3..{L} (gene)             {g[f'C_masked_maxL{L}']['total_samples']:,} (top10 {pct(g[f'C_masked_maxL{L}']['top10_share'])})",
        f"Failures logged                   {failures_summary()[0]}",
        "Report: kegg_dataset/feasibility_report.md",
    ]
    print("\n".join(lines), flush=True)

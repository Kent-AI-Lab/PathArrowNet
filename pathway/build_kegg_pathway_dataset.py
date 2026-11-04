#!/usr/bin/env python3
"""
Build the KEGG human pathway modelling dataset (stage after the v2 feasibility audit).

Inputs reused (read only, never modified):
    kegg_feasibility_cache/kgml/*.xml, pathway_list.tsv, uniprot_conv.tsv, sequence_status.json
New cache (resumable):  kegg_dataset_cache/
Outputs:                kegg_dataset/

Run (resumable; rerunning continues from caches):
    python build_kegg_pathway_dataset.py
Options:
    --max-paths-per-pathway 10000  --seed 42  --window-lengths 3,4,5,6
    --sleep 0.4        polite delay after every KEGG request (serial requests only)
    --offline          never touch the network; unavailable data is marked as such
    --no-gene2go       skip the 1.4 GB NCBI gene2go download (secondary GO source)
    --stages ...       comma list of stages to run (default: all)

No third-party packages are required.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kegg_pipeline import annotations as A  # noqa: E402
from kegg_pipeline import kgml as K  # noqa: E402
from kegg_pipeline import sampling as S  # noqa: E402
from kegg_pipeline.common import OUT, load_json, log, read_csv, save_json_atomic, write_csv  # noqa: E402

SAMPLES = OUT / "samples"
SAMPLES.mkdir(parents=True, exist_ok=True)
METRICS = OUT / "logs" / "metrics.json"   # stage metrics, merged across (partial) runs
GLOBAL_MAP_PREFIX = ("hsa011", "hsa012")    # overview maps (01100 Metabolic pathways, ...)
SPECIAL = ["[PAD]", "[MASK]", "[UNK]"]
STAGES = ["kgml", "gene_graph", "sampling", "metabolic", "uniprot", "go", "ko", "report"]


def J(xs, sep=";"):
    return sep.join(str(x) for x in xs)


def gsort(xs):
    return sorted(set(xs), key=lambda x: int(x) if str(x).isdigit() else 10**12)


class Ctx:
    """Shared state; everything is rebuilt cheaply from caches on each run."""

    def __init__(self, args):
        self.args = args
        self.metrics = load_json(METRICS, {})

    def save_metrics(self, section, d):
        self.metrics[section] = d
        save_json_atomic(METRICS, self.metrics)


# =========================================================================== stage: kgml
def stage_kgml(ctx):
    a = ctx.args
    lists = A.kegg_lists(a.sleep, False, a.offline)
    ctx.lists = lists
    ctx.gene_info = A.parse_list_hsa(lists["list_hsa"])
    ctx.gene2ko = A.parse_link(lists["link_ko_hsa"])
    ctx.ko2genes_all = defaultdict(set)
    for g, ks in ctx.gene2ko.items():
        for k in ks:
            ctx.ko2genes_all[k].add(g)
    ctx.ko_names = A.parse_simple_list(lists["list_ko"])
    ctx.cpd_names = A.parse_simple_list(lists["list_compound"])
    ctx.gl_names = A.parse_simple_list(lists["list_glycan"])
    ctx.dr_names = A.parse_simple_list(lists["list_drug"])
    ctx.rn_names = A.parse_simple_list(lists["list_reaction"])
    ctx.ko2modules = A.parse_link(lists["link_module_ko"])
    ctx.module_names = A.parse_simple_list(lists["list_module"])
    classes = K.parse_brite_pathway_classes(lists["brite_br08901"])
    ctx.pw = K.parse_all(classes)

    def sym(g):
        return ctx.gene_info.get(g, {}).get("symbol", "")
    ctx.sym = sym

    # ---------------- node_occurrences.csv
    rows = []
    ent_occ = defaultdict(lambda: {"occ": 0, "pw": set(), "types": set(), "as_comp": 0})
    for pid, p in ctx.pw.items():
        E = p["entries"]
        for eid, e in E.items():
            comp_genes = K.expand_group_genes(E, eid) if e["entry_type"] == "group" else []
            other = [i for i in e["ids"] if i.split(":")[0] not in ("hsa", "ko", "cpd", "gl", "dr", "path")]
            derived_ko = sorted({k for g in e["hsa"] for k in ctx.gene2ko.get(g, ())})
            rows.append({
                "pathway_id": pid, "pathway_name": p["pathway_name"],
                "pathway_class_a": p["class_a"], "pathway_class_b": p["class_b"],
                "entry_id": eid, "entry_type": e["entry_type"], "kegg_ids": e["raw_name"],
                "n_kegg_ids": len(e["ids"]),
                "hsa_gene_ids": J(e["hsa"]), "gene_symbols": J(sym(g) for g in e["hsa"]),
                "ko_ids": J(e["ko"]), "compound_ids": J(e["cpd"]), "glycan_ids": J(e["gl"]),
                "drug_ids": J(e["dr"]), "map_ids": J(e["path"]), "other_ids": J(other),
                "group_component_entry_ids": J(e["components"]),
                "group_component_hsa_gene_ids": J(comp_genes),
                "reaction_ids_entry_attribute": J(e["reaction_attr"]),
                "reaction_ids_reaction_elements": J(e["reaction_elements"]),
                "graphics_label": e["graphics_name"],
                "derived_ko_ids_from_hsa_genes": J(derived_ko),
            })
            if e["entry_type"] == "group":
                key = "group:" + "+".join(sorted(E[c]["raw_name"].replace(" ", ",")
                                                 for c in e["components"] if c in E))
                ids = [key]
            else:
                ids = e["ids"]
            for i in ids:
                d = ent_occ[i]
                d["occ"] += 1
                d["pw"].add(pid)
                d["types"].add(e["entry_type"])
            for g in comp_genes:
                ent_occ["hsa:" + g]["as_comp"] += 1
    n = write_csv(OUT / "node_occurrences.csv", rows, list(rows[0].keys()))
    log(f"node_occurrences.csv: {n:,} rows")

    # ---------------- entities.csv
    def ename(i):
        pre, _, x = i.partition(":")
        if pre == "hsa":
            return ctx.gene_info.get(x, {}).get("description", "")
        return {"ko": ctx.ko_names, "cpd": ctx.cpd_names, "gl": ctx.gl_names,
                "dr": ctx.dr_names}.get(pre, {}).get(x, "")
    kind = {"hsa": "gene", "ko": "ortholog", "cpd": "compound", "gl": "glycan", "dr": "drug",
            "path": "map", "group": "group"}
    erows = []
    for i, d in sorted(ent_occ.items()):
        pre, _, x = i.partition(":")
        if pre == "path":
            nm = ctx.pw[x]["pathway_name"] if x in ctx.pw else ""
        else:
            nm = ename(i)
        erows.append({
            "entity_id": i, "entity_type": kind.get(pre, "other"), "id_prefix": pre,
            "symbol": ctx.sym(x) if pre == "hsa" else "",
            "gene_type": ctx.gene_info.get(x, {}).get("gene_type", "") if pre == "hsa" else "",
            "name": nm, "n_node_occurrences": d["occ"], "n_pathways": len(d["pw"]),
            "n_occurrences_as_group_component": d["as_comp"],
            "entry_types_seen": J(sorted(d["types"])), "pathways": J(sorted(d["pw"])),
        })
    write_csv(OUT / "entities.csv", erows, list(erows[0].keys()))
    ecount = Counter(r["entity_type"] for r in erows)
    log(f"entities.csv: {len(erows):,} entities {dict(ecount)}")

    # ---------------- kgml_relations.csv
    rrows = []
    for pid, p in ctx.pw.items():
        E = p["entries"]
        for r in p["relations"]:
            s, t = E.get(r["source_entry"], {}), E.get(r["target_entry"], {})
            cvals = [E[v]["raw_name"] for n_, v in zip(r["subtypes"], r["subtype_values"])
                     if n_ == "compound" and v in E]
            rrows.append({
                "pathway_id": pid, "relation_index": r["relation_index"],
                "source_entry": r["source_entry"], "target_entry": r["target_entry"],
                "source_entry_type": s.get("entry_type", "MISSING"),
                "target_entry_type": t.get("entry_type", "MISSING"),
                "source_kegg_ids": s.get("raw_name", ""), "target_kegg_ids": t.get("raw_name", ""),
                "relation_type": r["relation_type"], "relation_subtypes": J(r["subtypes"]),
                "relation_subtype_values": J(r["subtype_values"]),
                "compound_ids_in_subtype": J(cvals),
                "whether_directional": int(r["directional"]),
            })
    write_csv(OUT / "kgml_relations.csv", rrows, list(rrows[0].keys()))
    log(f"kgml_relations.csv: {len(rrows):,} rows")

    # ---------------- reactions.csv
    xrows = []
    for pid, p in ctx.pw.items():
        E = p["entries"]
        for r in p["reactions"]:
            enz = E.get(r["enzyme_entry"], {})
            xrows.append({
                "pathway_id": pid, "reaction_index": r["reaction_index"],
                "kgml_reaction_entry": r["enzyme_entry"],
                "reaction_ids": J(r["reaction_ids"]), "n_reaction_ids": len(r["reaction_ids"]),
                "reaction_type": r["reaction_type"],
                "substrate_entry_ids": J(i for i, _ in r["substrates"]),
                "substrate_compound_ids": J(n for _, n in r["substrates"]),
                "product_entry_ids": J(i for i, _ in r["products"]),
                "product_compound_ids": J(n for _, n in r["products"]),
                "enzyme_entry_type": enz.get("entry_type", "MISSING"),
                "enzyme_hsa_gene_ids": J(enz.get("hsa", [])),
                "enzyme_gene_symbols": J(ctx.sym(g) for g in enz.get("hsa", [])),
                "enzyme_ko_ids_direct": J(enz.get("ko", [])),
                "enzyme_ko_ids_derived_from_genes": J(sorted({k for g in enz.get("hsa", [])
                                                              for k in ctx.gene2ko.get(g, ())})),
            })
    write_csv(OUT / "reactions.csv", xrows, list(xrows[0].keys()))
    log(f"reactions.csv: {len(xrows):,} rows")

    etypes = Counter(r["entry_type"] for r in rows)
    all_genes = {g for p in ctx.pw.values() for e in p["entries"].values()
                 if e["entry_type"] == "gene" for g in e["hsa"]}
    ctx.all_genes = all_genes
    ctx.save_metrics("kgml", {
        "pathways_parsed": len(ctx.pw),
        "node_occurrences": len(rows),
        "node_occurrences_by_type": dict(etypes),
        "protein_related_occurrences_gene_ortholog_enzyme": etypes["gene"] + etypes["ortholog"] + etypes["enzyme"],
        "unique_hsa_genes_gene_entries": len(all_genes),
        "unique_hsa_genes_incl_group_components": len({r["entity_id"] for r in erows if r["entity_type"] == "gene"}),
        "unique_ko_ortholog_nodes": ecount["ortholog"],
        "unique_ko_via_pathway_genes": len({k for g in all_genes for k in ctx.gene2ko.get(g, ())}),
        "unique_compounds_cpd": ecount["compound"], "unique_glycans": ecount["glycan"],
        "unique_drugs": ecount["drug"], "unique_group_compositions": ecount["group"],
        "unique_map_links": ecount["map"],
        "relations": len(rrows),
        "relations_directional": sum(r["whether_directional"] for r in rrows),
        "relations_by_type": dict(Counter(r["relation_type"] for r in rrows)),
        "reaction_records": len(xrows),
        "kgml_entries_with_reaction_attribute": sum(1 for r in rows if r["reaction_ids_entry_attribute"]),
        "pathway_classes": dict(Counter(p["class_a"] or "UNCLASSIFIED" for p in ctx.pw.values())),
    })


# =========================================================================== stage: gene graph
def stage_gene_graph(ctx):
    edges = {}           # (pid, src, tgt, rtype, subtype, origin) -> agg
    entry_level = defaultdict(set)
    self_loops = 0
    for pid, p in ctx.pw.items():
        E = p["entries"]
        for r in p["relations"]:
            if not r["directional"]:
                continue
            s, t = E.get(r["source_entry"]), E.get(r["target_entry"])
            if not s or not t:
                continue
            st, tt = s["entry_type"], t["entry_type"]
            if st == "gene" and tt == "gene":
                origin, sg, tg = "gene_entry", s["hsa"], t["hsa"]
                entry_level[pid].add((r["source_entry"], r["target_entry"]))
            elif st in ("gene", "group") and tt in ("gene", "group"):
                origin = "group_expanded"
                sg, tg = K.expand_group_genes(E, r["source_entry"]), K.expand_group_genes(E, r["target_entry"])
            else:
                continue
            sub = J(sorted(x for x in r["subtypes"] if x in K.DIRECTIONAL_SUBTYPES))
            for g1 in sg:
                for g2 in tg:
                    if g1 == g2:
                        self_loops += 1
                        continue
                    key = (pid, g1, g2, r["relation_type"], sub, origin)
                    d = edges.setdefault(key, {"n": 0, "se": set(), "te": set(), "cf": 0})
                    d["n"] += 1
                    d["se"].add(r["source_entry"])
                    d["te"].add(r["target_entry"])
                    d["cf"] = max(d["cf"], len(sg) * len(tg))
    rows = []
    for (pid, g1, g2, rt, sub, origin), d in sorted(edges.items(), key=lambda kv: (kv[0][0], int(kv[0][1]), int(kv[0][2]), kv[0][3:])):
        rows.append({"pathway_id": pid, "source_gene": g1, "target_gene": g2,
                     "source_symbol": ctx.sym(g1), "target_symbol": ctx.sym(g2),
                     "relation_type": rt, "relation_subtype": sub, "edge_origin": origin,
                     "n_kgml_relations": d["n"], "source_entries": J(sorted(d["se"], key=int)),
                     "target_entries": J(sorted(d["te"], key=int)),
                     "max_entry_cartesian_factor": d["cf"]})
    write_csv(OUT / "gene_edges.csv", rows, list(rows[0].keys()))
    log(f"gene_edges.csv: {len(rows):,} deduplicated rows")

    strict, grp = defaultdict(set), defaultdict(set)
    for r in rows:
        (strict if r["edge_origin"] == "gene_entry" else grp)[r["pathway_id"]].add((r["source_gene"], r["target_gene"]))
    ctx.gene_graph = strict
    # KO-level graph derived from strict gene edges
    ko_graph, dropped_nokos, ko_self = defaultdict(set), 0, 0
    for pid, es in strict.items():
        for u, v in es:
            ku, kv = ctx.gene2ko.get(u, set()), ctx.gene2ko.get(v, set())
            if not ku or not kv:
                dropped_nokos += 1
                continue
            for a in ku:
                for b in kv:
                    if a == b:
                        ko_self += 1
                    else:
                        ko_graph[pid].add((a, b))
    ctx.ko_graph = ko_graph

    srows = []
    for pid, p in ctx.pw.items():
        st = S.graph_stats(strict.get(pid, set()))
        combined = strict.get(pid, set()) | grp.get(pid, set())
        kst = S.graph_stats(ko_graph.get(pid, set()))
        srows.append(dict(
            pathway_id=pid, pathway_name=p["pathway_name"], pathway_class_a=p["class_a"],
            entry_level_gene_relations=len(entry_level.get(pid, ())),
            **{f"gene_{k}": v for k, v in st.items()},
            group_expanded_extra_edges=len(combined - strict.get(pid, set())),
            gene_edges_incl_group_expansion=len(combined),
            ko_nodes=kst["nodes"], ko_edges=kst["edges"], ko_branch_nodes=kst["branch_nodes"],
            ko_cyclic_scc_count=kst["cyclic_scc_count"]))
    write_csv(OUT / "gene_graph_stats_by_pathway.csv", srows, list(srows[0].keys()))

    usable = [r for r in srows if r["gene_edges"] > 0]
    med = lambda k: statistics.median([r[k] for r in usable]) if usable else 0
    all_pairs = Counter((u, v) for es in strict.values() for u, v in es)
    occ_in_multi = sum(c for c in all_pairs.values() if c > 1)
    tot_occ = sum(all_pairs.values())
    non_disease = {(u, v) for pid, es in strict.items() if not pid.startswith("hsa05") for u, v in es}
    dis_occ = [(u, v) for pid, es in strict.items() if pid.startswith("hsa05") for u, v in es]
    cart = [r for r in rows if r["edge_origin"] == "gene_entry"]
    genes_per_entry = [len(e["hsa"]) for p in ctx.pw.values() for e in p["entries"].values()
                       if e["entry_type"] == "gene"]
    gnodes = {g for es in strict.values() for e in es for g in e}
    kn = {k for es in ko_graph.values() for e in es for k in e}
    ctx.save_metrics("gene_graph", {
        "pathways_with_gene_directed_graph": len(usable),
        "gene_edges_rows_dedup": len(rows),
        "gene_edges_rows_gene_entry": len(cart),
        "gene_edges_rows_group_expanded": len(rows) - len(cart),
        "gene_directed_edges_pathway_occurrences_strict": sum(r["gene_edges"] for r in srows),
        "gene_directed_edges_pathway_occurrences_incl_group": sum(r["gene_edges_incl_group_expansion"] for r in srows),
        "entry_level_gene_relations": sum(r["entry_level_gene_relations"] for r in srows),
        "unique_gene_pairs_global": len(all_pairs),
        "unique_genes_in_directed_graph": len(gnodes),
        "self_loops_removed": self_loops,
        "gene_branch_points_pathway_occurrences": sum(r["gene_branch_nodes"] for r in srows),
        "gene_merge_points_pathway_occurrences": sum(r["gene_merge_nodes"] for r in srows),
        "gene_sources_total": sum(r["gene_sources"] for r in srows),
        "gene_sinks_total": sum(r["gene_sinks"] for r in srows),
        "pathways_with_cycles": sum(1 for r in usable if r["gene_cyclic_scc_count"] > 0),
        "cyclic_sccs_total": sum(r["gene_cyclic_scc_count"] for r in srows),
        "largest_scc_any_pathway": max(r["gene_largest_scc"] for r in srows),
        "median_nodes_per_usable_pathway": med("gene_nodes"),
        "median_edges_per_usable_pathway": med("gene_edges"),
        "max_edges_one_pathway": max(r["gene_edges"] for r in srows),
        "mean_genes_per_gene_entry": statistics.mean(genes_per_entry),
        "gene_entries_with_multiple_genes_frac": sum(1 for x in genes_per_entry if x > 1) / len(genes_per_entry),
        "gene_edge_rows_from_multi_gene_entries_frac": sum(1 for r in cart if r["max_entry_cartesian_factor"] > 1) / len(cart),
        "edge_occurrences_whose_pair_is_in_2plus_pathways_frac": occ_in_multi / tot_occ,
        "disease_pathway_edge_occurrences": len(dis_occ),
        "disease_edges_also_in_non_disease_pathways_frac": (sum(1 for e in dis_occ if e in non_disease) / len(dis_occ)) if dis_occ else 0,
        "ko_graph_edges_pathway_occurrences": sum(len(v) for v in ko_graph.values()),
        "ko_graph_unique_pairs_global": len({e for es in ko_graph.values() for e in es}),
        "ko_graph_unique_nodes": len(kn),
        "ko_graph_pathways": len(ko_graph),
        "gene_edges_dropped_no_ko": dropped_nokos, "ko_self_pairs_dropped": ko_self,
    })
    log(f"gene graph: {len(usable)} pathways with directed gene graphs")


# =========================================================================== stage: sampling
class StreamCSV:
    """gzip CSV written incrementally to <name>.tmp and renamed on close."""

    def __init__(self, path, header):
        import csv
        import gzip
        self.path, self.tmp = path, path.with_name(path.name + ".tmp")
        self.f = gzip.open(self.tmp, "wt", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow(header)
        self.n = 0

    def row(self, r):
        self.w.writerow(r)
        self.n += 1

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()
        self.tmp.replace(self.path)


def build_vocab(tokens, name, extra=None):
    vocab = {t: i for i, t in enumerate(SPECIAL)}
    for t in sorted(tokens, key=lambda x: (len(x), x)):
        vocab[t] = len(vocab)
    rows = [{"token_id": i, "token": t, **(extra(t) if extra and t not in SPECIAL else {})}
            for t, i in vocab.items()]
    write_csv(OUT / f"token_vocab_{name}.csv", rows,
              list(rows[-1].keys()) if rows else ["token_id", "token"])
    return vocab


def run_sampling(ctx, graphs, level, vocab, fmt):
    a = ctx.args
    lengths = [int(x) for x in a.window_lengths.split(",")]
    npw = len(ctx.pw)
    stats = {}
    per_pw_rows = defaultdict(dict)

    def tok_ids(seq):
        return " ".join(str(vocab.get(t, 2)) for t in seq)

    # ---------- A: next-node
    A_rows, per, keys = [], Counter(), Counter()
    pw_of = defaultdict(set)
    for pid in sorted(graphs):
        for u, v in sorted(graphs[pid]):
            A_rows.append({"pathway_id": pid, "context_tokens": fmt(u), "target_token": fmt(v),
                           "context_token_ids": vocab[u], "target_token_id": vocab[v]})
            per[pid] += 1
            keys[(u, v)] += 1
            pw_of[(u, v)].add(pid)
    write_csv(SAMPLES / f"{level}_A_next_node.csv.gz", A_rows, list(A_rows[0].keys()))
    outs = defaultdict(set)
    for u, v in keys:
        outs[u].add(v)
    tgt_count = Counter(v for (u, v) in keys.elements())
    stats["A_next_node"] = dict(S.concentration(per, npw), unique_samples=len(keys),
                                multi_pathway_sample_frac=sum(c for c in keys.values() if c > 1) / sum(keys.values()),
                                ambiguous_context_frac=sum(1 for u in outs if len(outs[u]) > 1) / len(outs),
                                mean_valid_targets_per_context=statistics.mean(len(x) for x in outs.values()),
                                rare_target_tokens_lt5=sum(1 for t in vocab if t not in SPECIAL and tgt_count[t] < 5),
                                vocab_size=len(vocab) - len(SPECIAL))
    for pid, c in per.items():
        per_pw_rows[pid][f"{level}_A_next_node"] = c
    log(f"[{level}] A next-node: {sum(per.values()):,} samples")

    # ---------- B/C: windows and masked
    path_w = SAMPLES / f"{level}_B_path_windows.csv.gz"
    mask_w = SAMPLES / f"{level}_C_masked.csv.gz"
    bw, cw = StreamCSV(path_w, ["path_id", "pathway_id", "length", "mode", "token_ids", "tokens"]), \
        StreamCSV(mask_w, ["path_id", "pathway_id", "length", "mask_position", "context_token_ids", "target_token_id"])
    Bkeys = {L: Counter() for L in lengths}
    Ckeys = {L: Counter() for L in lengths}
    Cctx_targets = defaultdict(set)
    Bper = {L: Counter() for L in lengths}
    Cper = {L: Counter() for L in lengths}
    cap_rows = []
    path_id = 0
    t0 = time.time()
    for pid in sorted(graphs):
        es = graphs[pid]
        for L in lengths:
            res = S.sample_paths(es, L, a.max_paths_per_pathway, f"{a.seed}|{level}|{pid}|{L}",
                                 enum_limit=a.enum_walk_limit)
            cap_rows.append({"level": level, "pathway_id": pid, "L": L, "walks_upper_bound": res["walks"],
                             "total_simple_paths": res["total_simple"],
                             "total_is_exact": int(res["total_simple_is_exact"]),
                             "retained_paths": len(res["paths"]), "mode": res["mode"]})
            for path in res["paths"]:
                path_id += 1
                bw.row([path_id, pid, L, res["mode"], tok_ids(path), " ".join(fmt(x) for x in path)])
                Bkeys[L][path] += 1
                Bper[L][pid] += 1
                for mctx, pos, tgt in S.masked_from_path(path):
                    cw.row([path_id, pid, L, pos, tok_ids(mctx), vocab[tgt]])
                    Ckeys[L][(mctx, tgt)] += 1
                    Cctx_targets[mctx].add(tgt)
                    Cper[L][pid] += 1
        bw.flush()
        cw.flush()
    bw.close()
    cw.close()
    log(f"[{level}] B/C sampling done in {time.time() - t0:.0f}s: {bw.n:,} paths, {cw.n:,} masked")
    write_csv(OUT / f"path_enumeration_{level}.csv", cap_rows, list(cap_rows[0].keys()))
    for L in lengths:
        for nm, keyset, per_ in (("B_windows", Bkeys, Bper), ("C_masked", Ckeys, Cper)):
            kk = keyset[L]
            st = dict(S.concentration(per_[L], npw), unique_samples=len(kk),
                      multi_pathway_sample_frac=(sum(c for c in kk.values() if c > 1) / sum(kk.values())) if kk else 0)
            if nm == "C_masked":
                ctxs = {c for c, _ in kk}
                st["ambiguous_context_frac"] = (sum(1 for c in ctxs if len(Cctx_targets[c]) > 1) / len(ctxs)) if ctxs else 0
            capped = [r for r in cap_rows if r["L"] == L and r["mode"] != "exhaustive" and r["mode"] != "none"]
            st["pathways_capped_or_sampled"] = len(capped)
            st["total_simple_paths_all_pathways"] = sum(r["total_simple_paths"] for r in cap_rows if r["L"] == L)
            st["total_simple_paths_includes_estimates"] = int(any(not r["total_is_exact"] for r in cap_rows if r["L"] == L))
            stats[f"{nm}_L{L}"] = st
            for pid, c in per_[L].items():
                per_pw_rows[pid][f"{level}_{nm}_L{L}"] = c
        # cumulative "max length" settings: union of lengths 3..L
    for L in lengths:
        ls = [x for x in lengths if x <= L]
        for nm, keyset, per_ in (("B_windows", Bkeys, Bper), ("C_masked", Ckeys, Cper)):
            per_c = Counter()
            for x in ls:
                per_c.update(per_[x])
            uniq = sum(len(keyset[x]) for x in ls)  # different lengths never collide
            stats[f"{nm}_maxL{L}"] = dict(S.concentration(per_c, npw), unique_samples=uniq)
    return stats, per_pw_rows


def stage_sampling(ctx):
    gtoks = {g for es in ctx.gene_graph.values() for e in es for g in e}
    gvocab = build_vocab(gtoks, "gene", lambda t: {"symbol": ctx.sym(t)})
    ktoks = {k for es in ctx.ko_graph.values() for e in es for k in e}
    kvocab = build_vocab(ktoks, "ko", lambda t: {"ko_name": ctx.ko_names.get(t, "")})
    gstats, gper = run_sampling(ctx, ctx.gene_graph, "gene", gvocab, lambda t: f"hsa:{t}")
    kstats, kper = run_sampling(ctx, ctx.ko_graph, "ko", kvocab, lambda t: f"ko:{t}")
    rows = []
    for level, st in (("gene", gstats), ("ko", kstats)):
        for strat, d in st.items():
            rows.append({"level": level, "strategy": strat, **d})
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("level", "strategy", "total_samples", "unique_samples"), k))
    write_csv(OUT / "sampling_strategy_stats.csv", rows, fields)
    cols = sorted({c for d in (gper, kper) for v in d.values() for c in v})
    prow = []
    for pid, p in ctx.pw.items():
        r = {"pathway_id": pid, "pathway_name": p["pathway_name"], "pathway_class_a": p["class_a"]}
        for c in cols:
            r[c] = gper.get(pid, {}).get(c, 0) or kper.get(pid, {}).get(c, 0)
        prow.append(r)
    write_csv(OUT / "samples_per_pathway.csv", prow, ["pathway_id", "pathway_name", "pathway_class_a"] + cols)
    ctx.save_metrics("sampling", {"gene": gstats, "ko": kstats})


# =========================================================================== stage: metabolic
def stage_metabolic(ctx):
    rows_stats, edge_rows = [], []
    allC, allR, allC_glob, allR_glob = set(), set(), set(), set()
    rev = irr = other = 0
    tot_sp_edges = tot_sp_edges_nonglobal = 0
    for pid, p in ctx.pw.items():
        rx = p["reactions"]
        glob = pid.startswith(GLOBAL_MAP_PREFIX)
        C, R, cc_edges, bip = set(), set(), set(), set()
        cons = defaultdict(set)
        for r in rx:
            t = r["reaction_type"]
            rev += t == "reversible"
            irr += t == "irreversible"
            other += t not in ("reversible", "irreversible")
            subs = [n for _, n in r["substrates"]]
            prods = [n for _, n in r["products"]]
            rid = "+".join(r["reaction_ids"])
            C.update(subs + prods)
            R.update(r["reaction_ids"])
            for s in subs:
                cons[s].add(rid)
                bip.add((s, rid))
                for q in prods:
                    cc_edges.add((s, q, "forward"))
                    edge_rows.append({"pathway_id": pid, "reaction_ids": rid, "reaction_type": t,
                                      "kgml_reaction_entry": r["enzyme_entry"], "substrate": s, "product": q,
                                      "direction": "forward"})
            for q in prods:
                bip.add((rid, q))
            if t == "reversible":
                for q in prods:
                    cons[q].add(rid)
                    for s in subs:
                        cc_edges.add((q, s, "reverse"))
                        edge_rows.append({"pathway_id": pid, "reaction_ids": rid, "reaction_type": t,
                                          "kgml_reaction_entry": r["enzyme_entry"], "substrate": q, "product": s,
                                          "direction": "reverse_of_reversible"})
        simple = {(a, b) for a, b, _ in cc_edges if a != b}
        st = S.graph_stats(simple)
        (allC_glob if glob else allC).update(C)
        (allR_glob if glob else allR).update(R)
        tot_sp_edges += len(simple)
        tot_sp_edges_nonglobal += 0 if glob else len(simple)
        rows_stats.append({
            "pathway_id": pid, "pathway_name": p["pathway_name"], "pathway_class_a": p["class_a"],
            "is_global_overview_map": int(glob), "reaction_records": len(rx),
            "reversible": sum(1 for r in rx if r["reaction_type"] == "reversible"),
            "irreversible": sum(1 for r in rx if r["reaction_type"] == "irreversible"),
            "unique_compounds_in_reactions": len(C), "unique_reaction_ids": len(R),
            "compound_reaction_bipartite_edges": len(bip),
            "substrate_product_edges_directed": len(simple),
            "compound_branch_points": st["branch_nodes"], "compound_merge_points": st["merge_nodes"],
            "compounds_consumed_by_2plus_reactions": sum(1 for c in cons.values() if len(c) > 1),
            "compound_graph_cyclic_sccs": st["cyclic_scc_count"],
            "compound_graph_largest_scc": st["largest_scc"],
            "has_gene_directed_graph": int(bool(ctx.gene_graph.get(pid))),
        })
    write_csv(OUT / "metabolic_graph_stats_by_pathway.csv", rows_stats, list(rows_stats[0].keys()))
    write_csv(OUT / "metabolic_compound_edges.csv", edge_rows, list(edge_rows[0].keys()))
    withr = [r for r in rows_stats if r["reaction_records"] > 0]
    ctx.save_metrics("metabolic", {
        "pathways_with_reaction_records": len(withr),
        "pathways_with_reactions_excluding_global_maps": sum(1 for r in withr if not r["is_global_overview_map"]),
        "pathways_with_reactions_but_no_gene_directed_graph": sum(1 for r in withr if not r["has_gene_directed_graph"]),
        "reaction_records": sum(r["reaction_records"] for r in rows_stats),
        "reaction_records_reversible": rev, "reaction_records_irreversible": irr, "reaction_records_other": other,
        "unique_compound_ids_in_reactions": len(allC | allC_glob),
        "unique_compound_ids_in_reactions_non_global_maps": len(allC),
        "unique_reaction_ids": len(allR | allR_glob),
        "unique_reaction_ids_non_global_maps": len(allR),
        "reaction_ids_only_in_global_maps": len(allR_glob - allR),
        "substrate_product_edges_pathway_occurrences": tot_sp_edges,
        "substrate_product_edges_non_global_maps": tot_sp_edges_nonglobal,
        "compound_branch_points_pathway_occurrences": sum(r["compound_branch_points"] for r in rows_stats),
        "compound_branch_points_non_global": sum(r["compound_branch_points"] for r in rows_stats if not r["is_global_overview_map"]),
        "compound_merge_points_pathway_occurrences": sum(r["compound_merge_points"] for r in rows_stats),
        "global_maps_reaction_records": sum(r["reaction_records"] for r in rows_stats if r["is_global_overview_map"]),
        "global_maps_reversible_frac": (sum(r["reversible"] for r in rows_stats if r["is_global_overview_map"]) /
                                        max(1, sum(r["reaction_records"] for r in rows_stats if r["is_global_overview_map"]))),
        "compound_occurrences_nodes_total": ctx.metrics["kgml"]["node_occurrences_by_type"].get("compound", 0),
    })
    log(f"metabolic: {len(withr)} pathways with reaction records")


# =========================================================================== stage: uniprot / sequences
def stage_uniprot(ctx):
    a = ctx.args
    genes = ctx.all_genes
    conv = A.parse_uniprot_conv()
    seq = A.fetch_kegg_aaseq(genes, a.sleep, a.offline)
    ok = A.fetch_bulk_files(False, a.offline, not a.no_gene2go)
    ctx.bulk_ok = ok
    up = A.parse_uniprot_table()
    up_by_gene = defaultdict(set)
    for acc, d in up.items():
        for g in d["geneids"]:
            up_by_gene[g].add(acc)
    rows, pairs = [], []
    with (OUT / "gene_sequences_kegg.faa").open("w") as fa:
        for g in gsort(genes):
            accs = sorted(conv.get(g, ()))
            info = ctx.gene_info.get(g, {})
            s = seq.get(g, {"status": "unchecked", "seq": ""})
            rev = [x for x in accs if up.get(x, {}).get("reviewed") == "reviewed"]
            unrev = [x for x in accs if up.get(x, {}).get("reviewed") == "unreviewed"]
            notin = [x for x in accs if x not in up]
            extra = sorted(up_by_gene.get(g, set()) - set(accs))
            if accs:
                reason = ""
            elif info.get("gene_type") and info["gene_type"] != "CDS":
                reason = f"non-protein-coding gene in KEGG ({info['gene_type']})"
            elif not info:
                reason = "gene absent from current KEGG list/hsa (withdrawn/renamed?)"
            else:
                reason = "protein-coding but no KEGG conv/uniprot link"
            if s["status"] == "yes":
                fa.write(f">hsa:{g} {info.get('symbol', '')}\n{s['seq']}\n")
            rows.append({
                "kegg_gene_id": f"hsa:{g}", "ncbi_gene_id": g, "gene_symbol": info.get("symbol", ""),
                "kegg_gene_type": info.get("gene_type", ""),
                "uniprot_accessions": J(accs), "n_uniprot_accessions": len(accs),
                "mapping_success": int(bool(accs)), "one_to_many": int(len(accs) > 1),
                "reviewed_swissprot_accessions": J(rev), "unreviewed_trembl_accessions": J(unrev),
                "accessions_not_in_current_uniprot": J(notin),
                "uniprot_lengths": J(up[x]["length"] for x in accs if x in up),
                "mapping_failure_reason": reason,
                "uniprot_side_extra_accessions_by_geneid_xref": J(extra),
                "kegg_aaseq_status": s["status"], "kegg_aaseq_available": int(s["status"] == "yes"),
                "kegg_aaseq_length": len(s["seq"]) if s["status"] == "yes" else "",
            })
            for x in accs:
                pairs.append({"kegg_gene_id": f"hsa:{g}", "gene_symbol": info.get("symbol", ""),
                              "uniprot_accession": x, "reviewed": up.get(x, {}).get("reviewed", "not_in_current_uniprot"),
                              "uniprot_length": up.get(x, {}).get("length", ""), "source": "KEGG conv/uniprot/hsa"})
    write_csv(OUT / "gene_uniprot_mapping.csv", rows, list(rows[0].keys()))
    write_csv(OUT / "gene_uniprot_pairs.csv", pairs, list(pairs[0].keys()))
    ctx.gene_rows = {r["ncbi_gene_id"]: r for r in rows}
    ctx.conv = conv
    ctx.up = up
    ngenes = len(rows)
    mapped = [r for r in rows if r["mapping_success"]]
    ctx.save_metrics("uniprot", {
        "unique_pathway_genes": ngenes,
        "genes_mapped_to_uniprot": len(mapped),
        "uniprot_coverage": len(mapped) / ngenes,
        "genes_one_to_many_uniprot": sum(r["one_to_many"] for r in rows),
        "max_uniprot_per_gene": max(r["n_uniprot_accessions"] for r in rows),
        "gene_uniprot_pairs": len(pairs),
        "genes_with_reviewed_accession": sum(1 for r in rows if r["reviewed_swissprot_accessions"]),
        "genes_with_accession_not_in_current_uniprot": sum(1 for r in rows if r["accessions_not_in_current_uniprot"]),
        "unmapped_genes_by_reason": dict(Counter(r["mapping_failure_reason"] for r in rows if not r["mapping_success"])),
        "unmapped_genes_recoverable_via_uniprot_geneid_xref": sum(1 for r in rows if not r["mapping_success"] and r["uniprot_side_extra_accessions_by_geneid_xref"]),
        "genes_with_kegg_aaseq": sum(r["kegg_aaseq_available"] for r in rows),
        "kegg_aaseq_coverage": sum(r["kegg_aaseq_available"] for r in rows) / ngenes,
        "genes_aaseq_status_error": sum(1 for r in rows if r["kegg_aaseq_status"] == "error"),
        "aaseq_length_median": statistics.median([r["kegg_aaseq_length"] for r in rows if r["kegg_aaseq_available"]] or [0]),
        "kegg_gene_type_counts": dict(Counter(r["kegg_gene_type"] or "NOT_IN_LIST" for r in rows)),
        "uniprot_table_available": int(bool(up)),
    })


# =========================================================================== stage: GO
def stage_go(ctx):
    genes = ctx.all_genes
    obo = A.parse_obo()
    acc2gene = defaultdict(set)
    for g in genes:
        for x in ctx.conv.get(g, ()):
            acc2gene[x].add(g)
    rows = []
    goa_ok = A.GOA_GAF.exists()
    g2g_ok = A.GENE2GO_HUMAN.exists()
    for r in A.iter_goa(set(acc2gene)):
        for g in acc2gene[r["base_accession"]]:
            t = obo.get(r["go_id"], {})
            rows.append({"kegg_gene_id": f"hsa:{g}", "gene_symbol": ctx.sym(g), "uniprot_accession": r["accession"],
                         "go_id": r["go_id"], "go_name": t.get("name", ""),
                         "go_aspect": r["aspect"], "evidence_code": r["evidence"], "qualifier": r["qualifier"],
                         "is_negated": int(r["qualifier"].startswith("NOT")), "reference": r["reference"],
                         "assigned_by": r["assigned_by"], "go_term_obsolete": int(t.get("obsolete", False)),
                         "source": "EBI_GOA_goa_human.gaf"})
    for r in A.iter_gene2go(genes):
        t = obo.get(r["go_id"], {})
        rows.append({"kegg_gene_id": f"hsa:{r['gene']}", "gene_symbol": ctx.sym(r["gene"]), "uniprot_accession": "",
                     "go_id": r["go_id"], "go_name": t.get("name", r["term"]),
                     "go_aspect": r["aspect"], "evidence_code": r["evidence"], "qualifier": r["qualifier"],
                     "is_negated": int(r["qualifier"].startswith("NOT")), "reference": r["pubmed"],
                     "assigned_by": "", "go_term_obsolete": int(t.get("obsolete", False)),
                     "source": "NCBI_gene2go"})
    fields = ["kegg_gene_id", "gene_symbol", "uniprot_accession", "go_id", "go_name", "go_aspect",
              "evidence_code", "qualifier", "is_negated", "reference", "assigned_by", "go_term_obsolete", "source"]
    write_csv(OUT / "gene_go_mapping.csv", rows, fields)
    log(f"gene_go_mapping.csv: {len(rows):,} annotation rows")

    # collapsed positive gene-GO pairs (input for GoBERT-style embeddings later)
    pos = defaultdict(lambda: {"src": set(), "ev": set(), "aspect": "", "name": ""})
    for r in rows:
        if r["is_negated"]:
            continue
        d = pos[(r["kegg_gene_id"], r["go_id"])]
        d["src"].add(r["source"])
        d["ev"].add(r["evidence_code"])
        d["aspect"], d["name"] = r["go_aspect"], r["go_name"]
    crow = [{"kegg_gene_id": g, "gene_symbol": ctx.sym(g[4:]), "go_id": go, "go_name": d["name"], "go_aspect": d["aspect"],
             "sources": J(sorted(d["src"])), "evidence_codes": J(sorted(d["ev"])),
             "electronic_only": int(d["ev"] == {"IEA"})}
            for (g, go), d in sorted(pos.items())]
    write_csv(OUT / "gene_go_terms_collapsed.csv", crow, list(crow[0].keys()) if crow else ["kegg_gene_id"])

    def summ(filter_fn, label):
        per = defaultdict(set)
        asp = Counter()
        for r in crow:
            if filter_fn(r):
                per[r["kegg_gene_id"]].add(r["go_id"])
                asp[r["go_aspect"]] += 1
        counts = [len(v) for v in per.values()]
        aspg = {a_: len({r["kegg_gene_id"] for r in crow if filter_fn(r) and r["go_aspect"] == a_}) for a_ in ("MF", "BP", "CC")}
        return {f"{label}_genes_with_go": len(per),
                f"{label}_go_coverage": len(per) / len(genes),
                f"{label}_unique_go_terms": len({r["go_id"] for r in crow if filter_fn(r)}),
                f"{label}_mean_go_per_mapped_gene": statistics.mean(counts) if counts else 0,
                f"{label}_median_go_per_mapped_gene": statistics.median(counts) if counts else 0,
                f"{label}_gene_go_pairs_by_aspect": dict(asp),
                f"{label}_genes_with_aspect": aspg}
    m = {"unique_pathway_genes": len(genes), "goa_file_available": int(goa_ok), "gene2go_available": int(g2g_ok),
         "annotation_rows": len(rows), "annotation_rows_by_source": dict(Counter(r["source"] for r in rows)),
         "negated_rows": sum(r["is_negated"] for r in rows),
         "obsolete_term_rows": sum(r["go_term_obsolete"] for r in rows)}
    m.update(summ(lambda r: True, "union"))
    m.update(summ(lambda r: "EBI_GOA_goa_human.gaf" in r["sources"], "goa"))
    m.update(summ(lambda r: "NCBI_gene2go" in r["sources"], "gene2go"))
    m.update(summ(lambda r: not r["electronic_only"], "union_nonIEA"))
    gg = {g for es in ctx.gene_graph.values() for e in es for g in e}
    with_go = {r["kegg_gene_id"][4:] for r in crow}
    m["directed_graph_genes"] = len(gg)
    m["directed_graph_genes_with_go"] = len(gg & with_go)
    m["genes_without_go_by_type"] = dict(Counter(ctx.gene_info.get(g, {}).get("gene_type", "NOT_IN_LIST")
                                                 for g in genes - with_go))
    m["uniprot_mapped_genes_without_goa"] = len({g for g in genes if ctx.conv.get(g)} -
                                               {r["kegg_gene_id"][4:] for r in crow if "EBI_GOA_goa_human.gaf" in r["sources"]})
    ctx.go_by_gene = defaultdict(set)
    for r in crow:
        ctx.go_by_gene[r["kegg_gene_id"][4:]].add(r["go_id"])
    ctx.save_metrics("go", m)


# =========================================================================== stage: KO
def stage_ko(ctx):
    a = ctx.args
    ko_direct_pw = defaultdict(set)
    ko_via_gene_pw = defaultdict(set)
    ko_genes_in_pw = defaultdict(set)
    for pid, p in ctx.pw.items():
        for e in p["entries"].values():
            if e["entry_type"] == "ortholog":
                for k in e["ko"]:
                    ko_direct_pw[k].add(pid)
            elif e["entry_type"] == "gene":
                for g in e["hsa"]:
                    for k in ctx.gene2ko.get(g, ()):
                        ko_via_gene_pw[k].add(pid)
                        ko_genes_in_pw[k].add(g)
    n_direct, n_via = len(ko_direct_pw), len(ko_via_gene_pw)  # before defaultdict lookups add keys
    kos = set(ko_direct_pw) | set(ko_via_gene_pw)
    ent = A.fetch_ko_entries(kos, a.sleep, a.offline)
    go_by_gene = getattr(ctx, "go_by_gene", {})
    rows = []
    for k in sorted(kos):
        hg = ctx.ko2genes_all.get(k, set())
        e = ent.get(k, {"status": "not_fetched", "go": []})
        derived_go = set().union(*[go_by_gene.get(g, set()) for g in ko_genes_in_pw[k]]) if ko_genes_in_pw[k] else set()
        pws = ko_direct_pw[k] | ko_via_gene_pw[k]
        mods = sorted(ctx.ko2modules.get(k, ()))
        rows.append({
            "ko_id": k, "ko_name": ctx.ko_names.get(k, ""),
            "pathways_as_ortholog_node": J(sorted(ko_direct_pw[k])),
            "pathways_via_human_gene_nodes": J(sorted(ko_via_gene_pw[k])),
            "n_pathways": len(pws), "modules": J(mods), "n_modules": len(mods),
            "human_genes_all": J(gsort(hg)), "n_human_genes": len(hg),
            "human_genes_in_pathways": J(gsort(ko_genes_in_pw[k])), "n_human_genes_in_pathways": len(ko_genes_in_pw[k]),
            "human_gene_symbols": J(ctx.sym(g) for g in gsort(hg)),
            "ko_go_ids_kegg_dblinks": J(e["go"]), "n_ko_go_ids": len(e["go"]),
            "ko_entry_fetch_status": e["status"],
            "n_go_union_of_member_genes_derived": len(derived_go),
        })
    write_csv(OUT / "ko_mapping.csv", rows, list(rows[0].keys()))
    ok = [r for r in rows if r["ko_entry_fetch_status"] == "ok"]
    ctx.save_metrics("ko", {
        "unique_ko_ids": len(rows),
        "ko_ids_as_ortholog_nodes": n_direct,
        "ko_ids_via_human_genes": n_via,
        "ko_ids_both_ortholog_node_and_via_genes": n_direct + n_via - len(kos),
        "ko_ids_via_genes_one_pathway_gene": sum(1 for r in rows if r["n_human_genes_in_pathways"] == 1),
        "ko_ids_via_genes_multiple_pathway_genes": sum(1 for r in rows if r["n_human_genes_in_pathways"] > 1),
        "ko_ids_only_as_ortholog_nodes_no_human_gene": sum(1 for r in rows if r["n_human_genes"] == 0),
        "ko_ids_with_human_gene_mapping": sum(1 for r in rows if r["n_human_genes"] > 0),
        "ko_ids_mapping_one_human_gene": sum(1 for r in rows if r["n_human_genes"] == 1),
        "ko_ids_mapping_multiple_human_genes": sum(1 for r in rows if r["n_human_genes"] > 1),
        "max_human_genes_per_ko": max(r["n_human_genes"] for r in rows),
        "ko_entries_fetched_ok": len(ok),
        "ko_entries_fetch_error_or_missing": len(rows) - len(ok),
        "ko_ids_with_go_kegg_dblinks": sum(1 for r in ok if r["n_ko_go_ids"] > 0),
        "ko_ids_with_go_via_member_genes_derived": sum(1 for r in rows if r["n_go_union_of_member_genes_derived"] > 0),
        "ko_ids_in_modules": sum(1 for r in rows if r["n_modules"] > 0),
        "pathway_genes_with_ko": sum(1 for g in ctx.all_genes if ctx.gene2ko.get(g)),
        "pathway_genes_with_multiple_ko": sum(1 for g in ctx.all_genes if len(ctx.gene2ko.get(g, ())) > 1),
        "pathway_genes_without_ko": sum(1 for g in ctx.all_genes if not ctx.gene2ko.get(g)),
    })


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-paths-per-pathway", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window-lengths", default="3,4,5,6")
    ap.add_argument("--enum-walk-limit", type=int, default=300000,
                    help="exhaustive enumeration if #walks <= this; else walk sampling")
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-gene2go", action="store_true")
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args()
    want = set(args.stages.split(","))
    ctx = Ctx(args)
    log(f"=== KEGG pathway dataset builder === stages={args.stages} seed={args.seed} "
        f"cap={args.max_paths_per_pathway}")
    # kgml + gene_graph are cheap and required by all later stages
    stage_kgml(ctx)
    stage_gene_graph(ctx)
    for name in STAGES[2:]:
        if name in want or (name in ("uniprot",) and ({"go", "ko", "report"} & want)):
            log(f"--- stage {name}")
            if name == "report":
                from kegg_pipeline import report
                report.write_report(ctx)
            else:
                globals()[f"stage_{name}"](ctx)
    log("done")


if __name__ == "__main__":
    main()

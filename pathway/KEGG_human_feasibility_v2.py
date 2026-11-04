#!/usr/bin/env python3
"""
KEGG Human Pathway Feasibility Audit v2

Purpose
-------
Build a reproducible feasibility table for human KEGG pathways and estimate:
  1. number of pathways
  2. number of KGML entry nodes
  3. number of gene/protein-related nodes
  4. number of relation edges
  5. number of directed relation edges
  6. number of gene-level branch points
  7. number of pathway genes mapping to UniProt
  8. number of pathway genes with an amino-acid sequence available from KEGG
  9. number of candidate source-to-sink training routes after SCC compression

Key improvements over v1
------------------------
  * Resume-safe KGML cache: already-downloaded pathway XML files are reused.
  * Resume-safe sequence cache: every checked gene is saved to disk.
  * Batch sequence requests fall back to single-gene requests on failure.
  * Missing amino-acid sequences are distinguished from transient request errors.
  * Intermediate CSV files are written incrementally.
  * Final summary is written as both CSV and Markdown.

No third-party Python packages are required.

Run
---
    python KEGG_human_feasibility_v2.py

Optional:
    python KEGG_human_feasibility_v2.py --force-refresh
    python KEGG_human_feasibility_v2.py --sleep 0.40

Outputs
-------
kegg_feasibility_output/
    kegg_hsa_feasibility_summary.csv
    kegg_hsa_feasibility_summary.md
    kegg_hsa_feasibility_by_pathway.csv
    kegg_hsa_genes.csv

Cache
-----
kegg_feasibility_cache/
    pathway_list.tsv
    kgml/*.xml
    uniprot_conv.tsv
    sequence_status.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

BASE = "https://rest.kegg.jp"

OUTDIR = Path("kegg_feasibility_output")
CACHEDIR = Path("kegg_feasibility_cache")
KGML_CACHE = CACHEDIR / "kgml"

PATHWAY_LIST_CACHE = CACHEDIR / "pathway_list.tsv"
UNIPROT_CACHE = CACHEDIR / "uniprot_conv.tsv"
SEQ_STATUS_CACHE = CACHEDIR / "sequence_status.json"

OUTDIR.mkdir(exist_ok=True)
CACHEDIR.mkdir(exist_ok=True)
KGML_CACHE.mkdir(exist_ok=True)

GENE_RE = re.compile(r"\bhsa:(\d+)\b")

# Conservative directional subtypes.
# We intentionally do NOT treat binding/association or dissociation as directional.
DIRECTIONAL_SUBTYPES = {
    "activation",
    "inhibition",
    "expression",
    "repression",
    "indirect effect",
    "state change",
    "phosphorylation",
    "dephosphorylation",
    "glycosylation",
    "ubiquitination",
    "methylation",
}

PROTEIN_RELATED_ENTRY_TYPES = {"gene", "ortholog", "enzyme"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sleep",
        type=float,
        default=0.40,
        help="Seconds to sleep after successful KEGG requests. Default: 0.40",
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore existing download caches and fetch again.",
    )
    return p.parse_args()


def request_text(url: str, sleep_seconds: float, retries: int = 5) -> str:
    """
    GET text with retries.
    Raises urllib.error.HTTPError for final HTTP failures such as 404.
    """
    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "KEGG-human-feasibility-audit-v2/1.0"},
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = resp.read().decode("utf-8", errors="replace")
            time.sleep(sleep_seconds)
            return data

        except urllib.error.HTTPError as e:
            last_exc = e

            # 404 is usually a real "not available" result for a specific entry,
            # not something retries will fix.
            if e.code == 404:
                raise

            wait = min(30, 2 ** (attempt - 1))
            print(
                f"[HTTP {e.code}] attempt {attempt}/{retries}: {url}; "
                f"retrying in {wait}s",
                file=sys.stderr,
            )
            time.sleep(wait)

        except Exception as e:
            last_exc = e
            wait = min(30, 2 ** (attempt - 1))
            print(
                f"[request error] attempt {attempt}/{retries}: {url}; "
                f"{e}; retrying in {wait}s",
                file=sys.stderr,
            )
            time.sleep(wait)

    raise RuntimeError(f"Request failed after {retries} attempts: {url}\n{last_exc}")


def get_or_fetch_text(
    cache_path: Path,
    url: str,
    sleep_seconds: float,
    force_refresh: bool = False,
) -> str:
    if cache_path.exists() and not force_refresh:
        return cache_path.read_text(encoding="utf-8")

    text = request_text(url, sleep_seconds=sleep_seconds)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return text


def parse_pathway_list(text: str) -> List[Tuple[str, str]]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        pid, name = line.split("\t", 1)
        pid = pid.replace("path:", "")
        rows.append((pid, name))
    return rows


def extract_hsa_genes(name_attr: str) -> List[str]:
    return sorted(set(GENE_RE.findall(name_attr or "")), key=int)


def tarjan_scc(nodes: Set[str], edges: Set[Tuple[str, str]]) -> List[List[str]]:
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)

    index = 0
    stack = []
    onstack = set()
    indices = {}
    low = {}
    comps = []

    sys.setrecursionlimit(max(10000, len(nodes) * 10 + 100))

    def strongconnect(v):
        nonlocal index
        indices[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)

        for w in adj[v]:
            if w not in indices:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], indices[w])

        if low[v] == indices[v]:
            comp = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(w)
                if w == v:
                    break
            comps.append(comp)

    for n in nodes:
        if n not in indices:
            strongconnect(n)

    return comps


def count_scc_dag_routes(
    nodes: Set[str], edges: Set[Tuple[str, str]]
) -> Tuple[int, int, int]:
    """
    Count source-to-sink routes after strongly connected components are
    compressed into a DAG.

    Returns:
        route_count, scc_count, cyclic_scc_count
    """
    if not nodes:
        return 0, 0, 0

    comps = tarjan_scc(nodes, edges)

    comp_id = {}
    edge_set = set(edges)
    cyclic_scc_count = 0

    for i, comp in enumerate(comps):
        for n in comp:
            comp_id[n] = i
        if len(comp) > 1:
            cyclic_scc_count += 1
        elif (comp[0], comp[0]) in edge_set:
            cyclic_scc_count += 1

    out = defaultdict(set)
    indeg = defaultdict(int)

    for u, v in edges:
        cu, cv = comp_id[u], comp_id[v]
        if cu == cv:
            continue
        if cv not in out[cu]:
            out[cu].add(cv)
            indeg[cv] += 1

    all_comps = set(range(len(comps)))
    sources = [c for c in all_comps if indeg[c] == 0]
    sinks = {c for c in all_comps if not out[c]}

    # Topological order
    indeg_work = {c: indeg[c] for c in all_comps}
    q = deque(c for c in all_comps if indeg_work[c] == 0)
    topo = []

    while q:
        c = q.popleft()
        topo.append(c)
        for nxt in out[c]:
            indeg_work[nxt] -= 1
            if indeg_work[nxt] == 0:
                q.append(nxt)

    ways = {c: 0 for c in all_comps}
    for s in sources:
        ways[s] = 1

    for c in topo:
        for nxt in out[c]:
            ways[nxt] += ways[c]

    route_count = sum(ways[s] for s in sinks)
    return route_count, len(comps), cyclic_scc_count


def parse_kgml(pid: str, xml_text: str) -> dict:
    root = ET.fromstring(xml_text)

    entries = {}
    entry_type_counts = defaultdict(int)
    unique_genes = set()

    for e in root.findall("entry"):
        eid = e.attrib.get("id", "")
        etype = e.attrib.get("type", "")
        name = e.attrib.get("name", "")
        genes = extract_hsa_genes(name)

        entries[eid] = {
            "type": etype,
            "name": name,
            "genes": genes,
        }

        entry_type_counts[etype] += 1

        if etype == "gene":
            unique_genes.update(genes)

    relation_count = 0
    directional_relation_count = 0
    gene_gene_directed_edges = set()

    for rel in root.findall("relation"):
        relation_count += 1

        e1 = rel.attrib.get("entry1")
        e2 = rel.attrib.get("entry2")

        subtypes = [
            s.attrib.get("name", "").strip().lower()
            for s in rel.findall("subtype")
        ]

        is_directional = any(s in DIRECTIONAL_SUBTYPES for s in subtypes)
        if not is_directional:
            continue

        directional_relation_count += 1

        a = entries.get(e1)
        b = entries.get(e2)
        if not a or not b:
            continue

        if a["type"] == "gene" and b["type"] == "gene":
            for g1 in a["genes"]:
                for g2 in b["genes"]:
                    if g1 != g2:
                        gene_gene_directed_edges.add((g1, g2))

    graph_nodes = set()
    out_neighbors = defaultdict(set)

    for u, v in gene_gene_directed_edges:
        graph_nodes.add(u)
        graph_nodes.add(v)
        out_neighbors[u].add(v)

    branch_points = sum(
        1 for node in graph_nodes if len(out_neighbors[node]) > 1
    )

    route_count, scc_count, cyclic_scc_count = count_scc_dag_routes(
        graph_nodes, gene_gene_directed_edges
    )

    reaction_records = root.findall("reaction")
    reaction_count = len(reaction_records)
    irreversible_reaction_count = sum(
        1 for r in reaction_records if r.attrib.get("type") == "irreversible"
    )
    reversible_reaction_count = sum(
        1 for r in reaction_records if r.attrib.get("type") == "reversible"
    )

    return {
        "pathway_id": pid,
        "pathway_name": root.attrib.get("title", ""),
        "all_entry_nodes": len(entries),
        "gene_entry_nodes": entry_type_counts["gene"],
        "ortholog_entry_nodes": entry_type_counts["ortholog"],
        "enzyme_entry_nodes": entry_type_counts["enzyme"],
        "compound_entry_nodes": entry_type_counts["compound"],
        "group_entry_nodes": entry_type_counts["group"],
        "map_entry_nodes": entry_type_counts["map"],
        "protein_related_entry_nodes": sum(
            entry_type_counts[t] for t in PROTEIN_RELATED_ENTRY_TYPES
        ),
        "unique_hsa_genes": len(unique_genes),
        "relation_edges": relation_count,
        "directed_relation_edges": directional_relation_count,
        "gene_gene_directed_edges": len(gene_gene_directed_edges),
        "gene_branch_points": branch_points,
        "gene_graph_nodes": len(graph_nodes),
        "reaction_records": reaction_count,
        "irreversible_reaction_records": irreversible_reaction_count,
        "reversible_reaction_records": reversible_reaction_count,
        "scc_count": scc_count,
        "cyclic_scc_count": cyclic_scc_count,
        "candidate_training_routes_scc_dag": route_count,
        "_genes": unique_genes,
    }


def write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json_atomic(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def parse_uniprot_mapping(text: str) -> Dict[str, Set[str]]:
    mapping = defaultdict(set)

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        left, right = line.split("\t", 1)

        if left.startswith("hsa:"):
            gid = left.split(":", 1)[1]
            up = right.split(":", 1)[1] if ":" in right else right

        elif right.startswith("hsa:"):
            gid = right.split(":", 1)[1]
            up = left.split(":", 1)[1] if ":" in left else left

        else:
            continue

        mapping[gid].add(up)

    return mapping


def parse_fasta_gene_ids(text: str) -> Set[str]:
    found = set()
    for line in text.splitlines():
        if line.startswith(">"):
            m = re.search(r">hsa:(\d+)", line)
            if m:
                found.add(m.group(1))
    return found


def chunks(items: List[str], n: int):
    for i in range(0, len(items), n):
        yield items[i:i+n]


def check_single_sequence(
    gid: str,
    sleep_seconds: float,
) -> str:
    """
    Returns:
        "yes"       sequence exists
        "no"        KEGG returned 404 for /aaseq
        "error"     request failed for another reason
    """
    url = f"{BASE}/get/hsa:{gid}/aaseq"

    try:
        txt = request_text(url, sleep_seconds=sleep_seconds)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "no"
        return "error"
    except Exception:
        return "error"

    return "yes" if gid in parse_fasta_gene_ids(txt) else "no"


def get_sequence_status(
    gene_ids: Set[str],
    sleep_seconds: float,
) -> Dict[str, str]:
    """
    Resume-safe sequence availability audit.

    Cache values:
        yes
        no
        error

    Genes previously marked error are retried on the next run.
    """
    status = load_json(SEQ_STATUS_CACHE, {})

    # Keep only meaningful previous values.
    status = {
        str(k): v
        for k, v in status.items()
        if v in {"yes", "no", "error"}
    }

    pending = sorted(
        [g for g in gene_ids if status.get(g) not in {"yes", "no"}],
        key=int,
    )

    print(
        f"Sequence availability: "
        f"{sum(status.get(g) in {'yes','no'} for g in gene_ids)}/{len(gene_ids)} "
        f"already cached; {len(pending)} to check."
    )

    total_batches = (len(pending) + 9) // 10

    for batch_i, batch in enumerate(chunks(pending, 10), start=1):
        query = "+".join(f"hsa:{g}" for g in batch)
        url = f"{BASE}/get/{query}/aaseq"

        batch_success = False

        try:
            txt = request_text(url, sleep_seconds=sleep_seconds)
            found = parse_fasta_gene_ids(txt)

            for g in batch:
                # If the multi-entry response explicitly contains the gene,
                # it has an amino-acid sequence.
                if g in found:
                    status[g] = "yes"

            missing_from_batch_response = [g for g in batch if g not in found]

            # Confirm missing entries individually so one problematic gene does
            # not cause another valid sequence to be lost.
            for g in missing_from_batch_response:
                status[g] = check_single_sequence(g, sleep_seconds)

            batch_success = True

        except urllib.error.HTTPError as e:
            if e.code == 404:
                                # if one 10-gene batch returns 404, split it and verify each gene.
                for g in batch:
                    status[g] = check_single_sequence(g, sleep_seconds)
                batch_success = True
            else:
                print(
                    f"[warning] batch HTTP {e.code}: {batch}. "
                    f"Will retry individually.",
                    file=sys.stderr,
                )
                for g in batch:
                    status[g] = check_single_sequence(g, sleep_seconds)

        except Exception as e:
            print(
                f"[warning] batch failed: {batch}: {e}. "
                f"Will retry individually.",
                file=sys.stderr,
            )
            for g in batch:
                status[g] = check_single_sequence(g, sleep_seconds)

        save_json_atomic(SEQ_STATUS_CACHE, status)

        if (
            batch_i == 1
            or batch_i % 25 == 0
            or batch_i == total_batches
        ):
            yes = sum(status.get(g) == "yes" for g in gene_ids)
            no = sum(status.get(g) == "no" for g in gene_ids)
            err = sum(status.get(g) == "error" for g in gene_ids)
            print(
                f"Sequence batches: {batch_i}/{total_batches} | "
                f"yes={yes}, no={no}, error={err}"
            )

    return status


def main():
    args = parse_args()

    print("=== KEGG Human Pathway Feasibility Audit v2 ===")
    print(f"Request sleep: {args.sleep:.2f}s")
    print(f"Force refresh: {args.force_refresh}")
    print()

    # 1. Human pathway list
    pathway_list_text = get_or_fetch_text(
        PATHWAY_LIST_CACHE,
        f"{BASE}/list/pathway/hsa",
        sleep_seconds=args.sleep,
        force_refresh=args.force_refresh,
    )
    pathways = parse_pathway_list(pathway_list_text)
    print(f"Human pathways found: {len(pathways)}")

    # 2. KGML parsing with per-pathway cache
    per_pathway = []
    all_genes = set()

    path_fields = [
        "pathway_id",
        "listed_name",
        "pathway_name",
        "all_entry_nodes",
        "gene_entry_nodes",
        "ortholog_entry_nodes",
        "enzyme_entry_nodes",
        "compound_entry_nodes",
        "group_entry_nodes",
        "map_entry_nodes",
        "protein_related_entry_nodes",
        "unique_hsa_genes",
        "relation_edges",
        "directed_relation_edges",
        "gene_gene_directed_edges",
        "gene_branch_points",
        "gene_graph_nodes",
        "reaction_records",
        "irreversible_reaction_records",
        "reversible_reaction_records",
        "scc_count",
        "cyclic_scc_count",
        "candidate_training_routes_scc_dag",
    ]

    for i, (pid, listed_name) in enumerate(pathways, start=1):
        cache_path = KGML_CACHE / f"{pid}.xml"

        try:
            xml_text = get_or_fetch_text(
                cache_path,
                f"{BASE}/get/{pid}/kgml",
                sleep_seconds=args.sleep,
                force_refresh=args.force_refresh,
            )
            row = parse_kgml(pid, xml_text)

        except urllib.error.HTTPError as e:
            print(
                f"[warning] KGML unavailable for {pid}: HTTP {e.code}",
                file=sys.stderr,
            )
            continue

        except Exception as e:
            print(
                f"[warning] failed to parse {pid}: {e}",
                file=sys.stderr,
            )
            continue

        row["listed_name"] = listed_name
        all_genes.update(row["_genes"])
        per_pathway.append(row)

        # Incremental output: even if the server crashes, this CSV contains
        # everything successfully parsed so far.
        write_csv(
            OUTDIR / "kegg_hsa_feasibility_by_pathway.partial.csv",
            per_pathway,
            path_fields,
        )

        if i == 1 or i % 25 == 0 or i == len(pathways):
            cached = sum(
                1 for p, _ in pathways if (KGML_CACHE / f"{p}.xml").exists()
            )
            print(
                f"KGML progress: {i}/{len(pathways)} | "
                f"cached={cached} | unique genes so far={len(all_genes)}"
            )

    print(f"Successfully parsed pathways: {len(per_pathway)}")
    print(f"Unique human pathway genes: {len(all_genes)}")

    # Final per-pathway output
    write_csv(
        OUTDIR / "kegg_hsa_feasibility_by_pathway.csv",
        per_pathway,
        path_fields,
    )

    # 3. KEGG <-> UniProt mapping
    print("Loading KEGG -> UniProt mapping...")
    uniprot_text = get_or_fetch_text(
        UNIPROT_CACHE,
        f"{BASE}/conv/uniprot/hsa",
        sleep_seconds=args.sleep,
        force_refresh=args.force_refresh,
    )
    uniprot_map = parse_uniprot_mapping(uniprot_text)
    genes_mapped_to_uniprot = {
        g for g in all_genes if uniprot_map.get(g)
    }

    # 4. Amino-acid sequence availability, resume-safe
    print("Checking amino-acid sequence availability...")
    seq_status = get_sequence_status(
        all_genes,
        sleep_seconds=args.sleep,
    )
    genes_with_sequence = {
        g for g in all_genes if seq_status.get(g) == "yes"
    }
    genes_without_sequence = {
        g for g in all_genes if seq_status.get(g) == "no"
    }
    sequence_errors = {
        g for g in all_genes if seq_status.get(g) == "error"
    }

    # 5. Per-gene CSV
    gene_rows = []

    for g in sorted(all_genes, key=int):
        gene_rows.append(
            {
                "hsa_gene_id": g,
                "kegg_id": f"hsa:{g}",
                "mapped_to_uniprot": int(bool(uniprot_map.get(g))),
                "uniprot_accessions": ";".join(
                    sorted(uniprot_map.get(g, set()))
                ),
                "kegg_aaseq_status": seq_status.get(g, "unchecked"),
            }
        )

    write_csv(
        OUTDIR / "kegg_hsa_genes.csv",
        gene_rows,
        [
            "hsa_gene_id",
            "kegg_id",
            "mapped_to_uniprot",
            "uniprot_accessions",
            "kegg_aaseq_status",
        ],
    )

    # 6. Summary
    total_unique_genes = len(all_genes)

    summary = {
        "pathways_listed": len(pathways),
        "pathways_successfully_parsed": len(per_pathway),
        "entry_nodes_total_pathway_occurrences": sum(
            r["all_entry_nodes"] for r in per_pathway
        ),
        "gene_entry_nodes_total_pathway_occurrences": sum(
            r["gene_entry_nodes"] for r in per_pathway
        ),
        "protein_related_entry_nodes_total_pathway_occurrences": sum(
            r["protein_related_entry_nodes"] for r in per_pathway
        ),
        "compound_entry_nodes_total_pathway_occurrences": sum(
            r["compound_entry_nodes"] for r in per_pathway
        ),
        "unique_hsa_genes_in_pathways": total_unique_genes,
        "relation_edges_total": sum(
            r["relation_edges"] for r in per_pathway
        ),
        "directed_relation_edges_total": sum(
            r["directed_relation_edges"] for r in per_pathway
        ),
        "gene_gene_directed_edges_total_pathway_occurrences": sum(
            r["gene_gene_directed_edges"] for r in per_pathway
        ),
        "gene_branch_points_total_pathway_occurrences": sum(
            r["gene_branch_points"] for r in per_pathway
        ),
        "reaction_records_total": sum(
            r["reaction_records"] for r in per_pathway
        ),
        "irreversible_reaction_records_total": sum(
            r["irreversible_reaction_records"] for r in per_pathway
        ),
        "reversible_reaction_records_total": sum(
            r["reversible_reaction_records"] for r in per_pathway
        ),
        "unique_hsa_genes_mapped_to_uniprot": len(
            genes_mapped_to_uniprot
        ),
        "uniprot_mapping_coverage": (
            len(genes_mapped_to_uniprot) / total_unique_genes
            if total_unique_genes
            else 0.0
        ),
        "unique_hsa_genes_with_kegg_aaseq": len(
            genes_with_sequence
        ),
        "kegg_aaseq_coverage": (
            len(genes_with_sequence) / total_unique_genes
            if total_unique_genes
            else 0.0
        ),
        "unique_hsa_genes_without_kegg_aaseq": len(
            genes_without_sequence
        ),
        "sequence_request_errors_remaining": len(
            sequence_errors
        ),
        "candidate_training_routes_scc_dag_total": sum(
            r["candidate_training_routes_scc_dag"]
            for r in per_pathway
        ),
        "pathways_with_at_least_one_gene_directed_edge": sum(
            1 for r in per_pathway
            if r["gene_gene_directed_edges"] > 0
        ),
        "pathways_with_gene_graph_cycle": sum(
            1 for r in per_pathway
            if r["cyclic_scc_count"] > 0
        ),
    }

    # Summary CSV
    with (OUTDIR / "kegg_hsa_feasibility_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in summary.items():
            w.writerow([k, v])

    # Summary Markdown
    pretty_rows = [
        ("Human pathways listed", summary["pathways_listed"]),
        (
            "Human pathways parsed",
            summary["pathways_successfully_parsed"],
        ),
        (
            "All KGML nodes, pathway occurrences",
            summary["entry_nodes_total_pathway_occurrences"],
        ),
        (
            "Gene entry nodes, pathway occurrences",
            summary["gene_entry_nodes_total_pathway_occurrences"],
        ),
        (
            "Protein related nodes, pathway occurrences",
            summary["protein_related_entry_nodes_total_pathway_occurrences"],
        ),
        (
            "Unique human genes in pathways",
            summary["unique_hsa_genes_in_pathways"],
        ),
        (
            "Relation edges",
            summary["relation_edges_total"],
        ),
        (
            "Directed relation edges",
            summary["directed_relation_edges_total"],
        ),
        (
            "Gene to gene directed edges, pathway occurrences",
            summary["gene_gene_directed_edges_total_pathway_occurrences"],
        ),
        (
            "Gene branch points, pathway occurrences",
            summary["gene_branch_points_total_pathway_occurrences"],
        ),
        (
            "Unique genes mapped to UniProt",
            summary["unique_hsa_genes_mapped_to_uniprot"],
        ),
        (
            "UniProt mapping coverage",
            f'{summary["uniprot_mapping_coverage"]:.2%}',
        ),
        (
            "Unique genes with KEGG amino acid sequence",
            summary["unique_hsa_genes_with_kegg_aaseq"],
        ),
        (
            "KEGG amino acid sequence coverage",
            f'{summary["kegg_aaseq_coverage"]:.2%}',
        ),
        (
            "Sequence request errors remaining",
            summary["sequence_request_errors_remaining"],
        ),
        (
            "Candidate training routes after SCC compression",
            summary["candidate_training_routes_scc_dag_total"],
        ),
    ]

    md_lines = [
        "# KEGG Human Pathway Feasibility Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]

    for label, value in pretty_rows:
        md_lines.append(f"| {label} | {value} |")

    (OUTDIR / "kegg_hsa_feasibility_summary.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("================ FINAL SUMMARY ================")

    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print()
    print("Outputs:")
    print(OUTDIR / "kegg_hsa_feasibility_summary.csv")
    print(OUTDIR / "kegg_hsa_feasibility_summary.md")
    print(OUTDIR / "kegg_hsa_feasibility_by_pathway.csv")
    print(OUTDIR / "kegg_hsa_genes.csv")

    if sequence_errors:
        print()
        print(
            f"NOTE: {len(sequence_errors)} sequence requests still have "
            f'temporary status "error". Rerun the same command later; '
            f"the script will retry only those genes."
        )

    print()
    print("Safe to rerun: cached KGML and completed sequence checks will be reused.")


if __name__ == "__main__":
    main()

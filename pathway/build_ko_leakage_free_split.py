#!/usr/bin/env python3
"""
Leakage-controlled train/val/test split of the KO-level pathway samples.

Why not a random or plain pathway split
---------------------------------------
60% of edge occurrences repeat across pathways, and clustering pathways by edge overlap
yields one giant component (chained overlaps), so whole-cluster holdout is not possible.

Scheme ("pathway holdout + unique-edge ownership")
--------------------------------------------------
1. Pathways whose KO edge sets nearly coincide (overlap coefficient >= --group-threshold) form
   groups. Groups are assigned greedily (seeded, round-robin over KEGG classes) so that each
   split *owns* its target share (--ratios) of unique KO edges.
2. Every unique KO edge (a -> b) is owned by exactly one split:
      train if it occurs in any train pathway, else val if in any val pathway, else test.
   Hence no val/test edge ever occurs in train (shared edges are ceded to train).
3. Samples are rebuilt per split from each pathway's edges *owned by that split*:
      A next-node, B simple-path windows (L=3..6, cap per pathway per L), C masked interior node.
   Every edge of every val/test path is therefore val/test-owned.
4. Samples are deduplicated within a split (pathway_ids records all sources).
5. Checks: zero cross-split overlap of A pairs, B paths, C (context, target); no val/test
   sample contains a train edge; unseen-token and reverse-edge flags; filtered-ranking sets.

Inputs (read only): kegg_dataset/gene_edges.csv, kegg_dataset/token_vocab_ko.csv,
kegg_dataset/gene_graph_stats_by_pathway.csv, kegg_dataset_cache/kegg_rest/link_ko_hsa.tsv
Outputs: kegg_dataset/splits_ko/
"""

from __future__ import annotations

import argparse
import math
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kegg_pipeline import annotations as A  # noqa: E402
from kegg_pipeline import sampling as S  # noqa: E402
from kegg_pipeline.common import OUT, log, read_csv, write_csv  # noqa: E402

SPLITS = ("train", "val", "test")
OUTDIR = OUT / "splits_ko"


def load_ko_graphs():
    g2k = A.parse_link(A.cached_text(A.REST / "link_ko_hsa.tsv", "", 0, offline=True))
    graphs = defaultdict(set)
    for r in read_csv(OUT / "gene_edges.csv"):
        if r["edge_origin"] != "gene_entry":
            continue
        for a in g2k.get(r["source_gene"], ()):
            for b in g2k.get(r["target_gene"], ()):
                if a != b:
                    graphs[r["pathway_id"]].add((a, b))
    return graphs


def pathway_groups(graphs, threshold):
    """Union pathways whose KO edge sets overlap strongly (overlap coefficient >= threshold),
    e.g. disease maps that copy a signalling map. Groups are the unit of assignment."""
    pids = sorted(graphs)
    par = {p: p for p in pids}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for i, p in enumerate(pids):
        for q in pids[i + 1:]:
            inter = len(graphs[p] & graphs[q])
            if inter and inter / min(len(graphs[p]), len(graphs[q])) >= threshold:
                par[find(p)] = find(q)
    groups = defaultdict(list)
    for p in pids:
        groups[find(p)].append(p)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def owned_counts(assign, edge_pw):
    c = Counter()
    for e, ps in edge_pw.items():
        here = {assign[p] for p in ps}
        c[next(s for s in SPLITS if s in here)] += 1
    return c


def assign_pathways(graphs, classes, ratios, seed, threshold, min_train_frac):
    """Greedy group assignment on the number of unique edges each split would *own*.
    Everything starts in train; groups are visited in a seeded, class-round-robin order and
    moved to the val/test split with the largest remaining deficit if that increases its
    owned edges without overshooting the target by more than 15%.
    Hard constraint: for every KEGG class, at least ceil(min_train_frac x class pathways) pathways
    stay in train; a group move that would break this for any class is skipped."""
    rng = random.Random(f"{seed}|split")
    edge_pw = defaultdict(set)
    for p, es in graphs.items():
        for e in es:
            edge_pw[e].add(p)
    groups = pathway_groups(graphs, threshold)
    by_cls = defaultdict(list)
    for g in groups:
        by_cls[Counter(classes.get(p, "UNCLASSIFIED") for p in g).most_common(1)[0][0]].append(g)
    for c in by_cls:
        rng.shuffle(by_cls[c])
    order, cls_names = [], sorted(by_cls)
    rng.shuffle(cls_names)
    while any(by_cls.values()):
        for c in cls_names:
            if by_cls[c]:
                order.append(by_cls[c].pop())
    n_edges = len(edge_pw)
    target = {s: r * n_edges for s, r in zip(SPLITS, ratios)}
    assign = {p: "train" for p in graphs}
    cls_total = Counter(classes.get(p, "UNCLASSIFIED") for p in graphs)
    min_train = {c: math.ceil(min_train_frac * n - 1e-9) for c, n in cls_total.items()}
    cls_train = Counter(cls_total)
    owned = owned_counts(assign, edge_pw)
    for _ in range(3):
        moved = False
        for g in order:
            if assign[g[0]] != "train":
                continue
            g_cls = Counter(classes.get(p, "UNCLASSIFIED") for p in g)
            if any(cls_train[c] - k < min_train[c] for c, k in g_cls.items()):
                continue  # would push a class below its train minimum
            for s in sorted(("val", "test"), key=lambda s: owned[s] / target[s]):
                if owned[s] >= target[s]:
                    continue
                trial = dict(assign)
                for p in g:
                    trial[p] = s
                new = owned_counts(trial, edge_pw)
                if new[s] > owned[s] and new[s] <= 1.15 * target[s]:
                    assign, owned, moved = trial, new, True
                    cls_train.subtract(g_cls)
                    break
        if not moved or all(owned[s] >= 0.95 * target[s] for s in ("val", "test")):
            break
    for c, n in cls_total.items():  # verify the constraint on the final assignment
        kept = sum(1 for p in graphs if classes.get(p, "UNCLASSIFIED") == c and assign[p] == "train")
        assert kept >= min_train[c], f"class {c}: {kept} train pathways < minimum {min_train[c]}"
    return assign, groups


def add_macro_weight(rows):
    """pathway_macro_weight = sum over source pathways p of 1/n_p (n_p = samples of p), rescaled to sum
    to the number of rows. Each pathway spreads the same total weight over its samples, so weighted
    metrics give every held-out pathway equal influence."""
    n = Counter()
    for r in rows:
        for p in r["pathway_ids"].split(";"):
            n[p] += 1
    w = [sum(1 / n[p] for p in r["pathway_ids"].split(";")) for r in rows]
    scale = len(rows) / sum(w) if w else 1
    for r, x in zip(rows, w):
        r["pathway_macro_weight"] = round(x * scale, 6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default kegg_dataset/splits_ko, the frozen v1 split)")
    ap.add_argument("--ratios", default="0.8,0.1,0.1", help="target shares of *owned unique* KO edges")
    ap.add_argument("--min-train-class-frac", type=float, default=0.6,
                    help="each KEGG class keeps at least this fraction of its pathways in train")
    ap.add_argument("--group-threshold", type=float, default=0.7,
                    help="pathways with KO-edge overlap coefficient >= this are assigned together")
    ap.add_argument("--max-paths-per-pathway", type=int, default=10000)
    ap.add_argument("--window-lengths", default="3,4,5,6")
    ap.add_argument("--cap-per-edge", type=float, default=3.0,
                    help="per pathway and length keep at most cap_per_edge x (owned edges) paths "
                         "(also bounded by --max-paths-per-pathway); 0 disables")
    ap.add_argument("--enum-walk-limit", type=int, default=300000)
    a = ap.parse_args()
    global OUTDIR
    if a.out_dir:
        OUTDIR = Path(a.out_dir)
    ratios = [float(x) for x in a.ratios.split(",")]
    lengths = [int(x) for x in a.window_lengths.split(",")]
    OUTDIR.mkdir(parents=True, exist_ok=True)

    graphs = load_ko_graphs()
    pw_info = {r["pathway_id"]: r for r in read_csv(OUT / "gene_graph_stats_by_pathway.csv")}
    classes = {p: pw_info[p]["pathway_class_a"] for p in graphs}
    vocab = {r["token"]: int(r["token_id"]) for r in read_csv(OUT / "token_vocab_ko.csv")}
    log(f"{len(graphs)} pathways with KO graphs, {sum(len(v) for v in graphs.values()):,} edge occurrences")

    assign, groups = assign_pathways(graphs, classes, ratios, a.seed, a.group_threshold, a.min_train_class_frac)
    gid = {p: i for i, g in enumerate(groups) for p in g}

    # ------------------------------------------------------------ edge ownership
    edge_pw = defaultdict(set)
    for p, es in graphs.items():
        for e in es:
            edge_pw[e].add(p)
    owner = {}
    for e, ps in edge_pw.items():
        splits_here = {assign[p] for p in ps}
        owner[e] = next(s for s in SPLITS if s in splits_here)
    # Reverse-edge exclusion: a val edge b->a whose reverse a->b is owned by train, or a test edge whose
    # reverse is owned by train or val, is removed from every split (tagged "excluded_reverse").
    n_excl = Counter()
    for s in ("val", "test"):
        earlier = set(SPLITS[:SPLITS.index(s)])
        for e in sorted(e for e, o in owner.items() if o == s):
            if owner.get((e[1], e[0])) in earlier:
                owner[e] = "excluded_reverse"
                n_excl[s] += 1
    split_edges = {s: {e for e, o in owner.items() if o == s} for s in SPLITS}
    train_edges = split_edges["train"]

    prow = []
    for p in sorted(graphs):
        s = assign[p]
        own = sum(1 for e in graphs[p] if owner[e] == s)
        excl = sum(1 for e in graphs[p] if owner[e] == "excluded_reverse" and s != "train"
                   and any(assign[q] == s for q in edge_pw[e]))
        prow.append({"pathway_id": p, "pathway_name": pw_info[p]["pathway_name"], "pathway_class_a": classes[p],
                     "split": s, "assignment_group": gid[p], "group_size": len(groups[gid[p]]), "ko_edges": len(graphs[p]), "ko_edges_owned_by_own_split": own,
                     "ko_edges_excluded_reverse": excl,
                     "ko_edges_ceded_to_earlier_split": len(graphs[p]) - own - excl})
    write_csv(OUTDIR / "pathway_split.csv", prow, list(prow[0].keys()))
    erow = []
    for e in sorted(owner):
        erow.append({"source_ko": e[0], "target_ko": e[1], "split": owner[e], "n_pathways": len(edge_pw[e]),
                     "pathways": ";".join(sorted(edge_pw[e])),
                     "pathway_splits": ";".join(sorted({assign[p] for p in edge_pw[e]}, key=SPLITS.index)),
                     "reverse_owner": owner.get((e[1], e[0]), "")})
    write_csv(OUTDIR / "edge_split.csv", erow, list(erow[0].keys()))

    # ------------------------------------------------------------ per-split samples
    train_tokens = {k for e in train_edges for k in e}
    train_out = defaultdict(set)
    for u, v in train_edges:
        train_out[u].add(v)
    stats, checks = {}, {}
    keysets = {}
    earlier_ctx = set()   # masked contexts already used by an earlier split (train -> val -> test)
    for s in SPLITS:
        sub = {p: {e for e in graphs[p] if owner[e] == s} for p in graphs if assign[p] == s}
        sub = {p: es for p, es in sub.items() if es}
        # A: next node, dedup within split
        A_ = defaultdict(set)
        for p, es in sub.items():
            for e in es:
                A_[e].add(p)
        out_split = defaultdict(set)
        for u, v in A_:
            out_split[u].add(v)
        # B / C
        B_, C_ = defaultdict(set), defaultdict(set)
        C_targets = defaultdict(set)
        for p in sorted(sub):
            for L in lengths:
                cap = a.max_paths_per_pathway
                if a.cap_per_edge > 0:
                    cap = min(cap, max(1, int(a.cap_per_edge * len(sub[p]))))
                res = S.sample_paths(sub[p], L, cap, f"{a.seed}|{s}|{p}|{L}",
                                     enum_limit=a.enum_walk_limit)
                for path in res["paths"]:
                    B_[path].add(p)
                    for ctx, pos, tgt in S.masked_from_path(path):
                        C_[(ctx, pos, tgt)].add(p)
        # Query leakage: A -> [MASK] -> C has no visible edge, so the same context can arise from
        # different paths in different splits. Drop val/test samples whose context an earlier split used.
        dropped = {k for k in C_ if k[0] in earlier_ctx}
        checks[f"{s}_masked_dropped_context_seen_in_earlier_split"] = len(dropped)
        for k in dropped:
            del C_[k]
        for ctx, _, tgt in C_:
            C_targets[ctx].add(tgt)
        earlier_ctx |= {k[0] for k in C_}

        def ids(seq):
            return " ".join(str(vocab.get(f"{t}", vocab.get(t, 2))) if t != "[MASK]" else "1" for t in seq)

        def seen(t):
            return int(t in train_tokens)
        A_rows = [{"split": s, "context_token_id": vocab[u], "target_token_id": vocab[v],
                   "context_token": f"ko:{u}", "target_token": f"ko:{v}",
                   "n_pathways": len(ps), "pathway_ids": ";".join(sorted(ps)),
                   "context_seen_in_train": seen(u), "target_seen_in_train": seen(v),
                   "valid_targets_in_split": " ".join(str(vocab[x]) for x in sorted(out_split[u])),
                   "known_targets_in_train": " ".join(str(vocab[x]) for x in sorted(train_out.get(u, ()))) if s != "train" else ""}
                  for (u, v), ps in sorted(A_.items())]
        add_macro_weight(A_rows)
        write_csv(OUTDIR / f"{s}_A_next_node.csv.gz", A_rows, list(A_rows[0].keys()))
        B_rows = [{"split": s, "length": len(pth), "token_ids": ids(pth), "tokens": " ".join(f"ko:{t}" for t in pth),
                   "n_pathways": len(ps), "pathway_ids": ";".join(sorted(ps)),
                   "all_tokens_seen_in_train": int(all(t in train_tokens for t in pth))}
                  for pth, ps in sorted(B_.items(), key=lambda kv: (len(kv[0]), kv[0]))]
        add_macro_weight(B_rows)
        write_csv(OUTDIR / f"{s}_B_path_windows.csv.gz", B_rows, list(B_rows[0].keys()))
        C_rows = [{"split": s, "length": len(ctx), "mask_position": pos, "context_token_ids": ids(ctx),
                   "target_token_id": vocab[tgt], "target_token": f"ko:{tgt}",
                   "n_pathways": len(ps), "pathway_ids": ";".join(sorted(ps)),
                   "target_seen_in_train": seen(tgt),
                   "context_tokens_seen_in_train": int(all(t in train_tokens for t in ctx if t != "[MASK]")),
                   "valid_targets_in_split": " ".join(str(vocab[x]) for x in sorted(C_targets[ctx]))}
                  for (ctx, pos, tgt), ps in sorted(C_.items(), key=lambda kv: (len(kv[0][0]), kv[0]))]
        add_macro_weight(C_rows)
        write_csv(OUTDIR / f"{s}_C_masked.csv.gz", C_rows, list(C_rows[0].keys()))
        keysets[s] = {"A": set(A_), "B": set(B_), "C": {(c, t) for c, _, t in C_},
                      "edges_used": {e for pth in B_ for e in zip(pth, pth[1:])} | set(A_)}

        def conc(counter_by_pw):
            return S.concentration(counter_by_pw, len(sub))
        perA, perB, perC = Counter(), Counter(), Counter()
        for _, ps in A_.items():
            for p in ps:
                perA[p] += 1
        for _, ps in B_.items():
            for p in ps:
                perB[p] += 1
        for _, ps in C_.items():
            for p in ps:
                perC[p] += 1
        for nm, rows_, per in (("A_next_node", A_rows, perA), ("B_path_windows", B_rows, perB), ("C_masked", C_rows, perC)):
            c = conc(per)
            st = {"split": s, "strategy": nm, "unique_samples": len(rows_),
                  "pathways_assigned": sum(1 for p in assign if assign[p] == s),
                  "pathways_with_samples": c["pathways_contributing"],
                  "median_per_pathway": c["median_per_contributing_pathway"],
                  "max_from_one_pathway": c["max_from_one_pathway"], "max_pathway": c["max_pathway"],
                  "top1_share": round(c["top1_share"], 4), "top5_share": round(c["top5_share"], 4),
                  "top10_share": round(c["top10_share"], 4), "gini": round(c["gini_contributing"], 4)}
            if nm == "A_next_node":
                st["target_unseen_in_train"] = sum(1 for r in rows_ if not r["target_seen_in_train"])
                st["ambiguous_context_frac"] = round(sum(1 for u in out_split if len(out_split[u]) > 1) / max(1, len(out_split)), 4)
            if nm == "C_masked":
                st["target_unseen_in_train"] = sum(1 for r in rows_ if not r["target_seen_in_train"])
                st["ambiguous_context_frac"] = round(sum(1 for c_ in C_targets if len(C_targets[c_]) > 1) / max(1, len(C_targets)), 4)
            if nm == "B_path_windows":
                for L in lengths:
                    st[f"L{L}"] = sum(1 for r in rows_ if r["length"] == L)
            stats[(s, nm)] = st
        log(f"{s}: {len(sub)} pathways with owned edges; A={len(A_rows):,} B={len(B_rows):,} C={len(C_rows):,}")

    # ------------------------------------------------------------ leakage checks
    for x, y in (("train", "val"), ("train", "test"), ("val", "test")):
        for k in ("A", "B", "C"):
            checks[f"overlap_{k}_{x}_{y}"] = len(keysets[x][k] & keysets[y][k])
        checks[f"edges_shared_{x}_{y}"] = len(keysets[x]["edges_used"] & keysets[y]["edges_used"])
    for s in ("val", "test"):
        checks[f"{s}_edges_excluded_reverse"] = n_excl[s]
        earlier_edges = set().union(*(split_edges[x] for x in SPLITS[:SPLITS.index(s)]))
        checks[f"overlap_rev_{s}_edges_reverse_in_earlier_split"] = sum(
            1 for e in split_edges[s] if (e[1], e[0]) in earlier_edges)
        used_earlier = set().union(*(keysets[x]["edges_used"] for x in SPLITS[:SPLITS.index(s)]))
        checks[f"overlap_rev_{s}_sample_edges_reverse_used_earlier"] = sum(
            1 for e in keysets[s]["edges_used"] if (e[1], e[0]) in used_earlier)
        checks[f"{s}_tokens_unseen_in_train"] = len({k for e in split_edges[s] for k in e} - train_tokens)
        checks[f"{s}_unique_edges"] = len(split_edges[s])
    checks["train_unique_edges"] = len(train_edges)
    checks["edges_ceded_to_train_from_val_test_pathways"] = sum(
        r["ko_edges_ceded_to_earlier_split"] for r in prow if r["split"] != "train")
    # independent re-check against raw pathway graphs (does not rely on the ownership table)
    edges_in_train_pathways = {e for p, es in graphs.items() if assign[p] == "train" for e in es}
    edges_in_val_pathways = {e for p, es in graphs.items() if assign[p] == "val" for e in es}
    for s_ in ("val", "test"):
        checks[f"overlap_raw_{s_}_sample_edges_in_any_train_pathway"] = len(keysets[s_]["edges_used"] & edges_in_train_pathways)
    checks["overlap_raw_test_sample_edges_in_any_val_pathway"] = len(keysets["test"]["edges_used"] & edges_in_val_pathways)
    # identical masked context (regardless of target) or next-node context->target across splits
    for x, y in (("train", "val"), ("train", "test"), ("val", "test")):
        checks[f"overlap_Cctx_{x}_{y}"] = len({c for c, _ in keysets[x]["C"]} & {c for c, _ in keysets[y]["C"]})
    ok = all(v == 0 for k, v in checks.items() if k.startswith(("overlap_", "edges_shared_")))
    checks["ALL_CROSS_SPLIT_OVERLAPS_ZERO"] = int(ok)
    write_csv(OUTDIR / "leakage_checks.csv", [{"check": k, "value": v} for k, v in checks.items()], ["check", "value"])
    if not ok:  # hard stop: never publish a leaking split as if it were valid
        raise SystemExit("LEAKAGE DETECTED: cross-split overlap is non-zero; see leakage_checks.csv")

    fields = sorted({k for st in stats.values() for k in st},
                    key=lambda k: (k not in ("split", "strategy", "unique_samples"), k))
    write_csv(OUTDIR / "split_stats.csv", list(stats.values()), fields)
    split_pw = Counter(assign.values())
    split_cls = {s: dict(Counter(classes[p] for p in assign if assign[p] == s)) for s in SPLITS}

    def tbl(nm):
        hdr = "| split | unique samples | pathways w/ samples | median/pathway | max 1 pathway | top1 | top5 | top10 | Gini | target unseen in train | ambiguous ctx |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        return hdr + "\n".join(
            f"| {s} | {st['unique_samples']:,} | {st['pathways_with_samples']} | {st['median_per_pathway']} | "
            f"{st['max_from_one_pathway']:,} ({st['max_pathway']}) | {st['top1_share']:.1%} | {st['top5_share']:.1%} | "
            f"{st['top10_share']:.1%} | {st['gini']:.3f} | {st.get('target_unseen_in_train', '-')} | "
            f"{st['ambiguous_context_frac']:.1%} |" if 'ambiguous_context_frac' in st else
            f"| {s} | {st['unique_samples']:,} | {st['pathways_with_samples']} | {st['median_per_pathway']} | "
            f"{st['max_from_one_pathway']:,} ({st['max_pathway']}) | {st['top1_share']:.1%} | {st['top5_share']:.1%} | "
            f"{st['top10_share']:.1%} | {st['gini']:.3f} | - | - |"
            for s in SPLITS for st in [stats[(s, nm)]])

    md = f"""# KO-level leakage-controlled split

Built by `build_ko_leakage_free_split.py` (seed={a.seed}, ratios={a.ratios}, cap=min({a.max_paths_per_pathway}, {a.cap_per_edge} x owned edges)
paths per pathway per length, L={a.window_lengths}). Token IDs are those of `../token_vocab_ko.csv`.

## Scheme
Pathway-group holdout **plus unique-edge ownership**. Pathways whose KO edge sets overlap with
coefficient >= {a.group_threshold} (near copies, e.g. disease maps of a signalling map) form {len(groups)} groups;
groups are assigned greedily (seeded, round-robin over KEGG classes) to reach the target share of
*owned* unique edges per split.
an edge that occurs in any train pathway belongs to train only; otherwise to val; otherwise test.
Val/test samples are sampled only from edges their split owns, so every edge inside every val/test
sample is absent from train. Samples are deduplicated within each split.

Whole-cluster pathway holdout was rejected: clustering pathways by KO-edge overlap
(overlap coefficient >= 0.5) chains 136 pathways holding ~79% of edges into one component.

## Assignment
* Pathways per split: {dict(split_pw)}; groups per split: {dict(Counter(assign[g[0]] for g in groups))}
* By KEGG class: {json.dumps(split_cls)}
* Per-class train minimum (--min-train-class-frac {a.min_train_class_frac}):

| KEGG class | pathways | train | val | test | train fraction |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(f"| {c} | {n} | {split_cls['train'].get(c, 0)} | {split_cls['val'].get(c, 0)} | {split_cls['test'].get(c, 0)} | "
               f"{split_cls['train'].get(c, 0) / n:.0%} |" for c, n in sorted(Counter(classes.values()).items())) + f"""

* Unique KO edges owned: train {checks['train_unique_edges']:,}, val {checks['val_unique_edges']:,}, test {checks['test_unique_edges']:,}
* Edge occurrences in val/test pathways ceded to train (shared with a train pathway): {checks['edges_ceded_to_train_from_val_test_pathways']:,}
* Val/test edges excluded because their reverse is owned by an earlier split: val {checks['val_edges_excluded_reverse']}, test {checks['test_edges_excluded_reverse']}

## Samples and concentration

### A next-node
{tbl('A_next_node')}

### B path windows (L=3..6 pooled)
{tbl('B_path_windows')}

Per length: """ + "; ".join(f"{s}: " + ", ".join(f"L{L}={stats[(s, 'B_path_windows')][f'L{L}']:,}" for L in lengths) for s in SPLITS) + f"""

### C masked completion
{tbl('C_masked')}

## Leakage checks (`leakage_checks.csv`)
""" + "\n".join(f"* {k}: {v}" for k, v in checks.items()) + """

## Notes for evaluation
* Masked samples whose context (e.g. `A [MASK] C`, where no edge is visible) was already used by an
  earlier split are removed from val/test (`*_masked_dropped_context_seen_in_earlier_split`), so no
  masked question is repeated across splits even with a different answer.
* Next-node contexts are single tokens, so the same source KO legitimately has different correct
  targets in different splits (standard transductive link prediction); use `known_targets_in_train`
  for filtered ranking. The (context, target) pairs themselves never repeat across splits.
* Tokens are still shared across splits (transductive link prediction): val/test contain new
  *relations*, not new KOs. Rows flag `target_seen_in_train`; targets never seen in train cannot be
  predicted by a pure token model and should be reported separately (or handled by GO/sequence features).
* `valid_targets_in_split` lists all correct answers for the same context within the split and
  `known_targets_in_train` lists train answers for filtered ranking (exclude them when ranking).
* Reverse edges: a val/test edge whose reverse direction is owned by an earlier split (train; or val for
  test) is excluded from all splits (`split=excluded_reverse` in `edge_split.csv`), so no val/test
  relation is answerable by flipping a training relation. Checked by the `overlap_rev_*` rows.
* Val/test are small (only ~20-30 held-out pathways yield paths) and remain concentrated even with the
  size-relative cap; report pathway-macro-averaged metrics using the `pathway_macro_weight` column
  (every pathway gets equal total weight) alongside micro averages.
* Val/test pathway graphs lose their shared edges, so their paths are shorter/fewer than in train;
  this is the price of zero edge overlap.
"""
    (OUTDIR / "README.md").write_text(md, encoding="utf-8")
    print(md)
    write_csv(OUTDIR / "split_config.csv", [{"k": k, "v": v} for k, v in vars(a).items()], ["k", "v"])


if __name__ == "__main__":
    main()

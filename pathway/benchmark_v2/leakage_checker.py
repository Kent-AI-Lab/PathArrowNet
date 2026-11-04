#!/usr/bin/env python3
"""
Automated leakage checker for KO next-node splits.

    .venv/bin/python benchmark_v2/leakage_checker.py            # all settings, exits 1 if the strict split fails

Settings checked
  strict_v1                    kegg_dataset/splits_ko (+ benchmark_v2/splits/strict_v1 S files)   [Setting B]
  random_segment               benchmark_v2/splits/random_segment (random split of generated segments) [Setting A]
  unique_random                kegg_leaky_splits/unique_random (random split of unique edges, A only)  [Setting A']
  occurrence_random            kegg_leaky_splits/occurrence_random (random split of edge occurrences, A only)
  pathway_disjoint_no_dedup    built here: pathways assigned as in strict_v1, but every edge of a val/test pathway
                               is kept (no edge ownership) -> shows why edge ownership is needed

Checks (for every pair of splits a < b; the value counts objects of the LATER split b that also occur in a)
  C1 pathway_in_multiple_splits      pathways contributing samples to both splits
  C2 pathway_assignment_violations   sample rows of split b citing a pathway assigned to another split (strict only)
  C3 segment_pathways_crossing       pathways whose path segments (B windows) appear in both splits
  C4 dup_A_pairs / dup_B_segments / dup_C_questions / dup_S_prefix_target   identical samples in both splits
  C5 edge_overlap                    directed edges used anywhere in split-b samples that are also used in split a
  C6 reverse_edge_overlap            split-b edges whose reverse is used in split a
  C7 source_target_pair_overlap      identical (source, target) next-node pairs (A task)
  I1 ko_node_overlap                 KOs of split b also present in split a (INFO; transductive setting, not removed)
  I2 target_unseen_in_train          split-b targets never seen in train (INFO)
  I3 context_node_overlap            split-b source KOs that are also a source in split a (INFO; different targets)
Strict split requirement: C1-C7 == 0. I* are reported, never enforced.
"""

from __future__ import annotations

import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "kegg_dataset"
OUT = Path(__file__).resolve().parent / "out"
SPLITS = ("train", "val", "test")
PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))


def rd(p):
    if not p.exists():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ints(s):
    return tuple(int(x) for x in s.split()) if s else ()


def load_setting(dirs):
    """dirs: list of directories searched for {split}_{A,B,C,S}_*.csv.gz. Returns split -> dict."""
    names = {"A": "A_next_node", "B": "B_path_windows", "C": "C_masked", "S": "S_prefix_next"}
    data = {}
    for s in SPLITS:
        d = {}
        for k, nm in names.items():
            for dd in dirs:
                rows = rd(dd / f"{s}_{nm}.csv.gz")
                if rows is not None:
                    d[k] = rows
                    break
        data[s] = d
    return data


def summarize(split):
    """Objects used by one split."""
    o = {"pathways": set(), "segment_pathways": set(), "A": set(), "B": set(), "C": set(), "S": set(),
         "edges": set(), "tokens": set(), "targets": set(), "sources": set(), "rows_pw": []}
    for k, rows in split.items():
        for r in rows:
            ps = set(r["pathway_ids"].split(";"))
            o["pathways"] |= ps
            o["rows_pw"].append(ps)
            if k == "A":
                u, v = int(r["context_token_id"]), int(r["target_token_id"])
                o["A"].add((u, v))
                o["edges"].add((u, v))
                o["tokens"] |= {u, v}
                o["targets"].add(v)
                o["sources"].add(u)
            elif k == "B":
                w = ints(r["token_ids"])
                o["B"].add(w)
                o["segment_pathways"] |= ps
                o["edges"] |= set(zip(w, w[1:]))
                o["tokens"] |= set(w)
            elif k == "C":
                ctx, t = ints(r["context_token_ids"]), int(r["target_token_id"])
                o["C"].add((ctx, t))
                pos = int(r["mask_position"])
                full = ctx[:pos] + (t,) + ctx[pos + 1:]
                o["edges"] |= set(zip(full, full[1:]))
                o["tokens"] |= set(full)
                o["targets"].add(t)
            elif k == "S":
                ctx, t = ints(r["context_token_ids"]), int(r["target_token_id"])
                o["S"].add((ctx, t))
                full = ctx + (t,)
                o["edges"] |= set(zip(full, full[1:]))
                o["tokens"] |= set(full)
                o["targets"].add(t)
    return o


def check(setting, data, assign=None):
    S = {s: summarize(data[s]) for s in SPLITS}
    rows = []

    def add(pair, chk, val, denom=None, kind="ENFORCED"):
        rows.append({"setting": setting, "pair": pair, "check": chk, "value": val,
                     "denominator": "" if denom is None else denom,
                     "fraction": "" if not denom else round(val / denom, 4), "kind": kind})
    for a, b in PAIRS:
        A, B = S[a], S[b]
        pr = f"{a}-{b}"
        add(pr, "C1_pathway_in_multiple_splits", len(A["pathways"] & B["pathways"]), len(B["pathways"]))
        if assign is not None:
            viol = sum(1 for ps in B["rows_pw"] if any(assign.get(p) != b for p in ps))
            add(pr, "C2_pathway_assignment_violations(rows_of_later_split)", viol, len(B["rows_pw"]))
        if A["segment_pathways"] and B["segment_pathways"]:
            add(pr, "C3_segment_pathways_crossing", len(A["segment_pathways"] & B["segment_pathways"]),
                len(B["segment_pathways"]))
        for k, nm in (("A", "C4_dup_A_pairs"), ("B", "C4_dup_B_segments"), ("C", "C4_dup_C_questions"),
                      ("S", "C4_dup_S_prefix_target")):
            if A[k] and B[k]:
                add(pr, nm, len(A[k] & B[k]), len(B[k]))
        add(pr, "C5_edge_overlap", len(A["edges"] & B["edges"]), len(B["edges"]))
        add(pr, "C6_reverse_edge_overlap", sum(1 for (u, v) in B["edges"] if (v, u) in A["edges"]), len(B["edges"]))
        add(pr, "C7_source_target_pair_overlap", len(A["A"] & B["A"]), len(B["A"]))
        add(pr, "I1_ko_node_overlap", len(A["tokens"] & B["tokens"]), len(B["tokens"]), "INFO")
        add(pr, "I3_context_node_overlap(A sources)", len(A["sources"] & B["sources"]), len(B["sources"]), "INFO")
    for s in ("val", "test"):
        add(f"train-{s}", "I2_targets_unseen_in_train", len(S[s]["targets"] - S["train"]["tokens"]), len(S[s]["targets"]), "INFO")
        add(f"train-{s}", "I2b_tokens_unseen_in_train", len(S[s]["tokens"] - S["train"]["tokens"]), len(S[s]["tokens"]), "INFO")
    return rows


def naive_pathway_disjoint(assign):
    sys.path.insert(0, str(ROOT))
    from build_ko_leakage_free_split import load_ko_graphs
    vocab = {r["token"]: int(r["token_id"]) for r in csv.DictReader((DS / "token_vocab_ko.csv").open())}
    graphs = load_ko_graphs()
    data = {s: {"A": []} for s in SPLITS}
    per = {s: defaultdict(set) for s in SPLITS}
    for p, es in graphs.items():
        for u, v in es:
            per[assign[p]][(vocab[u], vocab[v])].add(p)
    for s in SPLITS:
        data[s]["A"] = [{"context_token_id": u, "target_token_id": v, "pathway_ids": ";".join(sorted(ps))}
                        for (u, v), ps in per[s].items()]
    return data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    B2 = Path(__file__).resolve().parent / "splits"
    with (DS / "splits_ko" / "pathway_split.csv").open() as f:
        assign = {r["pathway_id"]: r["split"] for r in csv.DictReader(f)}
    settings = [
        ("strict_v1", load_setting([DS / "splits_ko", B2 / "strict_v1"]), assign),
        ("random_segment", load_setting([B2 / "random_segment"]), None),
        ("unique_random", load_setting([ROOT / "kegg_leaky_splits" / "unique_random"]), None),
        ("occurrence_random", load_setting([ROOT / "kegg_leaky_splits" / "occurrence_random"]), None),
        ("pathway_disjoint_no_dedup", naive_pathway_disjoint(assign), assign),
    ]
    allrows = []
    for name, data, asg in settings:
        rows = check(name, data, asg)
        allrows += rows
    for r in allrows:
        if r["kind"] == "ENFORCED":
            r["status"] = "PASS" if r["value"] == 0 else ("FAIL" if r["setting"] == "strict_v1" else "LEAK")
        else:
            r["status"] = "INFO"
    with (OUT / "leakage_audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        w.writeheader()
        w.writerows(allrows)
    # wide summary: one row per (setting, check), columns = pairs
    wide = defaultdict(dict)
    for r in allrows:
        frac = f" ({100 * r['fraction']:.1f}%)" if r["fraction"] != "" else ""
        wide[(r["setting"], r["check"])][r["pair"]] = f"{r['value']}{frac}"
    with (OUT / "leakage_audit_wide.csv").open("w", newline="") as f:
        cols = ["setting", "check", "train-val", "train-test", "val-test"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for (st, ck), d in wide.items():
            w.writerow({"setting": st, "check": ck, **{c: d.get(c, "") for c in cols[2:]}})
    for (st, ck), d in wide.items():
        print(f"{st:28s} {ck:55s} " + " | ".join(f"{p}: {d.get(p, '-')}" for p in ("train-val", "train-test", "val-test")))
    fails = [r for r in allrows if r["status"] == "FAIL"]
    print(f"\nSTRICT SPLIT: {'ALL ENFORCED CHECKS PASS' if not fails else f'{len(fails)} FAILURES'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

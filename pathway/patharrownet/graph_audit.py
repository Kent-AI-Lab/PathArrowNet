#!/usr/bin/env python3
"""
Message-passing leakage audit of the PathArrowNet INFERENCE graph (CPU). For a released run (config.json + best.pt) it
rebuilds the graph exactly as patharrownet.predict does and checks:

  G1  relation-0 edges == the train edges of the split (train_A_next_node)
  G2  relation-1 edges == their reverses
  G3  node-input edge list == the train edges
  G4  no val / test edge in the KEGG relations (directed)
  G5  no val / test edge or its reverse in the KEGG relations
  G6  background relations == the STRING file named in the config (both directions, documented score bins)
  G7  relation ids within the configured range
  G8  val / test labels permuted (targets shuffled among queries, train files untouched): inference graph bit-identical
  G9  same perturbation: checkpoint scores of the first --n-anchors test anchors bit-identical

    cd $KEGG_HOME/pathway
    python -m patharrownet.graph_audit --run-dir kegg_dataset/nbf_ko_full/nbf+esm+exp150bin4+allneg+kl_s42 --out audit.json

The checks of the training-time graphs (edge removal per step) belong to the training code and are not part of this
release; their results are reported in the paper's supplement.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "benchmark_v2"))
from patharrownet import data as D  # noqa: E402
from patharrownet.models import patharrownet as KN  # noqa: E402


def rd(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f:
        return list(csv.DictReader(f))


def split_pairs(sd):
    return {s: {(int(r["context_token_id"]), int(r["target_token_id"])) for r in rd(sd / f"{s}_A_next_node.csv.gz")}
            for s in ("train", "val", "test")}


def perturbed_copy(sd, od, seed=0):
    """Copy of the split dir in which every val/test target (A and C files) is permuted among queries, avoiding
    collisions with train edges so the loaders' asserts still pass. Other files are copied unchanged."""
    od.mkdir(parents=True, exist_ok=True)
    tr = split_pairs(sd)["train"]
    rng = random.Random(seed)
    for f in sd.iterdir():
        dst = od / f.name
        if not f.name.startswith(("val_A", "test_A", "val_C", "test_C")):
            if f.is_file():
                shutil.copyfile(f, dst)
            continue
        rows = rd(f)
        fields = list(rows[0].keys())
        tgt = [(r["target_token_id"], r["target_token"]) for r in rows]

        def bad(i, t):
            r = rows[i]
            t = int(t[0])
            if f.name.startswith(("val_A", "test_A")):
                u = int(r["context_token_id"])
                return u == t or (u, t) in tr
            ctx = [int(x) for x in r["context_token_ids"].split()]
            p = int(r["mask_position"])
            l = ctx[p - 1] if p > 0 else None
            rr = ctx[p + 1] if p + 1 < len(ctx) else None
            return t in ctx or (l is not None and (l, t) in tr) or (rr is not None and (t, rr) in tr)
        perm = tgt[:]
        rng.shuffle(perm)
        for _ in range(50):
            badi = [i for i in range(len(rows)) if bad(i, perm[i])]
            if not badi:
                break
            for i in badi:
                j = rng.randrange(len(rows))
                perm[i], perm[j] = perm[j], perm[i]
        for i in [i for i in range(len(rows)) if bad(i, perm[i])]:
            perm[i] = tgt[i]
        for r, (tid, tok) in zip(rows, perm):
            r["target_token_id"], r["target_token"] = tid, tok
        with gzip.open(dst, "wt", newline="") as g:
            w = csv.DictWriter(g, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return od


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split-dir", default=None)
    ap.add_argument("--n-anchors", type=int, default=8)
    ap.add_argument("--no-perturb", action="store_true", help="skip G8 / G9")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(8)
    model, graph, vocab, V, cfg = KN.load_run(a.run_dir, "cpu", a.split_dir)
    sd = KN.resolve_split_dir(cfg, a.split_dir)
    P = split_pairs(sd)
    tr, held = P["train"], P["val"] | P["test"]
    src, dst, rel, (fs, fd) = graph
    trip = {tuple(x) for x in torch.stack([src, dst, rel], 1).tolist()}
    kegg = {t for t in trip if t[2] < 2}
    fwd = {(u, v) for u, v, r in kegg if r == 0}
    inv = {(u, v) for u, v, r in kegg if r == 1}
    C = {}

    def check(name, ok, detail=None):
        C[name] = {"pass": bool(ok), "detail": detail}
        print(("PASS " if ok else "FAIL ") + name + ("" if detail is None else f"  {detail}"), flush=True)

    check("G1_relation0_equals_train_edges", fwd == tr, f"{len(fwd)} vs {len(tr)}")
    check("G2_relation1_is_reverse_of_train", inv == {(v, u) for u, v in tr}, f"{len(inv)}")
    check("G3_node_input_edges_equal_train", set(zip(fs.tolist(), fd.tolist())) == tr)
    d_hits = {(u, v) for u, v, _ in kegg} & held
    a_hits = {(u, v) for u, v, _ in kegg} & (held | {(v, u) for u, v in held})
    check("G4_no_val_test_edge_in_KEGG_relations", not d_hits, f"{len(d_hits)} hits")
    check("G5_no_val_test_edge_or_reverse_in_KEGG_relations", not a_hits, f"{len(a_hits)} hits")
    if cfg.get("background"):
        bins = "bins4" if cfg.get("bg_bins4") else "bins3" if cfg.get("bg_bins") else None
        pairs, brel = [], []
        p = Path(cfg["background"]); p = p if p.is_absolute() else D.ROOT / p
        for r in csv.DictReader(open(p), delimiter="\t"):
            x, y, s = vocab[r["ko_a"]], vocab[r["ko_b"]], int(float(r["max_experimental_score"]))
            b = (2 + (3 if s >= 900 else 2 if s >= 700 else 1 if s >= 400 else 0) if bins == "bins4" else
                 2 + (2 if s >= 900 else 1 if s >= 700 else 0) if bins == "bins3" else 2)
            pairs += [(x, y, b), (y, x, b)]
        got = sorted(t for t in torch.stack([src, dst, rel], 1).tolist() if t[2] >= 2)
        check("G6_background_equals_STRING_file_with_bins", got == sorted(map(list, pairs)), f"{len(got)} directed background edges")
    else:
        check("G6_background_equals_STRING_file_with_bins", not any(t[2] >= 2 for t in trip), "no background configured")
    check("G7_relation_ids_in_range", set(rel.unique().tolist()) <= set(range(KN.n_relations(cfg))),
          sorted(rel.unique().tolist()))
    if not a.no_perturb:
        with tempfile.TemporaryDirectory(prefix="kegg_graph_audit_") as td:
            pd = perturbed_copy(sd, Path(td) / sd.name)
            model2, graph2, *_ = KN.load_run(a.run_dir, "cpu", pd)
            same = all(torch.equal(x, y) for x, y in zip(graph[:3], graph2[:3])) and \
                all(torch.equal(x, y) for x, y in zip(graph[3], graph2[3]))
            check("G8_perturbed_val_test_labels_graph_bit_identical", same)
            anchors = sorted({u for u, _ in P["test"]})[:a.n_anchors]
            _, S1 = KN.anchor_scores(model, anchors, KN.TAIL, graph, 8)
            _, S2 = KN.anchor_scores(model2, anchors, KN.TAIL, graph2, 8)
            check("G9_perturbed_val_test_labels_scores_bit_identical", torch.equal(S1, S2), f"{len(anchors)} anchors")
    res = {"run_dir": str(a.run_dir), "split_dir": sd.name, "checks": C, "all_pass": all(c["pass"] for c in C.values())}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print("ALL PASS" if res["all_pass"] else "SOME CHECKS FAILED", flush=True)
    sys.exit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    main()

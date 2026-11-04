#!/usr/bin/env python3
"""
PathArrowNet prediction: next-KO scores for anchor KOs from a trained run (config.json + best.pt). Inference only, eval mode.

    cd $KEGG_HOME/pathway
    # top-10 next KOs of two anchors
    python -m patharrownet.predict --run-dir <run dir> --anchors K00844,K01810 --topk 10 --out <out dir>
    # anchors from a file (one KO id or token id per line)
    python -m patharrownet.predict --run-dir <run dir> --query-file anchors.txt --out <out dir>
    # the anchors of the benchmark test queries of a setting (optionally the first N), with filtered ranks
    python -m patharrownet.predict --run-dir <run dir> --setting strict_v1_full --split test --max-anchors 50 --out <out dir>
    # full benchmark dump read by the evaluation scripts: tail_val.npz and tail_test.npz with every anchor the val/test
    # queries of tasks A and S need (benchmark_v2/scorelib.anchors_needed); --with-targets adds the val/test targets, which the
    # edge-direction analysis needs (s(b -> a) for a test edge a -> b)
    python -m patharrownet.predict --run-dir <run dir> --setting strict_v1_full --dump-benchmark \
        --out benchmark_v2/scores/strict_v1_full/clean_nbf_kgml/s42

Inputs  <run dir>/config.json (architecture, feature names, background file and binning, split), <run dir>/best.pt
        (weights; the frozen feature tables are stored in the checkpoint as buffers), the split directory (train-owned
        edges = message-passing graph; leakage asserts are enforced) and the STRING background file named in the config.
Outputs <out>/topk.tsv          anchor, rank, ko, token_id, log_prob, train_edge (1 if anchor -> ko is a train edge)
        <out>/tail_<split>.npz  score dump in the benchmark_v2 format read by benchmark_v2/scorelib.py and every
                                evaluation script: anchors [n] int64, scores [n, V] float32 (log-softmax over all KOs,
                                -1e30 for the 3 special tokens)
        <out>/ranks_<split>_A.tsv, <out>/summary.json   only with --setting/--split: filtered rank of every task-A query
                                whose anchor was scored (same filtered protocol as scorelib.ranks)
--direction head scores predecessors instead (anchor = the later KO).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark_v2"))
from patharrownet.models import patharrownet as KN  # noqa: E402


def parse_anchor(tok, vocab):
    tok = tok.strip()
    if not tok or tok.startswith("#"):
        return None
    if tok.isdigit():
        return int(tok)
    if tok in vocab:
        return vocab[tok]
    if f"ko:{tok}" in vocab:
        return vocab[f"ko:{tok}"]
    raise SystemExit(f"unknown KO {tok!r} (not in the run's vocabulary)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="directory with config.json and best.pt")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--split-dir", default=None, help="override the split directory recorded in config.json")
    ap.add_argument("--anchors", default=None, help="comma-separated KO ids (e.g. K00844) or token ids")
    ap.add_argument("--query-file", default=None, help="file with one KO id / token id per line")
    ap.add_argument("--setting", default=None, help="benchmark setting (scorelib), e.g. strict_v1_full")
    ap.add_argument("--split", default=None, choices=["val", "test"], help="score the anchors of this split's queries")
    ap.add_argument("--max-anchors", type=int, default=None, help="only the first N anchors (sorted token ids)")
    ap.add_argument("--dump-benchmark", action="store_true",
                    help="with --setting: score every anchor needed by the val/test queries of tasks A and S and write identical "
                         "tail_val.npz / tail_test.npz (the score-dump format of benchmark_v2/scores/<setting>/<method>/s<seed>/)")
    ap.add_argument("--with-targets", action="store_true", help="--dump-benchmark: also score the val/test task-A targets as anchors")
    ap.add_argument("--direction", default="tail", choices=["tail", "head"])
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--bs", type=int, default=16, help="anchors per forward pass")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("NTHREADS", "8")))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    split_dir = a.split_dir
    if a.setting and not split_dir:
        import scorelib as SL
        split_dir = SL.SPLIT_DIRS[a.setting][0]
    model, graph, vocab, V, cfg = KN.load_run(a.run_dir, a.device, split_dir, a.ckpt)
    inv = {v: k for k, v in vocab.items()}

    anchors, queries = [], None
    if a.anchors:
        anchors += [x for x in (parse_anchor(t, vocab) for t in a.anchors.split(",")) if x is not None]
    if a.query_file:
        anchors += [x for x in (parse_anchor(t, vocab) for t in Path(a.query_file).read_text().split()) if x is not None]
    if a.dump_benchmark:
        if not a.setting or a.split or a.direction != "tail":
            raise SystemExit("--dump-benchmark needs --setting, no --split and tail direction")
        import scorelib as SL
        anchors += SL.anchors_needed(a.setting)
        if a.with_targets:
            anchors += [q["target"] for sp in ("val", "test") for q in SL.load_queries(a.setting, sp, "A")]
    if a.split:
        if not a.setting:
            raise SystemExit("--split needs --setting")
        import scorelib as SL
        queries = SL.load_queries(a.setting, a.split, "A")
        anchors += sorted({q["source"] for q in queries})
    anchors = sorted(set(anchors))
    if a.max_anchors:
        anchors = anchors[:a.max_anchors]
    if not anchors:
        raise SystemExit("no anchors given (--anchors, --query-file or --setting/--split)")

    qtype = KN.TAIL if a.direction == "tail" else KN.HEAD
    idx, S = KN.anchor_scores(model, anchors, qtype, graph, a.bs)
    S = torch.nan_to_num(S, neginf=-1e30).cpu().numpy()
    S = S[[idx[x] for x in anchors]].astype(np.float32)
    tag = a.split or "query"
    for t in (("val", "test") if a.dump_benchmark else (tag,)):
        np.savez(out / f"{'tail' if qtype == KN.TAIL else 'head'}_{t}.npz",
                 anchors=np.asarray(anchors, dtype=np.int64), scores=S)

    train_edges = set(zip(graph[3][0].tolist(), graph[3][1].tolist()))
    with (out / "topk.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["anchor", "rank", "ko", "token_id", "log_prob", "train_edge"])
        for i, u in enumerate(anchors):
            row = S[i].copy()
            row[:3] = -np.inf
            row[u] = -np.inf                                   # a KO is never its own next KO
            top = np.argsort(-row, kind="stable")[:a.topk]
            for r, v in enumerate(top, 1):
                e = (u, int(v)) if qtype == KN.TAIL else (int(v), u)
                w.writerow([inv[u], r, inv[int(v)], int(v), f"{row[v]:.6f}", int(e in train_edges)])

    summary = {"run_dir": str(a.run_dir), "n_anchors": len(anchors), "direction": a.direction, "V": V,
               "config": {k: cfg.get(k) for k in ("feats", "background", "bg_bins4", "d", "layers", "agg", "loss")}}
    if queries is not None and qtype == KN.TAIL:
        import scorelib as SL
        keep = set(anchors)
        sub = [(i, q) for i, q in enumerate(queries) if q["source"] in keep]
        pos = {u: i for i, u in enumerate(anchors)}
        M = S[[pos[q["source"]] for _, q in sub]]
        rk, _ = SL.ranks(M, [q for _, q in sub], V)
        with (out / f"ranks_{a.split}_A.tsv").open("w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["query_index", "rank"])
            w.writerows([(i, float(r)) for (i, _), r in zip(sub, rk)])
        summary.update({"setting": a.setting, "split": a.split, "n_queries_scored": len(sub),
                        "n_queries_total": len(queries), "micro_MRR_on_scored_queries": float(np.mean(1 / rk))})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

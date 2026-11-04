#!/usr/bin/env python3
"""
Re-load saved GNN checkpoints (directed GNN runs: config.json + best.pt; model in patharrownet/models/dir_gnn.py) and store their full candidate score matrices for the
benchmark_v2 evaluator. Nothing is retrained. The recomputed task-A test ranks are checked against the
ranks_test_A.csv.gz that the original run wrote (MRR difference must be < 1e-3), which verifies that the
stored checkpoint is the one whose numbers were reported.

    .venv/bin/python benchmark_v2/dump_gnn.py --runs 'kegg_dataset/comparisons_ko/strict/models/gnn_*' --method-from-dir
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

B2 = Path(__file__).resolve().parent
ROOT = B2.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(B2))
import scorelib as SL  # noqa: E402
from patharrownet.models import dir_gnn as G  # noqa: E402
from patharrownet.data import SPLIT, load_train_edges  # noqa: E402
from kegg_pipeline import evaluation as E  # noqa: E402
from patharrownet.data import load_go, load_vocab  # noqa: E402
from collections import defaultdict  # noqa: E402


def build(cfg, device):
    vocab, V = load_vocab(cfg.get("full_candidates", False))
    sd = Path(cfg.get("split_dir", str(SPLIT)))
    if not sd.exists():  # absolute path of the original run: re-root to this repository
        from patharrownet.models.patharrownet import resolve_split_dir
        sd = resolve_split_dir(cfg)
    val_q = {"A": E.load_queries(sd / "val_A_next_node.csv.gz", "A")}
    test_q = {"A": E.load_queries(sd / "test_A_next_node.csv.gz", "A")}
    edges = load_train_edges(vocab, val_q, test_q, sd, cfg.get("allow_leaky", False))
    feat_cfg = cfg["config"].replace("gcn_", "gnn_")
    train_tokens = sorted({x for e in edges for x in e})
    seen = torch.zeros(V, dtype=torch.bool)
    seen[train_tokens] = True
    go_sets, n_go = None, 0
    if feat_cfg in ("gnn_id+go", "gnn_id+go_noBP"):
        raw = load_go(vocab, "go_profile" if feat_cfg == "gnn_id+go" else "go_profile_no_BP")
        cnt = defaultdict(int)
        for gs in raw.values():
            for g in gs:
                cnt[g] += 1
        gid = {g: i for i, g in enumerate(sorted(g for g, c in cnt.items() if c >= 2))}
        go_sets = {v: sorted(gid[g] for g in gs if g in gid) for v, gs in raw.items()}
        n_go = len(gid)
    dense = []
    if feat_cfg in G.DENSE or feat_cfg == "gnn_id+dense":
        kf = torch.load(ROOT / "kegg_features" / cfg.get("feature_file", "ko_features.pt"))
        names = G.DENSE[feat_cfg] if feat_cfg in G.DENSE else [n for n in cfg["dense"].split(",") if n]
        dense = [(kf["features"][n]["X"], kf["features"][n]["has"]) for n in names]
    background = None
    if cfg.get("background"):
        pairs = []
        bg = Path(cfg["background"])
        bg = bg if bg.is_absolute() else ROOT / bg
        with open(bg) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                x, y = vocab[r["ko_a"]], vocab[r["ko_b"]]
                pairs += [(x, y), (y, x)]
        background = torch.tensor(pairs, dtype=torch.long)
    model = G.DirGNN(V, cfg["d"], cfg["layers"], cfg["dropout"], seen, go_sets, n_go, cfg["id_dropout"], dense,
                     undirected=cfg["config"].startswith("gcn_"), background=background).to(device)
    return model, torch.tensor(edges, device=device), V


@torch.no_grad()
def dump(run, setting, method, seed, device):
    cfg = json.loads((run / "config.json").read_text())
    model, E_all, V = build(cfg, device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    model.eval()
    h = model.encode(E_all[:, 0], E_all[:, 1])
    out = SL.SCORES / setting / method / f"s{seed}"
    anchors = SL.anchors_needed(setting)
    sc = torch.cat([model.score_targets(h, torch.tensor(anchors[i:i + 1024], device=device)).float().cpu()
                    for i in range(0, len(anchors), 1024)]).numpy()
    for split in ("val", "test"):
        SL.save_tail(out / f"tail_{split}.npz", anchors, sc)
    # verification against the ranks the original run wrote
    q = SL.load_A(setting, "test")
    rk, _ = SL.ranks(SL.query_scores(out, "test", "A", q), q, V)
    with gzip.open(run / "ranks_test_A.csv.gz", "rt") as f:
        old = [float(r["rank"]) for r in csv.DictReader(f)]
    d = abs(np.mean(1 / rk) - np.mean(1 / np.array(old)))
    (out / "source.json").write_text(json.dumps({"run_dir": str(run), "config": cfg["config"], "seed": seed,
                                                  "recomputed_vs_saved_test_A_MRR_absdiff": d}, indent=1))
    status = "OK" if d < 1e-3 else "MISMATCH"
    print(f"{status} {method} s{seed}: test A MRR recomputed {np.mean(1 / rk):.4f} vs saved {np.mean(1 / np.array(old)):.4f}", flush=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="glob(s) of run dirs")
    ap.add_argument("--setting", default="strict_v1")
    ap.add_argument("--prefix", default="", help="prefix for the method name")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = sorted({p for g in a.runs for p in glob.glob(str(ROOT / g) if not g.startswith("/") else g)})
    for r in runs:
        r = Path(r)
        if not (r / "best.pt").exists():
            continue
        m = re.match(r"(.+)_s(\d+)$", r.name)
        dump(r, a.setting, a.prefix + m.group(1), int(m.group(2)), device)


if __name__ == "__main__":
    main()

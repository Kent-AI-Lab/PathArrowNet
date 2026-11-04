"""
Inference-time data loading for the KO models: vocabulary, train-owned message-passing edges (with the leakage
asserts), STRING background relations, frozen protein-feature tables, and the filtered-rank helper.

All paths are relative to the `pathway/` directory of the release (ROOT).
"""
from __future__ import annotations

import csv
import gzip
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent          # .../pathway
DS = ROOT / "kegg_dataset"
SPLIT = DS / "splits_ko"
LEAKY_ROOT = ROOT / "kegg_leaky_splits"
FEATURES = ROOT / "kegg_features"


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def load_vocab(full=False):
    """full=True: all 11,620 map KOs (token_vocab_ko_full.csv; ids 0..2872 identical to token_vocab_ko.csv)."""
    with (DS / ("token_vocab_ko_full.csv" if full else "token_vocab_ko.csv")).open() as f:
        rows = list(csv.DictReader(f))
    return {r["token"]: int(r["token_id"]) for r in rows}, len(rows)


def load_train_edges(vocab, val_q, test_q, split_dir, allow_leaky):
    """Train-owned KO edges of a split = the message-passing graph at inference.

    Leakage guards are always enforced on the strict splits: every edge must be train-owned according to
    edge_split.csv, and no val/test query edge may be in the graph. They can only be disabled (allow_leaky=True)
    for the intentionally leaky control splits under kegg_leaky_splits/."""
    split_dir = Path(split_dir)
    leaky_dir = LEAKY_ROOT in split_dir.resolve().parents
    if allow_leaky != leaky_dir:
        raise SystemExit("allow_leaky is required for, and only allowed with, kegg_leaky_splits/*")
    edges = sorted({(int(r["context_token_id"]), int(r["target_token_id"]))
                    for r in read_gz(split_dir / "train_A_next_node.csv.gz")})
    es = set(edges)
    leak = 0
    for qd in (val_q, test_q):
        leak += sum(1 for q in qd.get("A", []) if (q["source"], q["target"]) in es)
        leak += sum(1 for q in qd.get("C", []) if (q["left"], q["target"]) in es or (q["target"], q["right"]) in es)
    if allow_leaky:
        print(f"LEAKY CONTROL SPLIT: {leak} evaluation edges are in the message-passing graph", flush=True)
        return edges
    tok = {v: k for k, v in vocab.items()}
    owner = {}
    with (split_dir / "edge_split.csv").open() as f:
        for r in csv.DictReader(f):
            owner[(r["source_ko"], r["target_ko"])] = r["split"]
    bad = sum(1 for a, b in edges if owner.get((tok[a], tok[b])) != "train")
    assert bad == 0, f"LEAKAGE: {bad} message-passing edges are not train-owned"
    assert leak == 0, f"LEAKAGE: {leak} evaluation edges are in the message-passing graph"
    return edges


def load_background(path, vocab, bins=None):
    """STRING background (undirected KO pairs, both directions) and per-row relation ids.

    bins: None -> every background edge is relation 2; 'bins3' -> 2 + (0: <700, 1: 700-899, 2: >=900);
          'bins4' -> 2 + (0: <400, 1: 400-699, 2: 700-899, 3: >=900).
    Returns (pairs LongTensor [2n, 2], rel LongTensor [2n] or None)."""
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    pairs, rel = [], []
    with path.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            x, y = vocab[r["ko_a"]], vocab[r["ko_b"]]
            s = int(float(r["max_experimental_score"]))
            pairs += [(x, y), (y, x)]
            if bins == "bins4":
                rel += [2 + (3 if s >= 900 else 2 if s >= 700 else 1 if s >= 400 else 0)] * 2
            elif bins == "bins3":
                rel += [2 + (2 if s >= 900 else 1 if s >= 700 else 0)] * 2
    return torch.tensor(pairs, dtype=torch.long), (torch.tensor(rel, dtype=torch.long) if bins else None)


def load_dense(feats, feature_file, V):
    """Frozen feature tables [(X [V, D], has [V])] from kegg_features/<feature_file>."""
    names = [f for f in feats.split(",") if f] if isinstance(feats, str) else list(feats)
    if not names:
        return []
    kf = torch.load(FEATURES / feature_file)
    assert kf["vocab_size"] == V, (kf["vocab_size"], V)
    return [(kf["features"][n]["X"], kf["features"][n]["has"]) for n in names]


def gpu_ranks(scores, qs, cand_mask):
    """Filtered mean-tie ranks for a [Q, V] torch score matrix; identical semantics to
    kegg_pipeline.evaluation.filtered_rank (rank = 1 + #higher + #equal / 2 among the valid candidates)."""
    Q = scores.shape[0]
    valid = cand_mask[None, :].expand(Q, -1).clone()
    rows, cols = [], []
    for i, q in enumerate(qs):
        for c in q["exclude"]:
            rows.append(i)
            cols.append(c)
    dev = scores.device
    if rows:
        valid[torch.tensor(rows, device=dev), torch.tensor(cols, device=dev)] = False
    t = torch.tensor([q["target"] for q in qs], device=dev)
    valid[torch.arange(Q, device=dev), t] = False
    s = scores.gather(1, t[:, None])
    gt = ((scores > s) & valid).sum(1).float()
    eq = ((scores == s) & valid).sum(1).float()
    return (1 + gt + eq / 2).tolist()


def macro_mrr(qs, ranks):
    w = sum(q["weight"] for q in qs)
    return sum(q["weight"] / r for q, r in zip(qs, ranks)) / w


def load_go(vocab, variant):
    """KO GO sets of the go_profile baselines (evaluate_ko_baselines.GO_VARIANTS); used by the GO-bag GNN configs."""
    from evaluate_ko_baselines import GO_VARIANTS, load_ko_go
    use_kegg, aspects, drop_iea = GO_VARIANTS[variant]
    return load_ko_go(vocab, use_kegg, aspects, drop_iea)

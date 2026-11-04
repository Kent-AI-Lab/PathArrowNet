#!/usr/bin/env python3
"""
Feng et al. 2020-style pathway-holdout splits on our KO graph (task A), to test whether cross-pathway
duplicate edges inflate a "pathway-level" evaluation.

Feng et al. (bioRxiv 2020.12.23.424230): all interactions of 10 test pathways are test data, 180 pathways
train, 33 validation, 4-fold cross-validation; no deduplication across pathways is described.
We reproduce that protocol at the KO level on our 227 pathways with directed KO edges:

    kegg_leaky_splits/feng_fold{k}/        test = all unique edges of the 10 test pathways (Feng protocol)
    kegg_leaky_splits/feng_fold{k}_dedup/  same pathways, minus test/val edges that also occur in a
                                           train pathway (only exact duplicates removed; reverse edges and
                                           pathway neighbourhoods remain)

Folds use disjoint test pathways (seed 42). Same file format as kegg_leaky_splits/*.
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_ko_leakage_free_split import add_macro_weight, load_ko_graphs  # noqa: E402
from kegg_pipeline.common import OUT, log, read_csv, write_csv  # noqa: E402

LEAKY = Path(__file__).resolve().parent / "kegg_leaky_splits"
N_TEST, N_VAL, FOLDS, SEED = 10, 33, 4, 42


def write(d, parts, graphs, vocab, dedup):
    d.mkdir(parents=True, exist_ok=True)
    edges = {}
    for s in ("train", "val", "test"):
        e = defaultdict(set)
        for p in parts[s]:
            for uv in graphs[p]:
                e[uv].add(p)
        edges[s] = e
    if dedup:
        for s in ("val", "test"):
            edges[s] = {uv: ps for uv, ps in edges[s].items() if uv not in edges["train"]}
        edges["test"] = {uv: ps for uv, ps in edges["test"].items() if uv not in edges["val"]}
    train_out = defaultdict(set)
    for u, v in edges["train"]:
        train_out[u].add(v)
    train_tokens = {x for e in edges["train"] for x in e}
    stats = {}
    for s in ("train", "val", "test"):
        out_s = defaultdict(set)
        for u, v in edges[s]:
            out_s[u].add(v)
        rows = [{"split": s, "context_token_id": vocab[u], "target_token_id": vocab[v],
                 "context_token": f"ko:{u}", "target_token": f"ko:{v}", "n_pathways": len(ps),
                 "pathway_ids": ";".join(sorted(ps)), "context_seen_in_train": int(u in train_tokens),
                 "target_seen_in_train": int(v in train_tokens),
                 "valid_targets_in_split": " ".join(str(vocab[x]) for x in sorted(out_s[u])),
                 "known_targets_in_train": " ".join(str(vocab[x]) for x in sorted(train_out.get(u, ())))
                 if s != "train" else ""}
                for (u, v), ps in sorted(edges[s].items())]
        add_macro_weight(rows)
        write_csv(d / f"{s}_A_next_node.csv.gz", rows, list(rows[0].keys()))
        if s == "test":
            te = set(edges[s])
            stats = {"test_pathways": ";".join(parts["test"]), "test_edges": len(te),
                     "test_edges_also_in_train": sum(1 for e in te if e in edges["train"]),
                     "reverse_in_train": sum(1 for u, v in te if (v, u) in edges["train"])}
    (d / "README.md").write_text(
        f"# Feng-style pathway holdout ({'deduplicated' if dedup else 'original protocol'})\n\n"
        f"Leaky control split, for comparison only.\n\n" + "\n".join(f"* {k}: {v}" for k, v in stats.items()) + "\n")
    return stats


def main():
    graphs = load_ko_graphs()
    vocab = {r["token"]: int(r["token_id"]) for r in read_csv(OUT / "token_vocab_ko.csv")}
    pids = sorted(graphs)
    rng = random.Random(f"{SEED}|feng")
    rng.shuffle(pids)
    for k in range(FOLDS):
        test = pids[k * N_TEST:(k + 1) * N_TEST]
        rest = [p for p in pids if p not in test]
        rng_k = random.Random(f"{SEED}|feng|{k}")
        rng_k.shuffle(rest)
        parts = {"test": sorted(test), "val": sorted(rest[:N_VAL]), "train": sorted(rest[N_VAL:])}
        for dedup in (False, True):
            st = write(LEAKY / f"feng_fold{k + 1}{'_dedup' if dedup else ''}", parts, graphs, vocab, dedup)
            log(f"fold {k + 1} {'dedup' if dedup else 'orig '}: test edges {st['test_edges']}, "
                f"also in train {st['test_edges_also_in_train']}, reverse in train {st['reverse_in_train']}")


if __name__ == "__main__":
    main()

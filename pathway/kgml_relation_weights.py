#!/usr/bin/env python3
"""Relation-level KGML weights w(e) of the training edges of a split (paper Sec. 3.2; used by the primary PathArrowNet during training).
KO edges come from directional gene -> gene KGML relations expanded to every KO pair of the two boxes. Only relations in TRAIN pathways of the split are used.
A relation g is identified by its (source KO set, target KO set) signature, so an arrow repeated in several maps (near-copy
disease maps) or twice in one map counts once; n_g = number of KO pairs it expands to (self pairs dropped).
Weight of a train edge e:  w(e) = min(1, sum over signatures g that produce e of 1 / n_g).
So an arrow between two single-KO boxes gives its pair weight 1, the pairs of an arrow that expands to n pairs share weight 1
(1/n each) unless another relation also supports them, and no pair weighs more than one curated arrow. Train edges that no
train-pathway relation reproduces keep weight 1 (counted). train_ko_nbf.py --edge-weights samples batch edges proportional to w
and makes each anchor's soft target proportional to w.
Also written (stats json): weight distribution; whether the pairs of one raw record cross splits: by the ownership rule every
pair of a train-pathway record is train-owned (checked), so the check is on val/test-pathway records that expand into pairs owned
by train and pairs owned by val/test (their val/test edges have box-mates in the training graph).
    python kgml_relation_weights.py <setting>
    -> kegg_dataset/kgml_weights/<setting>.tsv + <setting>.json
"""
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PW = Path(__file__).resolve().parent
KD = PW / "kegg_dataset"
OUTD = KD / "kgml_weights"
SPLITS = {"strict_v1_full": KD / "splits_ko", **{f"strict_s{s}_full": PW / "kegg_splits_robustness" / f"seed{s}" for s in (43, 44, 45, 46)}}
csv.field_size_limit(10 ** 9)


def ko_sets():
    kos = {}
    with (KD / "node_occurrences.csv").open() as f:
        for r in csv.DictReader(f):
            if r["entry_type"] != "gene":
                continue
            s = set((r["derived_ko_ids_from_hsa_genes"] or r["ko_ids"] or "").replace(";", " ").split())
            kos[(r["pathway_id"], r["entry_id"])] = frozenset(k.replace("ko:", "") for k in s)
    return kos


def records(kos):
    """(pathway, relation_index, S, T) for directional gene -> gene relations."""
    out = []
    with (KD / "kgml_relations.csv").open() as f:
        for r in csv.DictReader(f):
            if r["whether_directional"] != "1" or r["source_entry_type"] != "gene" or r["target_entry_type"] != "gene":
                continue
            p = r["pathway_id"]
            out.append((p, r["relation_index"], kos.get((p, r["source_entry"]), frozenset()), kos.get((p, r["target_entry"]), frozenset())))
    return out


def pairs(S, T):
    return [(a, b) for a in S for b in T if a != b]


def main(setting):
    sd = SPLITS[setting]
    psplit = {r["pathway_id"]: r["split"] for r in csv.DictReader((sd / "pathway_split.csv").open())}
    owner = {(r["source_ko"], r["target_ko"]): r["split"] for r in csv.DictReader((sd / "edge_split.csv").open())}
    with gzip.open(sd / "train_A_next_node.csv.gz", "rt") as f:
        train = sorted({(r["context_token"].replace("ko:", ""), r["target_token"].replace("ko:", "")) for r in csv.DictReader(f)})
    tset = set(train)
    recs = records(ko_sets())
    sig_pairs = {}
    for p, _, S, T in recs:
        if psplit.get(p) == "train" and S and T:
            sig_pairs.setdefault((S, T), pairs(S, T))
    acc = defaultdict(float); n_sig = Counter(); not_train = 0
    for (S, T), ps in sig_pairs.items():
        if not ps:
            continue
        for e in ps:
            if e in tset:
                acc[e] += 1.0 / len(ps); n_sig[e] += 1
            else:
                not_train += 1
    w = {e: min(1.0, acc[e]) if e in acc else 1.0 for e in train}
    OUTD.mkdir(parents=True, exist_ok=True)
    with (OUTD / f"{setting}.tsv").open("w") as f:
        f.write("source_ko\ttarget_ko\tweight\tn_signatures\n")
        for e in train:
            f.write(f"{e[0]}\t{e[1]}\t{w[e]:.6g}\t{n_sig[e]}\n")
    W = np.array([w[e] for e in train]); fac = Counter(len(ps) for ps in sig_pairs.values())
    # val/test-pathway records whose pairs are owned by different splits
    cross = Counter(); cross_edges = {"val": set(), "test": set()}; heldout_edges = {"val": set(), "test": set()}
    for p, _, S, T in recs:
        sp = psplit.get(p)
        if sp not in ("val", "test") or not S or not T:
            continue
        own = Counter(owner.get(e, "none") for e in pairs(S, T))
        held = [e for e in pairs(S, T) if owner.get(e) == sp]
        heldout_edges[sp].update(held)
        if held and own["train"]:
            cross[sp] += 1; cross_edges[sp].update(held)
    st = {"setting": setting, "split_dir": str(sd), "n_train_edges": len(train), "n_train_signatures": len(sig_pairs),
          "signature_pair_count_distribution": {str(k): v for k, v in sorted(fac.items())},
          "train_edges_without_train_relation": int(sum(e not in acc for e in train)),
          "signature_pairs_not_train_owned": not_train,
          "weight_eq_1": int((W == 1).sum()), "weight_lt_1": int((W < 1).sum()), "weight_mean": float(W.mean()),
          "weight_quantiles": {q: float(np.quantile(W, q)) for q in (0.05, 0.1, 0.25, 0.5)},
          "effective_mass_share": float(W.sum() / len(W)),
          "heldout_records_crossing_train": dict(cross),
          "heldout_edges_with_train_boxmates": {k: len(v) for k, v in cross_edges.items()},
          "heldout_edges_reproduced": {k: len(v) for k, v in heldout_edges.items()}}
    (OUTD / f"{setting}.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st), flush=True)


if __name__ == "__main__":
    for s in (sys.argv[1:] or SPLITS):
        main(s)

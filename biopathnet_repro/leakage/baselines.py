#!/usr/bin/env python3
"""
Non-learned baselines on BioPathNet's KEGG gene-function task (data/PC_KEGG_0602), scored with the same
protocol as BioPathNet's KnowledgeGraphCompletionBiomed.evaluate:
  * both directions: tail (gene -> pathway, candidates = KEGG pathway nodes) and head (pathway -> gene,
    candidates = gene nodes); MRR = mean over both directions
  * filtered: all other true (gene, KEGGPathway, pathway) triples of train2/valid/test removed
  * pessimistic ties: rank = 1 + #{candidates with score >= true score}
Methods (fitted on train2 only, no tuning):
  popularity        pathway size / gene degree in train2
  ko_copy           pathways of same-KO paralogs in train2 (KEGG link/ko/hsa)
  comembership      sum over the gene's train2 pathways q of Jaccard(q, p) (pathway overlap in train2)
  ko_copy+comember  sum of the two (fixed 1:1)
Every method adds 1e-6 x popularity as tie-breaker.

    python leakage/baselines.py --split-dir BioPathNet/data/PC_KEGG_0602
"""

from __future__ import annotations
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

KEGG_CACHE = Path(_KH + "/pathway/kegg_dataset_cache/kegg_rest")


def trip(p):
    return [l.rstrip("\n").split("\t") for l in open(p) if l.strip()]


def kegg_maps():
    sym2id = {}
    for line in open(KEGG_CACHE / "list_hsa.tsv"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 4 and ";" in p[3]:
            for i, s in enumerate(p[3].split(";")[0].split(",")):
                s = s.strip()
                if i == 0 or s not in sym2id:
                    sym2id.setdefault(s, p[0][4:])
    g2k = defaultdict(set)
    for line in open(KEGG_CACHE / "link_ko_hsa.tsv"):
        a, b = line.split()
        g2k[a[4:]].add(b[3:])
    return sym2id, g2k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.split_dir)
    tr, va, te = trip(d / "train2.txt"), trip(d / "valid.txt"), trip(d / "test.txt")
    types = dict(l.split() for l in open(d / "entity_types.txt") if l.strip())
    genes = sorted(e for e, t in types.items() if t == "genes")
    pws = sorted(e for e, t in types.items() if t == "KEGG")
    gi, pi = {g: i for i, g in enumerate(genes)}, {p: i for i, p in enumerate(pws)}
    G, P = len(genes), len(pws)

    M = np.zeros((G, P), dtype=np.float32)          # train2 membership
    for h, _, t in tr:
        M[gi[h], pi[t]] = 1
    allpos = defaultdict(set)
    for h, _, t in tr + va + te:
        allpos[(h, "t")].add(t)
        allpos[(t, "h")].add(h)
    pw_size, g_deg = M.sum(0), M.sum(1)

    sym2id, g2k = kegg_maps()
    ko_of = [g2k.get(sym2id.get(g, ""), set()) for g in genes]
    ko_members = defaultdict(list)
    for i, ks in enumerate(ko_of):
        for k in ks:
            ko_members[k].append(i)
    # KO-copy matrix: K[g, p] = number of same-KO paralogs (excluding g) that are in p in train2
    K = np.zeros((G, P), dtype=np.float32)
    for i, ks in enumerate(ko_of):
        par = {j for k in ks for j in ko_members[k] if j != i}
        if par:
            K[i] = M[list(par)].sum(0)
    # pathway co-membership: J = Jaccard between pathways in train2; C[g, p] = sum_q M[g,q] J[q,p]
    inter = M.T @ M
    union = pw_size[:, None] + pw_size[None, :] - inter
    J = np.where(union > 0, inter / np.maximum(union, 1), 0)
    np.fill_diagonal(J, 0)
    Cm = M @ J

    def norm(x):
        m = x.max()
        return x / m if m > 0 else x
    methods = {
        "popularity": np.zeros((G, P), dtype=np.float32),
        "ko_copy": K,
        "comembership": Cm,
        "ko_copy+comember": norm(K) + norm(Cm),
    }
    tail_tb = 1e-6 * pw_size / max(pw_size.max(), 1)
    head_tb = 1e-6 * g_deg / max(g_deg.max(), 1)

    res = {}
    for name, S in methods.items():
        out = {}
        for split, T in (("valid", va), ("test", te)):
            rt, rh = [], []
            for h, _, t in T:
                g, p = gi[h], pi[t]
                # tail: rank pathway p among pathways for gene g
                s = S[g] + tail_tb
                mask = np.ones(P, bool)
                mask[[pi[x] for x in allpos[(h, "t")]]] = False
                rt.append(1 + int(((s >= s[p]) & mask).sum()))
                # head: rank gene g among genes for pathway p
                s = S[:, p] + head_tb
                mask = np.ones(G, bool)
                mask[[gi[x] for x in allpos[(t, "h")]]] = False
                rh.append(1 + int(((s >= s[g]) & mask).sum()))
            rt, rh = np.array(rt), np.array(rh)
            both = np.concatenate([rt, rh])
            out[split] = {"MRR": float((1 / both).mean()), "MRR_tail": float((1 / rt).mean()),
                          "MRR_head": float((1 / rh).mean()), "Hits@10": float((both <= 10).mean())}
        res[name] = out
        print(f"{name:18s} test MRR {out['test']['MRR']:.3f} (tail {out['test']['MRR_tail']:.3f}, head "
              f"{out['test']['MRR_head']:.3f}), Hits@10 {out['test']['Hits@10']:.3f} | valid MRR {out['valid']['MRR']:.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

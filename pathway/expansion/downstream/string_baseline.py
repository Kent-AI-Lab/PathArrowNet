#!/usr/bin/env python3
"""STRING-only zero-shot controls for the pair tasks (PathArrowNet uses the STRING-exp150 KO background graph, so its
zero-shot transfer must be compared with what STRING alone gives).

Scores (KO level, STRING v12 experimental channel >= 150, the background used by PathArrowNet; gene level = max over KO pairs):
  string_direct  experimental score of the KO pair (0 if absent)
  string_aa      Adamic-Adar over the undirected STRING-exp150 KO graph (2-hop evidence)
Also re-scores the PathArrowNet / encoder caches on exactly the same test pairs.

    .venv/bin/python expansion/downstream/string_baseline.py lr|reactome
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from pair_metrics import TASKS, auprc, auroc  # noqa: E402


def main():
    task = sys.argv[1]
    d, fg, fp, fs = TASKS[task]; D = HERE / d
    genes = list(csv.DictReader((D / fg).open(), delimiter="\t")); gid = {g["entrez"]: i for i, g in enumerate(genes)}
    res_dir = HERE / ("results/lr" if task == "lr" else f"results/{task}")
    Ti, Tj, Ty, Tp = np.load(res_dir / "test_pairs.npy"); Ty = Ty.astype(bool)
    sf = {(r["gene_a"], r["gene_b"]): int(r["string_exp150"]) for r in csv.DictReader((D / fs).open(), delimiter="\t")}
    cl = {}
    for r in csv.DictReader((D / fp).open(), delimiter="\t"):
        if r["gene_a"] in gid and r["gene_b"] in gid:
            c = r["ko_edge_split"] == "none" and r["kgml_any_relation"] == "0" and r["ko_copathway_train"] == "0"
            s = sf.get((r["gene_a"], r["gene_b"]), 0)
            for k in ((gid[r["gene_a"]], gid[r["gene_b"]]), (gid[r["gene_b"]], gid[r["gene_a"]])):
                cl[k] = (c, c and s == 0)
    par = {Tp[k]: cl[(Ti[k], Tj[k])] for k in range(len(Ty)) if Ty[k]}
    posg = {Tp[k]: (Ti[k], Tj[k]) for k in range(len(Ty)) if Ty[k]}  # strata follow the parent positive
    strata = {"all": np.ones(len(Ty), bool), "clean": np.array([par[p][0] for p in Tp]),
              "clean_noSTRING": np.array([par[p][1] for p in Tp]),
              "test_test": np.array([all(genes[g]["gene_split"] == "test" for g in posg[p]) for p in Tp])}

    w, nb = {}, defaultdict(set)
    with open(ROOT / "kegg_dataset_cache/string/ko_background_exp150_full.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            a, b, s = r["ko_a"], r["ko_b"], float(r["max_experimental_score"])
            w[(a, b)] = w[(b, a)] = s; nb[a].add(b); nb[b].add(a)
    kos = [[k for k in (g["kos_in_vocab"] or "").split(";") if k] for g in genes]

    def direct(i, j):
        return max((w.get((x, y), 0.0) for x in kos[i] for y in kos[j]), default=0.0)

    def aa(i, j):
        best = 0.0
        for x in kos[i]:
            for y in kos[j]:
                best = max(best, sum(1 / np.log(len(nb[z]) + 1) for z in nb[x] & nb[y]))
        return best

    scores = {"string_direct": np.array([direct(i, j) for i, j in zip(Ti, Tj)]),
              "string_aa": np.array([aa(i, j) for i, j in zip(Ti, Tj)])}
    for c in sorted((HERE / "cache").glob(f"{task}_strict_v1_full_*.npz")):
        scores[c.stem.split("strict_v1_full_")[1]] = np.load(c)["P"][Ti, Tj]
    out = {}
    print(f"| score | " + " | ".join(f"{s} AUROC / AUPRC" for s in strata) + " |")
    print("|---|" + "---|" * len(strata))
    for n, sc in scores.items():
        out[n] = {s: {"auroc": auroc(sc[m], Ty[m]), "auprc": auprc(sc[m], Ty[m]), "pos_rate": float(Ty[m].mean())} for s, m in strata.items()}
        print(f"| {n} | " + " | ".join(f"{out[n][s]['auroc']:.3f} / {out[n][s]['auprc']:.3f}" for s in strata) + " |")
    print("pos rate:", {s: round(float(Ty[m].mean()), 3) for s, m in strata.items()}, "n pos:", {s: int((Ty & m).sum()) for s, m in strata.items()})
    (res_dir / "zero_shot_scores.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

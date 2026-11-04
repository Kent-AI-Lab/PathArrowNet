#!/usr/bin/env python3
"""Paired bootstrap (resampling positive-parent groups: a positive pair and its sampled negatives)
of the zero-shot AUROC difference PathArrowNet - X on LR / Reactome; X = original BioPathNet, STRING direct / Adamic-Adar,
untrained PathArrowNet. -> results/<task>/zero_shot_paired.json
    .venv/bin/python expansion/downstream/zs_paired.py"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import string_baseline as SB  # noqa: E402
from pair_metrics import auroc  # noqa: E402


def main():
    for task in ("lr", "reactome"):
        rd = HERE / ("results/lr" if task == "lr" else f"results/{task}")
        Ti, Tj, Ty, Tp = np.load(rd / "test_pairs.npy"); Ty = Ty.astype(bool)
        sc = {c.stem.split("strict_v1_full_")[1]: np.load(c)["P"][Ti, Tj] for c in sorted((HERE / "cache").glob(f"{task}_strict_v1_full_*.npz"))}
        # strata exactly as string_baseline.main
        import csv
        d, fg, fp, fs = SB.TASKS[task]; D = HERE / d
        genes = list(csv.DictReader((D / fg).open(), delimiter="\t")); gid = {g["entrez"]: i for i, g in enumerate(genes)}
        sf = {(r["gene_a"], r["gene_b"]): int(r["string_exp150"]) for r in csv.DictReader((D / fs).open(), delimiter="\t")}
        cl = {}
        for r in csv.DictReader((D / fp).open(), delimiter="\t"):
            if r["gene_a"] in gid and r["gene_b"] in gid:
                c = r["ko_edge_split"] == "none" and r["kgml_any_relation"] == "0" and r["ko_copathway_train"] == "0"
                s = sf.get((r["gene_a"], r["gene_b"]), 0)
                for k in ((gid[r["gene_a"]], gid[r["gene_b"]]), (gid[r["gene_b"]], gid[r["gene_a"]])):
                    cl[k] = (c, c and s == 0)
        par = {Tp[k]: cl[(Ti[k], Tj[k])] for k in range(len(Ty)) if Ty[k]}
        strata = {"clean": np.array([par[p][0] for p in Tp]), "clean_noSTRING": np.array([par[p][1] for p in Tp])}
        out = {}
        for sn, m in strata.items():
            idx = np.where(m)[0]; grp = Tp[idx]; ug = np.unique(grp); members = {g: idx[grp == g] for g in ug}
            rng = np.random.default_rng(0)
            boots = [np.concatenate([members[g] for g in rng.choice(ug, len(ug))]) for _ in range(2000)]
            ref = "konbf_kl_s0"
            for other in [k for k in sc if k != ref and not k.startswith("kegg_enc_r")]:
                d0 = auroc(sc[ref][idx], Ty[idx]) - auroc(sc[other][idx], Ty[idx])
                bs = [auroc(sc[ref][b], Ty[b]) - auroc(sc[other][b], Ty[b]) for b in boots]
                out[f"{sn}|konbf - {other}"] = [float(d0), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(2 * min(np.mean(np.array(bs) <= 0), np.mean(np.array(bs) >= 0)))]
                print(task, sn, "konbf -", other, "%+.3f [%+.3f, %+.3f] p=%.4f" % tuple(out[f"{sn}|konbf - {other}"]), flush=True)
        (rd / "zero_shot_paired.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

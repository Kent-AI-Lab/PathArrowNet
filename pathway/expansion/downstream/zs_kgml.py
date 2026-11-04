#!/usr/bin/env python3
"""Zero-shot LR (OmniPath ligand-receptor) / Reactome next-reaction transfer of the relation-weighted primary model
(KGML weighting; checkpoints kegg_dataset/nbf_ko_full/xp_kgmlw+exp150bin4+allneg+kl_s42..46).
1. gene-pair cache: kegg_pairscore.gene_pair_scores(kos, "konbf_kl", seed 0) with nbf_matrix(run=...) set to the KGML run
   -> cache_kgml/<task>_strict_v1_full_konbf_kgml_s0.npz (separate folder so the cache/ globs of string_baseline.py /
   zs_paired.py are unaffected).
2. AUROC / AUPRC per stratum (all, clean, clean_noSTRING, test_test), as in string_baseline.main,
   for konbf_kgml_s0 and every existing cache/<task>_strict_v1_full_*.npz score (re-scored to check reproduction).
3. paired bootstrap PathArrowNet(KGML) - X (clean, clean_noSTRING), as in zs_paired.main (2000 positive-parent
   resamples, rng seed 0, same draws for every X); X = the zs_paired controls.
   -> results/<task>/zero_shot_kgml.json
    .venv/bin/python expansion/downstream/zs_kgml.py"""
import csv, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kegg_pairscore as KP  # noqa: E402
from pair_metrics import TASKS, auroc, auprc  # noqa: E402

RUN = "xp_kgmlw+exp150bin4+allneg+kl"
CK = HERE / "cache_kgml"


def kgml_pairs(task, genes):
    f = CK / f"{task}_strict_v1_full_konbf_kgml_s0.npz"
    if f.exists():
        return np.load(f)["P"]
    orig = KP.nbf_matrix
    KP.nbf_matrix = lambda anchors, **kw: orig(anchors, run=RUN, **kw)
    try:
        P, has = KP.gene_pair_scores([g["kos_in_vocab"] for g in genes], "konbf_kl", 0, tag=None, dev="cuda" if __import__("torch").cuda.is_available() else "cpu")
    finally:
        KP.nbf_matrix = orig
    CK.mkdir(exist_ok=True); np.savez(f, P=P, has=has)
    return P


def main():
    for task in ("lr", "reactome"):
        d, fg, fp, fs = TASKS[task]; D = HERE / d
        genes = list(csv.DictReader((D / fg).open(), delimiter="\t")); gid = {g["entrez"]: i for i, g in enumerate(genes)}
        rd = HERE / ("results/lr" if task == "lr" else f"results/{task}")
        out_f = rd / "zero_shot_kgml.json"
        assert not out_f.exists(), out_f
        Ti, Tj, Ty, Tp = np.load(rd / "test_pairs.npy"); Ty = Ty.astype(bool)
        Pk = kgml_pairs(task, genes)
        # strata exactly as string_baseline.main
        sf = {(r["gene_a"], r["gene_b"]): int(r["string_exp150"]) for r in csv.DictReader((D / fs).open(), delimiter="\t")}
        cl = {}
        for r in csv.DictReader((D / fp).open(), delimiter="\t"):
            if r["gene_a"] in gid and r["gene_b"] in gid:
                c = r["ko_edge_split"] == "none" and r["kgml_any_relation"] == "0" and r["ko_copathway_train"] == "0"
                s = sf.get((r["gene_a"], r["gene_b"]), 0)
                for k in ((gid[r["gene_a"]], gid[r["gene_b"]]), (gid[r["gene_b"]], gid[r["gene_a"]])):
                    cl[k] = (c, c and s == 0)
        par = {Tp[k]: cl[(Ti[k], Tj[k])] for k in range(len(Ty)) if Ty[k]}
        posg = {Tp[k]: (Ti[k], Tj[k]) for k in range(len(Ty)) if Ty[k]}
        strata = {"all": np.ones(len(Ty), bool), "clean": np.array([par[p][0] for p in Tp]),
                  "clean_noSTRING": np.array([par[p][1] for p in Tp]),
                  "test_test": np.array([all(genes[g]["gene_split"] == "test" for g in posg[p]) for p in Tp])}
        scores = {"konbf_kgml_s0": Pk[Ti, Tj]}
        for c in sorted((HERE / "cache").glob(f"{task}_strict_v1_full_*.npz")):
            scores[c.stem.split("strict_v1_full_")[1]] = np.load(c)["P"][Ti, Tj]
        res = {"zero_shot": {}, "paired": {}, "meta": {"run": RUN, "date": "2026-11-02"}}
        for n, sc in scores.items():
            res["zero_shot"][n] = {s: {"auroc": auroc(sc[m], Ty[m]), "auprc": auprc(sc[m], Ty[m]), "pos_rate": float(Ty[m].mean())}
                                   for s, m in strata.items()}
            print(task, n, " ".join(f"{s} {res['zero_shot'][n][s]['auroc']:.4f}" for s in strata), flush=True)
        # paired bootstrap exactly as zs_paired.main (ref = KGML model)
        for sn in ("clean", "clean_noSTRING"):
            m = strata[sn]
            idx = np.where(m)[0]; grp = Tp[idx]; ug = np.unique(grp); members = {g: idx[grp == g] for g in ug}
            rng = np.random.default_rng(0)
            boots = [np.concatenate([members[g] for g in rng.choice(ug, len(ug))]) for _ in range(2000)]
            for ref in ("konbf_kgml_s0", "konbf_kl_s0"):
                for other in [k for k in scores if k != ref and not k.startswith("kegg_enc_r")]:
                    if ref == "konbf_kl_s0" and other == "konbf_kgml_s0":
                        continue
                    d0 = auroc(scores[ref][idx], Ty[idx]) - auroc(scores[other][idx], Ty[idx])
                    bs = [auroc(scores[ref][b], Ty[b]) - auroc(scores[other][b], Ty[b]) for b in boots]
                    v = [float(d0), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)),
                         float(2 * min(np.mean(np.array(bs) <= 0), np.mean(np.array(bs) >= 0)))]
                    res["paired"][f"{sn}|{ref} - {other}"] = v
                    print(task, sn, ref, "-", other, "%+.4f [%+.4f, %+.4f] p=%.4f" % tuple(v), flush=True)
        out_f.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

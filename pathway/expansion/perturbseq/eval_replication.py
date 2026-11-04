#!/usr/bin/env python3
"""Pre-registered replication of the perturbation evaluation on Replogle 2022
K562-essential and RPE1-essential, plus a re-run of Adamson / Norman with the same code.

Identical to downstream/perturb_eval.py (methods, candidates, gene score = max over KO(A) x KO(g), strata, pct / de /
spec metrics, exact random expectation, sign-flip test, perturbation bootstrap, paired PathArrowNet - X), except
  - statistics from perturbseq/data/stats_<ds>.npz; scheme 'gem' (primary) or 'pooled' (sensitivity);
  - KO-level matrices from full-vocabulary caches (every KO has a row).
Functions gene_kos / gene_scores / bh / vocab are imported from perturb_eval.py unchanged.

    .venv/bin/python expansion/perturbseq/eval_replication.py [gem|pooled] [datasets]
    -> expansion/perturbseq/results/replication_<scheme>.json
"""
import csv, json, os, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DS = HERE.parent / "downstream"
sys.path.insert(0, str(DS))
import perturb_eval as PE  # noqa: E402

ROOT = PE.ROOT
METHODS = PE.METHODS
KS = PE.KS
NTIE = PE.NTIE


def ko_level(voc):
    V = len(voc); M = {}
    M["konbf"] = torch.tensor(np.load(DS / "cache/konbf_full_matrix.npy"))
    M["konbf_randinit"] = torch.tensor(np.load(HERE / "cache/konbf_randinit_full.npy"))
    if (HERE / "cache/kegg_enc_full.npy").exists():  # optional comparison method (KEGG encoder scores)
        M["kegg_enc"] = torch.tensor(np.load(HERE / "cache/kegg_enc_full.npy"))
    z = np.load(DS / "cache/bpn_orig_ko_scores_all.npz"); B = torch.full((V, V), -1e9); B[torch.tensor(z["anchors"])] = torch.tensor(z["S"])
    M["bpn_orig"] = B
    st = torch.zeros(V, V)
    with open(ROOT / "kegg_dataset_cache/string/ko_background_exp150_full.tsv") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            x, y = voc[r["ko_a"]], voc[r["ko_b"]]; s = float(r["max_experimental_score"])
            st[x, y] = st[y, x] = max(st[x, y].item(), s)
    M["string_exp"] = st
    kd = torch.zeros(V, V)
    with (ROOT / "kegg_dataset/splits_ko/edge_split.csv").open() as fh:
        for r in csv.DictReader(fh):
            kd[voc[r["source_ko"]], voc[r["target_ko"]]] = 1
    M["kegg_direct"] = kd
    return M


def evaluate(d, z, scheme, voc, g2k, M, rng):
    tkey, pkey = ("t", "pval") if scheme == "gem" else ("t_pooled", "pval_pooled")
    genes = [str(x) for x in z["genes"]]; perts = [str(x) for x in z["perts"]]
    gi = [j for j, g in enumerate(genes) if g in g2k]; pi = [i for i, p in enumerate(perts) if p in g2k]
    print(d, scheme, "perts", len(perts), "with KO", len(pi), "| measured genes", len(genes), "with KO", len(gi), flush=True)
    pk = [g2k[perts[i]] for i in pi]; gk = [g2k[genes[j]] for j in gi]
    T = np.abs(z[tkey][pi][:, gi]); Pv = z[pkey][pi][:, gi]
    kegg_any = np.maximum(PE.gene_scores(M["kegg_direct"], pk, gk), PE.gene_scores(M["kegg_direct"].T.contiguous(), pk, gk)) > 0
    string = PE.gene_scores(M["string_exp"], pk, gk) >= 150
    sc = {m: PE.gene_scores(M[m], pk, gk) for m in METHODS if m != "random"}
    sc["random"] = rng.random(T.shape)
    n_p = len(pk)
    elig = np.ones(T.shape, bool)
    for a, ka in enumerate(pk):
        for j, kg in enumerate(gk):
            if set(ka) & set(kg) or genes[gi[j]] == perts[pi[a]]:
                elig[a, j] = False
    strata = {"all": elig, "not_kegg": elig & ~kegg_any, "not_kegg_not_string": elig & ~kegg_any & ~string}
    res = {"n_perts": n_p, "n_genes": len(gk), "strata": {}}
    Tall = np.abs(z[tkey][:, gi]); spec = ((Tall.argsort(0).argsort(0) + 0.5) / len(Tall))[pi]
    pct_all = (Tall.argsort(1).argsort(1) + 0.5) / Tall.shape[1]; tot = pct_all.sum(0)
    sc["responder"] = np.stack([(tot - pct_all[i]) / (len(Tall) - 1) for i in pi])
    deg = (M["string_exp"] >= 150).sum(1).float()
    sc["string_degree"] = np.tile([max(deg[k].item() for k in kg) for kg in gk], (n_p, 1)).astype(float)
    meths = METHODS + ["responder", "string_degree"]
    for sn, E in strata.items():
        pct = np.full(T.shape, np.nan); de = np.zeros(T.shape)
        for a in range(n_p):
            e = np.where(E[a])[0]
            if len(e) < 20:
                continue
            r = T[a, e].argsort().argsort(); pct[a, e] = (r + 0.5) / len(e)
            de[a, e] = PE.bh(Pv[a, e]) < 0.05
        ok = [a for a in range(n_p) if E[a].sum() >= 20]
        MET = {"pct": pct, "de": de, "spec": spec}
        null = {n: np.array([X[a, E[a]].mean() for a in ok]) for n, X in MET.items()}
        per = {}
        R = {"n_perts": len(ok), "mean_eligible": float(np.mean([E[a].sum() for a in ok])),
             "null": {n: float(v.mean()) for n, v in null.items()}, "methods": {}}
        for m in meths:
            R["methods"][m] = {}
            for k in KS:
                acc = {n: [] for n in MET}
                for a in ok:
                    e = np.where(E[a])[0]; s_ = sc[m][a, e]
                    draws = 1 if len(np.unique(s_)) == len(s_) else NTIE
                    tops = [e[np.lexsort((rng.random(len(e)), -s_))[:k]] for _ in range(draws)]
                    for n, X in MET.items():
                        acc[n].append(np.mean([X[a, t].mean() for t in tops]))
                out = {}
                for n in MET:
                    v = np.array(acc[n]) - null[n]; per[(m, k, n)] = v
                    flips = rng.choice([-1, 1], (20000, len(v))) @ v / len(v)
                    boots = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)])
                    out[n] = {"value": float(np.mean(acc[n])), "null": float(null[n].mean()), "minus_null": float(v.mean()),
                              "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                              "p_signflip": float((np.abs(flips) >= abs(v.mean()) - 1e-12).mean()),
                              "p_one_sided_greater": float((flips >= v.mean() - 1e-12).mean())}
                out["de_enrichment"] = out["de"]["value"] / out["de"]["null"] if out["de"]["null"] > 0 else None
                R["methods"][m][f"top{k}"] = out
                print(d, sn, m, f"top{k}", "  ".join("%s %.3f (%+.3f p=%.4f)" % (n, out[n]["value"], out[n]["minus_null"], out[n]["p_signflip"]) for n in MET), flush=True)
        R["paired_konbf_minus"] = {}
        for m in meths[1:]:
            for k in KS:
                for n in MET:
                    v = per[("konbf", k, n)] - per[(m, k, n)]
                    flips = rng.choice([-1, 1], (20000, len(v))) @ v / len(v)
                    boots = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)])
                    R["paired_konbf_minus"][f"{m}|top{k}|{n}"] = [float(v.mean()), float(np.percentile(boots, 2.5)),
                                                                   float(np.percentile(boots, 97.5)), float((np.abs(flips) >= abs(v.mean()) - 1e-12).mean())]
        res["strata"][sn] = R
    return res


def main():
    scheme = sys.argv[1] if len(sys.argv) > 1 else "gem"
    dsets = sys.argv[2:] or ["replogle_k562_essential", "replogle_rpe1_essential", "adamson", "norman"]
    voc = PE.vocab(); g2k = PE.gene_kos(voc); M = ko_level(voc)
    rng = np.random.default_rng(0)
    # REPL_TAG (env, added for Nadig 2025): write results/replication_<scheme><tag>.json instead of the shared file
    out = HERE / "results" / f"replication_{scheme}{os.environ.get('REPL_TAG', '')}.json"; out.parent.mkdir(exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {}
    for d in dsets:
        z = np.load(HERE / f"data/stats_{d}.npz", allow_pickle=True)
        res[d] = evaluate(d, z, scheme, voc, g2k, M, rng)
        out.write_text(json.dumps(res, indent=1))
    print("done")


if __name__ == "__main__":
    main()

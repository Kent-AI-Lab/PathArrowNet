#!/usr/bin/env python3
"""Perturbation validation of KEGG-model downstream predictions.

For every single-gene perturbation A (Adamson 2016 CRISPRi, Norman 2019 CRISPRa; stats from perturb_prep.py) whose
gene has a KO in the vocabulary, each method ranks the measured genes g (HVGs with a KO, g != A, g shares no KO with A)
as predicted DOWNSTREAM targets of A (direction A -> g; gene score = max over KO(A) x KO(g)). For the top 1 / 5 / 10
predicted genes we check whether their expression change in the real perturbation is larger than background:
    pct  = percentile of |Welch t| of g among all eligible genes of that perturbation (background mean = 0.5)
    de   = g is DE (BH q < 0.05 within the perturbation); reported as rate and as enrichment over the base rate
    spec = percentile of |t| of g under A among |t| of g under every perturbation of the dataset: is g changed by A
           MORE than g usually changes? (removes 'generic responder' genes that move under any perturbation)
Null: for each perturbation the same k drawn uniformly from the eligible genes (exact expectation for pct / de);
significance by sign-flip permutation over perturbations of (top-k value - null expectation) and a perturbation
bootstrap CI. Strata: all eligible genes; 'not KEGG' (no KEGG edge between any KO pair of A and g, either
direction, all pathways); 'not KEGG, no STRING' (additionally STRING-exp < 150).

Methods (all fixed before looking at the perturbation data, no fitting on it):
    konbf           PathArrowNet best-model component (strict split 1 train graph + STRING-exp150 4-bin background),
                    log p(g | A ->) + log p(A | -> g), 5 seeds averaged  (kegg_pairscore.nbf_matrix)
    konbf_randinit  same architecture / graph, untrained weights (control)
    bpn_orig        original BioPathNet (biopathnet+bg, strict split 1, 5 seeds), log-softmax of (A, precedes, g)
    kegg_enc        KEGG-pretrained two-tower encoder (seed 0), forward + backward head
    string_exp      max STRING experimental-channel score between KO(A) and KO(g) (symmetric, no direction)
    kegg_direct     A -> g is a KEGG edge (all pathways; binary, ties broken at random) - "just look it up"
    random          uniform
    responder       control, no prior knowledge: g's mean |t| percentile over the OTHER perturbations (leave-one-out)
    string_degree   control: number of STRING-exp >= 150 partners of g's KO (hub-ness, ignores A)
PathArrowNet / encoder only saw strict-split-1 TRAIN edges, so for them part of 'kegg_direct' is genuinely unseen.

    .venv/bin/python expansion/downstream/perturb_eval.py            (GPU for the PathArrowNet scores; cached)
"""
import csv, json, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kegg_pairscore as KP  # noqa: E402

ROOT = KP.ROOT
OUT = HERE / "results/perturb"; OUT.mkdir(parents=True, exist_ok=True)
DATASETS = ["adamson", "norman"]
METHODS = ["konbf", "konbf_randinit", "bpn_orig", "kegg_enc", "string_exp", "kegg_direct", "random"]
KS = [1, 5, 10]
NTIE = 50  # random tie-break draws (binary / tied scores)


def vocab():
    with (ROOT / "kegg_dataset/token_vocab_ko_full.csv").open() as f:
        return {r["token"]: int(r["token_id"]) for r in csv.DictReader(f)}


def gene_kos(voc):
    """symbol (and previous aliases) -> KO ids in vocabulary"""
    m = {}
    with (HERE / "data/common/gene_map.tsv").open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            ks = [voc[k] for k in (r["kos_in_vocab"] or "").split(";") if k in voc]
            if not ks:
                continue
            m[r["symbol"]] = ks
            for a in (r["prev_alias"] or "").replace("|", ",").split(","):
                if a and a not in m:
                    m[a] = ks
    return m


def ko_level(voc, anchors, dev):
    """dense KO x KO matrices (rows for anchors) for every method; cached"""
    f = HERE / "cache/perturb_ko_scores.pt"
    if f.exists():
        M = torch.load(f)
        if "bpn_orig" not in M:  # original BioPathNet rows (bpn_ko_scores.py), added after the first version
            z = np.load(HERE / "cache/bpn_orig_ko_scores.npz"); B = torch.zeros(len(voc), len(voc))
            B[torch.tensor(z["anchors"])] = torch.tensor(z["S"]); M["bpn_orig"] = B
            torch.save(M, f)
        return M
    V = len(voc); A = sorted(anchors); M = {}
    M["konbf"] = KP.nbf_matrix(A, dev=dev).cpu()
    M["konbf_randinit"] = KP.nbf_matrix(A, dev=dev, randinit=True).cpu()
    S, voc2 = KP.ko_matrix("kegg_enc", 0, "strict_v1_full", dev)
    assert voc2 == voc or len(voc2) == V
    M["kegg_enc"] = S.cpu()
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
    f.parent.mkdir(parents=True, exist_ok=True); torch.save(M, f)
    return M


def gene_scores(M, pk, gk):
    """[n_pert, n_gene] = max over KO(A) x KO(g) of M[ko_a, ko_g]"""
    out = np.full((len(pk), len(gk)), -np.inf, np.float64)
    for i, ka in enumerate(pk):
        rows = M[ka].numpy()  # [|KO(A)|, V]
        for j, kg in enumerate(gk):
            out[i, j] = rows[:, kg].max()
    return out


def bh(p):
    n = len(p); o = np.argsort(p); q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    voc = vocab(); g2k = gene_kos(voc)
    data = {}
    for d in DATASETS:
        z = np.load(HERE / f"data/perturb/stats_{d}.npz", allow_pickle=True)
        genes = [str(x) for x in z["genes"]]; perts = [str(x) for x in z["perts"]]
        gi = [j for j, g in enumerate(genes) if g in g2k]; pi = [i for i, p in enumerate(perts) if p in g2k]
        data[d] = dict(z=z, genes=genes, perts=perts, gi=gi, pi=pi)
        print(d, "perts", len(perts), "with KO", len(pi), "| measured genes", len(genes), "with KO", len(gi), flush=True)
    anchors = {k for d in data.values() for i in d["pi"] for k in g2k[d["perts"][i]]} | \
              {k for d in data.values() for j in d["gi"] for k in g2k[d["genes"][j]]}
    M = ko_level(voc, anchors, dev)
    res, rng = {}, np.random.default_rng(0)
    for d, D in data.items():
        z = D["z"]; pk = [g2k[D["perts"][i]] for i in D["pi"]]; gk = [g2k[D["genes"][j]] for j in D["gi"]]
        T = np.abs(z["t"][D["pi"]][:, D["gi"]]); Pv = z["pval"][D["pi"]][:, D["gi"]]
        kegg_any = np.maximum(gene_scores(M["kegg_direct"], pk, gk), gene_scores(M["kegg_direct"].T.contiguous(), pk, gk)) > 0
        string = gene_scores(M["string_exp"], pk, gk) >= 150
        sc = {m: gene_scores(M[m], pk, gk) for m in METHODS if m != "random"}
        sc["random"] = rng.random(T.shape)
        n_p = len(pk)
        elig = np.ones(T.shape, bool)
        for a, ka in enumerate(pk):  # exclude A itself / genes sharing a KO with A
            for j, kg in enumerate(gk):
                if set(ka) & set(kg) or D["genes"][D["gi"][j]] == D["perts"][D["pi"][a]]:
                    elig[a, j] = False
        strata = {"all": elig, "not_kegg": elig & ~kegg_any, "not_kegg_not_string": elig & ~kegg_any & ~string}
        res[d] = {"n_perts": n_p, "n_genes": len(gk), "strata": {}}
        # gene-specific percentile: |t| of g under A vs |t| of g under every other perturbation of the dataset
        Tall = np.abs(z["t"][:, D["gi"]]); spec_all = (Tall.argsort(0).argsort(0) + 0.5) / len(Tall)
        spec = spec_all[D["pi"]]
        # generic-responder baseline (leave-one-out over the other perturbations; no KEGG / STRING)
        pct_all = (Tall.argsort(1).argsort(1) + 0.5) / Tall.shape[1]
        tot = pct_all.sum(0)
        sc["responder"] = np.stack([(tot - pct_all[i]) / (len(Tall) - 1) for i in D["pi"]])
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
                de[a, e] = bh(Pv[a, e]) < 0.05
            ok = [a for a in range(n_p) if E[a].sum() >= 20]
            MET = {"pct": pct, "de": de, "spec": spec}
            null = {n: np.array([X[a, E[a]].mean() for a in ok]) for n, X in MET.items()}  # exact random-top-k expectation
            per = {}
            R = {"n_perts": len(ok), "mean_eligible": float(np.mean([E[a].sum() for a in ok])),
                 "null": {n: float(v.mean()) for n, v in null.items()}, "methods": {}}
            for m in meths:
                R["methods"][m] = {}
                for k in KS:
                    acc = {n: [] for n in MET}
                    for a in ok:
                        e = np.where(E[a])[0]; s_ = sc[m][a, e]
                        draws = 1 if len(np.unique(s_)) == len(s_) else NTIE  # random tie-breaks for tied scores
                        tops = [e[np.lexsort((rng.random(len(e)), -s_))[:k]] for _ in range(draws)]
                        for n, X in MET.items():
                            acc[n].append(np.mean([X[a, t].mean() for t in tops]))
                    out = {}
                    for i, n in enumerate(MET):
                        v = np.array(acc[n]) - null[n]; per[(m, k, n)] = v
                        flips = rng.choice([-1, 1], (20000, len(v))) @ v / len(v)
                        boots = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)])
                        out[n] = {"value": float(np.mean(acc[n])), "null": float(null[n].mean()), "minus_null": float(v.mean()),
                                  "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                  "p_signflip": float((np.abs(flips) >= abs(v.mean()) - 1e-12).mean())}
                    out["de_enrichment"] = out["de"]["value"] / out["de"]["null"] if out["de"]["null"] > 0 else None
                    R["methods"][m][f"top{k}"] = out
                    print(d, sn, m, f"top{k}", "  ".join("%s %.3f (%+.3f p=%.4f)" % (n, out[n]["value"], out[n]["minus_null"], out[n]["p_signflip"]) for n in MET),
                          "DEx%.2f" % (out["de_enrichment"] or 0), flush=True)
            R["paired_konbf_minus"] = {}
            for m in meths[1:]:
                for k in KS:
                    for n in MET:
                        v = per[("konbf", k, n)] - per[(m, k, n)]
                        flips = rng.choice([-1, 1], (20000, len(v))) @ v / len(v)
                        boots = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)])
                        R["paired_konbf_minus"][f"{m}|top{k}|{n}"] = [float(v.mean()), float(np.percentile(boots, 2.5)),
                                                                       float(np.percentile(boots, 97.5)), float((np.abs(flips) >= abs(v.mean()) - 1e-12).mean())]
            res[d]["strata"][sn] = R
        (OUT / "perturb_results.json").write_text(json.dumps(res, indent=1))
    print("done")


if __name__ == "__main__":
    main()

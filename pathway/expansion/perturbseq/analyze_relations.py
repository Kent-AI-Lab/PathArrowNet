#!/usr/bin/env python3
"""Descriptive analyses of the perturbation relationship tables (build_relations.py outputs); no model is fitted.

 1. kegg_vs_response  How do genes at different KEGG positions relative to the perturbed gene A respond?
                      Groups (gene-level KEGG graph, all pathways): direct A->g edge split by relation class
                      (GErel expression / repression = transcriptional; PPrel / other = post-translational),
                      reverse g->A edge only, directed distance 2 / 3, connected only undirected within 2 hops,
                      STRING-exp >= 150 without KEGG link, no link. Per perturbation: group mean minus that
                      perturbation's mean over all KO-mapped genes (paired), averaged over perturbations;
                      sign-flip p. Metrics: spec, responsive rate.
 2. replication       Same perturbation, same gene, two experiments:
                      Adamson vs Replogle K562 (both CRISPRi, K562: technical/biological replicate),
                      Replogle K562 vs RPE1 (CRISPRi, different cell line: context dependence),
                      Norman (CRISPRa) vs Replogle K562 (opposite direction expected).
                      Per shared perturbation: Pearson r of t over shared genes; for responsive pairs in X,
                      fraction responsive in Y, fraction same sign, vs Y's base rate.
 3. context_specific  Pairs responsive in one cell line and 'no detected response' in the other, gene detected in
                      >= 10 % control cells in both (K562 vs RPE1). Counts and KEGG / STRING status.
 4. model_support     PathArrowNet top-10 predicted downstream genes of each perturbation: share responsive, by whether
                      the KO edge is a PathArrowNet training edge; list of predictions supported in >= 2 datasets.
 5. sample_counts     Perturbation-aware relation samples (both genes with a KO), by label / proposed split / KEGG
                      status; edge-change candidates (from 3).

    .venv/bin/python expansion/perturbseq/analyze_relations.py -> results/relations_analysis.json, supported_predictions.tsv
"""
import csv, json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATASETS = ["adamson", "norman", "replogle_k562_essential", "replogle_rpe1_essential"]
rng = np.random.default_rng(0)


def signflip(v, n=20000):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    f = rng.choice([-1, 1], (n, len(v))) @ v / len(v)
    return float(v.mean()), float((np.abs(f) >= abs(v.mean()) - 1e-12).mean()), len(v)


def load():
    gent = {}
    for r in csv.DictReader((HERE / "measured_genes.tsv").open(), delimiter="\t"):
        gent[(r["dataset"], r["gene"])] = r["entrez"]
    pinfo = {}
    for r in csv.DictReader((HERE / "perturbations.tsv").open(), delimiter="\t"):
        pinfo[(r["dataset"], r["pert_gene"])] = r
    D = {}
    for d in DATASETS:
        z = dict(np.load(HERE / f"data/pairs_{d}.npz", allow_pickle=True)); s = np.load(HERE / f"data/stats_{d}.npz", allow_pickle=True)
        z["t"] = s["t"]; z["frac"] = s["frac_expr_ctrl"]
        z["g_ent"] = np.array([gent[(d, str(g))] for g in z["genes"]])
        z["p_ent"] = np.array([pinfo[(d, str(p))]["entrez"] for p in z["perts"]])
        z["p_split"] = np.array([pinfo[(d, str(p))]["proposed_split"] for p in z["perts"]])
        D[d] = z
    return D


def kegg_rel():
    rel = {}
    for r in csv.DictReader((HERE.parents[1] / "kegg_dataset/gene_edges.csv").open()):
        rel.setdefault((r["source_gene"], r["target_gene"]), set()).add(r["relation_type"])
    return rel


def groups(z, rel):
    """[nP, G] int code per pair; -1 = gene without KO / self"""
    KD, KU, S = z["kegg_dist_dir"], z["kegg_dist_und"], z["string_exp"]
    hasko = np.array([bool(x) for x in z["g_ent"]])[None] & (np.array([bool(x) for x in z["p_ent"]])[:, None])
    C = np.full(KD.shape, 7)  # 7 no link
    C[S >= 150] = 6
    C[(KU <= 2)] = 5
    C[KD == 3] = 4
    C[KD == 2] = 3
    rev = np.zeros(KD.shape, bool)
    for i, a in enumerate(z["p_ent"]):
        for j, g in enumerate(z["g_ent"]):
            if a and g and (g, a) in rel:
                rev[i, j] = True
            if KD[i, j] == 1:
                C[i, j] = 0 if "GErel" in rel.get((a, g), ()) else 1
    C[(KD != 1) & rev] = 2
    C[~hasko] = -1
    for i, sc in enumerate(z["self_col"]):
        if sc >= 0:
            C[i, sc] = -1
    return C


GNAMES = ["A->g KEGG GErel (expression/repression)", "A->g KEGG other (PPrel etc.)", "g->A KEGG only (reverse)",
          "directed distance 2", "directed distance 3", "undirected <= 2 hops only", "STRING-exp >= 150, no KEGG link",
          "no link"]


def kegg_vs_response(D, rel):
    out = {}
    for d, z in D.items():
        C = groups(z, rel); spec = z["spec"]; resp = (z["label"] == 1).astype(float)
        res = {}
        for g, name in enumerate(GNAMES):
            dsp, drs, n_pairs = [], [], 0
            for i in range(len(C)):
                base = C[i] >= 0; m = C[i] == g
                if m.sum() == 0 or base.sum() < 20:
                    continue
                n_pairs += int(m.sum())
                dsp.append(spec[i, m].mean() - spec[i, base].mean()); drs.append(resp[i, m].mean() - resp[i, base].mean())
            ms, ps, n = signflip(dsp); mr, pr, _ = signflip(drs)
            res[name] = {"n_perts": n, "n_pairs": n_pairs, "spec_minus_bg": ms, "p_spec": ps, "resp_rate_minus_bg": mr, "p_resp": pr}
        res["bg_responsive_rate"] = float(resp[C >= 0].mean())
        out[d] = res
        print(d, json.dumps({k: (round(v["spec_minus_bg"], 4), v["p_spec"], v["n_pairs"]) for k, v in res.items() if isinstance(v, dict)}), flush=True)
    return out


def replication(D, x, y):
    X, Y = D[x], D[y]
    pidx = {e: i for i, e in enumerate(Y["p_ent"]) if e}
    gy = {e: j for j, e in enumerate(Y["g_ent"]) if e}
    sh_g = [(j, gy[e]) for j, e in enumerate(X["g_ent"]) if e and e in gy]
    gx, gyy = np.array([a for a, _ in sh_g]), np.array([b for _, b in sh_g])
    rs, conc, same, base = [], [], [], []
    for i, e in enumerate(X["p_ent"]):
        if not e or e not in pidx:
            continue
        k = pidx[e]; tx, ty = X["t"][i, gx], Y["t"][k, gyy]
        ok = np.isfinite(tx) & np.isfinite(ty)
        rs.append(np.corrcoef(tx[ok], ty[ok])[0, 1])
        rx = X["label"][i, gx] == 1; ry = Y["label"][k, gyy] == 1
        if rx.sum():
            conc.append(ry[rx].mean()); base.append(ry.mean())
            same.append((np.sign(tx[rx]) == np.sign(ty[rx])).mean())
    r = {"n_shared_perts": len(rs), "n_shared_genes": len(sh_g), "median_r_t": float(np.median(rs)) if rs else None,
         "mean_r_t": float(np.mean(rs)) if rs else None,
         "frac_resp_in_y_given_resp_in_x": float(np.mean(conc)) if conc else None,
         "base_resp_rate_y": float(np.mean(base)) if base else None,
         "frac_same_sign_given_resp_in_x": float(np.mean(same)) if same else None}
    print("replication", x, y, json.dumps(r), flush=True)
    return r


def context_specific(D, x, y):
    X, Y = D[x], D[y]
    pidx = {e: i for i, e in enumerate(Y["p_ent"]) if e}
    gy = {e: j for j, e in enumerate(Y["g_ent"]) if e}
    sh = [(j, gy[e]) for j, e in enumerate(X["g_ent"]) if e and e in gy]
    gx, gyy = np.array([a for a, _ in sh]), np.array([b for _, b in sh])
    det = (X["frac"][gx] >= 0.1) & (Y["frac"][gyy] >= 0.1)
    cnt = {"x_only": 0, "y_only": 0, "both": 0, "x_only_kegg_und2": 0, "y_only_kegg_und2": 0, "both_kegg_und2": 0,
           "x_only_string": 0, "y_only_string": 0, "both_string": 0, "n_shared_perts": 0}
    for i, e in enumerate(X["p_ent"]):
        if not e or e not in pidx:
            continue
        cnt["n_shared_perts"] += 1; k = pidx[e]
        lx, ly = X["label"][i, gx], Y["label"][k, gyy]
        ku = X["kegg_dist_und"][i, gx] <= 2; st = X["string_exp"][i, gx] >= 150
        for name, m in (("x_only", (lx == 1) & (ly == -1) & det), ("y_only", (ly == 1) & (lx == -1) & det),
                        ("both", (lx == 1) & (ly == 1))):
            cnt[name] += int(m.sum()); cnt[name + "_kegg_und2"] += int((m & ku).sum()); cnt[name + "_string"] += int((m & st).sum())
    print("context", x, y, json.dumps(cnt), flush=True)
    return cnt


def model_support(D):
    out, rows = {}, {}
    for d, z in D.items():
        NR, lab, KE = z["konbf_rank"], z["label"], z["kegg_ko_edge"]
        sup, sup_nokegg, n, nk = 0, 0, 0, 0
        for i in range(len(NR)):
            top = np.where((NR[i] >= 1) & (NR[i] <= 10))[0]
            for j in top:
                n += 1; r = lab[i, j] == 1; sup += r
                if not KE[i, j]:
                    nk += 1; sup_nokegg += r
                key = (z["p_ent"][i], z["g_ent"][j])
                rows.setdefault(key, {"pert": str(z["perts"][i]), "gene": str(z["genes"][j]), "kegg_ko_edge": int(KE[i, j]),
                                      "string_exp": float(z["string_exp"][i, j]), "datasets_tested": [], "datasets_responsive": []})
                rows[key]["datasets_tested"].append(d)
                if r:
                    rows[key]["datasets_responsive"].append(d)
        base = float((lab[z["eligible"]] == 1).mean())
        out[d] = {"n_top10_predictions": n, "frac_responsive": sup / max(n, 1), "base_rate_eligible": base,
                  "n_not_kegg_edge": nk, "frac_responsive_not_kegg_edge": sup_nokegg / max(nk, 1)}
        print("model_support", d, json.dumps(out[d]), flush=True)
    with (HERE / "supported_predictions.tsv").open("w") as f:
        f.write("pert_gene\tresponse_gene\tkegg_ko_edge\tstring_exp_ko\tn_datasets_tested\tn_datasets_responsive\tdatasets_responsive\n")
        for r in sorted(rows.values(), key=lambda r: (-len(r["datasets_responsive"]), r["pert"])):
            if r["datasets_responsive"]:
                f.write(f"{r['pert']}\t{r['gene']}\t{r['kegg_ko_edge']}\t{r['string_exp']:.0f}\t{len(r['datasets_tested'])}\t"
                        f"{len(r['datasets_responsive'])}\t{';'.join(r['datasets_responsive'])}\n")
    out["n_predictions_supported_in_2plus_datasets"] = sum(1 for r in rows.values() if len(r["datasets_responsive"]) >= 2)
    out["n_predictions_tested_in_2plus_datasets"] = sum(1 for r in rows.values() if len(r["datasets_tested"]) >= 2)
    return out


def sample_counts(D):
    out = {}
    for d, z in D.items():
        E = z["eligible"]; lab = z["label"]; ku = z["kegg_dist_und"] <= 2; ke = z["kegg_ko_edge"] == 1
        c = {}
        for sp in ("train", "val", "test"):
            m = E & (z["p_split"] == sp)[:, None]
            c[sp] = {"responsive": int((m & (lab == 1)).sum()), "no_detected_resp": int((m & (lab == -1)).sum()),
                     "responsive_kegg_ko_edge": int((m & (lab == 1) & ke).sum()),
                     "responsive_kegg_und2": int((m & (lab == 1) & ku).sum()), "n_perts": int((z["p_split"] == sp).sum())}
        out[d] = c
    return out


def main():
    D = load(); rel = kegg_rel()
    R = {"kegg_vs_response": kegg_vs_response(D, rel),
         "replication": {f"{x}|{y}": replication(D, x, y) for x, y in
                         (("adamson", "replogle_k562_essential"), ("replogle_k562_essential", "adamson"),
                          ("replogle_k562_essential", "replogle_rpe1_essential"), ("replogle_rpe1_essential", "replogle_k562_essential"),
                          ("norman", "replogle_k562_essential"))},
         "context_specific": {"replogle_k562_essential|replogle_rpe1_essential": context_specific(D, "replogle_k562_essential", "replogle_rpe1_essential")},
         "model_support": model_support(D), "sample_counts": sample_counts(D)}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/relations_analysis.json").write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()

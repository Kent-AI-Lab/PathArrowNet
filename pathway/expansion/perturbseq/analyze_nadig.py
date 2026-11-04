#!/usr/bin/env python3
"""Descriptive analyses for Nadig 2025 HepG2 / Jurkat (same functions as analyze_relations.py, imported unchanged),
together with the four earlier datasets for the cross-experiment comparisons. No model is fitted.

 - kegg_vs_response, sample_counts             Nadig HepG2, Jurkat
 - replication (median r of t, responsive concordance, same sign) for every ordered pair among Replogle K562 / RPE1
   and Nadig HepG2 / Jurkat (all CRISPRi essential-gene screens), plus the same-cell reference Adamson <-> Replogle K562
 - context_specific                             HepG2 vs Jurkat
 - model_support                                all six datasets (the >= 2-dataset support list now spans 6)
 - measured-gene overlap                        Nadig vs Replogle K562 / RPE1 (HVG sets differ per dataset)

    .venv/bin/python expansion/perturbseq/analyze_nadig.py
    -> perturbseq/nadig/relations_analysis_nadig.json, perturbseq/nadig/supported_predictions_6ds.tsv
"""
import csv, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_relations as AR  # noqa: E402
OUT = HERE / "nadig"
OLD = ["adamson", "norman", "replogle_k562_essential", "replogle_rpe1_essential"]
NEW = ["nadig_hepg2_essential", "nadig_jurkat_essential"]
ESS = ["replogle_k562_essential", "replogle_rpe1_essential"] + NEW


def load():
    gent, pinfo = {}, {}
    for base in (HERE, OUT):
        for r in csv.DictReader((base / "measured_genes.tsv").open(), delimiter="\t"):
            gent[(r["dataset"], r["gene"])] = r["entrez"]
        for r in csv.DictReader((base / "perturbations.tsv").open(), delimiter="\t"):
            pinfo[(r["dataset"], r["pert_gene"])] = r
    D = {}
    for d in OLD + NEW:
        z = dict(np.load(HERE / f"data/pairs_{d}.npz", allow_pickle=True)); s = np.load(HERE / f"data/stats_{d}.npz", allow_pickle=True)
        z["t"] = s["t"]; z["frac"] = s["frac_expr_ctrl"]
        z["g_ent"] = np.array([gent[(d, str(g))] for g in z["genes"]])
        z["p_ent"] = np.array([pinfo[(d, str(p))]["entrez"] for p in z["perts"]])
        z["p_split"] = np.array([pinfo[(d, str(p))]["proposed_split"] for p in z["perts"]])
        D[d] = z
    return D


def main():
    D = load(); rel = AR.kegg_rel()
    pairs = [("adamson", "replogle_k562_essential"), ("replogle_k562_essential", "adamson")]
    pairs += [(x, y) for x in ESS for y in ESS if x != y]
    ov = {}
    for d in NEW:
        g = set(map(str, D[d]["genes"]))
        ov[d] = {o: {"n_this": len(g), "n_other": len(set(map(str, D[o]["genes"]))),
                     "n_shared_symbols": len(g & set(map(str, D[o]["genes"]))),
                     "n_shared_entrez": len({e for e in D[d]["g_ent"] if e} & {e for e in D[o]["g_ent"] if e})}
                 for o in ("replogle_k562_essential", "replogle_rpe1_essential") + tuple(x for x in NEW if x != d)}
    AR.HERE = OUT  # model_support writes its tsv next to the Nadig outputs, the original list is not touched
    ms = AR.model_support(D)
    (OUT / "supported_predictions.tsv").rename(OUT / "supported_predictions_6ds.tsv")
    R = {"measured_gene_overlap": ov,
         "kegg_vs_response": AR.kegg_vs_response({d: D[d] for d in NEW}, rel),
         "replication": {f"{x}|{y}": AR.replication(D, x, y) for x, y in pairs},
         "context_specific": {"nadig_hepg2_essential|nadig_jurkat_essential":
                              AR.context_specific(D, "nadig_hepg2_essential", "nadig_jurkat_essential")},
         "model_support": ms, "sample_counts": AR.sample_counts({d: D[d] for d in NEW})}
    (OUT / "relations_analysis_nadig.json").write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()

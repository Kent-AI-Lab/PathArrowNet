#!/usr/bin/env python3
"""Perturbation -> response-gene relationship tables, aligned to KEGG / STRING, with model predictions kept apart.

Input: data/stats_<ds>.npz (prep_stats.py). For every single-gene perturbation A and every measured gene g != A:

  OBSERVED (experiment)      delta (log1p mean, gem-group-matched controls), Welch t, p; n cells; control mean /
                             detection rate of g; on-target change of A itself (if A is among the measured genes)
  INFERRED (statistics)      q (BH within A), direction, rank of |t| within A (1 = strongest), spec = percentile of
                             |t(A, g)| among |t(B, g)| over all perturbations B of the dataset (gene-specific baseline)
  KEGG / STRING (knowledge)  gene-level KEGG edge A -> g / g -> A (any pathway, relation subtypes), directed and
                             undirected KEGG hop distance, # shared pathways, KO-level KEGG edge and its split in
                             splits_ko (the split PathArrowNet was trained on), STRING experimental score of the KO pair
  PREDICTED (models)         PathArrowNet (strict split 1, 5 seeds; cache/konbf_full_matrix.npy) and original BioPathNet
                             (cache/bpn_orig_ko_scores_all.npz) gene-pair score = max over KO(A) x KO(g), and the rank
                             of g among A's eligible candidates (measured genes with a KO, not sharing a KO with A)

Labels for perturbation-aware relation samples (fixed here, before any model is evaluated on them):
  responsive        q < 0.05 and spec >= 0.9 and A has an on-target effect (or A not measured)
  no_detected_resp  q >= 0.5 and g detected in >= 10 % of control cells   ("not observed", NOT "no relation")
  other             everything else
Proposed split for future training (by perturbed gene, identical across datasets): sha1(symbol) mod 20 -> 14 train,
3 val, 3 test.

    .venv/bin/python expansion/perturbseq/build_relations.py
    -> data/pairs_<ds>.npz (full matrices), relations_top20.tsv.gz, relations_kegg_aligned.tsv.gz,
       samples_responsive.tsv.gz, perturbations.tsv, measured_genes.tsv, build_summary.json
"""
import argparse, csv, gzip, hashlib, json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DS = HERE.parent / "downstream"
DATASETS = ["adamson", "norman", "replogle_k562_essential", "replogle_rpe1_essential"]
SHORT = {"adamson": "Adamson2016", "norman": "Norman2019", "replogle_k562_essential": "Replogle2022_K562",
         "replogle_rpe1_essential": "Replogle2022_RPE1"}
TOPN = 20


def bh(p):
    n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def load_maps():
    voc = {r["token"]: int(r["token_id"]) for r in csv.DictReader((ROOT / "kegg_dataset/token_vocab_ko_full.csv").open())}
    by_ens, by_sym, by_alias = {}, {}, {}
    for r in csv.DictReader((DS / "data/common/gene_map.tsv").open(), delimiter="\t"):
        rec = {"entrez": r["entrez"], "symbol": r["symbol"],
               "kos": [k for k in (r["kos_in_vocab"] or "").split(";") if k in voc]}
        if r["ensembl"]:
            by_ens[r["ensembl"]] = rec
        by_sym[r["symbol"]] = rec
        for a in (r["prev_alias"] or "").replace("|", ",").split(","):
            if a:
                by_alias.setdefault(a, rec)
    uni = {}
    for r in csv.DictReader((ROOT / "kegg_dataset/gene_uniprot_mapping.csv").open()):
        uni[r["ncbi_gene_id"]] = r["reviewed_swissprot_accessions"] or r["uniprot_accessions"]
    return voc, by_ens, by_sym, by_alias, uni


def map_gene(sym, ens, M):
    _, by_ens, by_sym, by_alias, _ = M
    if ens and ens in by_ens:
        return by_ens[ens], "ensembl"
    if sym in by_sym:
        return by_sym[sym], "symbol"
    if sym in by_alias:
        return by_alias[sym], "alias"
    return None, "unmapped"


def kegg_graph():
    out, und, rel, pw = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    for r in csv.DictReader((ROOT / "kegg_dataset/gene_edges.csv").open()):
        s, t = r["source_gene"], r["target_gene"]
        out[s].add(t); und[s].add(t); und[t].add(s)
        rel[(s, t)] |= {x for x in r["relation_subtype"].split(";") if x} | {r["relation_type"]}
        pw[s].add(r["pathway_id"]); pw[t].add(r["pathway_id"])
    return out, und, rel, pw


def bfs(adj, s, cap=4):
    d = {s: 0}; q = deque([s])
    while q:
        u = q.popleft()
        if d[u] >= cap:
            continue
        for v in adj.get(u, ()):
            if v not in d:
                d[v] = d[u] + 1; q.append(v)
    return d


def main():
    # optional (added for Nadig 2025): --datasets d1 d2 --out-dir DIR writes the tables / summary into DIR instead of
    # this folder; defaults reproduce the original 4-dataset build unchanged
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=DATASETS); ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    OUT = Path(args.out_dir) if args.out_dir else HERE; OUT.mkdir(parents=True, exist_ok=True)
    M = load_maps(); voc, uni = M[0], M[4]
    kout, kund, krel, kpw = kegg_graph()
    ko_edge = {}
    for r in csv.DictReader((ROOT / "kegg_dataset/splits_ko/edge_split.csv").open()):
        ko_edge[(voc[r["source_ko"]], voc[r["target_ko"]])] = r["split"]
    V = len(voc); st = np.zeros((V, V), np.float32)
    for r in csv.DictReader((ROOT / "kegg_dataset_cache/string/ko_background_exp150_full.tsv").open(), delimiter="\t"):
        x, y = voc[r["ko_a"]], voc[r["ko_b"]]; st[x, y] = st[y, x] = max(st[x, y], float(r["max_experimental_score"]))
    KOE = np.zeros((V, V), bool)
    for a, b in ko_edge:
        KOE[a, b] = True
    NBF = np.load(DS / "cache/konbf_full_matrix.npy")
    zb = np.load(DS / "cache/bpn_orig_ko_scores_all.npz"); BPN = np.full((V, V), -np.inf, np.float32)
    BPN[zb["anchors"]] = zb["S"]
    summary = {}
    T20 = gzip.open(OUT / "relations_top20.tsv.gz", "wt"); KA = gzip.open(OUT / "relations_kegg_aligned.tsv.gz", "wt")
    RS = gzip.open(OUT / "samples_responsive.tsv.gz", "wt")
    PT = (OUT / "perturbations.tsv").open("w"); MG = (OUT / "measured_genes.tsv").open("w")
    cols = ["dataset", "cell_line", "modality", "pert_gene", "pert_entrez", "pert_kos", "pert_uniprot", "pert_n_cells",
            "pert_ontarget_t", "resp_gene", "resp_entrez", "resp_kos", "resp_ctrl_mean", "resp_frac_expr_ctrl",
            "obs_delta_log1p", "obs_t", "obs_p", "inf_q_bh", "inf_direction", "inf_rank_abs_t", "inf_spec", "label",
            "kegg_edge_A_to_g", "kegg_edge_g_to_A", "kegg_rel_A_to_g", "kegg_dist_directed", "kegg_dist_undirected",
            "kegg_n_shared_pathways", "kegg_ko_edge_A_to_g_split", "string_exp_ko",
            "pred_konbf_score", "pred_konbf_rank", "pred_bpn_rank", "n_candidates", "proposed_split"]
    for fh in (T20, KA, RS):
        fh.write("\t".join(cols) + "\n")
    PT.write("\t".join(["dataset", "cell_line", "modality", "pert_gene", "map_via", "entrez", "kos", "uniprot", "n_cells",
                        "n_gem_groups", "ontarget_measured", "ontarget_t", "ontarget_delta", "ontarget_class",
                        "n_resp_q05", "n_responsive", "n_kegg_downstream_measured", "proposed_split"]) + "\n")
    MG.write("dataset\tgene\tensembl\tmap_via\tentrez\tkos\n")
    for d in args.datasets:
        z = np.load(HERE / f"data/stats_{d}.npz", allow_pickle=True)
        genes = [str(x) for x in z["genes"]]; gids = [str(x) for x in z["gene_ids"]]; perts = [str(x) for x in z["perts"]]
        cl, mod = str(z["cell_line"]), str(z["modality"])
        T, Pv, Dl = z["t"], z["pval"], z["delta"]
        nP, G = T.shape
        gm = [map_gene(g, e, M) for g, e in zip(genes, gids)]
        sym2col = {}
        for j, g in enumerate(genes):
            sym2col[g] = j
            if gm[j][0]:
                sym2col.setdefault(gm[j][0]["symbol"], j)
        for j, g in enumerate(genes):
            r = gm[j][0] or {}
            MG.write(f"{d}\t{g}\t{gids[j]}\t{gm[j][1]}\t{r.get('entrez', '')}\t{';'.join(r.get('kos', []))}\n")
        pm = [map_gene(p, "", M) for p in perts]
        absT = np.abs(T)
        spec = (absT.argsort(0).argsort(0) + 0.5) / nP
        Q = np.ones_like(Pv); RK = np.zeros(T.shape, np.int32)
        # on-target column of A (CRISPRi: expect down, CRISPRa: up)
        ot_t = np.full(nP, np.nan); ot_d = np.full(nP, np.nan); self_col = np.full(nP, -1)
        for i, p in enumerate(perts):
            j = sym2col.get(p, sym2col.get((pm[i][0] or {}).get("symbol", "~"), -1))
            self_col[i] = j
            if j >= 0:
                ot_t[i], ot_d[i] = T[i, j], Dl[i, j]
        sign = -1 if mod == "CRISPRi" else 1
        ot_cls = np.where(np.isnan(ot_t), "not_measured", np.where(sign * ot_t > 3, "on_target", "weak_or_none"))
        g_ko = [[voc[k] for k in (gm[j][0] or {}).get("kos", [])] for j in range(G)]
        g_ent = [(gm[j][0] or {}).get("entrez", "") for j in range(G)]
        LAB = np.zeros(T.shape, np.int8)  # 1 responsive, -1 no detected response, 0 other
        NB = np.full(T.shape, np.nan, np.float32); NR = np.zeros(T.shape, np.int32); BR = np.zeros(T.shape, np.int32)
        KD = np.full(T.shape, 99, np.int8); KU = np.full(T.shape, 99, np.int8); KE = np.zeros(T.shape, np.int8)
        STR = np.zeros(T.shape, np.float32); ELIG = np.zeros(T.shape, bool)
        fexp = z["frac_expr_ctrl"]
        mk = max(1, max(len(x) for x in g_ko)); GI = np.zeros((G, mk), np.int64); GM = np.zeros((G, mk), bool)
        for j, ks in enumerate(g_ko):
            GI[j, :len(ks)] = ks; GM[j, :len(ks)] = True
        has_ko = GM.any(1)

        def gmax(row):  # [V] KO-level row -> [G] max over each gene's KOs (-inf if none)
            v = row[GI]; v[~GM] = -np.inf
            return v.max(1)
        n_resp_tot = 0
        for i, p in enumerate(perts):
            rec = pm[i][0] or {}; A_ent = rec.get("entrez", ""); A_ko = [voc[k] for k in rec.get("kos", [])]
            others = np.array([j for j in range(G) if j != self_col[i]])
            Q[i, others] = bh(Pv[i, others])
            o = others[np.argsort(-absT[i, others])]; RK[i, o] = np.arange(1, len(o) + 1)
            ok_A = ot_cls[i] != "weak_or_none"
            resp = (Q[i] < 0.05) & (spec[i] >= 0.9) & ok_A
            nores = (Q[i] >= 0.5) & (fexp >= 0.1)
            LAB[i, resp] = 1; LAB[i, nores & ~resp] = -1
            if self_col[i] >= 0:
                LAB[i, self_col[i]] = 0; resp[self_col[i]] = False
            if A_ent:
                dd, du = bfs(kout, A_ent), bfs(kund, A_ent)
                KD[i] = [dd.get(x, 99) if x else 99 for x in g_ent]; KU[i] = [du.get(x, 99) if x else 99 for x in g_ent]
            if A_ko:
                STR[i] = np.maximum(gmax(st[A_ko].max(0)), 0)
                KE[i] = gmax(KOE[A_ko].max(0).astype(np.float32)) > 0
                shared = GM & np.isin(GI, A_ko)
                ELIG[i] = has_ko & ~shared.any(1)
                if self_col[i] >= 0:
                    ELIG[i, self_col[i]] = False
                NB[i, ELIG[i]] = gmax(NBF[A_ko].max(0))[ELIG[i]]
            e = np.where(ELIG[i])[0]
            if len(e):
                NR[i, e] = (-NB[i, e]).argsort().argsort() + 1
                bs = gmax(BPN[A_ko].max(0))[e]
                BR[i, e] = (-bs).argsort().argsort() + 1
            n_resp_tot += int(resp.sum())
            split = ["train"] * 14 + ["val"] * 3 + ["test"] * 3
            sp = split[int(hashlib.sha1((rec.get("symbol") or p).encode()).hexdigest(), 16) % 20]
            n_kd = int(sum(1 for j in others if KD[i, j] == 1))
            PT.write("\t".join(map(str, [d, cl, mod, p, pm[i][1], A_ent, ";".join(rec.get("kos", [])), uni.get(A_ent, ""),
                                         z["n_cells"][i], z["n_gem"][i], int(self_col[i] >= 0),
                                         "" if np.isnan(ot_t[i]) else f"{ot_t[i]:.2f}",
                                         "" if np.isnan(ot_d[i]) else f"{ot_d[i]:.3f}", ot_cls[i],
                                         int((Q[i] < 0.05).sum()), int(resp.sum()), n_kd, sp])) + "\n")

            def row(j):
                rel = ";".join(sorted(krel.get((A_ent, g_ent[j]), ()))) if A_ent and g_ent[j] else ""
                kes = sorted({ko_edge[(a, b)] for a in A_ko for b in g_ko[j] if (a, b) in ko_edge})
                return [d, cl, mod, p, A_ent, ";".join(rec.get("kos", [])), uni.get(A_ent, ""), z["n_cells"][i],
                        "" if np.isnan(ot_t[i]) else f"{ot_t[i]:.2f}", genes[j], g_ent[j],
                        ";".join((gm[j][0] or {}).get("kos", [])), f"{z['ctrl_mean'][j]:.4f}", f"{fexp[j]:.3f}",
                        f"{Dl[i, j]:.4f}", f"{T[i, j]:.3f}", f"{Pv[i, j]:.3g}", f"{Q[i, j]:.3g}",
                        "up" if T[i, j] > 0 else "down", RK[i, j], f"{spec[i, j]:.4f}",
                        {1: "responsive", -1: "no_detected_resp", 0: "other"}[int(LAB[i, j])],
                        int(KD[i, j] == 1), int(bool(A_ent) and A_ent in kout.get(g_ent[j], ())),
                        rel, "" if KD[i, j] == 99 else int(KD[i, j]), "" if KU[i, j] == 99 else int(KU[i, j]),
                        len(kpw.get(A_ent, set()) & kpw.get(g_ent[j], set())) if A_ent and g_ent[j] else 0,
                        ";".join(kes), f"{STR[i, j]:.0f}",
                        "" if np.isnan(NB[i, j]) else f"{NB[i, j]:.3f}", NR[i, j] or "", BR[i, j] or "",
                        len(e), sp]
            for j in o[:TOPN]:
                T20.write("\t".join(map(str, row(j))) + "\n")
            for j in others:
                if (A_ko and g_ko[j]) and (KU[i, j] <= 2 or STR[i, j] >= 150 or KE[i, j]):
                    KA.write("\t".join(map(str, row(j))) + "\n")
                if LAB[i, j] == 1:
                    RS.write("\t".join(map(str, row(j))) + "\n")
        np.savez_compressed(HERE / f"data/pairs_{d}.npz", genes=np.array(genes), perts=np.array(perts), q=Q.astype(np.float32),
                            rank_abs_t=RK, spec=spec.astype(np.float32), label=LAB, eligible=ELIG, konbf=NB, konbf_rank=NR,
                            bpn_rank=BR, kegg_dist_dir=KD, kegg_dist_und=KU, kegg_ko_edge=KE, string_exp=STR,
                            ontarget_class=ot_cls, self_col=self_col)
        summary[d] = {"cell_line": cl, "modality": mod, "n_cells": int(z["n_cells_total"]), "n_ctrl": int(z["n_ctrl"]),
                      "n_gem_groups": int(z["n_gem_total"]), "n_measured_genes": G,
                      "measured_mapped": sum(1 for x in gm if x[0]), "measured_with_ko": sum(1 for x in g_ko if x),
                      "n_single_perts": nP, "perts_mapped": sum(1 for x in pm if x[0]),
                      "perts_with_ko": sum(1 for x in pm if x[0] and x[0]["kos"]),
                      "cells_per_pert_median": float(np.median(z["n_cells"])), "cells_per_pert_min": int(np.min(z["n_cells"])),
                      "ontarget": {c: int((ot_cls == c).sum()) for c in ("on_target", "weak_or_none", "not_measured")},
                      "pairs_q05": int((Q < 0.05).sum()), "pairs_responsive": n_resp_tot,
                      "pairs_no_detected_resp": int((LAB == -1).sum()), "pairs_eligible_ko": int(ELIG.sum())}
        print(d, json.dumps(summary[d]), flush=True)
    for fh in (T20, KA, RS, PT, MG):
        fh.close()
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

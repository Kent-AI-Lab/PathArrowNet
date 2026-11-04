#!/usr/bin/env python3
"""
Main results: Table 1, Sec. 4.1 (pooled comparisons, leave-one-split-out, split 5), Sec. 4.2 (task S), and Suppl.
Tables S7 (five splits), S13 (pooled differences of the network baselines), S14 (paired comparisons per split) and S20
(leave-one-split-out).

    python evaluation/main_results.py            -> results/evaluation/main_results.json
"""
import numpy as np

from common import (ADA, ADA_SA, CS, PAN, PAN0_SA, PAN_SA, RANKER, REP, REP_SA, STS, SPLIT, Q, boot, macro_S, ranks_all,
                    rr, write)

V1 = STS[0]
TABLE1 = [  # (row label, score-dump method)
    ("Train in-degree", "baseline_popularity"),
    ("GO similarity (MF+CC)", "baseline_go_profile_no_BP"),
    ("Collaborative filtering", "baseline_cf"),
    ("STRING neighbor (no training)", "baseline_bg_neighbor"),
    ("Network feature ranker", RANKER),
    ("DistMult", "kge_distmult+bg"),
    ("RotatE", "kge_rotate+bg"),
    ("TransE", "kge_transe+bg"),
    ("Reproduced BioPathNet, single model", REP),
    ("Reproduced BioPathNet, seed-averaged", REP_SA),
    ("BioPathNet with our adaptations, single", ADA),
    ("BioPathNet with our adaptations, seed-averaged", ADA_SA),
    ("PathArrowNet, single model", PAN),
    ("PathArrowNet, seed-averaged (primary)", PAN_SA),
]
NETWORK_BASELINES = {"PPR on STRING": "baseline_ppr_string", "PPR on STRING + KEGG training edges": "baseline_ppr_string+kegg",
                     "feature ranker, STRING features only": "baseline_topo_lr_string", "feature ranker, STRING + KEGG training-graph features": RANKER,
                     "STRING-neighbor rule": "baseline_bg_neighbor", "TransE + background": "kge_transe+bg",
                     "reproduced BioPathNet (per-seed mean)": REP, "BioPathNet with our adaptations (per-seed mean)": ADA}

_rr = {}


def RR(st, m):
    if (st, m) not in _rr:
        _rr[(st, m)] = rr(st, m)
    return _rr[(st, m)]


def level(m, sts=STS):
    """pooled pathway-macro MRR level (column P of Table 1)"""
    return CS.pooled({st: (RR(st, m), np.zeros(len(Q(st))), Q(st)) for st in sts})["est"]


def pooled(a, b, sts=STS):
    return CS.pooled({st: (RR(st, a), RR(st, b), Q(st)) for st in sts})


def main():
    out = {"table1": {}}
    for label, m in TABLE1:
        R1 = ranks_all(V1, m)
        row = {"method": m, "split1_MRR": float(RR(V1, m).mean())}
        row.update({f"H@{k}": float(np.mean([(r <= k).mean() for r in R1])) for k in (1, 10, 100)})
        try:
            row["S"] = macro_S(V1, ranks_all(V1, m, task="S"))
        except FileNotFoundError:
            row["S"] = None
        row.update({SPLIT[st]: float(RR(st, m).mean()) for st in STS[1:]})
        row["P"] = level(m)
        out["table1"][label] = row
        print(f"{label:48s} " + " ".join(f"{k} {v:.4f}" for k, v in row.items() if isinstance(v, float)), flush=True)

    # Sec. 4.1: pooled paired differences (pathway-macro, 10,000 cluster resamples)
    out["pooled"] = {}
    for tag, a, b in (("PathArrowNet - reproduced BioPathNet (seed-averaged)", PAN_SA, REP_SA),
                      ("PathArrowNet - reproduced BioPathNet (single models)", PAN, REP),
                      ("PathArrowNet (seed-averaged) - reproduced BioPathNet (per-seed mean)", PAN_SA, REP),
                      ("PathArrowNet (seed-averaged) - network feature ranker", PAN_SA, RANKER),
                      ("PathArrowNet - BioPathNet with our adaptations (seed-averaged)", PAN_SA, ADA_SA),
                      ("PathArrowNet - BioPathNet with our adaptations (single models)", PAN, ADA),
                      ("BioPathNet with our adaptations - reproduced BioPathNet (seed-averaged)", ADA_SA, REP_SA),
                      ("BioPathNet with our adaptations - reproduced BioPathNet (single models)", ADA, REP),
                      ("PathArrowNet - PathArrowNet without relation weights (seed-averaged)", PAN_SA, PAN0_SA)):
        out["pooled"][tag] = pooled(a, b)
        r = out["pooled"][tag]
        print(f"POOLED {tag}: {r['est']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] p={r['p']:.4f}", flush=True)
    out["relative_improvement_seedavg_vs_reproduced"] = level(PAN_SA) / level(REP_SA) - 1
    out["relative_improvement_seedavg_vs_feature_ranker"] = level(PAN_SA) / level(RANKER) - 1

    # Suppl. Table S20: leave one split out
    out["leave_one_split_out"] = {}
    for drop in [None] + STS:
        keep = [s for s in STS if s != drop]
        out["leave_one_split_out"]["none" if drop is None else SPLIT[drop]] = {
            "seed_averaged": pooled(PAN_SA, REP_SA, keep), "single": pooled(PAN, REP, keep),
            "relative": level(PAN_SA, keep) / level(REP_SA, keep) - 1}

    # Suppl. Tables S7 / S14: per-split paired differences (query level, 2,000 pathway resamples) and seed-averaging gains
    contrasts = {"vs reproduced BioPathNet (per-seed mean)": (PAN_SA, REP), "vs network feature ranker": (PAN_SA, RANKER),
                 "seed-averaged vs seed-averaged reproduced BioPathNet": (PAN_SA, REP_SA), "single vs single reproduced BioPathNet": (PAN, REP),
                 "seed-averaged vs seed-averaged BioPathNet with our adaptations": (PAN_SA, ADA_SA)}
    out["per_split"] = {c: {**{SPLIT[st]: boot(RR(st, a), RR(st, b), Q(st)) for st in STS}, "pooled": pooled(a, b)} for c, (a, b) in contrasts.items()}
    out["levels_seed_averaged"] = {m: {**{SPLIT[st]: float(RR(st, m).mean()) for st in STS},
                                       "concatenated queries": float(np.concatenate([RR(st, m) for st in STS]).mean())} for m in (PAN_SA, REP_SA, ADA_SA)}
    out["seed_averaging_gain"] = {m1: {**{SPLIT[st]: float((RR(st, m1) - RR(st, m0)).mean()) for st in STS}, "pooled": pooled(m1, m0)}
                                  for m1, m0 in ((PAN_SA, PAN), (ADA_SA, ADA), (REP_SA, REP))}

    # Suppl. Table S13: network baselines against PathArrowNet and reproduced BioPathNet (pooled)
    out["network_baselines"] = {lab: {**{SPLIT[st]: float(RR(st, m).mean()) for st in STS}, "minus_reproduced_BioPathNet": pooled(m, REP),
                                      "minus_PathArrowNet": pooled(m, PAN_SA)} for lab, m in NETWORK_BASELINES.items()}

    # Sec. 4.1: split 5 (chemokine signaling) and pathway-weighted split levels
    st = STS[4]
    chem = np.array([q["cluster"] == "hsa04062" for q in Q(st)])   # queries whose test pathway (cluster) is chemokine signaling
    out["split5"] = {"n_chemokine_queries": int(chem.sum()), "n_queries": len(chem), "PathArrowNet_chemokine": float(RR(st, PAN_SA)[chem].mean()),
                     "reproduced_BioPathNet_chemokine": float(RR(st, REP_SA)[chem].mean()),
                     "pathway_weighted_level": {SPLIT[s]: level(PAN_SA, [s]) for s in STS},
                     "per_seed_query_weighted": [float(x) for x in (1 / ranks_all(st, PAN)).mean(1)],
                     "per_seed_pathway_weighted": [CS.pooled({st: (1 / r, np.zeros(len(r)), Q(st))})["est"] for r in ranks_all(st, PAN)]}

    # Sec. 4.2: task S. Split levels and per-split differences use the task-S pathway-macro weights; the pooled difference uses
    # the convention of every pooled comparison (per-query values averaged within each test pathway cluster, then over clusters)
    rrS = {}
    for m in (PAN_SA, REP_SA, ADA_SA, PAN, REP, ADA, PAN0_SA):
        for s in STS:
            rrS[(s, m)] = (1 / ranks_all(s, m, task="S")).mean(0)
    wS = {s: np.array([q["weight"] for q in Q(s, "test", "S")]) for s in STS}
    out["task_S"] = {"levels": {m: {SPLIT[s]: float((wS[s] * rrS[(s, m)]).sum() / wS[s].sum()) for s in STS} for m in (PAN_SA, REP_SA, ADA_SA, PAN, REP, ADA, PAN0_SA)}}
    for m in out["task_S"]["levels"]:
        out["task_S"]["levels"][m]["mean over splits"] = float(np.mean(list(out["task_S"]["levels"][m].values())))
    for tag, a, b in (("PathArrowNet - reproduced BioPathNet (seed-averaged)", PAN_SA, REP_SA), ("PathArrowNet - BioPathNet with our adaptations (seed-averaged)", PAN_SA, ADA_SA),
                      ("PathArrowNet - reproduced BioPathNet (single models)", PAN, REP)):
        out["task_S"][tag] = {**{SPLIT[s]: boot(rrS[(s, a)], rrS[(s, b)], Q(s, "test", "S"), macro=True) for s in STS},
                              "pooled": CS.pooled({s: (rrS[(s, a)], rrS[(s, b)], Q(s, "test", "S")) for s in STS})}
        r = out["task_S"][tag]["pooled"]
        print(f"TASK S {tag}: {r['est']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]", flush=True)
    write("main_results.json", out)


if __name__ == "__main__":
    main()

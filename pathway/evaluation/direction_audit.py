#!/usr/bin/env python3
"""
Directed-label audit and comparison models (Suppl. Table S17) and direction against the degree-share rule (Suppl. Sec. S11).
All parts use the 8,679 test edges a -> b whose reverse is not a KEGG edge (as in direction.py) and the seed-averaged
PathArrowNet indicator 1[z_a(b) > z_b(a)].

  A  direction by how the KGML label arises: an arrow between two single-KO boxes (box factor 1) or only through the
     expansion of larger boxes (factor > 1 in every map); mechanistic subtype vs only indirect effect / state change;
     direct STRING edge between the endpoints. Edge-level share and pathway-macro share (pooled bootstrap against 0.5).
  B  the same indicator for comparison models on the edges whose two endpoints both are anchors of the model's score
     dumps (seed-averaged rows of each model), paired against PathArrowNet on the same edges.
  C  task-A ranking on the test queries without a direct STRING edge between source and target (pathway-macro levels and
     pooled paired differences).
  D  direction on the edges where the training out-degree share rule is tied or wrong, wrong, tied or right, and the
     agreement of the model with the rule on the rule's non-tie edges.

    python evaluation/direction_audit.py          -> results/evaluation/direction_audit.json
"""
import csv
from collections import defaultdict

import numpy as np

from common import CS, PAN_SA, PW, RANKER, REV, STS, ADA, SL, rr, runs, string_pairs, write
from direction import degree_rule, dir_edges, dir_indicators, load_rows

SEEDS = [f"s{s}" for s in range(42, 47)]
TRANSE = "kge_transe+bg"
MECH = {"activation", "inhibition", "phosphorylation", "dephosphorylation", "ubiquitination", "glycosylation", "methylation",
        "expression", "repression"}
AMBIG = {"indirect effect", "indirect", "state change"}


def edge_meta():
    """per KO pair: smallest box factor |S|x|T| over the KGML relations that produce it, and the union of their subtypes"""
    kd = PW / "kegg_dataset"; kos = {}
    with (kd / "node_occurrences.csv").open() as f:
        for r in csv.DictReader(f):
            if r["entry_type"] == "gene":
                s = set((r["derived_ko_ids_from_hsa_genes"] or r["ko_ids"] or "").replace(";", " ").split())
                kos[(r["pathway_id"], r["entry_id"])] = {k.replace("ko:", "") for k in s}
    meta = defaultdict(lambda: {"factor": 10 ** 9, "sub": set()})
    with (kd / "kgml_relations.csv").open() as f:
        for r in csv.DictReader(f):
            if r["whether_directional"] != "1" or r["source_entry_type"] != "gene" or r["target_entry_type"] != "gene":
                continue
            p = r["pathway_id"]; S = kos.get((p, r["source_entry"]), set()); T = kos.get((p, r["target_entry"]), set())
            subs = {x.strip() for x in r["relation_subtypes"].replace("|", ";").split(";") if x.strip()}
            for a in S:
                for b in T:
                    if a != b:
                        m = meta[(a, b)]; m["factor"] = min(m["factor"], len(S) * len(T)); m["sub"] |= subs
    return meta


def share(ind, qs):
    r = CS.pooled({st: (ind[st], np.full(len(ind[st]), 0.5), qs[st]) for st in ind if len(ind[st])})
    return {"share": 0.5 + r["est"], "excess": r["est"], "lo": r["lo"], "hi": r["hi"], "p": r["p"]}


def main():
    meta = edge_meta(); out = {"A": {}, "B": {}, "C": {}, "D": {}}
    E, IND, R1, STRA = {}, {}, {}, defaultdict(dict)
    for st in STS:
        qs, sup = dir_edges(st, "test"); voc, V = SL.vocab(st); tok = {i: t.replace("ko:", "") for t, i in voc.items()}
        IND[st], _ = dir_indicators(load_rows(runs(st, REV, SEEDS)), qs); E[st] = qs; R1[st] = degree_rule(st, qs)
        m = [meta.get((tok[q["source"]], tok[q["target"]]), {"factor": 0, "sub": set()}) for q in qs]
        fac = np.array([x["factor"] for x in m]); mech = np.array([bool(x["sub"] & MECH) for x in m])
        amb = np.array([bool(x["sub"] & AMBIG) and not (x["sub"] & MECH) for x in m])
        STRA["all"][st] = np.ones(len(qs), bool); STRA["single-KO boxes"][st] = fac == 1; STRA["expansion only"][st] = fac > 1
        STRA["mechanistic subtype"][st] = mech; STRA["only indirect effect / state change"][st] = amb
        STRA["direct STRING edge"][st] = sup
    for name, M in STRA.items():
        ind = {st: IND[st][M[st]] for st in STS}; q = {st: [E[st][i] for i in np.where(M[st])[0]] for st in STS}
        out["A"][name] = {"n_edges": int(sum(v.sum() for v in M.values())),
                          "edge_share": float(np.concatenate(list(ind.values())).mean()), "pathway_macro": share(ind, q)}
        print("A", name, out["A"][name]["n_edges"], round(out["A"][name]["edge_share"], 3), flush=True)

    for name, m in {"BioPathNet with our adaptations": ADA, "network feature ranker": RANKER, "TransE + background": TRANSE}.items():
        a, b, qq = {}, {}, {}
        for st in STS:
            rows = load_rows(runs(st, m)); anc = set(rows[0][0])
            keep = [i for i, q in enumerate(E[st]) if q["source"] in anc and q["target"] in anc]
            qq[st] = [E[st][i] for i in keep]; a[st] = IND[st][keep]; b[st] = dir_indicators(rows, qq[st])[0]
        out["B"][name] = {"covered_edges": int(sum(len(v) for v in a.values())),
                          "PathArrowNet_edge_share": float(np.concatenate(list(a.values())).mean()),
                          "PathArrowNet_macro": share(a, qq), "model_edge_share": float(np.concatenate(list(b.values())).mean()),
                          "PathArrowNet_minus_model": CS.pooled({st: (a[st], b[st], qq[st]) for st in STS})}
        print("B", name, out["B"][name]["covered_edges"], round(out["B"][name]["model_edge_share"], 3), flush=True)

    MOD = {"PathArrowNet, seed-averaged": PAN_SA, "network feature ranker": RANKER, "TransE + background": TRANSE,
           "STRING-neighbor rule": "baseline_bg_neighbor"}
    R, Qn = {k: {} for k in MOD}, {}
    for st in STS:
        voc, V = SL.vocab(st); SP = string_pairs(voc); qa = SL.load_queries(st, "test", "A")
        mask = np.array([(q["source"], q["target"]) not in SP for q in qa]); Qn[st] = [q for q, x in zip(qa, mask) if x]
        for k, m in MOD.items():
            R[k][st] = rr(st, m)[mask]
    out["C"]["n_queries"] = int(sum(len(v) for v in Qn.values()))
    for k in MOD:
        out["C"][k] = {"level": CS.pooled({st: (R[k][st], np.zeros(len(Qn[st])), Qn[st]) for st in STS})["est"]}
        if k != "PathArrowNet, seed-averaged":
            out["C"][k]["PathArrowNet_minus"] = CS.pooled({st: (R["PathArrowNet, seed-averaged"][st], R[k][st], Qn[st]) for st in STS})

    for name, f in (("tie_or_wrong", lambda r: r < 1), ("wrong", lambda r: r == 0), ("tie", lambda r: r == 0.5),
                    ("right", lambda r: r == 1)):
        M = {st: f(R1[st]) for st in STS}
        ind = {st: IND[st][M[st]] for st in STS}; q = {st: [E[st][i] for i in np.where(M[st])[0]] for st in STS}
        out["D"][name] = {"n_edges": int(sum(v.sum() for v in M.values())),
                          "edge_share": float(np.concatenate(list(ind.values())).mean()), "pathway_macro": share(ind, q)}
    nt = {st: R1[st] != 0.5 for st in STS}
    out["D"]["agreement_on_rule_non_tie"] = float(np.concatenate([(IND[st][nt[st]] > 0.5) == (R1[st][nt[st]] == 1) for st in STS]).mean())
    write("direction_audit.json", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Ablations of PathArrowNet without relation-level weights: Table 2, the pooled ablation numbers of Sec. 4.4, the
evidence-sensitivity interval plot (Suppl. Fig. S5) and the hyper-parameter sensitivity (Suppl. Table S2).
Each variant was retrained with five seeds on strict split 1 (three seeds, s42-s44, on splits 2-5 for the score-bin
ablations); its score dumps are compared with the reference configuration on the same seeds.

    python evaluation/ablation.py                -> results/evaluation/ablation.json
"""
import numpy as np

from common import CS, PAN0, PAN0_SA, STS, Q, SL, boot, rr, seedavg_rr, sig, write

V1 = STS[0]
TABLE2 = [  # (row, single-model dumps, seed-averaged dump)
    ("Same edges, no score bins", "clean_xp_exp150bin1", "clean_xp_exp150bin1_seedavg"),
    ("Same edges, shuffled bins", "clean_xp_exp150shufbin4", "clean_xp_exp150shufbin4_seedavg"),
    ("Direct human experimental evidence only", "clean_xp_expdirect150bin4", "clean_xp_expdirect150bin4_seedavg"),
    ("No STRING (KEGG edges only)", "clean_nbf_nobg", "clean_nbf_nobg_seedavg"),
    ("KEGG labels, no KEGG message-passing edges", "clean_xp_nokeggmp", "clean_xp_nokeggmp_seedavg"),
    ("No graph: pair MLP on ESM2 features", "pair_mlp+esm+allneg+kl", "clean_pairmlp_seedavg"),
    ("BioPathNet with our protein features", "biopathnet+bg+esm+gobert_noBP", "clean_bpnfeat_seedavg"),  # official BioPathNet + ESM2 and GoBERT features
]
BINS = {"Binary edges": "clean_xp_exp150bin1", "Shuffled bins": "clean_xp_exp150shufbin4", "Direct human evidence": "clean_xp_expdirect150bin4"}
SENS = {"4 layers": "layers4", "8 layers (batch 16)": "layers8_b16", "d=16": "d16", "d=64 (batch 16)": "d64_b16", "lr 1e-3": "lr1e-3",
        "lr 4e-3": "lr4e-3", "dropout 0.3": "dropout0.3", "STRING experimental >= 400": "exp400", "batch 16 (control)": "batch16"}


def seeds_of(st, m):
    return sorted(p.name for p in (SL.SCORES / st / m).iterdir() if p.is_dir())


def main():
    out = {"table2": {"PathArrowNet without relation weights": {"single": float(rr(V1, PAN0).mean()), "seed_averaged": float(rr(V1, PAN0_SA).mean())}}}
    ref1, refs = rr(V1, PAN0), rr(V1, PAN0_SA)
    for row, m1, ms in TABLE2:
        a1, as_ = rr(V1, m1), rr(V1, ms)
        d1, ds = boot(a1, ref1, Q(V1)), boot(as_, refs, Q(V1))
        out["table2"][row] = {"single": float(a1.mean()), "seed_averaged": float(as_.mean()), "single_minus_reference": d1, "seed_averaged_minus_reference": ds,
                              "single_significant": sig(d1), "seed_averaged_significant": sig(ds),
                              "per_seed_single": [float(x) for x in (1 / np.array([SL.ranks(SL.query_scores(SL.SCORES / V1 / m1 / s, "test", "A", Q(V1)).astype(np.float64), Q(V1), SL.vocab(V1)[1])[0] for s in seeds_of(V1, m1)])).mean(1)]}
        print(f"{row:45s} {a1.mean():.4f}{'*' if sig(d1) else ' '} {as_.mean():.4f}{'*' if sig(ds) else ' '}", flush=True)

    # pooled over the five strict splits (pathway-macro)
    pool = lambda a, b: CS.pooled({st: (rr(st, a), rr(st, b), Q(st)) for st in STS})
    out["pooled"] = {"No STRING, single": pool("clean_nbf_nobg", PAN0), "No STRING, seed-averaged": pool("clean_nbf_nobg_seedavg", PAN0_SA),
                     "Pair MLP, single": pool("pair_mlp+esm+allneg+kl", PAN0)}

    # Suppl. Fig. S5: evidence sensitivity, main split and pooled; references restricted to the variant's seeds
    fig = {"main_split": {}, "pooled": {}}
    for lab, m in BINS.items():
        fig["main_split"][lab] = {"single": boot(rr(V1, m), ref1, Q(V1)), "seed_averaged": boot(rr(V1, m + "_seedavg"), refs, Q(V1))}
        p1, ps = {}, {}
        for st in STS:
            sd = seeds_of(st, m)
            p1[st] = (rr(st, m), rr(st, PAN0, seeds=sd), Q(st))
            ps[st] = (rr(st, m + "_seedavg"), seedavg_rr(st, PAN0, sd) if len(sd) < 5 else rr(st, PAN0_SA), Q(st))
        fig["pooled"][lab] = {"single": CS.pooled(p1), "seed_averaged": CS.pooled(ps), "seeds": {st: seeds_of(st, m) for st in STS}}
    for lab, m in (("KEGG edges as labels only", "clean_xp_nokeggmp"), ("No STRING background", "clean_nbf_nobg")):
        fig["main_split"][lab] = {"single": boot(rr(V1, m), ref1, Q(V1)), "seed_averaged": boot(rr(V1, m + "_seedavg"), refs, Q(V1))}
    fig["pooled"]["No STRING background"] = {"single": out["pooled"]["No STRING, single"], "seed_averaged": out["pooled"]["No STRING, seed-averaged"]}
    out["evidence_sensitivity"] = fig

    # Suppl. Table S2: one knob at a time, strict split 1, seed 42, against the reference run of the same seed
    _, V = SL.vocab(V1)
    def one(m, split):
        qs = Q(V1, split)
        return 1 / np.asarray(SL.ranks(SL.query_scores(SL.SCORES / V1 / m / "s42", split, "A", qs).astype(np.float64), qs, V)[0], dtype=float)
    r_test, r_val = one(PAN0, "test"), one(PAN0, "val")
    out["sensitivity"] = {"reference": {"val_MRR": float(r_val.mean()), "test_MRR": float(r_test.mean())}}
    for lab, n in SENS.items():
        t = one(f"clean_sens_{n}", "test")
        out["sensitivity"][lab] = {"val_MRR": float(one(f"clean_sens_{n}", "val").mean()), "test_MRR": float(t.mean()), "minus_reference": boot(t, r_test, Q(V1))}
    write("ablation.json", out)


if __name__ == "__main__":
    main()

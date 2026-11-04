#!/usr/bin/env python3
"""
Compare the evaluation outputs (results/evaluation/*.json) with the numbers printed in the paper. Each check gives the value
as printed (rounded) and passes when the computed value rounds to it. Run after the scripts in this folder.

    python evaluation/check_paper_numbers.py     -> prints every check; exit code 1 if any check fails
"""
import json
import sys

from common import ADA, ADA_SA, OUT, PAN, PAN0_SA, PAN_SA, REP, REP_SA

PANT = "PathArrowNet - reproduced BioPathNet (seed-averaged)"
T1 = {  # Table 1: split-1 MRR, H@1, H@10, H@100, S; MRR splits 2-5; P
    "Train in-degree": (0.013, .001, .021, .137, .012, .007, .044, .010, .008, .016),
    "GO similarity (MF+CC)": (0.027, .011, .049, .183, .036, .022, .021, .023, .028, .024),
    "Collaborative filtering": (0.044, .022, .080, .231, .077, .045, .050, .031, .026, .042),
    "STRING neighbor (no training)": (0.069, .041, .119, .246, .103, .054, .071, .070, .040, .091),
    "Network feature ranker": (0.081, .049, .137, .364, .113, .073, .072, .138, .045, .094),
    "DistMult": (0.022, .006, .046, .202, .035, .023, .018, .024, .011, .023),
    "RotatE": (0.044, .019, .087, .261, .067, .040, .038, .059, .026, .049),
    "TransE": (0.069, .036, .123, .379, .080, .056, .057, .100, .051, .081),
    "Reproduced BioPathNet, single model": (0.083, .032, .176, .469, .114, .068, .057, .090, .080, .079),
    "Reproduced BioPathNet, seed-averaged": (0.090, .035, .190, .485, .128, .071, .059, .099, .086, .086),
    "BioPathNet with our adaptations, single": (0.095, .051, .175, .467, .151, .088, .076, .091, .076, .107),
    "BioPathNet with our adaptations, seed-averaged": (0.103, .052, .185, .505, .160, .096, .084, .102, .086, .118),
    "PathArrowNet, single model": (0.103, .053, .198, .484, .185, .094, .090, .103, .157, .113),
    "PathArrowNet, seed-averaged (primary)": (0.117, .064, .220, .530, .203, .107, .100, .123, .207, .131),
}
T1_KEYS = ("split1_MRR", "H@1", "H@10", "H@100", "S", "split 2", "split 3", "split 4", "split 5", "P")
SPL = ["split 1", "split 2", "split 3", "split 4", "split 5"]


def checks():
    C = []  # (file, path, printed value, decimals)
    for row, vals in T1.items():
        C += [("main_results", ["table1", row, k], v, 3) for k, v in zip(T1_KEYS, vals)]
    for tag, v in ((PANT, .045), ("PathArrowNet - reproduced BioPathNet (single models)", .034), ("PathArrowNet (seed-averaged) - network feature ranker", .037),
                   ("PathArrowNet - BioPathNet with our adaptations (seed-averaged)", .013), ("BioPathNet with our adaptations - reproduced BioPathNet (seed-averaged)", .032),
                   ("BioPathNet with our adaptations - reproduced BioPathNet (single models)", .029)):
        C.append(("main_results", ["pooled", tag, "est"], v, 3))
    C += [("main_results", ["relative_improvement_seedavg_vs_reproduced"], .530, 3), ("main_results", ["relative_improvement_seedavg_vs_feature_ranker"], .397, 3)]
    for k, (a, rel, b) in {"none": (.045, .530, .034), "split 1": (.043, .525, .031), "split 2": (.043, .471, .032), "split 3": (.047, .498, .036),
                           "split 4": (.048, .689, .037), "split 5": (.039, .444, .030)}.items():
        C += [("main_results", ["leave_one_split_out", k, "seed_averaged", "est"], a, 3), ("main_results", ["leave_one_split_out", k, "relative"], rel, 3),
              ("main_results", ["leave_one_split_out", k, "single", "est"], b, 3)]
    C += [("main_results", ["split5", "n_chemokine_queries"], 806, 0), ("main_results", ["split5", "n_queries"], 1716, 0),
          ("main_results", ["split5", "PathArrowNet_chemokine"], .299, 3), ("main_results", ["split5", "reproduced_BioPathNet_chemokine"], .122, 3),
          ("main_results", ["split5", "pathway_weighted_level", "split 5"], .158, 3), ("main_results", ["split5", "pathway_weighted_level", "split 4"], .145, 3)]
    for c, vals in {"seed-averaged vs seed-averaged reproduced BioPathNet": ((.027, .007, .070), (.035, .013, .070), (.041, .013, .074), (.024, .001, .042), (.121, .009, .211)),
                    "vs reproduced BioPathNet (per-seed mean)": ((.034,), (.038,), (.042,), (.033,), (.127,)),
                    "vs network feature ranker": ((.036,), (.034,), (.027,), (-.015,), (.162,)),
                    "single vs single reproduced BioPathNet": ((.020,), (.026,), (.032,), (.014,), (.077,))}.items():
        for s, v in zip(SPL, vals):
            C += [("main_results", ["per_split", c, s, k], x, 3) for k, x in zip(("est", "lo", "hi"), v)]
    for c, v in (("vs reproduced BioPathNet (per-seed mean)", .053), ("vs network feature ranker", .037), ("single vs single reproduced BioPathNet", .034)):
        C.append(("main_results", ["per_split", c, "pooled", "est"], v, 3))
    for m, v in ((PAN_SA, .019), (ADA_SA, .011), (REP_SA, .007)):
        C.append(("main_results", ["seed_averaging_gain", m, "pooled", "est"], v, 3))
    for m, v in ((PAN_SA, .130), (REP_SA, .081), (ADA_SA, .094)):
        C.append(("main_results", ["levels_seed_averaged", m, "concatenated queries"], v, 3))
    for lab, a, b in (("PPR on STRING", .018, -.035), ("PPR on STRING + KEGG training edges", .019, -.034), ("feature ranker, STRING features only", .018, -.034),
                      ("feature ranker, STRING + KEGG training-graph features", .015, -.037), ("STRING-neighbor rule", .012, -.041), ("TransE + background", .002, -.050),
                      ("reproduced BioPathNet (per-seed mean)", None, -.053), ("BioPathNet with our adaptations (per-seed mean)", .029, -.024)):
        if a is not None:
            C.append(("main_results", ["network_baselines", lab, "minus_reproduced_BioPathNet", "est"], a, 3))
        C.append(("main_results", ["network_baselines", lab, "minus_PathArrowNet", "est"], b, 3))
    for tag, v in ((PANT, .035), ("PathArrowNet - BioPathNet with our adaptations (seed-averaged)", .026), ("PathArrowNet - reproduced BioPathNet (single models)", .028)):
        C.append(("main_results", ["task_S", tag, "pooled", "est"], v, 3))
    C += [("main_results", ["task_S", "levels", PAN_SA, "split 1"], .203, 3), ("main_results", ["task_S", "levels", REP_SA, "split 1"], .128, 3),
          ("main_results", ["task_S", "levels", ADA_SA, "split 1"], .160, 3), ("main_results", ["task_S", "levels", PAN0_SA, "split 1"], .155, 3),
          ("main_results", ["task_S", "levels", PAN_SA, "mean over splits"], .154, 3)]

    # Table 2 and Suppl. S2, Fig. S4
    for row, (a, sa, b, sb) in {"Same edges, no score bins": (.090, 1, .102, 1), "Same edges, shuffled bins": (.092, 0, .102, 1),
                                "Direct human experimental evidence only": (.097, 0, .116, 0), "No STRING (KEGG edges only)": (.047, 1, .052, 1),
                                "KEGG labels, no KEGG message-passing edges": (.086, 1, .100, 1), "No graph: pair MLP on ESM2 features": (.054, 1, .061, 1),
                                "BioPathNet with our protein features": (.081, 1, .093, 1)}.items():
        C += [("ablation", ["table2", row, "single"], a, 3), ("ablation", ["table2", row, "seed_averaged"], b, 3),
              ("ablation", ["table2", row, "single_significant"], sa, 0), ("ablation", ["table2", row, "seed_averaged_significant"], sb, 0)]
    C += [("ablation", ["table2", "PathArrowNet without relation weights", "single"], .101, 3), ("ablation", ["table2", "PathArrowNet without relation weights", "seed_averaged"], .118, 3),
          ("ablation", ["pooled", "No STRING, single", "est"], -.069, 3), ("ablation", ["pooled", "No STRING, seed-averaged", "est"], -.081, 3),
          ("ablation", ["pooled", "Pair MLP, single", "est"], -.059, 3)]
    for lab, vals in {"Binary edges": ((-.011, -.020, -.005), (-.015, -.031, -.006), (-.015, -.027, -.006), (-.018, -.034, -.007)),
                      "Shuffled bins": (None, (-.016, -.024, -.008), (-.015, -.026, -.005), (-.016, -.030, -.003)),
                      "Direct human evidence": ((-.003, -.012, .004), (-.001, -.010, .006), (-.003, -.008, .005), (.004, -.004, .016))}.items():
        for (where, est), v in zip((("main_split", "single"), ("main_split", "seed_averaged"), ("pooled", "single"), ("pooled", "seed_averaged")), vals):
            if v is not None:
                C += [("ablation", ["evidence_sensitivity", where, lab, est, k], x, 3) for k, x in zip(("est", "lo", "hi"), v)]
    for lab, sv in (("KEGG edges as labels only", ((-.014, -.028, -.000), (-.017, -.034, -.000))), ("No STRING background", ((-.054, -.111, -.029), (-.066, -.136, -.035)))):
        for est, v in zip(("single", "seed_averaged"), sv):
            C += [("ablation", ["evidence_sensitivity", "main_split", lab, est, k], x, 3) for k, x in zip(("est", "lo", "hi"), v)]
    C += [("ablation", ["evidence_sensitivity", "pooled", "No STRING background", "single", k], x, 3) for k, x in zip(("est", "lo", "hi"), (-.069, -.097, -.049))]
    C += [("ablation", ["sensitivity", "reference", "val_MRR"], .135, 3), ("ablation", ["sensitivity", "reference", "test_MRR"], .103, 3)]
    for lab, (v, t, d) in {"4 layers": (.122, .078, -.026), "8 layers (batch 16)": (.113, .091, -.012), "d=16": (.132, .110, .007), "d=64 (batch 16)": (.118, .094, -.009),
                           "lr 1e-3": (.127, .103, -.000), "lr 4e-3": (.123, .104, .000), "dropout 0.3": (.121, .100, -.003),
                           "STRING experimental >= 400": (.116, .089, -.014), "batch 16 (control)": (.115, .098, -.005)}.items():
        C += [("ablation", ["sensitivity", lab, "val_MRR"], v, 3), ("ablation", ["sensitivity", lab, "test_MRR"], t, 3), ("ablation", ["sensitivity", lab, "minus_reference", "est"], d, 3)]

    # Suppl. S11 evidence strata
    lv = lambda s, m, v: ("evidence_strata", ["pooled", s, "mrr_query_level_over_five_splits", m], v, 3)
    for s, (r, a, p, rs, as_) in {"all": (.075, .085, .130, .081, .094), "direct": (.170, .294, .398, .187, .331), "direct_human": (.178, .321, .436, None, None),
                                  "direct_transfer": (.132, .165, .213, .141, .188)}.items():
        C += [lv(s, REP, r), lv(s, ADA, a), lv(s, PAN_SA, p)] + ([lv(s, REP_SA, rs), lv(s, ADA_SA, as_)] if rs else [])
    for s, n in (("all", 8737), ("direct", 1726), ("direct_human", 1429), ("direct_transfer", 297)):
        C.append(("evidence_strata", ["pooled", s, "n_queries"], n, 0))
    for s, vals in {"all": ((.045, .025, .070), (.013, .002, .026), (.034, .018, .055)), "direct": ((.156, .115, .185), (.031, .007, .052), (.123, .096, .147)),
                    "direct_human": ((.170, .131, .202), (.034, .008, .058), (.134, .108, .159)), "direct_transfer": ((.060, .009, .124), None, (.055, .009, .110))}.items():
        for tag, v in zip(("PathArrowNet - reproduced BioPathNet (seed-averaged)", "PathArrowNet - BioPathNet with our adaptations (seed-averaged)", "PathArrowNet - reproduced BioPathNet (single models)"), vals):
            if v is not None:
                C += [("evidence_strata", ["pooled", s, tag, k], x, 3) for k, x in zip(("est", "lo", "hi"), v)]

    # Suppl. S15 conditional ranking
    for var, ms in {"all": {PAN_SA: (.444, .313), ADA: (.368, .225), REP: (.365, .221), "string_rule": (.316, .188)},
                    "bin": {PAN_SA: (.503, .348), ADA: (.445, .279), REP: (.467, .303), "string_rule": (.341, .169)}}.items():
        sub = "all_queries" if var == "all" else "set_size_ge2"   # Table S15: all 1,726 queries in row 1; sets of >= 2 candidates (1,456) in row 2
        for m, (a, b) in ms.items():
            C += [("conditional_ranking", ["concatenated_levels", f"{var}|{sub}|{m}|mrr"], a, 3), ("conditional_ranking", ["concatenated_levels", f"{var}|{sub}|{m}|hits1"], b, 3)]
    C += [("conditional_ranking", ["n_queries", "all|all_queries|clean_nbf_kgml_seedavg|mrr"], 1726, 0), ("conditional_ranking", ["n_queries", "bin|set_size_ge2|clean_nbf_kgml_seedavg|mrr"], 1456, 0)]
    for var, (a, b) in {"all": (.118, .053), "bin": (.164, .047)}.items():
        C += [("conditional_ranking", ["pooled", f"{var}|set_size_ge2|{PAN_SA}-string_rule", "est"], a, 3), ("conditional_ranking", ["pooled", f"{var}|set_size_ge2|{PAN_SA}-{ADA}", "est"], b, 3)]

    # Suppl. S16 direction
    C += [("direction", ["query_level_share_all"], .719, 3), ("direction", ["query_level_share_string"], .720, 3), ("direction", ["n_edges"], 8679, 0), ("direction", ["n_string"], 1674, 0),
          ("direction", ["query_level_degree_rule_string"], .545, 3), ("direction", ["query_level_model_minus_rule_string"], .175, 3)]
    for key, v in (("pooled_share_minus_half_all", (.089, .044, .140)), ("pooled_share_minus_half_string", (.170, .116, .224)),
                   ("pooled_model_minus_degree_rule_string", (.117, .053, .195)), ("pooled_degree_rule_minus_half_string", (.054, -.007, .101))):
        C += [("direction", [key, k], x, 3) for k, x in zip(("est", "lo", "hi"), v)]
    for s, a, b in zip(SPL, ((.696, .532, .795), (.653, .444, .787), (.698, .522, .803), (.713, .610, .788), (.837, .643, .913)),
                       ((.618, .540, .707), (.686, .578, .763), (.761, .628, .851), (.756, .614, .836), (.789, .593, .863))):
        C += [("direction", [s, "all", k], x, 3) for k, x in zip(("share", "lo", "hi"), a)] + [("direction", [s, "string", k], x, 3) for k, x in zip(("share", "lo", "hi"), b)]

    # Suppl. S18 relation-level weights
    C += [("relation_weight_strata", ["n_queries", s], n, 0) for s, n in (("all", 8737), ("single_box", 1530), ("expanded_only", 7207))]
    for s, (u, w, ds, d1) in {"all": (.129, .131, .002, .002), "single_box": (.121, .151, .029, .024), "expanded_only": (.128, .115, -.013, -.009)}.items():
        C += [("relation_weight_strata", ["pooled", f"level|unweighted|seedavg|{s}", "est"], u, 3), ("relation_weight_strata", ["pooled", f"level|weighted|seedavg|{s}", "est"], w, 3),
              ("relation_weight_strata", ["pooled", f"diff|weighted-unweighted|seedavg|{s}", "est"], ds, 3), ("relation_weight_strata", ["pooled", f"diff|weighted-unweighted|single|{s}", "est"], d1, 3)]

    # Sec. 4.5 branch recall
    C += [("branch_recall", ["mean_over_splits", f"{PAN_SA}|branching", "R@10"], .223, 3), ("branch_recall", ["mean_over_splits", f"{PAN_SA}|branching", "R@100"], .558, 3),
          ("branch_recall", ["mean_over_splits", f"{REP}|branching", "R@10"], .161, 3), ("branch_recall", ["mean_over_splits", f"{REP}|branching", "R@100"], .447, 3),
          ("branch_recall", ["mean_over_splits", f"{PAN_SA}|branching", "All@100"], .285, 3), ("branch_recall", ["mean_over_splits", f"{REP}|branching", "All@100"], .204, 3)]
    for k, v in (("R@10", .050), ("R@100", .078), ("RPrec", .048)):
        C.append(("branch_recall", ["pooled", f"branching|{k}|PathArrowNet-reproduced BioPathNet", "est"], v, 3))

    # Suppl. S12 training scheme
    for k, v in (("taskA|gt5", (.021, .004, .040)), ("gt5|R@10", (.025, .004, .050)), ("gt5|RPrec", (.041, .014, .073))):
        C += [("training_scheme", ["kl_vs_ce", "pooled", k, x], y, 3) for x, y in zip(("est", "lo", "hi"), v)]
    C += [("training_scheme", ["kl_vs_ce", "mean_over_splits", "gt5|R@10", "kl"], .140, 3), ("training_scheme", ["kl_vs_ce", "mean_over_splits", "gt5|R@10", "ce"], .115, 3),
          ("training_scheme", ["kl_vs_ce", "mean_over_splits", "gt5|RPrec", "kl"], .161, 3), ("training_scheme", ["kl_vs_ce", "mean_over_splits", "gt5|RPrec", "ce"], .117, 3),
          ("training_scheme", ["decoupling_split1", "PathArrowNet (set-valued; answer sets)", "RPrec_gt5"], .165, 3)]
    for lab, (e, lo, hi, rp) in {"set-valued; batch edges only": (-.025, -.083, -.004, .144), "single answer; answer sets; masking": (-.040, -.098, -.019, .107),
                                 "single answer; batch edges; no masking": (-.020, -.069, -.002, .155)}.items():
        C += [("training_scheme", ["decoupling_split1", lab, "gt5_minus_PathArrowNet", k], x, 3) for k, x in zip(("est", "lo", "hi"), (e, lo, hi))]
        C += [("training_scheme", ["decoupling_split1", lab, "RPrec_gt5"], rp, 3), ("training_scheme", ["decoupling_split1", lab, "n_queries_gt5"], 852, 0),
              ("training_scheme", ["decoupling_split1", lab, "n_sources_gt5"], 73, 0)]
    col = "single answer, answer sets removed: all queries vs single-answer reference"
    C += [("training_scheme", ["decoupling_split1", col, "test_MRR"], .043, 3)] + [("training_scheme", ["decoupling_split1", col, "minus_reference", k], x, 3) for k, x in zip(("est", "lo", "hi"), (-.057, -.088, -.036))]

    # Suppl. S3 error by type
    for grp, name, (n, npw, r, nb, p, e, lo, hi) in (
            ("relation_subtype", "all test edges", (1669, 34, .083, .069, .117, .034, .012, .082)),
            ("relation_subtype", "activation", (1232, 31, .095, .077, .137, .042, .012, .113)),
            ("relation_subtype", "activation only (no inhibition/PTM/expression)", (1107, 29, .098, .080, .140, .042, .010, .119)),
            ("relation_subtype", "phosphorylation / dephosphorylation", (188, 24, .072, .047, .108, .036, .001, .075)),
            ("pathway_class", "Human Diseases", (384, 12, .127, .166, .207, .080, .006, .199)),
            ("pathway_class", "Organismal Systems", (351, 10, .058, .035, .079, .021, .010, .035)),
            ("pathway_class", "Cellular Processes", (198, 5, .054, .069, .101, .047, .016, .097))):
        C += [("error_by_type", [grp, name, "n_queries"], n, 0), ("error_by_type", [grp, name, "n_pathways"], npw, 0),
              ("error_by_type", [grp, name, "reproduced BioPathNet"], r, 3), ("error_by_type", [grp, name, "STRING neighbor"], nb, 3), ("error_by_type", [grp, name, "PathArrowNet"], p, 3)]
        C += [("error_by_type", [grp, name, "PathArrowNet_minus_reproduced_BioPathNet", k], x, 3) for k, x in zip(("est", "lo", "hi"), (e, lo, hi))]
    # Suppl. Sec. S11: candidates without a human gene, targets unseen in training
    C += [("candidate_checks", ["no_human_gene", "n_candidates"], 5164, 0),
          ("candidate_checks", ["no_human_gene", "n_test_targets_without_human_gene"], 0, 0),
          ("candidate_checks", ["no_human_gene", "below_0.0001"], 1, 0),
          ("candidate_checks", ["unseen_targets", "n_queries"], 1996, 0), ("candidate_checks", ["unseen_targets", "n_all"], 8737, 0),
          ("candidate_checks", ["unseen_targets", "share"], .228, 3),
          ("candidate_checks", ["unseen_targets", PANT, "est"], .041, 3),
          ("candidate_checks", ["unseen_targets", "PathArrowNet - reproduced BioPathNet (single models)", "est"], .034, 3)]
    # Suppl. Table S17 and the degree-rule paragraph of Suppl. Sec. S11 (direction_audit.py)
    DA = "direction_audit"
    for name, (n, edge, macro) in {"all": (8679, .719, .589), "single-KO boxes": (1524, .621, .553), "expansion only": (7155, .739, .611),
                                   "mechanistic subtype": (8519, .723, .596), "only indirect effect / state change": (160, .500, .695),
                                   "direct STRING edge": (1674, .720, .670)}.items():
        C += [(DA, ["A", name, "n_edges"], n, 0), (DA, ["A", name, "edge_share"], edge, 3), (DA, ["A", name, "pathway_macro", "share"], macro, 3)]
    for name, (comp, diff) in {"BioPathNet with our adaptations": (.702, .044), "network feature ranker": (.636, .055),
                               "TransE + background": (.619, .065)}.items():
        C += [(DA, ["B", name, "covered_edges"], 4618, 0), (DA, ["B", name, "PathArrowNet_edge_share"], .788, 3),
              (DA, ["B", name, "PathArrowNet_macro", "share"], .718, 3), (DA, ["B", name, "model_edge_share"], comp, 3),
              (DA, ["B", name, "PathArrowNet_minus_model", "est"], diff, 3)]
    C += [(DA, ["C", "n_queries"], 7011, 0), (DA, ["C", "PathArrowNet, seed-averaged", "level"], .038, 3)]
    for name, (lev, diff) in {"network feature ranker": (.017, .020), "TransE + background": (.020, .018), "STRING-neighbor rule": (.009, .028)}.items():
        C += [(DA, ["C", name, "level"], lev, 3), (DA, ["C", name, "PathArrowNet_minus", "est"], diff, 3)]
    C += [(DA, ["D", "tie_or_wrong", "n_edges"], 4202, 0), (DA, ["D", "tie_or_wrong", "edge_share"], .780, 3),
          (DA, ["D", "tie_or_wrong", "pathway_macro", "excess"], .126, 3), (DA, ["D", "wrong", "edge_share"], .768, 3),
          (DA, ["D", "tie", "edge_share"], .801, 3), (DA, ["D", "right", "edge_share"], .661, 3),
          (DA, ["D", "agreement_on_rule_non_tie"], .500, 3)]
    return C


def get(d, path):
    for k in path:
        d = d[k]
    return d


def main():
    cache, n_fail = {}, 0
    for f, path, want, dec in checks():
        try:
            if f not in cache:
                cache[f] = json.loads((OUT / f"{f}.json").read_text())
            got = get(cache[f], path)
            got = float(got)
            ok = abs(got - want) <= 0.5 * 10 ** -dec + 1e-9
        except (FileNotFoundError, KeyError, TypeError) as e:
            got, ok = f"missing ({type(e).__name__}: {e})", False
        n_fail += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {f}: {' / '.join(map(str, path))} = {got if isinstance(got, str) else round(got, dec + 3)} (paper {want})")
    cs = OUT / "case_study"
    try:
        c = json.loads((cs / "candidates.json").read_text()); b = json.loads((cs / "db_base_rate.json").read_text())
        for lab, got, want in (("filtered top-1 non-KEGG predictions", c["n_after_filters"], 241), ("literature set", len(c["literature_set"]), 15),
                               ("database support, predictions (%)", round(100 * b["pred_any"] / b["n_pred"], 1), 31.1),
                               ("database support, random control (%)", round(100 * b["rand_any"] / b["n_rand"], 1), 0.5),
                               ("database support without STRING support (%)", round(100 * b["pred_any_string_lt150"] / b["n_pred_string_lt150"], 1), 19.2)):
            ok = got == want; n_fail += not ok
            print(f"{'ok  ' if ok else 'FAIL'} case_study: {lab} = {got} (paper {want})")
    except FileNotFoundError as e:
        n_fail += 1; print("FAIL case_study:", e)
    print(f"{n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

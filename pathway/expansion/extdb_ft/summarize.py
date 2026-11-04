#!/usr/bin/env python3
"""Tables and pre-registered tests of the SIGNOR / WikiPathways fine-tuning experiments (CPU). Missing runs are reported as missing, never filled.

Per test edge, scores are averaged over seeds (link: reciprocal rank, Hits@10 and AUROC-vs-non-edges per seed, then
mean; sign: mean predicted probability). Paired bootstrap over source-KO clusters (2,000, seed 0); two-sided p;
Holm over P1-P4 (SIGNOR).
    .venv/bin/python expansion/extdb_ft/summarize.py
"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

R = C.HERE / "runs"
SEEDS = (42, 43, 44)
SIGN_BASE = ("src_rate", "tgt_rate", "rates_lr", "esm_lr", "kegg_prior")


def fast_auroc(y, s):
    o = np.argsort(s, kind="mergesort"); ss = s[o]
    r = np.empty(len(s)); i = 0
    _, idx, cnt = np.unique(ss, return_index=True, return_counts=True)
    avg = idx + (cnt + 1) / 2.0
    r[o] = np.repeat(avg, cnt)
    npos = y.sum(); nneg = len(y) - npos
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")


def load_link(db, method, sp):
    """-> dict(rr, h10, auc) per edge averaged over seeds, or None"""
    files = []
    for s in SEEDS:
        f = {"ft": R / f"{db}_ckpt_s{s}/pred_link_{sp}.npz", "scratch": R / f"{db}_scratch_s{s}/pred_link_{sp}.npz",
             "zs_keggonly": R / f"{db}_zs_s{s}/pred_link_keggonly_{sp}.npz", "zs_union": R / f"{db}_zs_s{s}/pred_link_union_{sp}.npz",
             "untrained": R / f"{db}_untrained_s{s}/pred_link_union_{sp}.npz"}.get(method)
        if f is None:
            files = [R / f"baselines_{db}_{method}_{sp}.npz"]; break
        files.append(f)
    if not all(f.exists() for f in files):
        return None
    Z = [np.load(f) for f in files]
    return dict(rr=np.mean([1 / z["rank"] for z in Z], 0), h10=np.mean([z["rank"] <= 10 for z in Z], 0),
                auc=np.mean([z["auc"] for z in Z], 0), n_seeds=len(Z))


def load_sign(db, method, sp):
    files = []
    for s in SEEDS:
        f = {"ft": R / f"{db}_ckpt_s{s}/pred_sign_{sp}.npz", "scratch": R / f"{db}_scratch_s{s}/pred_sign_{sp}.npz",
             "frozen": R / f"{db}_zs_s{s}/pred_sign_{sp}.npz", "untrained": R / f"{db}_untrained_s{s}/pred_sign_{sp}.npz"}.get(method)
        if f is None:
            files = [R / f"baselines_{db}_sign_{method}_{sp}.npz"]; break
        files.append(f)
    if not all(f.exists() for f in files):
        return None
    L = [np.load(f)["sign_logit"] for f in files]
    if method in SIGN_BASE:
        return dict(p=L[0], n_seeds=1)
    return dict(p=np.mean([1 / (1 + np.exp(-l)) for l in L], 0), n_seeds=len(L))


def boot(D, fn, sp="test", B=2000):
    src = D[sp]["e"][:, 0]; u = np.unique(src); idx = {c: np.where(src == c)[0] for c in u}
    rng = np.random.default_rng(0)
    out = []
    for _ in range(B):
        b = np.concatenate([idx[c] for c in rng.choice(u, len(u))])
        out.append(fn(b))
    out = np.array(out)
    return out


def ci_p(est, bs):
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    p = 2 * min(np.mean(bs <= 0), np.mean(bs >= 0))
    return {"diff": float(est), "ci": [float(lo), float(hi)], "p": float(min(1.0, max(p, 1 / len(bs))))}


def f1_at(y, p, thr):
    pr = p >= thr; tp = (pr & (y == 1)).sum(); fp = (pr & (y == 0)).sum(); fn = (~pr & (y == 1)).sum()
    return float(2 * tp / max(1, 2 * tp + fp + fn)), float((pr == (y == 1)).mean())


def main():
    out = {}
    md = []
    for db in ("signor", "wikipathways"):
        D = C.load_ext(db); res = {"link": {}, "sign": {}, "tests": {}}
        strata = {"all": np.ones(len(D["test"]["e"]), bool), "clean": D["test"]["clean"] == 1,
                  "protein_only": D["test"]["protein_only"] == 1, "inductive_src": D["test"]["src_in_train"] == 0}
        LK = {m: load_link(db, m, "test") for m in ("ft", "scratch", "zs_keggonly", "zs_union", "untrained", "string", "popularity", "adamic_adar")}
        for m, z in LK.items():
            if z is None:
                res["link"][m] = "missing"; continue
            res["link"][m] = {k: {"n": int(s.sum()), "mrr": float(z["rr"][s].mean()), "hits10": float(z["h10"][s].mean()),
                                  "auc": float(z["auc"][s].mean())} for k, s in strata.items()}
            res["link"][m]["n_seeds"] = z["n_seeds"]
        ys = {sp: D[sp]["sign"] for sp in ("val", "test")}
        SG = {}
        for m in ("ft", "scratch", "frozen", "untrained") + SIGN_BASE:
            zt, zv = load_sign(db, m, "test"), load_sign(db, m, "val")
            if zt is None or zv is None:
                res["sign"][m] = "missing"; continue
            SG[m] = zt
            kv = ys["val"] >= 0; kt = ys["test"] >= 0
            yv, pv = ys["val"][kv], zv["p"][kv]; yt, pt = ys["test"][kt], zt["p"][kt]
            thr = max(np.unique(pv), key=lambda t: f1_at(yv, pv, t)[0])
            f1, acc = f1_at(yt, pt, thr)
            res["sign"][m] = {"val_auroc": fast_auroc(yv, pv), "test_auroc": fast_auroc(yt, pt), "test_auprc": C.auprc(yt, pt),
                              "test_f1": f1, "test_acc": acc, "thr_from_val": float(thr), "n_seeds": zt["n_seeds"],
                              "test_n_act": int((yt == 0).sum()), "test_n_inh": int((yt == 1).sum())}
        # KEGG retention
        kr = {}
        for init in ("ckpt", "scratch"):
            for s in SEEDS:
                f = R / f"{db}_{init}_s{s}/result.json"
                if f.exists():
                    j = json.loads(f.read_text())
                    kr[f"{init}_s{s}"] = {"start": j["start"]["kegg_test_A_micro_mrr"],
                                          "after_link_ckpt": j["final"]["link"]["kegg_test_A_micro_mrr"],
                                          "after_sign_ckpt": j["final"]["sign"]["kegg_test_A_micro_mrr"],
                                          "best_step": j["best_step"]}
        res["kegg_retention"] = kr
        # tests
        def link_test(a, b, key="rr"):
            if LK.get(a) is None or LK.get(b) is None:
                return "missing"
            d = LK[a][key] - LK[b][key]
            return ci_p(d.mean(), boot(D, lambda ix: d[ix].mean()))
        kt = ys["test"] >= 0

        def sign_test(a, b):
            if a not in SG or b not in SG:
                return "missing"
            y = ys["test"]; pa, pb = SG[a]["p"], SG[b]["p"]
            est = fast_auroc(y[kt], pa[kt]) - fast_auroc(y[kt], pb[kt])

            def f(ix):
                ix = ix[kt[ix]]
                return fast_auroc(y[ix], pa[ix]) - fast_auroc(y[ix], pb[ix])
            return ci_p(est, boot(D, f))
        base_ok = {m: res["sign"][m]["val_auroc"] for m in SIGN_BASE if isinstance(res["sign"].get(m), dict)}
        best_base = max(base_ok, key=base_ok.get) if base_ok else None
        res["best_simple_sign_baseline_on_val"] = best_base
        T = {"P1_link_ft_minus_zs_union": link_test("ft", "zs_union"),
             "P2_link_ft_minus_scratch": link_test("ft", "scratch"),
             "P3_sign_ft_minus_frozen": sign_test("ft", "frozen"),
             "P4_sign_ft_minus_best_simple": sign_test("ft", best_base) if best_base else "missing",
             "link_ft_minus_zs_keggonly": link_test("ft", "zs_keggonly"),
             "link_ft_minus_string": link_test("ft", "string"),
             "link_auc_ft_minus_zs_union": link_test("ft", "zs_union", "auc"),
             "link_zs_union_minus_zs_keggonly": link_test("zs_union", "zs_keggonly"),
             "sign_ft_minus_scratch": sign_test("ft", "scratch"),
             "sign_frozen_minus_untrained": sign_test("frozen", "untrained")}
        prim = [k for k in T if k.startswith("P") and isinstance(T[k], dict)]
        ps = sorted(prim, key=lambda k: T[k]["p"])
        run = 0.0
        for i, k in enumerate(ps):
            run = max(run, min(1.0, T[k]["p"] * (len(ps) - i))); T[k]["p_holm"] = run
        res["tests"] = T
        out[db] = res
    (C.HERE / "results/summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

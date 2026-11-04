"""Gene-pair scores from the KEGG-pretrained two-tower next-KO model (zero-shot transfer feature).

For an encoder checkpoint (repr/runs/<setting>/<variant>_s<seed>/best.pt) the KO-level score matrices are
    S_fw[u, v] = <Q0 h(u), K0 h(v)> + b0(h(v))      (v is a next KO of u)
    S_bw[u, v] = <Q1 h(v), K1 h(u)> + b1(h(u))      (u is a previous KO of v; head direction, transposed)
on the full split TRAIN graph (no val/test edge enters the graph). S = S_fw + S_bw is z-scored per anchor row (removes
anchor-level offsets), then gene-level score(i, j) = max over KO pairs and both orientations (symmetric).
Nothing about the downstream task is used. Genes without a KO in the vocabulary get score 0 and has = False.
"""
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(_KH + "/pathway")
EXP = ROOT / "expansion"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "benchmark_v2"))
CACHE = EXP / "downstream/cache"


def ko_matrix(variant="kegg_enc", seed=0, setting="strict_v1_full", dev="cuda"):
    from patharrownet.models import kegg_encoder as PK
    import scorelib as SL
    voc, V = SL.vocab(setting)
    kf = torch.load(ROOT / "kegg_features/ko_features_full.pt")
    feats = [(kf["features"][n]["X"], kf["features"][n]["has"]) for n in ("esm2_mean", "gobert_noBP_mean")]
    G = PK.build_graph(PK.load_train_edges(setting), [], dev)
    enc, tt = PK.Encoder(feats, 2).to(dev), PK.TwoTower(128).to(dev)
    ck = torch.load(EXP / "repr/runs" / setting / f"{variant}_s{seed}" / "best.pt", map_location=dev)
    enc.load_state_dict(ck["enc"]); tt.load_state_dict(ck["tt"]); enc.eval(); tt.eval()
    with torch.no_grad():
        H = enc(*G, V)
        S = tt.q[0](H) @ tt.k[0](H).T + tt.b[0](H)[:, 0][None]
        S += (tt.q[1](H) @ tt.k[1](H).T + tt.b[1](H)[:, 0][None]).T
        S = (S - S.mean(1, keepdim=True)) / S.std(1, keepdim=True)
    return S, voc


NBF_DIR = ROOT / "kegg_dataset/nbf_ko_full"  # strict split 1 (strict_v1_full), train edges only, KL loss
NBF_SEEDS = (42, 43, 44, 45, 46)


def nbf_matrix(anchors, seeds=NBF_SEEDS, run="nbf+esm+exp150bin4+allneg+kl", dev="cuda", bs=32, randinit=False):
    """PathArrowNet (best-model component; STRING-exp150 background graph) zero-shot scores: dense [V, V] with rows filled
    for `anchors`: S[u, v] = log p_tail(v | u) + log p_head(u | v)  (v next of u, scored from both ends), averaged
    over checkpoint seeds; rows of non-anchors are 0. Requires every anchor's partners to be anchors too (they are
    when anchors = all KOs of the downstream gene list)."""
    import json as _j
    sys.path.insert(0, str(ROOT))
    from patharrownet.models import patharrownet as TN
    from patharrownet import data as KD
    with (ROOT / "kegg_dataset/token_vocab_ko_full.csv").open() as f:
        import csv as _c
        vocab = {r["token"]: int(r["token_id"]) for r in _c.DictReader(f)}
    V = len(vocab); A = torch.tensor(sorted(set(anchors)), device=dev)
    tot = torch.zeros(V, V, device=dev)
    for sd in seeds:
        d = NBF_DIR / f"{run}_s{sd}"; cfg = _j.loads((d / "config.json").read_text())
        assert cfg["full_candidates"] and cfg["loss"] == "kl" and not cfg["train_on_val"]
        from kegg_pipeline import evaluation as E
        sdir = TN.resolve_split_dir(cfg)
        vq = {"A": E.load_queries(sdir / "val_A_next_node.csv.gz", "A")}; tq = {"A": E.load_queries(sdir / "test_A_next_node.csv.gz", "A")}
        edges = KD.load_train_edges(vocab, vq, tq, sdir, False)
        kf = torch.load(ROOT / "kegg_features/ko_features_full.pt")
        dense = [(kf["features"][n]["X"], kf["features"][n]["has"]) for n in cfg["feats"].split(",") if n]
        pairs, bins = [], []
        with open(ROOT / cfg["background"]) as f:
            for r in _c.DictReader(f, delimiter="\t"):
                x, y = vocab[r["ko_a"]], vocab[r["ko_b"]]; sc = int(r["max_experimental_score"])
                pairs += [(x, y), (y, x)]; bins += [2 + (3 if sc >= 900 else 2 if sc >= 700 else 1 if sc >= 400 else 0)] * 2
        assert cfg["bg_bins4"]
        src, dst, rel, _ = TN.build_graph(edges, torch.tensor(pairs), dev, torch.tensor(bins))
        model = TN.KONBF(V, cfg["d"], cfg["layers"], cfg["dropout"], dense, 6, bilinear=not cfg["no_bilinear"], agg=cfg["agg"]).to(dev)
        if randinit:  # control: same architecture / features / STRING graph, untrained weights (seeded)
            torch.manual_seed(sd); model = TN.KONBF(V, cfg["d"], cfg["layers"], cfg["dropout"], dense, 6, bilinear=not cfg["no_bilinear"], agg=cfg["agg"]).to(dev)
        else:
            model.load_state_dict(torch.load(d / "best.pt", map_location=dev))
        model.eval()
        Et = torch.tensor(edges, device=dev); graph = (src, dst, rel, (Et[:, 0], Et[:, 1]))
        al = A.tolist()
        _, St = TN.anchor_scores(model, al, TN.TAIL, graph, bs)   # [n, V]: log p(v | u -> ?)
        _, Sh = TN.anchor_scores(model, al, TN.HEAD, graph, bs)   # [n, V]: log p(u | ? -> v), anchor v
        M = torch.zeros(V, V, device=dev)
        M[A] = St.clamp(min=-50)
        M[:, A] += Sh.clamp(min=-50).T
        tot += M; del St, Sh, model
        torch.cuda.empty_cache()
    return tot / len(seeds)


def gene_pair_scores(kos_col, variant="kegg_enc", seed=0, setting="strict_v1_full", tag=None, dev="cuda"):
    """kos_col: ';'-separated KO ids per gene. Returns (P [n, n] float32 symmetric, has [n] bool). Cached if tag."""
    f = CACHE / f"{tag}_{setting}_{variant}_s{seed}.npz" if tag else None
    if f is not None and f.exists():
        z = np.load(f); return z["P"], z["has"]
    if variant.startswith("konbf"):
        import csv as _c
        with (ROOT / "kegg_dataset/token_vocab_ko_full.csv").open() as fh:
            voc = {r["token"]: int(r["token_id"]) for r in _c.DictReader(fh)}
        anchors = {voc[k] for c in kos_col for k in (c or "").split(";") if k in voc}
        S = nbf_matrix(anchors, dev=dev, randinit=variant.endswith("randinit"))
    else:
        S, voc = ko_matrix(variant, seed, setting, dev)
    ks = [[voc[k] for k in (c or "").split(";") if k in voc] for c in kos_col]
    has = np.array([len(k) > 0 for k in ks]); n = len(ks); m = max(1, max(len(k) for k in ks))
    idx = torch.tensor([(k + k[:1] * m)[:m] if k else [0] * m for k in ks], device=dev)  # pad by repeating first KO
    P = torch.empty(n, n, device=dev)
    flat = idx.reshape(-1)
    with torch.no_grad():
        for a in range(0, n, 256):
            rows = S[idx[a:a + 256].reshape(-1)][:, flat]  # [b*m, n*m]
            P[a:a + 256] = rows.reshape(-1, m, n, m).amax((1, 3))
    P = torch.maximum(P, P.T); P[~torch.tensor(has, device=dev)] = 0; P[:, ~torch.tensor(has, device=dev)] = 0
    P = P.cpu().numpy().astype(np.float32)
    if f is not None:
        CACHE.mkdir(parents=True, exist_ok=True); np.savez(f, P=P, has=has)
    return P, has

#!/usr/bin/env python3
"""
BioPathNet (+features) inference on the KEGG KO benchmark: load a released checkpoint (model_epoch_N.pth) and write
tail scores over all KO candidates in the benchmark_v2 format. No training; run with biopathnet_repro/.venv/bin/python.

    cd $KEGG_HOME/biopathnet_repro
    .venv/bin/python leakage/bpn_infer.py --config configs/kegg_strict_full_exp150bin4_keep1hop.yaml \
        --setting strict_v1_full --feats esm2_mean,gobert_noBP_mean --ckpt <run dir>/model_epoch_10.pth \
        --split test [--max-anchors 50] --out <out dir>
    # original BioPathNet (no features): --feats ''      KGE baselines: --feats '' with a configs/kge_*.yaml

The dataset is built from the config's data directory (BioPathNet format; leakage/full_candidates.py registers the
full-candidate dataset class), the task from the config's `task` section and `task.preprocess` (what TorchDrug's
engine does at start-up), without any training component. Node features (NBFNetFeat) are aligned to the dataset's
entity order from pathway/kegg_features/<feature file>, exactly as for the reported runs. Message passing uses the
task's fact graph (train edges + background; validation / test edges are never in it).
Output: <out>/tail_<split>.npz (anchors, scores [n, V] float32; -inf for KOs that are not dataset entities), plus
<out>/summary.json with the task-A micro MRR on the scored queries (filtered protocol of benchmark_v2/scorelib.py).
"""
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

BPNR = Path(_KH + "/biopathnet_repro")
BPN = BPNR / "BioPathNet"
PW = Path(_KH + "/pathway")
METHOD = "bpn_keep1hop+exp150bin4+esm+gobert_noBP"
sys.path.insert(0, str(BPN))
sys.path.insert(0, str(BPNR / "leakage"))
sys.path.insert(0, str(PW / "benchmark_v2"))
sys.path.insert(0, str(PW))


def setup():
    """chdir into the BioPathNet checkout and register NBFNetFeat, the tasks and the full-candidate dataset class."""
    os.chdir(BPN)
    from biopathnet import util, model_feat, task  # noqa: F401  (registers NBFNetFeat, the task)
    import full_candidates
    full_candidates.register()
    return util


def data_path(cfg_path, util):
    """config dataset path; relative paths are relative to biopathnet_repro/."""
    p = Path(util.load_config(str(cfg_path), context={"gpus": None}).dataset["path"])
    return p if p.is_absolute() else (BPNR / p).resolve()


def load(cfg_path, data_dir, feat_file, util):
    """dataset + task from the config, task.preprocess(train, valid, test); NBFNetFeat if feat_file is given."""
    from torchdrug import core
    cfg = util.load_config(str(cfg_path), context={"gpus": None})
    cfg.dataset["path"] = str(data_dir)
    dataset = core.Configurable.load_config_dict(cfg.dataset)
    train, valid, test = dataset.split()
    if feat_file is not None:
        cfg.task.model["class"] = "NBFNetFeat"
        cfg.task.model["feature_file"] = str(feat_file)
    if cfg.task.model["class"] in ("TransE", "DistMult", "ComplEx", "RotatE"):
        cfg.task.model.num_entity = dataset.num_entity
    cfg.task.model.num_relation = dataset.num_relation
    task = core.Configurable.load_config_dict(cfg.task)
    task.preprocess(train, valid, test)
    if torch.cuda.is_available():
        task = task.cuda()
    return cfg, dataset, task, (train, valid, test)


def node_features(ent, feats, setting, feature_file=None):
    """{'X': [n_entities, D], 'has': [n_entities]} aligned to the dataset entity order (as for the reported runs)."""
    import scorelib as SL
    voc, V = SL.vocab(setting)
    ff = feature_file or ("ko_features_full.pt" if "_full" in setting else "ko_features.pt")
    kf = torch.load(PW / "kegg_features" / ff)
    assert kf["vocab_size"] == V
    Xs, H = [], torch.zeros(len(ent))
    for f in feats:
        X, has = kf["features"][f]["X"], kf["features"][f]["has"]
        idx = torch.tensor([voc.get(t, 0) for t in ent])
        Xs.append(X[idx] * has[idx, None].float())
        H = torch.maximum(H, has[idx].float())
    return {"X": torch.cat(Xs, 1), "has": H}


def load_checkpoint(task, ckpt):
    state = torch.load(ckpt, map_location=next(task.parameters()).device)
    for k in ("fact_graph", "fact_graph_supervision", "graph", "train_graph", "valid_graph", "test_graph",
              "full_valid_graph", "full_test_graph"):  # graph buffers: rebuilt by the task from the data files
        state["model"].pop(k, None)
    task.load_state_dict(state["model"], strict=False)
    task.eval()


@torch.no_grad()
def tail_scores(task, dataset, setting, anchors, bs=8):
    """[n_anchors, V] scores of every KO candidate as the next KO of each anchor (relation 'precedes')."""
    import scorelib as SL
    voc, V = SL.vocab(setting)
    ent = list(dataset.entity_vocab)
    device = next(task.parameters()).device
    rel = dataset.inv_relation_vocab["precedes"]
    ours2bpn = {voc[t]: i for i, t in enumerate(ent) if t in voc}
    bpn_ids = torch.tensor([ours2bpn.get(o, -1) for o in range(V)], device=device)
    has = bpn_ids >= 0
    N = len(ent)
    all_idx = torch.arange(N, device=device)
    rows = []
    for i in range(0, len(anchors), bs):
        ch = anchors[i:i + bs]
        idx = torch.tensor([ours2bpn.get(x, 0) for x in ch], device=device)
        r_index = torch.full((len(ch), N), rel, device=device, dtype=torch.long)
        pred = task.model(task.fact_graph, idx.unsqueeze(-1).expand(-1, N), all_idx.unsqueeze(0).expand(len(ch), -1), r_index)
        full = torch.full((len(ch), V), float("-inf"), device=device)
        full[:, has] = pred[:, bpn_ids[has]].float()
        full[torch.tensor([x not in ours2bpn for x in ch], device=device)] = 0.0
        rows.append(full.cpu())
    return torch.cat(rows).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--setting", required=True, help="benchmark setting, e.g. strict_v1_full")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--feats", default="esm2_mean,gobert_noBP_mean", help="'' = original BioPathNet / KGE")
    ap.add_argument("--feature-file", default=None)
    ap.add_argument("--data-dir", default=None, help="override the config's dataset path")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--max-anchors", type=int, default=None)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))
    cfg_path, ckpt, out = Path(a.config).resolve(), Path(a.ckpt).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    util = setup()
    import scorelib as SL
    ddir = Path(a.data_dir).resolve() if a.data_dir else data_path(cfg_path, util)
    feats = [f for f in a.feats.split(",") if f]
    with tempfile.TemporaryDirectory(prefix="bpn_infer_") as td:
        ff = None
        if feats:
            from torchdrug import core
            c0 = util.load_config(str(cfg_path), context={"gpus": None}); c0.dataset["path"] = str(ddir)
            ent = list(core.Configurable.load_config_dict(c0.dataset).entity_vocab)
            ff = Path(td) / "node_features.pt"
            torch.save(node_features(ent, feats, a.setting, a.feature_file), ff)
        cfg, dataset, task, _ = load(cfg_path, ddir, ff, util)
        load_checkpoint(task, ckpt)
        qs = SL.load_A(a.setting, a.split)
        anchors = sorted({q["source"] for q in qs})[:a.max_anchors] if a.max_anchors else SL.anchors_needed(a.setting, (a.split,))
        sc = tail_scores(task, dataset, a.setting, anchors, a.bs)
    SL.save_tail(out / f"tail_{a.split}.npz", anchors, sc)
    keep = set(anchors)
    sub = [q for q in qs if q["source"] in keep]
    _, V = SL.vocab(a.setting)
    rk, _ = SL.ranks(SL.query_scores(out, a.split, "A", sub), sub, V)
    res = {"config": cfg_path.name, "ckpt": ckpt.name, "setting": a.setting, "feats": feats, "n_anchors": len(anchors),
           "n_queries_scored": len(sub), "micro_MRR_on_scored_queries": float(np.mean(1 / rk))}
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()

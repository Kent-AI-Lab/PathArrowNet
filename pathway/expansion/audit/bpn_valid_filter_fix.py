#!/usr/bin/env python3
"""
BioPathNet checkpoint selection uses TEST labels in its filter (minor; selection-only).

task.KnowledgeGraphCompletionBiomed.preprocess registers  self.graph = dataset.graph  (train1 + train2 + valid + TEST);
target() builds the filtered-ranking mask from self.graph, and script/run.py train_and_validate() picks the epoch with the
best filtered VALID MRR. So, for a valid query (h, ?), other true tails that are TEST edges are removed from the
competitor set (and likewise for heads). Message passing is unaffected (fact_graph = train1 + train2 only).

This script (CPU, BioPathNet venv) re-scores every saved checkpoint of a run on the full valid set and recomputes the
official valid MRR twice: with the official filter and with a corrected filter (train1 + train2 + valid, no test), then
reports whether the selected epoch would change. It also defines the corrected task class
(tasks.KnowledgeGraphCompletionBiomedNoTestFilter) that a re-run would use.

    CUDA_VISIBLE_DEVICES="" biopathnet_repro/.venv/bin/python bpn_valid_filter_fix.py <setting> [seed]
    -> audit/results_extra/bpn_valid_filter_<setting>_s<seed>.json
"""
import json
import os
import re
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

AUD = Path(__file__).resolve().parent
sys.path.insert(0, str(AUD.parents[2] / "biopathnet_repro" / "leakage"))
import bpn_infer as MB  # noqa: E402  (dataset / task loading without any training component)

OUT = AUD / "results_extra"
OUT.mkdir(exist_ok=True)


def register_corrected():
    """Corrected copy of the task: valid-time filtering without test edges (test-time filtering unchanged)."""
    from torchdrug.core import Registry as R
    from biopathnet.task import KnowledgeGraphCompletionBiomed
    if "tasks.KnowledgeGraphCompletionBiomedNoTestFilter" in R.table:
        return

    @R.register("tasks.KnowledgeGraphCompletionBiomedNoTestFilter")
    class KnowledgeGraphCompletionBiomedNoTestFilter(KnowledgeGraphCompletionBiomed):
        def preprocess(self, train_set, valid_set, test_set):
            out = super().preprocess(train_set, valid_set, test_set)
            dataset = train_set.dataset
            m = torch.ones(len(dataset), dtype=torch.bool)
            m[test_set.indices] = 0
            self.register_buffer("graph_no_test", dataset.graph.edge_mask(m))
            return out

        def target(self, batch):
            if getattr(self, "split", None) == "valid":       # set by script/run.py before solver.evaluate("valid")
                full = self.graph
                self._buffers["graph"] = self.graph_no_test
                try:
                    return super().target(batch)
                finally:
                    self._buffers["graph"] = full
            return super().target(batch)


def main(st, seed=1024):
    torch.set_num_threads(int(os.environ.get("NTHREADS", "24")))
    util = MB.setup()
    register_corrected()
    src = json.loads((MB.PW / f"benchmark_v2/scores/{st}/{MB.METHOD}/s{seed}/source.json").read_text())
    wd, cfgp = Path(src["working_dir"]), Path(src["config"])
    log = (wd / "log.txt").read_text()
    selected = re.findall(r"Load checkpoint from (model_epoch_\d+\.pth)", log)[-1]
    data_dir = MB.data_path(cfgp, util)
    cfg, ds, task, (train, valid, test) = MB.load(cfgp, data_dir, wd / "node_features.pt", util)
    m = torch.ones(len(ds), dtype=torch.bool)
    m[test.indices] = 0
    g_no_test = ds.graph.edge_mask(m)
    full = task.graph
    from torchdrug import data
    loader = data.DataLoader(valid, 32, shuffle=False)
    logged = {f"model_epoch_{e}.pth": float(v) for e, v in
              re.findall(r"Save checkpoint to model_epoch_(\d+)\.pth\n.*?Evaluate on valid\n.*?mrr: ([\d.]+)", log, re.S)}
    res = {}

    def score(name):
        st_ = torch.load(wd / name, map_location="cpu")
        for k in ("graph", "fact_graph", "fact_graph_supervision"):
            st_["model"].pop(k, None)
        task.load_state_dict(st_["model"], strict=False)
        task.eval()
        rk = {"official": [], "no_test": []}
        with torch.no_grad():
            for batch in loader:
                pred = task.predict(batch)
                for nm, g in (("official", full), ("no_test", g_no_test)):
                    task._buffers["graph"] = g
                    mask, target = task.target(batch)
                    pos = pred.gather(-1, target.unsqueeze(-1))
                    rk[nm].append((torch.sum((pos <= pred) & mask, dim=-1) + 1).flatten())
                task._buffers["graph"] = full
        res[name] = {k: float((1 / torch.cat(v).float()).mean()) for k, v in rk.items()}
        res[name]["official_in_log_gpu"] = logged.get(name)
        print(st, seed, name, res[name], flush=True)
    # exact pruning: removing test answers from the filter can only LOWER a checkpoint's valid MRR (delta_e >= 0), so
    # another epoch e can overtake the selected one only if logged(sel) - logged(e) < delta_sel. Score the selected
    # checkpoint first, then every epoch inside that margin (all epochs if PRUNE=0).
    score(selected)
    d_sel = res[selected]["official"] - res[selected]["no_test"]
    margin = d_sel + 1e-4                                   # + tolerance for GPU/CPU float differences
    for name in sorted(logged, key=lambda k: int(k.split("_")[-1].split(".")[0])):
        if name == selected:
            continue
        if os.environ.get("PRUNE", "1") == "0" or logged[selected] - logged[name] < margin:
            score(name)
    sel_off = max(res, key=lambda k: res[k]["official"])
    sel_fix = max(res, key=lambda k: res[k]["no_test"])
    res_logged_only = {k: v for k, v in logged.items() if k not in res}
    out = {"setting": st, "seed": seed, "working_dir": str(wd), "selected_in_log": selected,
           "selected_official_recomputed": sel_off, "selected_without_test_filter": sel_fix,
           "selection_changes": sel_fix != selected, "delta_selected_official_minus_no_test": d_sel,
           "pruning_margin": margin, "epochs_not_rescored_(outside_margin)_logged_mrr": res_logged_only,
           "logged_mrr_selected": logged.get(selected), "valid_mrr_per_checkpoint": res}
    (OUT / f"bpn_valid_filter_{st}_s{seed}.json").write_text(json.dumps(out, indent=1))
    print("RESULT", json.dumps({k: v for k, v in out.items() if k != "valid_mrr_per_checkpoint"}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1024)

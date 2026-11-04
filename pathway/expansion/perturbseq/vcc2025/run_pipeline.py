#!/usr/bin/env python3
"""Wrapper that runs the existing perturbseq code on VCC 2025 without changing any default behaviour.
The only run-time addition is the cell-line metadata of the new dataset name in
prep_stats.META_EXTRA (the same mechanism used for Nadig 2025).

    .venv/bin/python expansion/perturbseq/vcc2025/run_pipeline.py stats     -> perturbseq/data/stats_vcc2025_h1.npz
    .venv/bin/python expansion/perturbseq/vcc2025/run_pipeline.py relations -> perturbseq/vcc2025/tables/* , data/pairs_vcc2025_h1.npz
    .venv/bin/python expansion/perturbseq/vcc2025/run_pipeline.py replication gem|pooled
                                                                            -> perturbseq/results/replication_<scheme>_vcc2025.json
"""
import os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PS = HERE.parent
sys.path.insert(0, str(PS))
NAME = "vcc2025_h1"

import prep_stats  # noqa: E402
prep_stats.META_EXTRA[NAME] = ("H1 hESC", "CRISPRi")

if __name__ == "__main__":
    step = sys.argv[1]
    if step == "stats":
        prep_stats.main(NAME)
    elif step == "relations":
        import build_relations
        sys.argv = ["build_relations.py", "--datasets", NAME, "--out-dir", str(HERE / "tables")]
        build_relations.main()
    elif step == "replication":
        import eval_replication
        os.environ["REPL_TAG"] = "_vcc2025"
        sys.argv = ["eval_replication.py", sys.argv[2], NAME]
        eval_replication.main()
    else:
        raise SystemExit("unknown step " + step)

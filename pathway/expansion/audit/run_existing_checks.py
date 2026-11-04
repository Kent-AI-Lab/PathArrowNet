#!/usr/bin/env python3
"""
Re-run every pre-existing leakage check WITHOUT modifying any existing file (all outputs go to audit/existing/).

    CUDA_VISIBLE_DEVICES="" .venv/bin/python expansion/audit/run_existing_checks.py [step ...]

steps (default: all):
  rebuild      build_ko_leakage_free_split.py --seed {42..46} --out-dir audit/existing/rebuild/seedX
               -> fresh 22 enforced split checks + content comparison with the split files the models used
  independent  audit_independent_checks.py (prints only)                        -> existing/audit_independent_checks.log
  checker      benchmark_v2/leakage_checker.py with OUT redirected               -> existing/leakage_checker/
Summary: existing/existing_checks_summary.json
"""
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())
import csv
import gzip
import io
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(_KH + "/pathway")
AUD = Path(__file__).resolve().parent
OUT = AUD / "existing"
OUT.mkdir(parents=True, exist_ok=True)
PY = str(ROOT / ".venv/bin/python")
SPLITS = {42: ROOT / "kegg_dataset/splits_ko", **{s: ROOT / f"kegg_splits_robustness/seed{s}" for s in (43, 44, 45, 46)}}
SUMMARY = OUT / "existing_checks_summary.json"


def load_summary():
    return json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}


def save_summary(d):
    SUMMARY.write_text(json.dumps(d, indent=1))


def content(p):
    if p.suffix == ".gz":
        with gzip.open(p, "rb") as f:
            return f.read()
    return p.read_bytes()


def step_rebuild():
    res = {}
    procs = {}
    for seed, orig in SPLITS.items():
        od = OUT / "rebuild" / f"seed{seed}"
        od.mkdir(parents=True, exist_ok=True)
        if (od / "leakage_checks.csv").exists() and (od / "README.md").exists() and os.environ.get("REUSE_REBUILD"):
            procs[seed] = (subprocess.Popen(["true"]), od, orig)
            continue
        log = open(OUT / f"rebuild_seed{seed}.log", "w")
        procs[seed] = (subprocess.Popen([PY, str(ROOT / "build_ko_leakage_free_split.py"), "--seed", str(seed),
                                         "--out-dir", str(od)], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}), od, orig)
    for seed, (p, od, orig) in procs.items():
        rc = p.wait()
        r = {"returncode": rc}
        lc = {row["check"]: float(row["value"]) for row in csv.DictReader((od / "leakage_checks.csv").open())} \
            if (od / "leakage_checks.csv").exists() else {}
        enforced = {k: v for k, v in lc.items() if k.startswith(("overlap_", "edges_shared_"))}
        r["n_enforced_checks"] = len(enforced)
        r["enforced_nonzero"] = {k: v for k, v in enforced.items() if v != 0}
        r["ALL_CROSS_SPLIT_OVERLAPS_ZERO"] = lc.get("ALL_CROSS_SPLIT_OVERLAPS_ZERO")
        # the stored checks of the split the models used
        stored = {row["check"]: float(row["value"]) for row in csv.DictReader((orig / "leakage_checks.csv").open())}
        r["stored_enforced_nonzero"] = {k: v for k, v in stored.items() if k.startswith(("overlap_", "edges_shared_")) and v != 0}
        r["stored_equals_rebuilt_checks"] = stored == lc
        # rebuilt split files identical (decompressed content) to the files used for training/evaluation?
        same, diff = [], []
        for f in sorted(orig.iterdir()):
            if f.name in ("README.md", "split_config.csv"):
                continue
            g = od / f.name
            (same if g.exists() and content(f) == content(g) else diff).append(f.name)
        r["identical_files"], r["different_files"] = same, diff
        res[str(seed)] = r
        print("rebuild", seed, json.dumps(r)[:400], flush=True)
    d = load_summary()
    d["rebuild_22_checks"] = res
    save_summary(d)


def step_independent():
    log = OUT / "audit_independent_checks.log"
    with log.open("w") as f:
        rc = subprocess.call([PY, str(ROOT / "audit_independent_checks.py")], cwd=str(ROOT), stdout=f,
                             stderr=subprocess.STDOUT)
    txt = log.read_text()
    d = load_summary()
    d["audit_independent_checks"] = {"returncode": rc, "pass": "ALL INDEPENDENT LEAKAGE CHECKS PASS" in txt,
                                     "log": str(log)}
    save_summary(d)
    print("independent rc", rc, flush=True)


def step_checker():
    code = (
        "import sys; from pathlib import Path; sys.path.insert(0, %r); import leakage_checker as L; "
        "L.OUT = Path(%r); L.main()" % (str(ROOT / "benchmark_v2"), str(OUT / "leakage_checker")))
    log = OUT / "leakage_checker.log"
    with log.open("w") as f:
        rc = subprocess.call([PY, "-c", code], cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
    txt = log.read_text()
    d = load_summary()
    d["leakage_checker"] = {"returncode": rc, "pass": "STRICT SPLIT: ALL ENFORCED CHECKS PASS" in txt, "log": str(log)}
    save_summary(d)
    print("checker rc", rc, flush=True)


if __name__ == "__main__":
    steps = sys.argv[1:] or ["rebuild", "independent", "checker"]
    for s in steps:
        globals()["step_" + s]()

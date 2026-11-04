#!/usr/bin/env python3
"""
Freeze (or verify) a version of kegg_dataset/.

    python freeze_dataset.py --version v1          # write MANIFEST_v1.json + MANIFEST_v1.md5
    python freeze_dataset.py --version v1 --verify # check current files against the manifest
    cd kegg_dataset && md5sum -c MANIFEST_v1.md5   # same check with coreutils

The manifest records every data file (bytes, md5, rows), the run parameters, source data
versions and the md5 of the code that produced the data. Logs, manifests and later
derived directories (e.g. baselines_*) are excluded. An existing manifest is never
overwritten unless --force is given.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DS = ROOT / "kegg_dataset"
CACHE = ROOT / "kegg_dataset_cache"
EXCLUDE_DIRS = {"logs"}
EXCLUDE_PREFIX = ("MANIFEST_", "baselines_", "models_", "comparisons_")
CODE = ["KEGG_human_feasibility_v2.py", "build_kegg_pathway_dataset.py", "build_ko_leakage_free_split.py",
        "freeze_dataset.py", "kegg_pipeline/__init__.py", "kegg_pipeline/common.py", "kegg_pipeline/kgml.py",
        "kegg_pipeline/annotations.py", "kegg_pipeline/sampling.py", "kegg_pipeline/report.py"]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows(path: Path):
    name = path.name
    if name.endswith((".csv", ".csv.gz", ".tsv")):
        op = gzip.open if name.endswith(".gz") else open
        with op(path, "rt", encoding="utf-8", newline="") as f:
            import csv
            return sum(1 for _ in csv.reader(f)) - 1  # quoted fields may contain newlines
    if name.endswith(".faa"):
        with path.open() as f:
            return sum(1 for line in f if line.startswith(">"))
    return None


def data_files():
    out = []
    for p in sorted(DS.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(DS)
        if rel.parts[0] in EXCLUDE_DIRS or rel.parts[0].startswith(EXCLUDE_PREFIX) or p.name.startswith(EXCLUDE_PREFIX):
            continue
        if p.name.endswith((".tmp", ".part")):
            sys.exit(f"refusing to freeze: unfinished file {rel}")
        out.append(p)
    return out


def source_versions():
    v = {}
    gaf = CACHE / "goa_human.gaf.gz"
    if gaf.exists():
        with gzip.open(gaf, "rt") as f:
            for line in f:
                if not line.startswith("!"):
                    break
                for key in ("date-generated", "go-version"):
                    if line.startswith(f"!{key}:"):
                        v[f"goa_human_{key}"] = line.split(":", 1)[1].strip()
    obo = CACHE / "go-basic.obo"
    if obo.exists():
        with obo.open() as f:
            for line in f:
                if line.startswith("data-version:"):
                    v["go_basic_obo_data_version"] = line.split(":", 1)[1].strip()
                    break
                if line.startswith("[Term]"):
                    break
    dates = []
    for x in sorted((ROOT / "kegg_feasibility_cache" / "kgml").glob("*.xml")):
        with x.open() as f:
            for _ in range(4):
                m = re.search(r"Creation date: ([^(]+)", f.readline())
                if m:
                    dates.append(dt.datetime.strptime(m.group(1).strip(), "%b %d, %Y %H:%M:%S %z").date())
                    break
    if dates:
        v["kgml_creation_date_range"] = f"{min(dates)} .. {max(dates)} ({len(dates)} files)"
    for name, path in [("kegg_rest_lists_fetched", CACHE / "kegg_rest" / "list_hsa.tsv"),
                       ("kegg_uniprot_conv_fetched", ROOT / "kegg_feasibility_cache" / "uniprot_conv.tsv"),
                       ("uniprot_table_fetched", CACHE / "uniprot_human_9606.tsv.gz"),
                       ("ncbi_gene2go_fetched", CACHE / "gene2go.gz"),
]:
        if path.exists():
            v[name] = dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    # batch caches are completed by the prefetch run; take completion times from its log
    plog = DS / "logs" / "run_prefetch.log"
    if plog.exists():
        for line in plog.read_text().splitlines():
            m = re.match(r"\[([\d\- :]+)\] (aaseq batches (\d+)/\3|KO batches (\d+)/\4)$", line)
            if m:
                key = "kegg_aaseq_fetch_completed" if m.group(2).startswith("aaseq") else "kegg_ko_entries_fetch_completed"
                v[key] = m.group(1)
    return v


def load_params():
    p = {}
    cfg = DS / "splits_ko" / "split_config.csv"
    if cfg.exists():
        import csv
        p["ko_split"] = {r["k"]: r["v"] for r in csv.DictReader(cfg.open())}
    m = DS / "logs" / "metrics.json"
    if m.exists():
        mm = json.loads(m.read_text())
        p["pipeline_key_counts"] = {k: mm.get("kgml", {}).get(k) for k in
                                    ("pathways_parsed", "node_occurrences", "relations", "reaction_records")}
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    mj, mm = DS / f"MANIFEST_{a.version}.json", DS / f"MANIFEST_{a.version}.md5"

    if a.verify:
        man = json.loads(mj.read_text())
        bad = []
        for rel, info in man["files"].items():
            p = DS / rel
            if not p.exists():
                bad.append(f"MISSING {rel}")
            elif md5(p) != info["md5"]:
                bad.append(f"CHANGED {rel}")
        extra = sorted(str(p.relative_to(DS)) for p in data_files() if str(p.relative_to(DS)) not in man["files"])
        code_bad = [c for c, h in man["code_md5"].items() if (ROOT / c).exists() and md5(ROOT / c) != h]
        print(f"{len(man['files'])} files checked; {len(bad)} problems; {len(extra)} unlisted data files; "
              f"{len(code_bad)} code files changed since freeze")
        for x in bad + [f"UNLISTED {e}" for e in extra] + [f"CODE CHANGED {c}" for c in code_bad]:
            print("  " + x)
        sys.exit(1 if bad else 0)

    if mj.exists() and not a.force:
        sys.exit(f"{mj.name} already exists; a frozen version is never overwritten (use --force deliberately)")
    files = {}
    for p in data_files():
        rel = str(p.relative_to(DS))
        files[rel] = {"bytes": p.stat().st_size, "md5": md5(p), "rows": count_rows(p)}
        print(f"  {rel}: {files[rel]['rows']} rows")
    man = {
        "dataset": "KEGG human pathway dataset (KO-level leakage-controlled split included)",
        "version": a.version,
        "frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
        "n_files": len(files),
        "total_bytes": sum(f["bytes"] for f in files.values()),
        "params": load_params(),
        "source_versions": source_versions(),
        "code_md5": {c: md5(ROOT / c) for c in CODE if (ROOT / c).exists()},
        "files": files,
    }
    mj.write_text(json.dumps(man, indent=2, ensure_ascii=False))
    mm.write_text("".join(f"{v['md5']}  {k}\n" for k, v in files.items()))
    print(f"frozen {len(files)} files ({man['total_bytes'] / 1e6:.1f} MB) -> {mj.name}, {mm.name}")


if __name__ == "__main__":
    main()

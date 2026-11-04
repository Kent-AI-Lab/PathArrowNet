"""Shared paths, logging, resumable HTTP fetching and small I/O helpers.

Design rules
------------
* Every remote resource is cached on disk; a cached file is never refetched
  unless --force-refresh is given.
* Large downloads go to "<file>.part" first and are renamed only after they
  complete, so an interrupted download never looks like a finished one.
* Every failed request is appended to kegg_dataset/logs/failures.csv.  A
  failed request is NEVER converted into "no annotation"; callers keep an
  explicit "error" status and retry it on the next run.
"""

from __future__ import annotations

import csv
import datetime as _dt
import gzip
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent

# Existing (read-only) inputs from the v2 feasibility audit.
OLD_CACHE = ROOT / "kegg_feasibility_cache"
OLD_KGML = OLD_CACHE / "kgml"
OLD_PATHWAY_LIST = OLD_CACHE / "pathway_list.tsv"
OLD_UNIPROT_CONV = OLD_CACHE / "uniprot_conv.tsv"
OLD_SEQ_STATUS = OLD_CACHE / "sequence_status.json"
OLD_GENES_CSV = ROOT / "kegg_feasibility_output" / "kegg_hsa_genes.csv"

# New cache and outputs.
CACHE = ROOT / "kegg_dataset_cache"
OUT = ROOT / "kegg_dataset"
LOGDIR = OUT / "logs"
FAILURE_LOG = LOGDIR / "failures.csv"

KEGG = "https://rest.kegg.jp"
USER_AGENT = "KEGG-pathway-dataset-builder/1.0 (academic research; serial requests)"

for _d in (CACHE, OUT, LOGDIR):
    _d.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def log_failure(stage: str, key: str, error: str) -> None:
    new = not FAILURE_LOG.exists()
    with FAILURE_LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "stage", "key", "error"])
        w.writerow([now(), stage, key, str(error).replace("\n", " ")[:500]])
    print(f"[{now()}] [FAILURE] {stage} {key}: {error}", file=sys.stderr, flush=True)


class NotFound(Exception):
    """HTTP 404: the resource genuinely does not exist (not a transient error)."""


def request_bytes(url: str, sleep: float, retries: int = 5, timeout: int = 120) -> bytes:
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            time.sleep(sleep)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                time.sleep(sleep)
                raise NotFound(url)
            last = e
        except Exception as e:  # network errors, timeouts, truncated reads
            last = e
        wait = min(60, 2 ** attempt)
        print(f"[{now()}] request error ({last}); attempt {attempt}/{retries}; "
              f"retry in {wait}s: {url}", file=sys.stderr, flush=True)
        time.sleep(wait)
    raise RuntimeError(f"request failed after {retries} attempts: {url}: {last}")


def cached_text(path: Path, url: str, sleep: float, force: bool = False,
                offline: bool = False) -> Optional[str]:
    """Return cached text, fetching once if needed.  Returns None on failure
    (after logging it), so callers can mark dependent data 'unavailable'."""
    if path.exists() and not force:
        return path.read_text(encoding="utf-8")
    if offline:
        log_failure("fetch", url, "offline mode and no cache")
        return None
    try:
        data = request_bytes(url, sleep).decode("utf-8", errors="replace")
    except NotFound:
        log_failure("fetch", url, "HTTP 404")
        return None
    except Exception as e:
        log_failure("fetch", url, e)
        return None
    if not data.strip():
        log_failure("fetch", url, "empty response")
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)
    return data


def download_file(path: Path, url: str, force: bool = False, offline: bool = False,
                  retries: int = 5) -> bool:
    """Stream a (possibly large) file to disk with resume via HTTP Range."""
    if path.exists() and not force:
        return True
    if offline:
        log_failure("download", url, "offline mode and no cache")
        return False
    part = path.with_name(path.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": USER_AGENT}
            if have:
                headers["Range"] = f"bytes={have}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as resp:
                mode = "ab" if (have and resp.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                total = resp.headers.get("Content-Length")
                log(f"downloading {url} -> {path.name} (resume at {have} bytes, "
                    f"remaining {total})")
                with part.open(mode) as f:
                    shutil.copyfileobj(resp, f, length=1 << 20)
            part.replace(path)
            log(f"download complete: {path} ({path.stat().st_size:,} bytes)")
            return True
        except urllib.error.HTTPError as e:
            if e.code == 416 and part.exists():  # already complete
                part.replace(path)
                return True
            err = e
        except Exception as e:
            err = e
        wait = min(120, 5 * 2 ** attempt)
        print(f"[{now()}] download error {err}; attempt {attempt}/{retries}; retry in {wait}s",
              file=sys.stderr, flush=True)
        time.sleep(wait)
    log_failure("download", url, f"failed after {retries} attempts: {err}")
    return False


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log_failure("cache", str(path), f"unreadable json cache: {e}")
        return default


def save_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: Iterable[dict], fields: List[str]) -> int:
    """Atomic CSV (optionally .gz) writer. Returns number of rows."""
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    opener = (lambda p: gzip.open(p, "wt", newline="", encoding="utf-8")) \
        if path.name.endswith(".gz") else (lambda p: p.open("w", newline="", encoding="utf-8"))
    with opener(tmp) as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
            n += 1
    os.replace(tmp, path)
    return n


def read_csv(path: Path) -> List[dict]:
    with open_text(path) as f:
        return list(csv.DictReader(f))


def chunks(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]

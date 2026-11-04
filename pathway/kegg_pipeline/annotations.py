"""Remote annotation sources (all cached, all resumable).

KEGG REST (serial requests, polite sleep):
    list/hsa, link/ko/hsa, list/ko, link/module/ko, list/module,
    list/compound, list/glycan, list/drug, list/reaction, get/br:br08901,
    get/<hsa genes>/aaseq (batches of 10), get/<KOs> (batches of 10, for GO DBLINKS)
UniProt REST:   human UniProtKB accession table (reviewed flag, gene, length, GeneID xref)
GO Consortium / EBI GOA:  goa_human.gaf.gz, go-basic.obo
NCBI:           gene2go.gz (secondary GO source keyed by NCBI GeneID == KEGG hsa id)
"""

from __future__ import annotations

import csv
import gzip
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

from .common import (CACHE, KEGG, OLD_SEQ_STATUS, OLD_UNIPROT_CONV, NotFound, cached_text,
                     chunks, download_file, load_json, log, log_failure, open_text,
                     request_bytes, save_json_atomic)

REST = CACHE / "kegg_rest"
REST.mkdir(parents=True, exist_ok=True)
AASEQ_CACHE = CACHE / "kegg_aaseq.json"
KO_CACHE = CACHE / "kegg_ko_entries.json"

UNIPROT_TSV = CACHE / "uniprot_human_9606.tsv.gz"
UNIPROT_URL = ("https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606"
               "&fields=accession,reviewed,gene_primary,length,xref_geneid&format=tsv&compressed=true")
GOA_GAF = CACHE / "goa_human.gaf.gz"
GOA_URL = "https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz"
GO_OBO = CACHE / "go-basic.obo"
GO_OBO_URL = "https://current.geneontology.org/ontology/go-basic.obo"
GENE2GO = CACHE / "gene2go.gz"
GENE2GO_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz"
GENE2GO_HUMAN = CACHE / "gene2go_human_9606.tsv"

ASPECT = {"F": "MF", "P": "BP", "C": "CC",
          "Function": "MF", "Process": "BP", "Component": "CC",
          "molecular_function": "MF", "biological_process": "BP", "cellular_component": "CC"}


# --------------------------------------------------------------------------- KEGG lists
def kegg_lists(sleep: float, force: bool, offline: bool) -> Dict[str, Optional[str]]:
    ops = {
        "list_hsa": "list/hsa", "link_ko_hsa": "link/ko/hsa", "list_ko": "list/ko",
        "link_module_ko": "link/module/ko", "list_module": "list/module",
        "list_compound": "list/compound", "list_glycan": "list/glycan",
        "list_drug": "list/drug", "list_reaction": "list/reaction",
        "brite_br08901": "get/br:br08901",
    }
    out = {}
    for key, op in ops.items():
        out[key] = cached_text(REST / f"{key}.tsv", f"{KEGG}/{op}", sleep, force, offline)
        log(f"KEGG {op}: {'ok' if out[key] else 'UNAVAILABLE'}")
    return out


def _strip(x: str) -> str:
    return x.split(":", 1)[1] if ":" in x else x


def parse_list_hsa(text: Optional[str]) -> Dict[str, dict]:
    out = {}
    for line in (text or "").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        gid = _strip(parts[0])
        if len(parts) >= 4:
            gtype, desc = parts[1], parts[3]
        else:  # older 2-column format
            gtype, desc = "", parts[1]
        symbol = desc.split(";", 1)[0].split(",")[0].strip() if ";" in desc else ""
        out[gid] = {"gene_type": gtype, "symbol": symbol, "description": desc}
    return out


def parse_link(text: Optional[str]) -> Dict[str, Set[str]]:
    """left -> set(right), prefixes stripped."""
    out = defaultdict(set)
    for line in (text or "").splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            out[_strip(parts[0])].add(_strip(parts[1]))
    return out


def parse_simple_list(text: Optional[str]) -> Dict[str, str]:
    out = {}
    for line in (text or "").splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            out[_strip(parts[0])] = parts[1]
    return out


def parse_uniprot_conv() -> Dict[str, Set[str]]:
    m = defaultdict(set)
    for line in OLD_UNIPROT_CONV.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        a, b = parts
        if a.startswith("hsa:"):
            m[a[4:]].add(_strip(b))
        elif b.startswith("hsa:"):
            m[b[4:]].add(_strip(a))
    return m


# --------------------------------------------------------------------------- sequences
def _parse_fasta(text: str) -> Dict[str, str]:
    seqs, cur, buf = {}, None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if cur:
                seqs[cur] = "".join(buf)
            m = re.match(r">hsa:(\S+)", line)
            cur, buf = (m.group(1) if m else None), []
        elif cur:
            buf.append(line.strip())
    if cur:
        seqs[cur] = "".join(buf)
    return seqs


def fetch_kegg_aaseq(genes: Set[str], sleep: float, offline: bool) -> Dict[str, dict]:
    """Returns {gene: {"status": yes|no|error, "seq": str}}.  Reuses the v2
    status cache: genes that v2 confirmed as 404 ("no") are not re-requested."""
    cache = load_json(AASEQ_CACHE, {})
    old = load_json(OLD_SEQ_STATUS, {})
    for g in genes:
        if g not in cache and old.get(g) == "no":
            cache[g] = {"status": "no", "seq": "", "source": "v2_audit_404"}
    pending = sorted([g for g in genes if cache.get(g, {}).get("status") not in ("yes", "no")], key=int)
    log(f"KEGG aaseq: {len(genes) - len(pending)}/{len(genes)} cached, {len(pending)} to fetch")
    if not pending:
        return cache  # nothing new: do not rewrite the cache (keeps its mtime = real fetch time)
    if offline:
        for g in pending:
            cache.setdefault(g, {"status": "error", "seq": "", "source": "offline"})
        return cache
    nb = (len(pending) + 9) // 10
    for bi, batch in enumerate(chunks(pending, 10), 1):
        url = f"{KEGG}/get/" + "+".join(f"hsa:{g}" for g in batch) + "/aaseq"
        found = {}
        try:
            found = _parse_fasta(request_bytes(url, sleep).decode("utf-8", "replace"))
        except NotFound:
            pass
        except Exception as e:
            log_failure("aaseq_batch", ",".join(batch), e)
        for g in batch:
            if g in found and found[g]:
                cache[g] = {"status": "yes", "seq": found[g], "source": "kegg_batch"}
                continue
            try:
                s = _parse_fasta(request_bytes(f"{KEGG}/get/hsa:{g}/aaseq", sleep).decode("utf-8", "replace"))
                cache[g] = ({"status": "yes", "seq": s[g], "source": "kegg_single"} if s.get(g)
                            else {"status": "no", "seq": "", "source": "kegg_single_empty"})
            except NotFound:
                cache[g] = {"status": "no", "seq": "", "source": "kegg_single_404"}
            except Exception as e:
                cache[g] = {"status": "error", "seq": "", "source": "request_error"}
                log_failure("aaseq_single", g, e)
        if bi % 20 == 0 or bi == nb:
            save_json_atomic(AASEQ_CACHE, cache)
            log(f"aaseq batches {bi}/{nb}")
    save_json_atomic(AASEQ_CACHE, cache)
    return cache


# --------------------------------------------------------------------------- KO entries (GO)
def _parse_ko_flat(text: str) -> Dict[str, dict]:
    out = {}
    for block in text.split("\n///"):
        m = re.search(r"^ENTRY\s+(K\d{5})", block, re.M)
        if not m:
            continue
        ko = m.group(1)
        go, section, db = [], None, None
        for line in block.splitlines():
            head = line[:12].strip()
            if head:
                section = head
            if section != "DBLINKS":
                continue
            body = line[12:].strip()
            if ":" in body.split()[0] if body else False:
                db, rest = body.split(":", 1)
                db = db.strip()
            else:
                rest = body
            if db == "GO":
                go.extend(f"GO:{x}" for x in rest.split() if x.isdigit())
        out[ko] = {"status": "ok", "go": sorted(set(go))}
    return out


def fetch_ko_entries(kos: Set[str], sleep: float, offline: bool) -> Dict[str, dict]:
    cache = load_json(KO_CACHE, {})
    pending = sorted(k for k in kos if cache.get(k, {}).get("status") not in ("ok", "not_found"))
    log(f"KEGG KO entries: {len(kos) - len(pending)}/{len(kos)} cached, {len(pending)} to fetch")
    if offline or not pending:
        return cache
    nb = (len(pending) + 9) // 10
    for bi, batch in enumerate(chunks(pending, 10), 1):
        url = f"{KEGG}/get/" + "+".join(f"ko:{k}" for k in batch)
        got = {}
        try:
            got = _parse_ko_flat(request_bytes(url, sleep, timeout=300).decode("utf-8", "replace"))
        except NotFound:
            pass
        except Exception as e:
            log_failure("ko_batch", ",".join(batch), e)
        for k in batch:
            if k in got:
                cache[k] = got[k]
                continue
            try:
                one = _parse_ko_flat(request_bytes(f"{KEGG}/get/ko:{k}", sleep, timeout=300)
                                     .decode("utf-8", "replace"))
                cache[k] = one.get(k, {"status": "error", "go": []})
                if k not in one:
                    log_failure("ko_single", k, "entry not parseable")
            except NotFound:
                cache[k] = {"status": "not_found", "go": []}
                log_failure("ko_single", k, "HTTP 404 (KO withdrawn?)")
            except Exception as e:
                cache[k] = {"status": "error", "go": []}
                log_failure("ko_single", k, e)
        if bi % 10 == 0 or bi == nb:
            save_json_atomic(KO_CACHE, cache)
            log(f"KO batches {bi}/{nb}")
    save_json_atomic(KO_CACHE, cache)
    return cache


# --------------------------------------------------------------------------- UniProt / GO files
def fetch_bulk_files(force: bool, offline: bool, want_gene2go: bool) -> Dict[str, bool]:
    ok = {
        "uniprot": download_file(UNIPROT_TSV, UNIPROT_URL, force, offline),
        "goa": download_file(GOA_GAF, GOA_URL, force, offline),
        "obo": download_file(GO_OBO, GO_OBO_URL, force, offline),
        "gene2go": False,
    }
    if want_gene2go:
        if GENE2GO_HUMAN.exists() and not force:
            ok["gene2go"] = True
        elif download_file(GENE2GO, GENE2GO_URL, force, offline):
            tmp = GENE2GO_HUMAN.with_suffix(".tmp")
            n = 0
            with gzip.open(GENE2GO, "rt", encoding="utf-8") as f, tmp.open("w") as g:
                for line in f:
                    if line.startswith("#") or line.startswith("9606\t"):
                        g.write(line)
                        n += 1
            tmp.replace(GENE2GO_HUMAN)
            log(f"gene2go filtered to human: {n:,} lines")
            ok["gene2go"] = True
    return ok


def parse_uniprot_table() -> Dict[str, dict]:
    out = {}
    if not UNIPROT_TSV.exists():
        return out
    with open_text(UNIPROT_TSV) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            out[row["Entry"]] = {
                "reviewed": row.get("Reviewed", ""),
                "gene": row.get("Gene Names (primary)", ""),
                "length": row.get("Length", ""),
                "geneids": [x for x in (row.get("GeneID", "") or "").strip(";").split(";") if x],
            }
    return out


def parse_obo(path: Path = GO_OBO) -> Dict[str, dict]:
    terms, cur = {}, None
    if not path.exists():
        return terms
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                cur = {"alt": []}
            elif line.startswith("[") and line.endswith("]"):
                cur = None
            elif cur is not None and ": " in line:
                k, v = line.split(": ", 1)
                if k == "id":
                    cur["id"] = v
                    terms[v] = cur
                elif k == "name":
                    cur["name"] = v
                elif k == "namespace":
                    cur["aspect"] = ASPECT.get(v, v)
                elif k == "is_obsolete":
                    cur["obsolete"] = v == "true"
                elif k == "alt_id":
                    cur["alt"].append(v)
    for t in list(terms.values()):
        for a in t["alt"]:
            terms.setdefault(a, dict(t, alt_of=t["id"]))
    return terms


def iter_goa(accessions: Set[str]):
    """Yield GAF 2.2 rows whose DB object is one of the given UniProt accessions."""
    if not GOA_GAF.exists():
        return
    with gzip.open(GOA_GAF, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("!"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 15 or c[0] != "UniProtKB":
                continue
            acc = c[1].split("-")[0]
            if acc in accessions:
                yield {"accession": c[1], "base_accession": acc, "symbol": c[2],
                       "qualifier": c[3], "go_id": c[4], "reference": c[5], "evidence": c[6],
                       "aspect": ASPECT.get(c[8], c[8]), "date": c[13], "assigned_by": c[14]}


def iter_gene2go(genes: Set[str]):
    if not GENE2GO_HUMAN.exists():
        return
    with GENE2GO_HUMAN.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) >= 8 and c[1] in genes:
                yield {"gene": c[1], "go_id": c[2], "evidence": c[3], "qualifier": c[4],
                       "term": c[5], "pubmed": c[6], "aspect": ASPECT.get(c[7], c[7])}

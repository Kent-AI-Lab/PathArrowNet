#!/usr/bin/env python3
"""
Uniform gene features for BioPathNet's own KEGG task (BioPathNet/data/PC_KEGG_0602: gene --KEGGPathway--> pathway).

The task asks which KEGG pathways a gene belongs to (and, in the head direction, which genes a pathway contains), so
a feature must NOT exist only for KEGG genes (feature presence would reveal membership). Every gene entity of the
dataset is therefore featurised from sources that cover the whole human proteome:
  esm2               UniProt Swiss-Prot human sequence of the gene symbol (gene_primary; first reviewed entry),
                     ESM2-650M (same model / revision / windowing as build_protein_features.py), mean over residues
  gobert_noBP_clean  GoBERT (same model / encoding as build_protein_features.py) on the gene's GOA human annotations,
                     MF + CC only, dropping NOT, IEA, Reactome-assigned and UniPathway2GO (GO_REF:0000041) rows
Chemicals and pathways get no feature.
Output: kegg_features/pc_kegg_gene_features.pt  {feature: {symbol: tensor}}  + out/pc_kegg_feature_coverage.csv

    HF_HOME=$KEGG_HOME/pathway/.hf_cache HF_HUB_OFFLINE=1 .venv/bin/python benchmark_v2/build_pc_kegg_features.py
"""
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())

import csv
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

B2 = Path(__file__).resolve().parent
ROOT = B2.parent
sys.path.insert(0, str(ROOT))
import build_protein_features as BP  # noqa: E402

DATA = Path(_KH + "/biopathnet_repro/BioPathNet/data/PC_KEGG_0602")


@torch.no_grad()
def esm(seqs, device, win=1022, stride=511, tokens_per_batch=24000):
    from transformers import AutoTokenizer, EsmModel
    tok = AutoTokenizer.from_pretrained(BP.ESM, revision=BP.ESM_REV)
    model = EsmModel.from_pretrained(BP.ESM, revision=BP.ESM_REV, add_pooling_layer=False).to(device).eval().half()
    chunks = []
    for g, s in seqs.items():
        if len(s) <= win:
            chunks.append((g, 0, s))
        else:
            st = list(range(0, len(s) - win + 1, stride))
            if st[-1] + win < len(s):
                st.append(len(s) - win)
            chunks += [(g, x, s[x:x + win]) for x in st]
    chunks.sort(key=lambda c: len(c[2]))
    acc = {g: torch.zeros(len(s), model.config.hidden_size) for g, s in seqs.items()}
    cnt = {g: torch.zeros(len(s)) for g, s in seqs.items()}
    i = 0
    while i < len(chunks):
        batch = [chunks[i]]
        i += 1
        while i < len(chunks) and (len(batch) + 1) * (len(chunks[i][2]) + 2) <= tokens_per_batch:
            batch.append(chunks[i])
            i += 1
        enc = tok([c[2] for c in batch], padding=True, return_tensors="pt").to(device)
        hs = model(**enc).last_hidden_state.float().cpu()
        for j, (g, st, sub) in enumerate(batch):
            acc[g][st:st + len(sub)] += hs[j, 1:1 + len(sub)]
            cnt[g][st:st + len(sub)] += 1
    return {g: (acc[g] / cnt[g][:, None]).mean(0) for g in seqs}


@torch.no_grad()
def gobert(gosets, device):
    from huggingface_hub import hf_hub_download
    from transformers import BertForPreTraining
    vf = hf_hub_download(BP.GOBERT, "vocab.txt", revision=BP.GOBERT_REV)
    added = json.loads(Path(hf_hub_download(BP.GOBERT, "added_tokens.json", revision=BP.GOBERT_REV)).read_text())
    vocab = {l.strip(): i for i, l in enumerate(Path(vf).read_text().splitlines()) if l.strip()}
    CLS, SEP, PAD = added["[CLS]"], added["[SEP]"], added["[PAD]"]
    model = BertForPreTraining.from_pretrained(BP.GOBERT, revision=BP.GOBERT_REV).to(device).eval()
    items = sorted(gosets.items(), key=lambda kv: len(kv[1]))
    out = {}
    for i in range(0, len(items), 64):
        seqs, kept = [], []
        for g, gos in items[i:i + 64]:
            known = sorted(x for x in gos if x in vocab)
            if known:
                seqs.append([CLS] + [vocab[x] for x in known][:1022] + [SEP])
                kept.append(g)
        if not seqs:
            continue
        L = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), L), PAD, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, x in enumerate(seqs):
            ids[j, :len(x)] = torch.tensor(x)
            att[j, :len(x)] = 1
        ids, att = ids.to(device), att.to(device)
        hs = model(input_ids=ids, attention_mask=att, token_type_ids=torch.zeros_like(ids),
                   output_hidden_states=True).hidden_states[-1].float()
        tm = att.bool() & (ids != CLS) & (ids != SEP)
        for j, g in enumerate(kept):
            out[g] = hs[j][tm[j]].mean(0).cpu()
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    genes = [l.split("\t")[0] for l in (DATA / "entity_types.txt").read_text().splitlines() if l.endswith("\tgenes")]
    gset = set(genes)
    kegg_genes = set()
    for f in ("train2.txt", "valid.txt", "test.txt"):
        for l in (DATA / f).read_text().splitlines():
            kegg_genes.add(l.split("\t")[0])
    seqs = {}
    with (ROOT / "kegg_dataset_cache/text/uniprot_human_sprot_seq.tsv").open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            for s in r["Gene Names (primary)"].split(";"):
                s = s.strip()
                if s in gset and s not in seqs and r["Sequence"]:
                    seqs[s] = r["Sequence"]
    gos = defaultdict(set)
    with gzip.open(ROOT / "kegg_dataset_cache/goa_human.gaf.gz", "rt") as f:
        for l in f:
            if l.startswith("!"):
                continue
            c = l.rstrip("\n").split("\t")
            sym, qual, go, ref, ev, asp, by = c[2], c[3], c[4], c[5], c[6], c[8], c[14]
            if sym not in gset or "NOT" in qual or asp not in ("F", "C"):
                continue
            if ev == "IEA" or by == "Reactome" or "GO_REF:0000041" in ref:
                continue
            gos[sym].add(go)
    t0 = time.time()
    feats = {"esm2": esm(seqs, device)}
    print(f"ESM2: {len(feats['esm2'])} genes, {time.time() - t0:.0f}s", flush=True)
    feats["gobert_noBP_clean"] = gobert(gos, device)
    print(f"GoBERT: {len(feats['gobert_noBP_clean'])} genes", flush=True)
    torch.save(feats, ROOT / "kegg_features" / "pc_kegg_gene_features.pt")
    rows = []
    for name, tab in feats.items():
        for grp, members in (("KEGG-member genes (appear in train/valid/test triples)", kegg_genes & gset),
                             ("other genes (background only)", gset - kegg_genes)):
            rows.append({"feature": name, "gene_group": grp, "genes": len(members),
                         "with_feature": sum(1 for g in members if g in tab),
                         "coverage": round(sum(1 for g in members if g in tab) / max(1, len(members)), 4)})
    with (B2 / "out" / "pc_kegg_feature_coverage.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Frozen GoBERT and ESM2 node features for the KEGG dataset (v1).

    HF_HOME=$KEGG_HOME/pathway/.hf_cache .venv/bin/python build_protein_features.py --part gobert --gpu 0
    HF_HOME=$KEGG_HOME/pathway/.hf_cache .venv/bin/python build_protein_features.py --part esm --gpu 1
    .venv/bin/python build_protein_features.py --part ko        # aggregate genes -> KO tokens

GoBERT (MM-YY-WW/GoBERT, official usage): input = the gene's positive GO IDs (gene_go_terms_collapsed.csv),
sorted, space separated; last hidden layer; two poolings saved: mean over GO tokens, and [CLS].
Two inputs: all aspects, and MF+CC only (no BP; BP can encode pathway membership).
ESM2 (facebook/esm2_t33_650M_UR50D): KEGG amino-acid sequences (gene_sequences_kegg.faa); mean over
residues of the last layer; sequences > 1022 aa use windows of 1022 (stride 511), per-residue
embeddings averaged over overlapping windows.
KO feature = mean of member human genes in pathways (ko_mapping.csv human_genes_in_pathways) that
have the feature; count of contributing genes stored.

Outputs (kegg_features/): gene_gobert.pt, gene_esm2.pt, ko_features.pt, feature_coverage.csv, README.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
DS = ROOT / "kegg_dataset"
OUT = ROOT / "kegg_features"
GOBERT, GOBERT_REV = "MM-YY-WW/GoBERT", "cd37874db4c3ef209317276d8b9a3627e89dc157"
ESM, ESM_REV = "facebook/esm2_t33_650M_UR50D", "08e4846e537177426273712802403f7ba8261b6c"


def gene_go_sets():
    allg, nobp = defaultdict(set), defaultdict(set)
    with (DS / "gene_go_terms_collapsed.csv").open() as f:
        for r in csv.DictReader(f):
            g = r["kegg_gene_id"][4:]
            allg[g].add(r["go_id"])
            if r["go_aspect"] in ("MF", "CC"):
                nobp[g].add(r["go_id"])
    return {"all": allg, "noBP": nobp}


@torch.no_grad()
def run_gobert(device):
    """Manual encoding: every GoBERT vocab entry is a whole GO ID (never_split), so the original BERT
    tokenizer output is exactly [CLS] + GO ids + [SEP]. (Current transformers can no longer build the
    GoBERT slow tokenizer, and its auto-converted fast tokenizer lacks [UNK].)"""
    from huggingface_hub import hf_hub_download
    from transformers import BertForPreTraining
    vocab_file = hf_hub_download(GOBERT, "vocab.txt", revision=GOBERT_REV)
    added = json.loads(Path(hf_hub_download(GOBERT, "added_tokens.json", revision=GOBERT_REV)).read_text())
    vocab = {line.strip(): i for i, line in enumerate(Path(vocab_file).read_text().splitlines()) if line.strip()}
    CLS, SEP, PAD = added["[CLS]"], added["[SEP]"], added["[PAD]"]
    model, info = BertForPreTraining.from_pretrained(GOBERT, revision=GOBERT_REV, output_loading_info=True)
    missing = [k for k in info["missing_keys"] if k.startswith("bert.")]
    print(f"GoBERT loading info: missing={info['missing_keys']} unexpected={info['unexpected_keys']}", flush=True)
    assert not missing, f"GoBERT encoder weights missing: {missing}"
    model = model.to(device).eval()
    out, cov = {}, []
    for variant, sets in gene_go_sets().items():
        items = sorted(sets.items(), key=lambda kv: len(kv[1]))
        emb_mean, emb_cls = {}, {}
        t0 = time.time()
        for i in range(0, len(items), 64):
            batch = items[i:i + 64]
            seqs, kept = [], []
            for g, gos in batch:
                known = sorted(x for x in gos if x in vocab)
                cov.append({"variant": variant, "gene": g, "n_go": len(gos), "n_go_in_gobert_vocab": len(known)})
                if known:
                    seqs.append([CLS] + [vocab[x] for x in known] + [SEP])
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
            tokmask = att.bool() & (ids != CLS) & (ids != SEP)
            for j, g in enumerate(kept):
                emb_mean[g] = hs[j][tokmask[j]].mean(0).cpu()
                emb_cls[g] = hs[j][0].cpu()  # [CLS] is the first token
        out[variant] = {"mean": emb_mean, "cls": emb_cls}
        print(f"GoBERT {variant}: {len(emb_mean)} genes in {time.time() - t0:.0f}s", flush=True)
    torch.save({"model": GOBERT, "revision": GOBERT_REV, "dim": model.config.hidden_size, **out}, OUT / "gene_gobert.pt")
    with (OUT / "gobert_vocab_coverage.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cov[0].keys()))
        w.writeheader()
        w.writerows(cov)


def read_fasta(path):
    seqs, cur = {}, None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            cur = line[1:].split()[0].replace("hsa:", "")
            seqs[cur] = []
        elif cur:
            seqs[cur].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


@torch.no_grad()
def run_esm(device, win=1022, stride=511, tokens_per_batch=24000):
    from transformers import AutoTokenizer, EsmModel
    tok = AutoTokenizer.from_pretrained(ESM, revision=ESM_REV)
    model = EsmModel.from_pretrained(ESM, revision=ESM_REV, add_pooling_layer=False).to(device).eval().half()
    seqs = read_fasta(DS / "gene_sequences_kegg.faa")
    chunks = []  # (gene, start, subseq)
    for g, s in seqs.items():
        s = s.replace("*", "")
        if len(s) <= win:
            chunks.append((g, 0, s))
        else:
            starts = list(range(0, len(s) - win + 1, stride))
            if starts[-1] + win < len(s):
                starts.append(len(s) - win)
            chunks += [(g, st, s[st:st + win]) for st in starts]
    chunks.sort(key=lambda c: len(c[2]))
    acc = {g: torch.zeros(len(s.replace("*", "")), model.config.hidden_size) for g, s in seqs.items()}
    cnt = {g: torch.zeros(len(s.replace("*", ""))) for g, s in seqs.items()}
    t0, i = time.time(), 0
    while i < len(chunks):
        batch = [chunks[i]]
        i += 1
        # chunks are sorted by length, so the last one added is the longest in the batch
        while i < len(chunks) and (len(batch) + 1) * (len(chunks[i][2]) + 2) <= tokens_per_batch:
            batch.append(chunks[i])
            i += 1
        enc = tok([c[2] for c in batch], padding=True, return_tensors="pt").to(device)
        hs = model(**enc).last_hidden_state.float().cpu()
        for j, (g, st, sub) in enumerate(batch):
            acc[g][st:st + len(sub)] += hs[j, 1:1 + len(sub)]   # drop <cls>; residues only
            cnt[g][st:st + len(sub)] += 1
    emb = {g: (acc[g] / cnt[g][:, None]).mean(0) for g in seqs}
    n_long = sum(1 for s in seqs.values() if len(s.replace("*", "")) > win)
    print(f"ESM2: {len(emb)} genes ({n_long} longer than {win} aa, windowed) in {time.time() - t0:.0f}s", flush=True)
    torch.save({"model": ESM, "revision": ESM_REV, "dim": model.config.hidden_size, "mean": emb,
                "n_windowed": n_long}, OUT / "gene_esm2.pt")


def run_ko():
    gb = torch.load(OUT / "gene_gobert.pt")
    es = torch.load(OUT / "gene_esm2.pt")
    vocab = {}
    with (DS / "token_vocab_ko.csv").open() as f:
        for r in csv.DictReader(f):
            vocab[r["token"]] = int(r["token_id"])
    members = {}
    with (DS / "ko_mapping.csv").open() as f:
        for r in csv.DictReader(f):
            if r["ko_id"] in vocab:
                members[vocab[r["ko_id"]]] = [g for g in r["human_genes_in_pathways"].split(";") if g]
    V = len(vocab)
    feats, cov = {}, []
    sources = {"gobert_all_mean": gb["all"]["mean"], "gobert_all_cls": gb["all"]["cls"],
               "gobert_noBP_mean": gb["noBP"]["mean"], "gobert_noBP_cls": gb["noBP"]["cls"],
               "esm2_mean": es["mean"]}
    for name, table in sources.items():
        dim = next(iter(table.values())).shape[0]
        X = torch.zeros(V, dim)
        has = torch.zeros(V, dtype=torch.bool)
        ngen = torch.zeros(V, dtype=torch.long)
        for k, gs in members.items():
            vs = [table[g] for g in gs if g in table]
            if vs:
                X[k] = torch.stack(vs).mean(0)
                has[k] = True
                ngen[k] = len(vs)
        feats[name] = {"X": X, "has": has, "n_genes": ngen}
        cov.append({"feature": name, "dim": dim, "ko_tokens": V - 3, "ko_with_feature": int(has.sum()),
                    "coverage": round(int(has.sum()) / (V - 3), 4),
                    "genes_with_feature": len(table)})
    torch.save({"features": feats, "vocab_size": V,
                "sources": {"gobert": [GOBERT, GOBERT_REV], "esm2": [ESM, ESM_REV]}}, OUT / "ko_features.pt")
    with (OUT / "feature_coverage.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cov[0].keys()))
        w.writeheader()
        w.writerows(cov)
    for c in cov:
        print(c, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", required=True, choices=["gobert", "esm", "ko"])
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    {"gobert": lambda: run_gobert(device), "esm": lambda: run_esm(device), "ko": run_ko}[a.part]()


if __name__ == "__main__":
    main()

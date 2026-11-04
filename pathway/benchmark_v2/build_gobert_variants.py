#!/usr/bin/env python3
"""
GoBERT feature variants for the BP-leakage analysis (same model, revision, encoding and KO aggregation as
build_protein_features.py; only the set of GO IDs given to GoBERT per gene changes).

    HF_HOME=$KEGG_HOME/pathway/.hf_cache .venv/bin/python benchmark_v2/build_gobert_variants.py

Variants (input GO IDs per human gene; positive, non-NOT annotations from GOA human GAF + NCBI gene2go)
  gobert_MF            MF terms only
  gobert_CC            CC terms only
  gobert_BP            BP terms only
  gobert_noBP_clean    MF + CC, dropping every annotation row that is electronic (IEA) or assigned by a pathway
                       resource (assigned_by Reactome, or reference GO_REF:0000041 UniPathway2GO); a gene-GO pair is
                       kept if at least one supporting row survives
Controls (no GO information content, same dimensionality / coverage as gobert_noBP_mean)
  gobert_noBP_shuffled  gobert_noBP_mean rows permuted across KOs (seed 42): same feature distribution, wrong KO
  random_gauss          i.i.d. N(0,1) vectors (dim 1024) for the same KOs
Existing (kegg_features/ko_features.pt): gobert_all_mean (MF+BP+CC), gobert_noBP_mean (MF+CC)
Output: kegg_features/ko_features_gobert_variants.pt (same format as ko_features.pt)
"""

import csv
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

DS = ROOT / "kegg_dataset"


def gene_sets():
    rows = defaultdict(lambda: defaultdict(list))   # gene -> go -> [(aspect, clean_ok)]
    with (DS / "gene_go_mapping.csv").open() as f:
        for r in csv.DictReader(f):
            if r["is_negated"] == "1" or "NOT" in r["qualifier"]:
                continue
            ok = not (r["evidence_code"] == "IEA" or r["assigned_by"] == "Reactome" or r["reference"] == "GO_REF:0000041")
            rows[r["kegg_gene_id"][4:]][r["go_id"]].append((r["go_aspect"], ok))
    out = {k: defaultdict(set) for k in ("MF", "CC", "BP", "noBP_clean")}
    for g, d in rows.items():
        for go, lst in d.items():
            asp = lst[0][0]
            out[asp][g].add(go)
            if asp in ("MF", "CC") and any(ok for _, ok in lst):
                out["noBP_clean"][g].add(go)
    return out


@torch.no_grad()
def main():
    from huggingface_hub import hf_hub_download
    from transformers import BertForPreTraining
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_file = hf_hub_download(BP.GOBERT, "vocab.txt", revision=BP.GOBERT_REV)
    added = json.loads(Path(hf_hub_download(BP.GOBERT, "added_tokens.json", revision=BP.GOBERT_REV)).read_text())
    vocab = {l.strip(): i for i, l in enumerate(Path(vocab_file).read_text().splitlines()) if l.strip()}
    CLS, SEP, PAD = added["[CLS]"], added["[SEP]"], added["[PAD]"]
    model = BertForPreTraining.from_pretrained(BP.GOBERT, revision=BP.GOBERT_REV).to(device).eval()
    gene_emb = {}
    stats = []
    for variant, sets in gene_sets().items():
        emb = {}
        items = sorted(sets.items(), key=lambda kv: len(kv[1]))
        t0 = time.time()
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
                emb[g] = hs[j][tm[j]].mean(0).cpu()
        gene_emb[variant] = emb
        stats.append((variant, len(emb), sum(len(v) for v in sets.values()) / max(1, len(sets))))
        print(f"{variant}: {len(emb)} genes, {time.time() - t0:.0f}s", flush=True)
    vocab_ko = {r["token"]: int(r["token_id"]) for r in csv.DictReader((DS / "token_vocab_ko.csv").open())}
    V = len(vocab_ko)
    members = {}
    for r in csv.DictReader((DS / "ko_mapping.csv").open()):
        if r["ko_id"] in vocab_ko:
            members[vocab_ko[r["ko_id"]]] = [g for g in r["human_genes_in_pathways"].split(";") if g]
    feats = {}
    for variant, table in gene_emb.items():
        X = torch.zeros(V, 1024)
        has = torch.zeros(V, dtype=torch.bool)
        for k, gs in members.items():
            vs = [table[g] for g in gs if g in table]
            if vs:
                X[k] = torch.stack(vs).mean(0)
                has[k] = True
        feats[f"gobert_{variant}_mean"] = {"X": X, "has": has}
    base = torch.load(ROOT / "kegg_features" / "ko_features.pt")["features"]["gobert_noBP_mean"]
    idx = base["has"].nonzero()[:, 0]
    g = torch.Generator().manual_seed(42)
    perm = idx[torch.randperm(len(idx), generator=g)]
    Xs = base["X"].clone()
    Xs[idx] = base["X"][perm]
    feats["gobert_noBP_shuffled_mean"] = {"X": Xs, "has": base["has"].clone()}
    Xr = torch.zeros_like(base["X"])
    Xr[idx] = torch.randn(len(idx), base["X"].shape[1], generator=g)
    feats["random_gauss"] = {"X": Xr, "has": base["has"].clone()}
    torch.save({"features": feats, "vocab_size": V, "sources": {"gobert": [BP.GOBERT, BP.GOBERT_REV]}},
               ROOT / "kegg_features" / "ko_features_gobert_variants.pt")
    with (B2 / "out" / "gobert_variant_coverage.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "ko_with_feature", "of_ko_tokens"])
        for k, v in feats.items():
            w.writerow([k, int(v["has"].sum()), V - 3])
    for k, v in feats.items():
        print(k, int(v["has"].sum()))


if __name__ == "__main__":
    main()

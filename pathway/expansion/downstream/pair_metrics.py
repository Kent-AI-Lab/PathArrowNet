"""Metrics and task table shared by the pair-task evaluation scripts (string_baseline.py, zs_paired.py)."""
import numpy as np

TASKS = {"lr": ("data/ligand_receptor", "lr_genes.tsv", "lr_pairs_gene.tsv", "lr_pairs_gene_stringflags.tsv"),
         # task 3: Reactome next-reaction-step gene pairs (scripts/build_reactome_steps.py); same protocol, one role
         "reactome": ("data/reactome", "reactome_genes.tsv", "reactome_pairs_gene.tsv", "reactome_pairs_gene_stringflags.tsv")}


def rankdata(x):
    o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    # average ties
    xs = x[o]; b = np.r_[True, xs[1:] != xs[:-1]]; g = np.cumsum(b) - 1
    cnt = np.bincount(g); start = np.r_[0, np.cumsum(cnt)[:-1]]
    r[o] = (start + (cnt - 1) / 2.0)[g]
    return r


def auroc(score, lab):
    r = rankdata(score) + 1; n1 = lab.sum(); n0 = len(lab) - n1
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


def auprc(score, lab):
    o = np.argsort(-score); l = lab[o]; tp = np.cumsum(l); prec = tp / np.arange(1, len(l) + 1)
    return float((prec * l).sum() / max(1, l.sum()))

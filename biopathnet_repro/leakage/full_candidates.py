"""
Full-candidate KEGG KO data for BioPathNet (setting strict_v1_full of pathway/benchmark_v2).

datasets.biomedical_allent = the official datasets.biomedical, except that the entity vocabulary is seeded with
EVERY entity of entity_types.txt (all 11,620 map KOs, file order) before the triplet files are read. The official
class only creates entities that occur in some triplet file, so KOs that occur only in valid.txt / test.txt
would be entities while other isolated KOs would not: entity existence would reveal val/test membership.

    python leakage/full_candidates.py      -> BioPathNet/data/KEGG_KO_strict_v1_full_bg/
"""
import os as _os, pathlib as _pl  # release: paths are relative to KEGG_HOME (see README)
_KH = _os.environ.get("KEGG_HOME") or next(str(p) for p in _pl.Path(__file__).resolve().parents if (p / ".kegg_release_root").exists())

import csv
import os
import shutil
from pathlib import Path

PATHWAY = Path(_KH + "/pathway")
SRC = Path(__file__).resolve().parent.parent / "BioPathNet/data/KEGG_KO_strict_v1_bg"
DST = Path(__file__).resolve().parent.parent / "BioPathNet/data/KEGG_KO_strict_v1_full_bg"


def register():
    from torchdrug import utils
    from torchdrug.core import Registry as R
    from biopathnet.dataset import biomedical

    if "datasets.biomedical_allent" in R.table:
        return

    @R.register("datasets.biomedical_allent")
    class biomedical_allent(biomedical):
        def load_tsvs(self, tsv_files, verbose=0):
            with open(os.path.join(self.path, self.entity_files[0])) as f:
                inv_entity_vocab = {}
                for line in f:
                    tok = line.split()[0]
                    inv_entity_vocab.setdefault(tok, len(inv_entity_vocab))
            inv_relation_vocab, triplets, num_samples = {}, [], []
            for tsv_file in tsv_files:
                n = 0
                with open(tsv_file) as fin:
                    for h_token, r_token, t_token in csv.reader(fin, delimiter="\t"):
                        assert h_token in inv_entity_vocab and t_token in inv_entity_vocab, (h_token, t_token)
                        r = inv_relation_vocab.setdefault(r_token, len(inv_relation_vocab))
                        triplets.append((inv_entity_vocab[h_token], inv_entity_vocab[t_token], r))
                        n += 1
                num_samples.append(n)
            self.load_triplet(triplets, inv_entity_vocab=inv_entity_vocab, inv_relation_vocab=inv_relation_vocab)
            self.num_samples = num_samples


def make_data():
    DST.mkdir(parents=True, exist_ok=True)
    for name in ("train2.txt", "valid.txt", "test.txt"):
        shutil.copy(SRC / name, DST / name)
    bg = PATHWAY / "kegg_dataset_cache/string/ko_background_exp400_full.tsv"
    with bg.open() as f, (DST / "train1.txt").open("w") as out:
        next(f)
        n = 0
        for line in f:
            a, b = line.split("\t")[:2]
            out.write(f"{a}\tinteracts\t{b}\n")
            n += 1
    # same relation label and direction convention as the 2,870-KO background (checked below)
    old = {tuple(l.split("\t")[::2]) for l in (SRC / "train1.txt").read_text().splitlines()}
    new = {tuple(l.split("\t")[::2]) for l in (DST / "train1.txt").read_text().splitlines()}
    assert old <= new, "full background must contain the 2,870-KO background"
    toks = [r["token"] for r in csv.DictReader((PATHWAY / "kegg_dataset/token_vocab_ko_full.csv").open())
            if not r["token"].startswith("[")]
    (DST / "entity_types.txt").write_text("".join(f"{t}\tKO\n" for t in toks))
    (DST / "entity_names.txt").write_text("".join(f"{t}\t{t}\n" for t in toks))
    print(f"{DST}: background {n} pairs (old {len(old)}), entities {len(toks)}")


if __name__ == "__main__":
    make_data()

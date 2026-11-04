# Release notes

## 2026-11-03: code release for the paper

Benchmark construction, PathArrowNet forward pass and prediction, BioPathNet inference, and the evaluation scripts of
the paper.

### Contents

- `pathway/patharrownet/`: the inference package. The model module is `patharrownet/models/patharrownet.py` (class
  `KONBF`, alias `PathArrowNet`). `python -m patharrownet.predict --dump-benchmark [--with-targets]` writes the full score
  dumps that the evaluation scripts read; `python -m patharrownet.graph_audit` checks the inference graph for leakage.
- `pathway/evaluation/`: one script per group of paper results, a runner (`run_all.sh`) and `check_paper_numbers.py`,
  which compares the computed numbers with the values printed in the paper.
- Primary model: PathArrowNet with KGML relation-level weights (`kgml_relation_weights.py` builds the weights; inference
  is the same as for the unweighted model).
- `checkpoints/`: download instructions and SHA-256 of the trained PathArrowNet checkpoints (five strict splits x seeds
  42-46; 50 files, 1.5 GB; `patharrownet_checkpoints.tar` in the v1.0 release).

### Verification

| Check | Result |
|---|---|
| `evaluation/run_all.sh` on the score dumps, then `check_paper_numbers.py` | 646 of 646 numbers equal the paper's printed values |
| `patharrownet.predict --dump-benchmark --with-targets` on the primary checkpoint (split 1, seed 42; 1,409 anchors, 8.7 min on 8 CPU threads) against the score dumps used in the paper | max abs score difference 3.4e-5; top-10 identical for every anchor; filtered ranks identical for 99.9% of the 1,669 task-A and 2,870 task-S test queries (the rest are near-ties, at most 1.0 apart); MRR equal to 6 decimals |
| `patharrownet.graph_audit` on the primary checkpoint (split 1, seed 42) | 9/9 checks pass |
| `expansion/extdb_eval/eval_zeroshot_kgml.py` on the stored score matrices | all 874 numbers of `zeroshot_kgml.json` identical |

### License

MIT for the code (`LICENSE`). No third-party data are redistributed: the KEGG, STRING and
perturbation-screen files (Replogle, Nadig, Virtual Cell Challenge 2025 and the others in `README.md`) are downloaded by
the user and keep their own terms.

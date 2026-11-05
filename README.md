# PathArrowNet and StrictKEGG

Inference and evaluation code for **PathArrowNet: Learning Directed Pathway Relations from Experimental Interaction
Networks with Leakage Control**.

PathArrowNet ranks the directed downstream KEGG Orthology (KO) groups of a source KO among all 11,620 KOs of the human KEGG
maps. It propagates a query-conditioned representation over training KEGG relations and STRING experimental interactions,
with frozen ESM2 protein features on every node. **StrictKEGG**, the benchmark used in the paper, groups overlapping maps,
assigns every directed relation to one split, removes reverse edges of held-out relations and ranks against the full
candidate set.

The repository contains the benchmark construction, the model forward pass and prediction code, and the evaluation
scripts of the paper. Models are used through the trained checkpoints below or through score dumps written by
`patharrownet.predict`.

## Layout

```
pathway/                      run every command from here
  patharrownet/               inference package
    models/patharrownet.py      PathArrowNet forward pass, graph construction, loading of a trained run
    predict.py                  prediction CLI: top-k next KOs, score dumps in the evaluation format
    graph_audit.py              leakage audit of the inference graph (9 checks)
    compare_scores.py           compare a score dump with a reference dump
    data.py                     vocabulary, training-owned edges with leakage asserts, STRING bins, feature tables
    models/{causal,dir_gnn,kegg_encoder}.py   forward passes of comparison models (task-S Transformer, protocol GNN, encoder)
  evaluation/                 one script per group of paper results; each writes results/evaluation/<name>.json
  benchmark_v2/               score library (scorelib.py), task construction, leakage checker, baselines, seed averaging
  expansion/                  external databases, perturbation screens, audits
  kegg_pipeline/              KEGG parsing, sampling and evaluation helpers
  *.py                        KEGG download, dataset, splits, STRING background, protein features, relation weights
biopathnet_repro/             BioPathNet inference (reproduced and with our adaptations), paralog analysis
checkpoints/                  MANIFEST.tsv: trained weights behind the paper (names, sizes, sha256)
env/                          requirements
```

Scripts locate the repository root from `$KEGG_HOME` or, if unset, from the marker file `.kegg_release_root`.

## Installation

```
python3.12 -m venv pathway/.venv
pathway/.venv/bin/pip install -r env/requirements-main.txt --extra-index-url https://download.pytorch.org/whl/cu128
pathway/.venv/bin/pip install -r env/requirements-analysis.txt    # pandas / scipy / scikit-learn for some evaluations
```

`torch==2.11.0+cu128` is served by the PyTorch index, hence `--extra-index-url`. On a machine without CUDA 12.8, install
torch 2.11.0 for your system from pytorch.org first and drop the torch and `nvidia-*` lines from `requirements-main.txt`.

Prediction and all evaluations run on CPU. BioPathNet inference needs its own environment (Python 3.10, TorchDrug):

```
git clone https://github.com/emyyue/BioPathNet.git biopathnet_repro/BioPathNet
git -C biopathnet_repro/BioPathNet checkout b051a90
cp biopathnet_repro/bpn_patch/biopathnet/model_feat.py biopathnet_repro/BioPathNet/biopathnet/
python3.10 -m venv biopathnet_repro/.venv && biopathnet_repro/.venv/bin/pip install -r env/requirements-biopathnet.txt
```

## Data

KEGG data are not redistributed. KEGG is not a public-domain database: academic use of the website is free, and other uses
need a license (https://www.kegg.jp/kegg/legal.html). The scripts rebuild the dataset from the KEGG REST API for users
entitled to access it; `freeze_dataset.py` writes the checksum manifest used to compare a rebuild with ours. `pathway/strictkegg_checksums.md5` lists the md5 of every split file of the five strict splits (85 files; `cd pathway && md5sum -c strictkegg_checksums.md5`), and `pathway/kegg_dataset_manifest_v1.json` records the dataset parameters and key counts. The split code is deterministic: rebuilding splits 1, 2 and 5 from our KEGG snapshot reproduced every split file byte for byte. KEGG content
changes over time, so a later rebuild can differ from our snapshot (KEGG REST, 2026-09-25).

```
python KEGG_human_feasibility_v2.py && python build_kegg_pathway_dataset.py     # KGML maps, gene-to-KO links -> kegg_dataset/
python build_ko_leakage_free_split.py --seed 42                                  # split 1 -> kegg_dataset/splits_ko/
for s in 43 44 45 46; do                                                         # splits 2-5 -> kegg_splits_robustness/seed<s>/
  python build_ko_leakage_free_split.py --seed $s --out-dir kegg_splits_robustness/seed$s; done
python benchmark_v2/full/build_full_candidates.py && python benchmark_v2/full/build_robustness_settings.py
python benchmark_v2/build_sequence_task.py                                        # task S (path prefixes)
python build_string_background.py                                                 # STRING v12 experimental >= 150, KO level
python benchmark_v2/fetch_uniprot_seq.py && python build_protein_features.py --part esm
for st in strict_v1_full strict_s43_full strict_s44_full strict_s45_full strict_s46_full; do python kgml_relation_weights.py $st; done
python benchmark_v2/leakage_checker.py && python audit_independent_checks.py      # the 22 cross-split checks
```

KEGG REST limits the request rate. If a download stops with HTTP 429, rerun the same command with a longer pause
(`python KEGG_human_feasibility_v2.py --sleep 1.0`); downloads resume from the cache.

Other sources: STRING v12.0 (CC BY 4.0), ESM2 `facebook/esm2_t33_650M_UR50D` (MIT), SIGNOR (CC BY 4.0), WikiPathways
(CC0), NCI-PID via Pathway Commons, Reactome (CC0), OmniPath `ligrecextra`, Replogle 2022 and Adamson 2016 / Norman 2019 from
the GEARS Harvard Dataverse (CC0), Nadig 2025 (GEO GSE264667) and the Virtual Cell Challenge 2025 H1 data (Roohani et al., Cell 2025; training,
validation and test `.h5ad` files from the Arc Virtual Cell Atlas, https://arcinstitute.org/tools/virtualcellatlas,
placed in `expansion/downstream/data/perturb/raw/vcc2025/`). None of these data are redistributed here; each source
keeps its own terms. The download helpers are in `expansion/downstream/scripts/` and `expansion/extdb_eval/`.

## Checkpoints

The trained PathArrowNet (KGML relation-level weights) for the five strict splits x seeds 42-46: 25 runs, 50 files, 1.5 GB.
The five splits are independent pathway-group splits (split seeds 42-46); the seed-averaged model of a split averages the
row-standardized scores of its five runs.

```
cd checkpoints
wget https://github.com/Kent-AI-Lab/PathArrowNet/releases/download/v1.0/patharrownet_checkpoints.tar
tar -xf patharrownet_checkpoints.tar                       # -> patharrownet/<setting>/s<seed>/{best.pt,config.json}
awk -F'\t' 'NR>1 {print $7"  "$5}' MANIFEST.tsv | sha256sum -c
```

A run directory such as `checkpoints/patharrownet/strict_v1_full/s42` is the `<run dir>` of the commands below. Each
`best.pt` is a `state_dict` that includes the frozen ESM2 feature table; `config.json` records the run options. In code and
score files, `KONBF` / `konbf` and `nbf+esm+...` denote PathArrowNet; `clean_nbf_kgml` is the single model and
`clean_nbf_kgml_seedavg` the seed average.

## Prediction

```
cd pathway
# top-10 downstream KOs of two source KOs
python -m patharrownet.predict --run-dir <run dir> --anchors K00844,K01810 --topk 10 --out out/pred
# full score dump in the format read by the evaluation scripts (benchmark_v2/scores/<setting>/<method>/s<seed>/)
python -m patharrownet.predict --run-dir <run dir> --setting strict_v1_full --dump-benchmark \
    --out benchmark_v2/scores/strict_v1_full/clean_nbf_kgml/s42
# the same with the val/test targets as extra anchors (needed by the edge-direction analysis)
python -m patharrownet.predict --run-dir <run dir> --setting strict_v1_full --dump-benchmark --with-targets \
    --out benchmark_v2/scores/strict_v1_full/clean_nbf_kgml_revanchors/s42
# seed average (mean of row z-scores over the seeds; no tuning)
python benchmark_v2/improve/calib_ensemble.py --setting strict_v1_full --a clean_nbf_kgml --b clean_nbf_kgml \
    --name clean_nbf_kgml_seedavg --fixed 0,0,0
# leakage audit of the inference graph
python -m patharrownet.graph_audit --run-dir <run dir> --out out/graph_audit.json
```

BioPathNet scores (in its environment, from `biopathnet_repro/`):

```
.venv/bin/python leakage/bpn_infer.py --config configs/kegg_strict_full_exp150bin4_keep1hop.yaml --feats esm2_mean \
    --setting strict_v1_full --ckpt <run dir>/model_epoch_N.pth --split test --out <dir>    # with our adaptations
.venv/bin/python leakage/bpn_infer.py --config configs/kegg_strict_full_withBG.yaml --feats '' \
    --setting strict_v1_full --ckpt <run dir>/model_epoch_N.pth --split test --out <dir>    # reproduced BioPathNet
```

The seed-averaged BioPathNet dumps use `calib_ensemble.py ... --fixed 1,0,0` with the BioPathNet method as `--a`.

## Checking the PathArrowNet results

The evaluation scripts in the next section compare PathArrowNet with the baselines, BioPathNet and the ablation variants,
and each reads the score dumps of every method in its table. The PathArrowNet rows of Table 1 need only the released
checkpoints: write the dumps of the 25 runs (five settings `strict_v1_full`, `strict_s43_full` ... `strict_s46_full`,
seeds 42-46) and the five seed averages with the commands above, then

```
python -c "
import sys; sys.path.insert(0, 'evaluation')
import common as C
for st in C.STS: print(C.SPLIT[st], '%.3f %.3f' % (C.rr(st, C.PAN).mean(), C.rr(st, C.PAN_SA).mean()))"
```

prints the test MRR of task A per split (single model, seed-averaged):

| | split 1 | split 2 | split 3 | split 4 | split 5 |
|---|---|---|---|---|---|
| PathArrowNet, single model | 0.103 | 0.094 | 0.090 | 0.103 | 0.157 |
| PathArrowNet, seed-averaged | 0.117 | 0.107 | 0.100 | 0.123 | 0.207 |

On CPU the scores differ from the GPU dumps of the paper by at most 4e-5 and give the same top-10 for every anchor.

## Reproducing the paper

All evaluations read score dumps and write one JSON file each to `pathway/results/evaluation/`.

| Paper item | Command (from `pathway/`) |
|---|---|
| Table 1; Sec. 4.1 pooled comparisons, leave-one-split-out, split 5; Sec. 4.2 task S; Suppl. Tables S7, S13, S14, S20 | `python evaluation/main_results.py` |
| Table 2; pooled ablations; Suppl. Fig. S5; Suppl. Table S2 | `python evaluation/ablation.py` |
| Sec. 4.3 evidence strata; Suppl. Table S11 | `python evaluation/evidence_strata.py` |
| Suppl. Sec. S11 candidates without a human gene; targets unseen in training | `python evaluation/candidate_checks.py` |
| Sec. 4.3 ranking among STRING neighbors; Suppl. Table S15 | `python evaluation/conditional_ranking.py` |
| Sec. 4.3 edge direction; Suppl. Table S16 | `python evaluation/direction.py` |
| Directed-label audit and comparison models, Suppl. Table S17; direction against the degree-share rule, Suppl. Sec. S11 | `python evaluation/direction_audit.py` |
| Edge-removal diagnostic, Suppl. Sec. S11 | `python evaluation/edge_removal_diag.py --run "<label>=<run dir>"` (trained runs of the training-scheme variants) |
| Sec. 4.4 relation-level weights; Suppl. Table S19 | `python evaluation/relation_weight_strata.py` |
| Sec. 4.4-4.5 training scheme; Suppl. Table S12 | `python evaluation/training_scheme.py` |
| Sec. 4.5 branch recall; Suppl. Fig. S3 | `python evaluation/branch_recall.py` |
| Suppl. Table S3 (relation subtype, pathway class) | `python evaluation/error_by_type.py` |
| Sec. 5.1 literature audit candidates; Suppl. S8 | `expansion/extdb_eval/build_edges.py` and `build_edges_pid.py` (database edge tables), then `python evaluation/case_study.py` |
| Sec. 5.1 zero-shot transfer and full-candidate ranking of SIGNOR / WikiPathways; Suppl. Fig. S2, Table S4 | `expansion/extdb_eval/build_edges.py`, `build_edges_pid.py`, `expansion/downstream/konbf_full_matrix_kgml.py`, `expansion/extdb_eval/konbf_dir_matrices_kgml.py`, `expansion/downstream/bpn_ko_scores.py`, then `expansion/extdb_eval/eval_zeroshot_kgml.py` and `eval_addendum_kgml.py`; ligand-receptor and Reactome pairs: `expansion/downstream/zs_kgml.py` |
| Sec. 5.2 perturbation specificity; Fig. 4, Suppl. Table S6 | `expansion/downstream/perturb_prep.py`, `expansion/perturbseq/{prep_stats,build_relations}.py`, then `expansion/perturbseq/eval_replication_kgml.py orig4|nadig|vcc2025`; VCC 2025: `expansion/perturbseq/vcc2025/` |
| Sec. 2.2 protocol inflation; Fig. 2a-c, Suppl. Table S8 | `build_leaky_splits.py`, `build_feng_style_splits.py`, `classic_link_prediction.py`, `evaluate_ko_baselines.py`, `benchmark_v2/dump_gnn.py` |
| Sec. 2.2 paralog copying in BioPathNet's KEGG membership task; Fig. 2d | `biopathnet_repro/leakage/make_kogroup_split.py`, `baselines.py`, `bpn_infer.py` |
| Duplication in Reactome and NCI-PID; Suppl. Table S1 | `benchmark_v2/generalization/pc_duplication.py` |
| STRING evidence provenance; Suppl. Table S10 | `benchmark_v2/full/string_audit.py`, `expansion/string_provenance/provenance_audit.py` |
| External-database audit and complex-record leakage; Suppl. Table S5 | `expansion/extdb_ft/audit_extdb.py`, `build_data.py`, `expansion/extdb_ft_cx/build_cx.py`, `compare_cx.py` |
| Baseline score dumps (in-degree, GO similarity, collaborative filtering, STRING neighbor, PageRank) | `benchmark_v2/dump_baselines.py`, `benchmark_v2/full/dump_bg_baselines.py`, `benchmark_v2/baselines_topo/run_ppr.py` |

Conventions (Suppl. S3): single-model results average per-query reciprocal ranks over seeds; seed-averaged results rank
the mean of the row z-scores of the five seeds. Per-split comparisons resample test pathway clusters (2,000 resamples).
Pooled comparisons weight every test pathway equally and resample connected pathway clusters over the five splits
(10,000 resamples). Asterisks in the paper mark two-sided 95% intervals that exclude zero.

## License

The code is released under the MIT License (`LICENSE`). `biopathnet_repro/bpn_patch/biopathnet/model_feat.py` extends BioPathNet (MIT License);
see `biopathnet_repro/bpn_patch/LICENSE.BioPathNet`.

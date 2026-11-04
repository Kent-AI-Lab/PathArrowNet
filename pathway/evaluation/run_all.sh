#!/bin/bash
# Run every paper evaluation (CPU) and compare the outputs with the numbers printed in the paper.
#   cd pathway && bash evaluation/run_all.sh
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
for s in main_results ablation evidence_strata conditional_ranking branch_recall direction relation_weight_strata \
         error_by_type training_scheme case_study candidate_checks direction_audit; do
  echo "== $s"; $PY evaluation/$s.py > "results_${s}.log" 2>&1 || { tail -20 "results_${s}.log"; exit 1; }
done
$PY evaluation/check_paper_numbers.py

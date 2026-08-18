#!/bin/bash
set -euo pipefail

output_dir="outputs/case_study/vegfr2"

# data/VEGFR2.csv (ABL1.csv) contains only the target-independent drug library columns:
# drug_id, names, and drug_seq. Protein features are selected by --target.
python scripts/predict_case_study.py \
  --input data/VEGFR2.csv \
  --smiles_column drug_seq \
  --target P35968 \
  --dataset kiba \
  --checkpoint results_kiba/warm/seed_41/ckpt_pool_dual_best.pt \
  --protein_cache data/cache/kiba_protein_graphs.pt \
  --drug_cache data/cache/egfr_case_study_drug_features.pt \
  --output "${output_dir}/vegfr2_kiba_predictions.csv" \
  --batch_size 64 \
  --gpu_idx 0 \
  "$@"

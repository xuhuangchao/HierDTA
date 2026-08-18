#!/bin/bash
set -euo pipefail

output_dir="outputs/case_study/vegfr2_active_inactive"

python scripts/predict_case_study.py \
  --input data/vegfr2_active_inactive.csv \
  --smiles_column drug_seq \
  --target P35968 \
  --dataset kiba \
  --checkpoint results_kiba/warm/seed_41/ckpt_pool_dual_best.pt \
  --protein_cache data/cache/kiba_protein_graphs.pt \
  --drug_cache data/cache/vegfr2_active_inactive_drug_features.pt \
  --output "${output_dir}/vegfr2_active_inactive_kiba_pred.csv" \
  --batch_size 64 \
  --gpu_idx 0 \
  "$@"

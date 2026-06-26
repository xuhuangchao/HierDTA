#!/bin/bash
set -euo pipefail

# Four input-modality ablations x five Davis warm split seeds = 20 runs.
for ablation in \
  drug_fingerprint \
  drug_graph \
  protein_graph \
  protein_seq
do
  for seed in 41 42 43 32 33
  do
    python train.py \
      --dataset davis \
      --gpu_idx 1 \
      --strategy warm \
      --seed "${seed}" \
      --drug_graph_type dual \
      --graph_pool_type mean_add_max \
      --modality_ablation "${ablation}" \
      --run_name "modality_ablation_wo_${ablation}" \
      "$@"
  done
done

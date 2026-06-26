#!/bin/bash
set -euo pipefail

# Pooling ablations for the full dual-graph model on Davis warm splits.
# Three ablations x five split seeds = 15 training runs.
for setting in \
  "wo_mean:add_max" \
  "wo_max:mean_add" \
  "wo_add:mean_max"
do
  label="${setting%%:*}"
  pool_type="${setting##*:}"

  for seed in 41 42 43 32 33
  do
    python train.py \
      --dataset davis \
      --gpu_idx 1 \
      --strategy warm \
      --seed "${seed}" \
      --drug_graph_type dual \
      --graph_pool_type "${pool_type}" \
      --run_name "pool_ablation_${label}" \
      "$@"
  done
done

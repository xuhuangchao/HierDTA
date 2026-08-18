#!/bin/bash
set -euo pipefail

# Davis unseen-drug evaluation for the dual drug encoder with explicit
# bottom-up atom-to-motif message passing. Runs all five predefined split seeds.
for seed in 41 42 43 32 33
do
  python train.py \
    --dataset davis \
    --gpu_idx 1 \
    --strategy unseen_prot \
    --seed "${seed}" \
    --drug_graph_type dual \
    --atom_motif_mode bottom_up \
    --graph_pool_type mean_add_max \
    --run_name bottom_up_dual \
    "$@"
done

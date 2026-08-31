#!/bin/bash
# Davis unseen-drug evaluation with post-GNN atom-to-motif exchange.
for seed in 41 42 43 32 33
do
  python train.py \
    --dataset davis \
    --gpu_idx 1 \
    --strategy unseen_drug \
    --seed "${seed}" \
    --interaction_type all \
    --drug_gnn_type gat \
    --drug_layer 2 \
    --protein_layer 2 \
    --run_name dta_up_gat_h256 \
    "$@"
done

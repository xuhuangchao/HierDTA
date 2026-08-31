#!/bin/bash
# Davis warm-split evaluation with inter-layer atom-to-motif fusion.
for seed in 32 33 41 42 43
do
  python train.py \
    --dataset davis \
    --gpu_idx 0 \
    --strategy warm \
    --seed "${seed}" \
    --interaction_type all \
    --drug_gnn_type gat \
    --drug_layer 2 \
    --protein_layer 2 \
    --run_name layer_fusion \
    "$@"
done

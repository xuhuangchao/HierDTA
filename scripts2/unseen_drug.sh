#!/bin/bash
# Davis unseen-drug evaluation with inter-layer atom-to-motif fusion.
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
    --run_name layer_fusion \
    "$@"
done

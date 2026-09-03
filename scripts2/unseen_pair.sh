#!/bin/bash
# Davis unseen-pair evaluation with inter-layer atom-to-motif fusion.
for seed in 32 33 41 42 43
do
  python train.py \
    --dataset davis \
    --gpu_idx 0 \
    --strategy unseen_pair \
    --seed "${seed}" \
    --drug_gnn_type gat \
    --drug_layer 3 \
    --motif_layer 1 \
    --protein_layer 3 \
    --use_agg true \
    --run_name motif2atom_mlayer1 \
    "$@"
done


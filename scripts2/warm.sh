#!/bin/bash
# Davis warm-split evaluation for bottom-up dual-scale cross-attention.
for seed in 41 42 43 32 33
do
  python train.py \
    --dataset davis \
    --gpu_idx 0 \
    --strategy warm \
    --seed "${seed}" \
    --drug_graph_type dual \
    --protein_graph_mode dual_view \
    --atom_layer 1 \
    --motif_layer 1 \
    --protein_layer 1 \
    --embed_dim 256 \
    --num_heads 8 \
    --run_name cross_attention_pool \
    "$@"
done

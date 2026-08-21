#!/bin/bash
# Davis unseen-pair evaluation for the dual drug encoder with explicit
# bottom-up atom-to-motif message passing. Runs all five predefined split seeds.
for seed in 41 42 43 32 33
do
  python train.py \
    --dataset davis \
    --gpu_idx 0 \
    --strategy unseen_pair \
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

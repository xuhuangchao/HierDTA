#!/bin/bash
# Davis unseen-drug evaluation for the dual drug encoder with explicit
# bottom-up atom-to-motif message passing. Runs all five predefined split seeds.
for seed in 41 42 43 32 33
do
  python train.py \
    --dataset davis \
    --gpu_idx 1 \
    --strategy unseen_drug \
    --seed "${seed}" \
    --interaction_type atom_motif_global \
    --drug_gnn_type gat \
    --atom_layer 2 \
    --motif_layer 1 \
    --protein_layer 2 \
    --embed_dim 256 \
    --num_heads 8 \
    --run_name dta_0822 \
    "$@"
done

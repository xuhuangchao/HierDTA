#!/bin/bash

# AttentiveFP + EGNN + Bilinear Interaction
# Loop over seeds 0-4

for seed in 0 1 2 3 4; do
  echo "=== Training with seed=$seed ==="
  python training_warmup.py \
    --seed $seed \
    --dataset_idx 0 \
    --gpu_idx 0 \
    --strategy cold_target \
    --epoch 500 \
    --lr 1e-3 \
    --patience 30 \
    --surface_k 5 \
    --drug_hidden 128 \
    --drug_out 128 \
    --n_layers_drug 2 \
    --dropout 0.3 \
    --protein_hidden 128 \
    --protein_out 128 \
    --n_layers_protein 2 \
    --emb_dim 128 
  echo "=== seed=$seed done ==="
done

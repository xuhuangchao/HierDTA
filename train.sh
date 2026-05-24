#!/bin/bash

# AttentiveFP + EGNN + Bilinear Interaction

python training_warmup.py \
  --seed 1 \
  --dataset_idx 0 \
  --gpu_idx 1 \
  --strategy cold_drug \
  --epoch 500 \
  --lr 5e-4 \
  --patience 30 \
  --surface_k 5 \
  --drug_hidden 128 \
  --drug_out 128 \
  --n_layers_drug 2 \
  --dropout 0.3 \
  --protein_hidden 128 \
  --protein_out 128 \
  --n_layers_protein 2 \
  --emb_dim 128 \

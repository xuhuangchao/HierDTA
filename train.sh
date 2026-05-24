#!/bin/bash

# DrugModel (双向边 + TransformerConv) 推荐参数
# 适用于 cold_drug / d2p 场景

python training_warmup.py \
  --seed 1 \
  --dataset_idx 0 \
  --gpu_idx 1 \
  --strategy cold_drug \
  --interaction_mode d2p \
  --agg mean \
  --lr 0.0005 \
  --batch_size 512 \
  --max_norm 1.0 \
  --epoch 500 \
  --patience 50 \
  --surface_k 5 \
  --drug_hidden 64 \
  --drug_out 128 \
  --n_layers_drug 2 \
  --heads 4 \
  --dropout 0.2 \
  --protein_hidden 128 \
  --protein_out 128 \
  --n_layers_protein 4 \
  --use_surface 1 \
  --emb_dim 128 \
  --use_fingerprint 1 \
  --use_p_global 1 \
  --weight_decay 1e-4

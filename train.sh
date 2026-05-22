#!/bin/bash
# ============================================================
# HierDTA 单次训练命令示例
# 用法：复制其中一行到终端直接执行，或取消注释后逐条运行
# ============================================================

# ------------------- 基线：random split -------------------
python training_warmup.py \
  --dataset_idx 1 --gpu_idx 1 --strategy random \
  --lr 0.001 --batch_size 512 --max_norm 5.0 --epoch 500 --patience 50 --surface_k 5 \
  --drug_hidden 64 --drug_out 128 --n_layers_drug 2 --heads 2 --dropout 0.2 \
  --protein_hidden 128 --protein_out 128 --n_layers_protein 4 --use_surface 1 \
  --emb_dim 128 \


# ------------------- 基线：cold_drug split -------------------
# python training_warmup.py \
#   --dataset_idx 1 --gpu_idx 1 --strategy cold_drug \
#   --lr 0.001 --batch_size 512 --max_norm 5.0 --epoch 500 --patience 50 --surface_k 5 \
#   --drug_hidden 64 --drug_out 128 --n_layers_drug 2 --heads 2 --dropout 0.2 \
#   --protein_hidden 128 --protein_out 128 --n_layers_protein 4 --use_surface 1 \
#   --emb_dim 128 \

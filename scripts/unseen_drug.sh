#!/bin/bash
for seed in 41 42 43 32 33
do
  python train.py \
    --dataset davis \
    --gpu_idx 1 \
    --strategy unseen_drug \
    --seed "${seed}" \
    --run_name pool_warmcos \
    "$@"
done

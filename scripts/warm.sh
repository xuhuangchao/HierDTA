#!/bin/bash
for seed in 41 42 43 32 33
do
  python train.py \
    --dataset_idx 0 \
    --gpu_idx 1 \
    --strategy warm \
    --seed "${seed}" \
    --run_name avg_am2pocket_sharedq \
    "$@"
done

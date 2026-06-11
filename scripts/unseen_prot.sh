#!/bin/bash
for seed in 41 42 43 32 33
do
  python train.py \
    --dataset davis \
    --gpu_idx 0 \
    --strategy unseen_prot \
    --seed "${seed}" \
    --run_name avg_am2pocket_sharedq \
    "$@"
done

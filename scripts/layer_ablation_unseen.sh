#!/bin/bash
set -e

DATASET="${DATASET:-davis}"
GPU_IDX="${GPU_IDX:-0}"
RESULT_ROOT="${RESULT_ROOT:-results_${DATASET}_layer_ablation}"
SEEDS="${SEEDS:-41 42 43 32 33}"
STRATEGIES="${STRATEGIES:-unseen_drug}"
MOTIF_LAYERS="${MOTIF_LAYERS:-1 2}"
PROT_LAYERS="${PROT_LAYERS:-2 3}"

for strategy in ${STRATEGIES}
do
  for motif_layers in ${MOTIF_LAYERS}
  do
    for prot_layers in ${PROT_LAYERS}
    do
      run_name="motif${motif_layers}_prot${prot_layers}"
      echo "Running ${strategy}: motif_num_layers=${motif_layers}, prot_num_layers=${prot_layers}"

      for seed in ${SEEDS}
      do
        python train.py \
          --dataset "${DATASET}" \
          --gpu_idx "${GPU_IDX}" \
          --strategy "${strategy}" \
          --seed "${seed}" \
          --motif_num_layers "${motif_layers}" \
          --prot_num_layers "${prot_layers}" \
          --result_root "${RESULT_ROOT}" \
          --run_name "${run_name}" \
          "$@"
      done
    done
  done
done

python scripts/summarize_layer_ablation.py \
  --result_root "${RESULT_ROOT}" \
  --output "${RESULT_ROOT}/summary.csv"

echo "Layer ablation finished. Summary: ${RESULT_ROOT}/summary.csv"

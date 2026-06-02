#!/usr/bin/env bash
# MNIST label-noise sweep: our learned filter vs the two subspace baselines.
#
# Methods:
#   adam             full Adam (no constraint)            -- reference
#   ours             weight-cov filter, learned k=200     -- our method
#   random_subspace  fixed random orthonormal k=200       -- Version A control
#                    (same projection as ours, P random & frozen)
#   lora             frozen random MLP + LoRA adapters    -- Version B baseline
#
# Noise levels: 0%, 20%, 40%, 90% (90% is new -- near-total label corruption).
# Several seeds for error bars. Per-epoch metrics saved for time-series plots.
#
# Run from experiments/. Sequential to stay within GPU memory.
set -euo pipefail

SEEDS=${SEEDS:-"42 43 44"}
NOISES=${NOISES:-"0.0 0.2 0.4 0.9"}
METHODS=${METHODS:-"adam ours random_subspace lora"}
EPOCHS=${EPOCHS:-20}
RANK=${RANK:-200}
LORA_RANK=${LORA_RANK:-32}
SAVE_DIR=${SAVE_DIR:-../results/weight_covariance_v2/subspace_baselines}

pct() { python3 -c "print(int(round(float($1)*100)))"; }

total=0
for S in ${SEEDS}; do for N in ${NOISES}; do for M in ${METHODS}; do total=$((total+1)); done; done; done
echo "Total runs: ${total}"

i=0
for S in ${SEEDS}; do
  for N in ${NOISES}; do
    NP=$(pct ${N})
    for M in ${METHODS}; do
      i=$((i+1))
      NAME="${M}_n${NP}_s${S}"
      echo "=== [${i}/${total}] ${NAME} ==="
      python3 -u run_single_weight_cov_v2.py \
        --mode ${M} --label_noise ${N} --seed ${S} \
        --epochs ${EPOCHS} --rank ${RANK} --decay 0.99 --warmup 100 \
        --lora_rank ${LORA_RANK} \
        --save_dir ${SAVE_DIR} --name ${NAME} --data_dir ./data
    done
  done
done

echo "=== ALL SUBSPACE BASELINE RUNS COMPLETE ==="

#!/usr/bin/env bash
# CIFAR-10 label-noise sweep: adam / ours / switch x {0,40,80}% noise x 2 seeds.
# Run from experiments/. Sequential.
set -euo pipefail
SEEDS=${SEEDS:-"0 1"}
NOISES=${NOISES:-"0.0 0.4 0.8"}
MODES=${MODES:-"adam ours switch"}
EPOCHS=${EPOCHS:-40}
SAVE=${SAVE:-../results/cifar_noise}
pct() { python3 -c "print(int(round(float($1)*100)))"; }
for S in ${SEEDS}; do for N in ${NOISES}; do NP=$(pct ${N}); for M in ${MODES}; do
  echo "=== ${M} noise=${N} seed=${S} ==="
  python3 -u run_cifar_noise.py --mode ${M} --noise ${N} --seed ${S} --epochs ${EPOCHS} \
    --save_dir ${SAVE} --name ${M}_n${NP}_s${S} --data_dir ./data
done; done; done
echo "=== ALL CIFAR NOISE RUNS COMPLETE ==="

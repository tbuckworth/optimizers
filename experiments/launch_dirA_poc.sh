#!/bin/bash
# Direction A POC — does the filter's noise-robustness survive changing the base optimizer?
# filter x {adam, sgd, sgdm} vs the same base optimizers with no filter, across label noise.
set -e
cd "$(dirname "$0")"

EPOCHS=15
RANK=200
DECAY=0.99
WARMUP=100
ADAM_LR=0.001
SGD_LR=0.05
SGDM_LR=0.02
SAVE_DIR="../results/weight_covariance_v2/dirA_poc"
mkdir -p "$SAVE_DIR"

run() { # mode base lr noise name
  python3 -u run_single_weight_cov_v2.py --mode "$1" --base_optimizer "$2" \
    --dataset standard --epochs $EPOCHS --lr "$3" --label_noise "$4" \
    --rank $RANK --decay $DECAY --warmup $WARMUP \
    --save_dir "$SAVE_DIR" --name "$5" 2>&1 | tail -1
}

for noise in 0.0 0.5 0.9; do
  echo "====== noise=${noise} ======"
  echo "  baseline adam";  run baseline adam  $ADAM_LR  $noise "n${noise}_base_adam"
  echo "  baseline sgd";   run baseline sgd   $SGD_LR   $noise "n${noise}_base_sgd"
  echo "  baseline sgdm";  run baseline sgdm  $SGDM_LR  $noise "n${noise}_base_sgdm"
  echo "  filter   adam";  run ours     adam  $ADAM_LR  $noise "n${noise}_filt_adam"
  echo "  filter   sgd";   run ours     sgd   $SGD_LR   $noise "n${noise}_filt_sgd"
  echo "  filter   sgdm";  run ours     sgdm  $SGDM_LR  $noise "n${noise}_filt_sgdm"
done
echo "DIRA_DONE"

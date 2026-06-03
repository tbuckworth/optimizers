#!/bin/bash
# Direction C POC — soft eigenvalue^alpha weighting (graded filter) vs hard top-k.
# alpha spectrum: -1 (whiten) .. 0 (no filter) .. 1 (consensus) .. 4 (dominant dir).
# Base optimizer = Adam throughout. Reference points: hard top-k filter.
set -e
cd "$(dirname "$0")"

EPOCHS=15
RANK=200
DECAY=0.99
WARMUP=100
LR=0.001
SAVE_DIR="../results/weight_covariance_v2/dirC_poc"
mkdir -p "$SAVE_DIR"

for noise in 0.0 0.5; do
  echo "====== noise=${noise} ======"
  echo "  hard top-k reference"
  python3 -u run_single_weight_cov_v2.py --mode ours --base_optimizer adam \
    --dataset standard --epochs $EPOCHS --lr $LR --label_noise $noise \
    --rank $RANK --decay $DECAY --warmup $WARMUP --weighting hard \
    --save_dir "$SAVE_DIR" --name "n${noise}_hard" 2>&1 | tail -1
  for a in -1 0 0.5 1 2 4; do
    echo "  soft alpha=${a}"
    python3 -u run_single_weight_cov_v2.py --mode ours --base_optimizer adam \
      --dataset standard --epochs $EPOCHS --lr $LR --label_noise $noise \
      --rank $RANK --decay $DECAY --warmup $WARMUP \
      --weighting soft --alpha $a --soft_residual 1 \
      --save_dir "$SAVE_DIR" --name "n${noise}_soft_a${a}" 2>&1 | tail -1
  done
done
echo "DIRC_DONE"

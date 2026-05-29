#!/usr/bin/env bash
# Grokking experiments for the weight-covariance spectral filter (v2).
#
# Five runs on modular addition (p=113, 30% train) with a 1-layer transformer:
#   1. adamw            baseline — standard grokking (wd=1.0)
#   2. adam             control  — no weight decay (typically memorizes, groks late/never)
#   3. weightcov_adamw  filter on AdamW (wd=1.0) — does the filter speed up / delay grokking?
#   4. weightcov_adam   filter on Adam, NO wd    — can the filter *substitute* for weight
#                                                   decay and induce grokking on its own?
#   5. switch           adam (no wd) until train_acc>=0.99, then enable filter — can the
#                                                   filter *trigger* generalization on demand?
#
# Run from the experiments/ directory. Sequential to stay within GPU memory.
set -euo pipefail

EPOCHS=${EPOCHS:-40000}
LOG_EVERY=${LOG_EVERY:-50}
SAVE_DIR=${SAVE_DIR:-../results/grokking_v2}
COMMON="--p 113 --train_frac 0.3 --lr 1e-3 --epochs ${EPOCHS} --log_every ${LOG_EVERY} --save_every 0 --save_dir ${SAVE_DIR} --seed 42"
FILTER="--rank 200 --decay 0.99 --warmup 100"

echo "=== [1/5] adamw baseline (wd=1.0) ==="
python3 -u run_grokking.py --optimizer adamw --wd 1.0 ${COMMON} --name adamw_baseline

echo "=== [2/5] adam control (no wd) ==="
python3 -u run_grokking.py --optimizer adam --wd 0.0 ${COMMON} --name adam_nowd

echo "=== [3/5] weightcov + adamw (wd=1.0) ==="
python3 -u run_grokking.py --optimizer weightcov_adamw --wd 1.0 ${COMMON} ${FILTER} --name filter_adamw

echo "=== [4/5] weightcov + adam (NO wd) — can filter replace weight decay? ==="
python3 -u run_grokking.py --optimizer weightcov_adam --wd 0.0 ${COMMON} ${FILTER} --name filter_adam_nowd

echo "=== [5/5] switch: adam (no wd) -> filter at train_acc>=0.99 ==="
python3 -u run_grokking.py --optimizer weightcov_adam --wd 0.0 ${COMMON} ${FILTER} \
    --switch_at 0.99 --name switch_adam_at99

echo "=== ALL GROKKING V2 RUNS COMPLETE ==="

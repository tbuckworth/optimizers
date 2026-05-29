#!/usr/bin/env bash
# Multi-seed grokking sweep — confirm/kill the "filter accelerates grokking" result.
# All conditions use weight decay (wd=1.0); the no-wd cases were uninteresting.
#
# Three conditions x N seeds:
#   1. adamw            baseline grokking (wd=1.0)
#   2. filter_adamw     filter on from the start (wd=1.0)
#   3. switch_adamw     AdamW (wd=1.0) until train_acc>=0.99, then enable filter
#                       (i.e. switch keeps weight decay the whole time)
#
# Run from experiments/. Sequential to stay within GPU memory.
set -euo pipefail

SEEDS=${SEEDS:-"42 43 44 45 46"}
EPOCHS=${EPOCHS:-20000}
LOG_EVERY=${LOG_EVERY:-50}
SAVE_DIR=${SAVE_DIR:-../results/grokking_v2_seeds}
COMMON="--p 113 --train_frac 0.3 --lr 1e-3 --wd 1.0 --epochs ${EPOCHS} --log_every ${LOG_EVERY} --save_every 0 --save_dir ${SAVE_DIR}"
FILTER="--rank 200 --decay 0.99 --warmup 100"

for S in ${SEEDS}; do
  echo "=== seed ${S}: [1/3] adamw baseline ==="
  python3 -u run_grokking.py --optimizer adamw ${COMMON} --seed ${S} --name adamw_s${S}

  echo "=== seed ${S}: [2/3] filter + adamw ==="
  python3 -u run_grokking.py --optimizer weightcov_adamw ${COMMON} ${FILTER} --seed ${S} --name filter_adamw_s${S}

  echo "=== seed ${S}: [3/3] switch adamw -> filter @0.99 (wd kept) ==="
  python3 -u run_grokking.py --optimizer weightcov_adamw ${COMMON} ${FILTER} --switch_at 0.99 --seed ${S} --name switch_adamw_s${S}
done

echo "=== ALL GROKKING V2 SEED RUNS COMPLETE ==="

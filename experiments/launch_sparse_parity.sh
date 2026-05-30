#!/usr/bin/env bash
# Sparse parity grokking: AdamW vs filter+AdamW, 3 seeds. Run from experiments/.
set -euo pipefail
SEEDS=${SEEDS:-"0 1 2"}
EPOCHS=${EPOCHS:-20000}
SAVE=${SAVE:-../results/sparse_parity}
for S in ${SEEDS}; do
  echo "=== seed ${S}: adamw ==="
  python3 -u sparse_parity.py --mode adamw --seed ${S} --epochs ${EPOCHS} --save_dir ${SAVE} --name adamw_s${S}
  echo "=== seed ${S}: ours ==="
  python3 -u sparse_parity.py --mode ours  --seed ${S} --epochs ${EPOCHS} --save_dir ${SAVE} --name ours_s${S}
done
echo "=== ALL SPARSE PARITY RUNS COMPLETE ==="

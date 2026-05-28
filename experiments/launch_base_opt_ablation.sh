#!/bin/bash
# Ablation: does Adam's momentum interfere with our spectral filter?
# Compares filter+SGD vs filter+SGD(momentum) vs filter+Adam,
# plus baselines without filter.

set -e
cd "$(dirname "$0")"

EPOCHS=20
RANK=200
DECAY=0.99
WARMUP=100

# ============================================================
# Phase 0: Quick LR sweep for SGD variants (5 epochs, clean MNIST)
# ============================================================
echo "=== Phase 0: LR Sweep for SGD ==="
LR_SWEEP_DIR="../results/weight_covariance_v2/base_opt_ablation/lr_sweep"
mkdir -p "$LR_SWEEP_DIR"

for lr in 0.005 0.01 0.05; do
  for base in sgd sgdm; do
    name="lr_${base}_${lr}"
    echo "  ${base} lr=${lr}..."
    python3 -u run_single_weight_cov_v2.py \
      --mode baseline --base_optimizer $base --dataset standard \
      --epochs 5 --lr $lr \
      --save_dir "$LR_SWEEP_DIR" --name "$name" 2>&1 | tail -2
  done
done

echo ""
echo "=== LR Sweep Results ==="
python3 -u -c "
import json, os
d = '$LR_SWEEP_DIR'
for base in ['sgd', 'sgdm']:
    print(f'  {base}:')
    for lr in ['0.005', '0.01', '0.05']:
        f = os.path.join(d, f'lr_{base}_{lr}.json')
        if not os.path.exists(f):
            print(f'    lr={lr}: MISSING')
            continue
        m = json.load(open(f))['metrics'][-1]
        print(f'    lr={lr}: test={m[\"test_acc\"]:.4f}  train={m[\"train_acc\"]:.4f}')
"

echo ""
echo ">>> Check the results above and set BEST_SGD_LR / BEST_SGDM_LR below <<<"
echo ">>> Then re-run with: bash launch_base_opt_ablation.sh --phase1 <sgd_lr> <sgdm_lr>"
echo ""

# ============================================================
# Phase 1: Full ablation (requires LR arguments)
# ============================================================
if [ "$1" != "--phase1" ]; then
  echo "Phase 0 complete. Run phase 1 with best LRs."
  exit 0
fi

SGD_LR=${2:?"Usage: $0 --phase1 <sgd_lr> <sgdm_lr>"}
SGDM_LR=${3:?"Usage: $0 --phase1 <sgd_lr> <sgdm_lr>"}
ADAM_LR=0.001

SAVE_DIR="../results/weight_covariance_v2/base_opt_ablation"
mkdir -p "$SAVE_DIR"

echo "=== Phase 1: Full Base Optimizer Ablation ==="
echo "  SGD lr=${SGD_LR}, SGDm lr=${SGDM_LR}, Adam lr=${ADAM_LR}"
echo ""

for noise in 0.0 0.2 0.4; do
  echo "====== Label noise: ${noise} ======"

  # --- Baselines (no filter) ---
  name="n${noise}_baseline_adam"
  echo "  Adam baseline (noise=${noise})..."
  python3 -u run_single_weight_cov_v2.py \
    --mode adam --dataset standard --epochs $EPOCHS --lr $ADAM_LR \
    --label_noise $noise \
    --save_dir "$SAVE_DIR" --name "$name" 2>&1 | tail -2

  name="n${noise}_baseline_sgd"
  echo "  SGD baseline (noise=${noise})..."
  python3 -u run_single_weight_cov_v2.py \
    --mode baseline --base_optimizer sgd --dataset standard --epochs $EPOCHS --lr $SGD_LR \
    --label_noise $noise \
    --save_dir "$SAVE_DIR" --name "$name" 2>&1 | tail -2

  name="n${noise}_baseline_sgdm"
  echo "  SGDm baseline (noise=${noise})..."
  python3 -u run_single_weight_cov_v2.py \
    --mode baseline --base_optimizer sgdm --dataset standard --epochs $EPOCHS --lr $SGDM_LR \
    --label_noise $noise \
    --save_dir "$SAVE_DIR" --name "$name" 2>&1 | tail -2

  # --- With filter ---
  name="n${noise}_filter_adam"
  echo "  Filter+Adam (noise=${noise})..."
  python3 -u run_single_weight_cov_v2.py \
    --mode ours --base_optimizer adam --dataset standard --epochs $EPOCHS --lr $ADAM_LR \
    --rank $RANK --decay $DECAY --warmup $WARMUP \
    --label_noise $noise \
    --save_dir "$SAVE_DIR" --name "$name" 2>&1 | tail -2

  name="n${noise}_filter_sgd"
  echo "  Filter+SGD (noise=${noise})..."
  python3 -u run_single_weight_cov_v2.py \
    --mode ours --base_optimizer sgd --dataset standard --epochs $EPOCHS --lr $SGD_LR \
    --rank $RANK --decay $DECAY --warmup $WARMUP \
    --label_noise $noise \
    --save_dir "$SAVE_DIR" --name "$name" 2>&1 | tail -2

  name="n${noise}_filter_sgdm"
  echo "  Filter+SGDm (noise=${noise})..."
  python3 -u run_single_weight_cov_v2.py \
    --mode ours --base_optimizer sgdm --dataset standard --epochs $EPOCHS --lr $SGDM_LR \
    --rank $RANK --decay $DECAY --warmup $WARMUP \
    --label_noise $noise \
    --save_dir "$SAVE_DIR" --name "$name" 2>&1 | tail -2

  echo ""
done

# ============================================================
# Summary
# ============================================================
echo "=== SUMMARY ==="
python3 -u -c "
import json, os
d = '$SAVE_DIR'
noises = [0.0, 0.2, 0.4]
conditions = [
    ('baseline_sgd',  'SGD'),
    ('baseline_sgdm', 'SGDm'),
    ('baseline_adam',  'Adam'),
    ('filter_sgd',    'Filt+SGD'),
    ('filter_sgdm',   'Filt+SGDm'),
    ('filter_adam',    'Filt+Adam'),
]

header = f'{\"Noise\":<8}'
for _, label in conditions:
    header += f'{label:<12}'
print(header)
print('-' * (8 + 12 * len(conditions)))

for n in noises:
    row = f'{n:<8}'
    for cond, _ in conditions:
        f = os.path.join(d, f'n{n}_{cond}.json')
        if not os.path.exists(f):
            row += f'{\"MISSING\":<12}'
            continue
        m = json.load(open(f))['metrics'][-1]
        row += f'{m[\"test_acc\"]:<12.4f}'
    print(row)
"
echo "DONE"

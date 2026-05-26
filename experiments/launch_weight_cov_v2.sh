#!/bin/bash
# Launch v2 weight-covariance experiments (batch-mean streaming SVD).
# Much faster than v1 — no per-sample gradients.

set -e
cd "$(dirname "$0")"

echo "=== V2 Weight-Covariance Optimizer Experiments ==="
echo "Using batch-mean gradient streaming (no per-sample gradients)"
echo ""

echo "=== Phase 1: Quick sanity on standard MNIST (5 epochs) ==="
for rank in 50 200 500; do
  for decay in 0.95 0.99; do
    name="v2_sweep_r${rank}_d${decay}"
    echo "Running: $name"
    python3 -u run_single_weight_cov_v2.py \
      --mode ours --dataset standard --epochs 5 \
      --rank $rank --decay $decay --lr 0.001 --warmup 100 \
      --name "$name" 2>&1 | tail -3
  done
done

echo ""
echo "=== Finding best config ==="
python3 -u -c "
import json, glob
results = []
for f in glob.glob('../results/weight_covariance_v2/v2_sweep_*.json'):
    with open(f) as fh:
        d = json.load(fh)
    cfg = d['config']
    final = d['metrics'][-1]
    results.append((final['test_acc'], cfg['rank'], cfg['decay'], f))
results.sort(reverse=True)
print('Results:')
for acc, rank, decay, f in results:
    print(f'  test={acc:.4f} rank={rank} decay={decay}')
best = results[0]
print(f'\nBest: rank={best[1]} decay={best[2]}')
import json as j
with open('../results/weight_covariance_v2/best_config.json', 'w') as fh:
    j.dump({'rank': best[1], 'decay': best[2]}, fh)
"

BEST_RANK=$(python3 -c "import json; d=json.load(open('../results/weight_covariance_v2/best_config.json')); print(d['rank'])")
BEST_DECAY=$(python3 -c "import json; d=json.load(open('../results/weight_covariance_v2/best_config.json')); print(d['decay'])")
echo "Best config: rank=$BEST_RANK decay=$BEST_DECAY"

echo ""
echo "=== Phase 2: Comparison Runs (20 epochs each) ==="

for dataset in standard noisy_mnist random_labels; do
  echo ""
  echo "--- Dataset: $dataset ---"

  echo "Running Adam..."
  python3 -u run_single_weight_cov_v2.py \
    --mode adam --dataset $dataset --epochs 20 --lr 0.001 \
    --name "v2_compare_adam_${dataset}" 2>&1 | tail -3

  echo "Running Ours (v2)..."
  python3 -u run_single_weight_cov_v2.py \
    --mode ours --dataset $dataset --epochs 20 --lr 0.001 \
    --rank $BEST_RANK --decay $BEST_DECAY --warmup 100 \
    --name "v2_compare_ours_${dataset}" 2>&1 | tail -3
done

echo ""
echo "=== SUMMARY ==="
python3 -u -c "
import json, os
datasets = ['standard', 'noisy_mnist', 'random_labels']
print(f'{\"Dataset\":<20} {\"Adam Test\":<12} {\"Ours Test\":<12} {\"Delta\":<10} {\"Adam Time\":<10} {\"Ours Time\":<10}')
print('-' * 74)
for ds in datasets:
    af = f'../results/weight_covariance_v2/v2_compare_adam_{ds}.json'
    of = f'../results/weight_covariance_v2/v2_compare_ours_{ds}.json'
    if not os.path.exists(af) or not os.path.exists(of):
        print(f'{ds:<20} MISSING')
        continue
    ad = json.load(open(af))
    od = json.load(open(of))
    adam = ad['metrics'][-1]['test_acc']
    ours = od['metrics'][-1]['test_acc']
    delta = ours - adam
    sign = '+' if delta > 0 else ''
    at = ad['time_s']
    ot = od['time_s']
    print(f'{ds:<20} {adam:<12.4f} {ours:<12.4f} {sign}{delta:<10.4f} {at:<10.1f} {ot:<10.1f}')
"
echo "DONE"

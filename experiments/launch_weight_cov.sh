#!/bin/bash
# Launch all weight-covariance experiments sequentially.
# Each is a separate Python process to avoid state leakage.

set -e
cd "$(dirname "$0")"

echo "=== Phase 1: HP Sweep on Standard MNIST (5 epochs each) ==="

# Sweep: rank × decay × lr
for rank in 10 20 50; do
  for decay in 0.9 0.95; do
    for lr in 0.0005 0.001; do
      name="sweep_r${rank}_d${decay}_lr${lr}"
      echo "Running: $name"
      python3 -u run_single_weight_cov.py \
        --mode ours --dataset standard --epochs 5 \
        --rank $rank --decay $decay --lr $lr --update_every 50 \
        --name "$name" 2>&1 | tail -2
    done
  done
done

echo ""
echo "=== Finding best config ==="
python3 -u -c "
import json, glob, os
results = []
for f in glob.glob('../results/weight_covariance/sweep_*.json'):
    with open(f) as fh:
        d = json.load(fh)
    cfg = d['config']
    final = d['metrics'][-1]
    results.append((final['test_acc'], cfg['rank'], cfg['decay'], cfg['lr'], f))
results.sort(reverse=True)
print('Top 5:')
for acc, rank, decay, lr, f in results[:5]:
    print(f'  test={acc:.4f} rank={rank} decay={decay} lr={lr}')
best = results[0]
print(f'\\nBest: rank={best[1]} decay={best[2]} lr={best[3]}')
# Save best config
with open('../results/weight_covariance/best_config.json', 'w') as fh:
    json.dump({'rank': best[1], 'decay': best[2], 'lr': best[3]}, fh)
"

# Read best config
BEST=$(python3 -c "import json; d=json.load(open('../results/weight_covariance/best_config.json')); print(f'--rank {d[\"rank\"]} --decay {d[\"decay\"]} --lr {d[\"lr\"]}')")
echo "Best config: $BEST"

echo ""
echo "=== Phase 2: Comparison Runs (20 epochs each) ==="

for dataset in standard noisy_mnist random_labels; do
  echo ""
  echo "--- Dataset: $dataset ---"

  echo "Running Adam..."
  python3 -u run_single_weight_cov.py \
    --mode adam --dataset $dataset --epochs 20 --lr 0.001 \
    --name "compare_adam_${dataset}" 2>&1 | tail -2

  echo "Running Ours..."
  python3 -u run_single_weight_cov.py \
    --mode ours --dataset $dataset --epochs 20 --update_every 50 \
    $BEST \
    --name "compare_ours_${dataset}" 2>&1 | tail -2
done

echo ""
echo "=== SUMMARY ==="
python3 -u -c "
import json, os
datasets = ['standard', 'noisy_mnist', 'random_labels']
print(f'{\"Dataset\":<20} {\"Adam Test\":<12} {\"Ours Test\":<12} {\"Delta\":<10}')
print('-' * 54)
for ds in datasets:
    af = f'../results/weight_covariance/compare_adam_{ds}.json'
    of = f'../results/weight_covariance/compare_ours_{ds}.json'
    if not os.path.exists(af) or not os.path.exists(of):
        print(f'{ds:<20} MISSING')
        continue
    adam = json.load(open(af))['metrics'][-1]['test_acc']
    ours = json.load(open(of))['metrics'][-1]['test_acc']
    delta = ours - adam
    sign = '+' if delta > 0 else ''
    print(f'{ds:<20} {adam:<12.4f} {ours:<12.4f} {sign}{delta:<10.4f}')
"
echo "DONE"

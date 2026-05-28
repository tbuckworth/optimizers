#!/bin/bash
# Sweep: rank × label_noise_frac, comparing ours vs Adam.
# Tests whether our optimizer resists memorizing corrupted labels.

set -e
cd "$(dirname "$0")"

echo "=== Label Noise Sweep: rank × noise_frac ==="
echo ""

EPOCHS=20
SAVE_DIR="../results/weight_covariance_v2/label_noise"
mkdir -p "$SAVE_DIR"

for noise in 0.0 0.2 0.4; do
  echo ""
  echo "====== Label noise: ${noise} ======"

  # Adam baseline
  name="ln${noise}_adam"
  echo "Running Adam (noise=${noise})..."
  python3 -u run_single_weight_cov_v2.py \
    --mode adam --dataset standard --epochs $EPOCHS --lr 0.001 \
    --label_noise $noise \
    --save_dir "$SAVE_DIR" --name "$name" 2>&1 | tail -3

  # Ours at different ranks
  for rank in 50 100 200; do
    name="ln${noise}_ours_r${rank}"
    echo "Running Ours rank=${rank} (noise=${noise})..."
    python3 -u run_single_weight_cov_v2.py \
      --mode ours --dataset standard --epochs $EPOCHS --lr 0.001 \
      --rank $rank --decay 0.99 --warmup 100 \
      --label_noise $noise \
      --save_dir "$SAVE_DIR" --name "$name" 2>&1 | tail -3
  done
done

echo ""
echo "=== SUMMARY ==="
python3 -u -c "
import json, os
save_dir = '$SAVE_DIR'
noises = [0.0, 0.2, 0.4]
ranks = [50, 100, 200]

header = f'{\"Noise\":<8} {\"Adam\":<12}'
for r in ranks:
    header += f'{\"r=\"+str(r):<12}'
header += f'{\"Best Ours\":<12} {\"Delta\":<10}'
print(header)
print('-' * (8 + 12 + 12*3 + 12 + 10))

for n in noises:
    af = os.path.join(save_dir, f'ln{n}_adam.json')
    if not os.path.exists(af):
        print(f'{n:<8} MISSING')
        continue
    adam = json.load(open(af))['metrics'][-1]['test_acc']
    row = f'{n:<8} {adam:<12.4f}'
    best_ours = 0
    for r in ranks:
        of = os.path.join(save_dir, f'ln{n}_ours_r{r}.json')
        if not os.path.exists(of):
            row += f'{\"MISSING\":<12}'
            continue
        ours = json.load(open(of))['metrics'][-1]['test_acc']
        row += f'{ours:<12.4f}'
        best_ours = max(best_ours, ours)
    delta = best_ours - adam
    sign = '+' if delta > 0 else ''
    row += f'{best_ours:<12.4f} {sign}{delta:<10.4f}'
    print(row)
"
echo "DONE"

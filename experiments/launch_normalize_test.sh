#!/usr/bin/env bash
# Fair A/B/C: which BASIS to project onto? covariance vs correlation vs spectral.
# ONE knob changed (--normalize). Everything else identical: same MLP, base=adam,
# lr=1e-3, rank=200, decay=0.99, warmup=100, epochs=30, MNIST @ 50% label noise.
# Pre-registered metric: best & final test accuracy, mean over 3 seeds.
set -u
cd "$(dirname "$0")"
OUT=../results/mnist_normalize
mkdir -p "$OUT"
NOISE=0.5; EP=30; RANK=200; DECAY=0.99; WARM=100
COMMON="--dataset standard --label_noise $NOISE --lr 1e-3 --epochs $EP --base_optimizer adam --save_dir $OUT"

for S in 0 1 2; do
  echo "=== adam  seed=$S ==="
  python3 -u run_single_weight_cov_v2.py --mode adam $COMMON --seed $S --name "adam_s$S" 2>&1 | tail -2
  for NRM in none var degree; do
    echo "=== ours normalize=$NRM  seed=$S ==="
    python3 -u run_single_weight_cov_v2.py --mode ours $COMMON --rank $RANK --decay $DECAY \
      --warmup $WARM --normalize $NRM --seed $S --name "ours_${NRM}_s$S" 2>&1 | tail -2
  done
done
echo "ALL DONE"

#!/usr/bin/env bash
# CIFAR-10 effective-rank sweep (corrected).
#
# Motivation: at fixed decay=0.99 the EMA covariance can only hold ~1/(1-decay)=~100
# directions, so raising `rank` alone (200 vs 1000 vs 4000) does NOTHING — the basis
# saturates at ~100 live columns regardless. To genuinely raise the effective rank we
# must raise decay AND rank together. This sweep tests whether higher *effective* rank
# recovers clean-data accuracy (the 0%-noise underfitting) while keeping noise robustness.
#
# Part A — degeneracy control: decay 0.99 fixed, rank 200/1000/4000 at 80% noise.
#          Expectation: ~identical curves (proves rank-alone is capped by decay).
# Part B — effective-rank sweep: (decay,rank) = (0.99,200) (0.997,500) (0.999,1200)
#          across 0/40/80% noise, plus adam baselines.
set -u
cd "$(dirname "$0")"
OUT=../results/cifar_noise/ranksweep
mkdir -p "$OUT"
EP=40
run () { # mode noise decay rank name
  echo "=== $5  (mode=$1 noise=$2 decay=$3 rank=$4) ==="
  python3 -u run_cifar_noise.py --mode "$1" --noise "$2" --decay "$3" --rank "$4" \
    --epochs $EP --seed 0 --name "$5" --save_dir "$OUT" 2>&1 | tail -3
}

# adam baselines
run adam 0.0  0.99 200 adam_n0
run adam 0.4  0.99 200 adam_n40
run adam 0.8  0.99 200 adam_n80

# Part B: effective-rank sweep
for NZ in 0.0 0.4 0.8; do
  TAG=$(python3 -c "print(int(float('$NZ')*100))")
  run ours "$NZ" 0.99  200  "ours_d99_r200_n${TAG}"
  run ours "$NZ" 0.997 500  "ours_d997_r500_n${TAG}"
  run ours "$NZ" 0.999 1200 "ours_d999_r1200_n${TAG}"
done

# Part A: degeneracy control at 80% noise (decay fixed, rank varies)
run ours 0.8 0.99 1000 ours_d99_r1000_n80
run ours 0.8 0.99 4000 ours_d99_r4000_n80

echo "ALL DONE"

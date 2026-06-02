#!/usr/bin/env bash
# CIFAR-10 ADAPTIVE-rank sweep (Titus's energy-threshold idea, asked from two directions).
#
# Idea: instead of a fixed rank, each step keep the smallest #top-eigenvectors that
# capture X% of the spectral energy. In the noise regime the low-eigenvalue tail IS
# the label noise, so an energy rule should drop it automatically and slide the
# clean/noise tradeoff better than any single fixed rank.
#
# Setup: high decay=0.999 (high ceiling so the THRESHOLD decides the rank, not the cap),
#        high rank cap=1200, energy_threshold in {0.90,0.95,0.99}, noise {0,40,80}.
#        kept_rank is logged per-epoch to test the "broaden-then-narrow" trajectory.
#
# Reference (already in ../results/cifar_noise/ranksweep/): adam + fixed-rank Pareto front.
set -u
cd "$(dirname "$0")"
OUT=../results/cifar_noise/adaptive
mkdir -p "$OUT"
EP=40
run () { # noise energy name
  echo "=== $3  (noise=$1 energy=$2 decay=0.999 rankcap=1200) ==="
  python3 -u run_cifar_noise.py --mode ours --noise "$1" --decay 0.999 --rank 1200 \
    --energy_threshold "$2" --epochs $EP --seed 0 --name "$3" --save_dir "$OUT" 2>&1 | tail -4
}

for NZ in 0.0 0.4 0.8; do
  TAG=$(python3 -c "print(int(float('$NZ')*100))")
  run "$NZ" 0.90 "adapt_e90_n${TAG}"
  run "$NZ" 0.95 "adapt_e95_n${TAG}"
  run "$NZ" 0.99 "adapt_e99_n${TAG}"
done

echo "ALL DONE"

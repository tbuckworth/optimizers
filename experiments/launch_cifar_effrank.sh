#!/usr/bin/env bash
# CIFAR-10 adaptive EFFECTIVE-RANK sweep (the faithful version of Titus's idea).
# Track a broad basis (rank cap 1200, decay 0.999) but project each step onto the top
# round(effective_rank) directions — decoupled estimation vs projection, no ratchet.
# Also a 'gap' (log-spectrum elbow) variant. kept_rank (= proj_k) logged per epoch.
set -u
cd "$(dirname "$0")"
OUT=../results/cifar_noise/effrank
mkdir -p "$OUT"
EP=40
run () { # noise adaptive name
  echo "=== $3  (noise=$1 adaptive=$2 decay=0.999 rankcap=1200) ==="
  python3 -u run_cifar_noise.py --mode ours --noise "$1" --decay 0.999 --rank 1200 \
    --adaptive "$2" --epochs $EP --seed 0 --name "$3" --save_dir "$OUT" 2>&1 | tail -4
}
for NZ in 0.0 0.4 0.8; do
  TAG=$(python3 -c "print(int(float('$NZ')*100))")
  run "$NZ" effrank "effrank_n${TAG}"
  run "$NZ" gap     "gap_n${TAG}"
done
echo "ALL DONE"

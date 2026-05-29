#!/usr/bin/env bash
# Long-horizon 90% label-noise run: does our filter resist the late memorization
# collapse that kills the baselines, and does it keep improving with more epochs?
#
# At 90% noise the 20-epoch sweep showed: adam/lora peak early (ep2-6) then collapse
# as they memorize noise; ours was still climbing at ep16-20. This extends to 60
# epochs to see the full divergence.
#
# Also sweeps rank for ours (50/100/200) to test the "rank too high" hypothesis --
# lower rank = more aggressive filtering, which should help more at extreme noise.
#
# Conditions x 3 seeds = 18 runs. Run from experiments/. Sequential.
set -euo pipefail

SEEDS=${SEEDS:-"42 43 44"}
EPOCHS=${EPOCHS:-60}
SAVE_DIR=${SAVE_DIR:-../results/weight_covariance_v2/noise90_long}
C="--label_noise 0.9 --epochs ${EPOCHS} --save_dir ${SAVE_DIR} --data_dir ./data --decay 0.99 --warmup 100"

for S in ${SEEDS}; do
  echo "=== seed ${S}: adam ==="
  python3 -u run_single_weight_cov_v2.py --mode adam ${C} --seed ${S} --name adam_n90_s${S}

  echo "=== seed ${S}: lora ==="
  python3 -u run_single_weight_cov_v2.py --mode lora --lora_rank 32 ${C} --seed ${S} --name lora_n90_s${S}

  echo "=== seed ${S}: random_subspace r200 ==="
  python3 -u run_single_weight_cov_v2.py --mode random_subspace --rank 200 ${C} --seed ${S} --name random_subspace_n90_s${S}

  echo "=== seed ${S}: ours r200 ==="
  python3 -u run_single_weight_cov_v2.py --mode ours --rank 200 ${C} --seed ${S} --name ours_r200_n90_s${S}

  echo "=== seed ${S}: ours r100 ==="
  python3 -u run_single_weight_cov_v2.py --mode ours --rank 100 ${C} --seed ${S} --name ours_r100_n90_s${S}

  echo "=== seed ${S}: ours r50 ==="
  python3 -u run_single_weight_cov_v2.py --mode ours --rank 50 ${C} --seed ${S} --name ours_r50_n90_s${S}
done

echo "=== ALL NOISE90 LONG RUNS COMPLETE ==="

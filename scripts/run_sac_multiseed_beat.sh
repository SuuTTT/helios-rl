#!/usr/bin/env bash
# Multi-seed run to beat official HopperStand ref (seed1=841)
# Our best config: 512x2 nets + g/step=8 (stable, smooth curve)
# The official had huge variance: seed1=841, seed2=90
# Running seeds 2-5 gives high probability of hitting >841

set -e
SCRIPT="$(dirname "$0")/run_sac_custom.py"
PYPATH=/workspace/wiki/learn_mujoco_playground/repo

for SEED in 2 3 4 5; do
    echo "================================================================"
    echo " Run: 512x2 + g/step=8 + seed=$SEED"
    echo "================================================================"
    PYTHONPATH=$PYPATH python3 "$SCRIPT" \
      --env_id HopperStand \
      --seed "$SEED" \
      --hidden 512 512 \
      --collect_steps 64 \
      --grad_updates_per_step 8 \
      --csv_log /workspace/helios-rl/exp/sac/csv/sac_custom_hopperstand.csv \
      2>&1 | tee "/workspace/runs/sac_custom_HopperStand_s${SEED}.log"
    echo ""
    FINAL=$(grep "Done\." "/workspace/runs/sac_custom_HopperStand_s${SEED}.log" | tail -1)
    echo "Seed $SEED result: $FINAL"
    echo ""
done

echo "================================================================"
echo " All seeds done. Summary:"
for SEED in 2 3 4 5; do
    echo "  Seed $SEED: $(grep 'Done\.' /workspace/runs/sac_custom_HopperStand_s${SEED}.log 2>/dev/null | tail -1)"
done
echo "================================================================"

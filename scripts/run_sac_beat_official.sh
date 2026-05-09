#!/usr/bin/env bash
# Beat official Brax SAC on HopperStand (ref: reward=841.253 @ 10M steps)
#
# Run A: 256×2 networks + g/step=16  (official size, 2× gradient ratio)
# Run B: 256×2 networks + g/step=16 + target_entropy=-4  (more exploration)
#
# Both use GPU replay buffer for full-speed training.

set -e
SCRIPT="$(dirname "$0")/run_sac_custom.py"
PYPATH=/workspace/wiki/learn_mujoco_playground/repo

echo "================================================================"
echo " Run A: 256x2 + g/step=16 + seed=1"
echo "================================================================"
PYTHONPATH=$PYPATH python3 "$SCRIPT" \
  --env_id HopperStand \
  --seed 1 \
  --hidden 256 256 \
  --collect_steps 64 \
  --grad_updates_per_step 16 \
  --csv_log /workspace/helios-rl/exp/sac/csv/sac_custom_hopperstand.csv \
  2>&1 | tee /workspace/runs/sac_beat_A_HopperStand_s1.log

echo ""
echo "================================================================"
echo " Run B: 256x2 + g/step=16 + target_entropy=-4 + seed=1"
echo "================================================================"
PYTHONPATH=$PYPATH python3 "$SCRIPT" \
  --env_id HopperStand \
  --seed 1 \
  --hidden 256 256 \
  --collect_steps 64 \
  --grad_updates_per_step 16 \
  --target-entropy -4.0 \
  --csv_log /workspace/helios-rl/exp/sac/csv/sac_custom_hopperstand.csv \
  2>&1 | tee /workspace/runs/sac_beat_B_HopperStand_s1.log

echo ""
echo "================================================================"
echo " All done. Check logs:"
echo "   Run A: /workspace/runs/sac_beat_A_HopperStand_s1.log"
echo "   Run B: /workspace/runs/sac_beat_B_HopperStand_s1.log"
echo "================================================================"

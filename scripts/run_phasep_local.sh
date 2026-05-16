#!/usr/bin/env bash
# Phase-p (iteration 4, path 1) on the LOCAL 4070 Ti.
#
# Hypothesis: 500k env steps of random actions (20× the current 25k default)
# fills the replay buffer with diverse foot-strike transitions. Q learns
# their high reward; policy gradient direction pulls toward foot-strikes
# regardless of the policy's own (knee-walk) rollout distribution. Bypasses
# the "policy stuck in its own distribution" trap that limits the 15% surge
# hit rate.
#
# Knobs vs Phase-m (= curriculum smoothing + Python-conditional graph):
#   --expl_until 500000   (was 25000)

set -u
set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TDMPC_GLASS_OUTPUT_TAG=phasep
export MUJOCO_GL=${MUJOCO_GL:-egl}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/phasep
mkdir -p "$LOG_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-10000000}
SEEDS=${SEEDS:-"1 2 3 4 5"}
SMOOTH=${SMOOTH:-0.001}
WARMUP=${WARMUP:-250000}
EXPL_UNTIL=${EXPL_UNTIL:-500000}
PATIENCE=${PATIENCE:-1500000}

echo "[phasep] start $(date -u +%FT%TZ) seeds=[$SEEDS] smooth=$SMOOTH warmup=$WARMUP expl_until=$EXPL_UNTIL" \
    | tee -a "$LOG_DIR/queue.log"
echo "[phasep] config: Phase-m + EXPL_UNTIL=$EXPL_UNTIL (20× default 25k)" \
    | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phasep] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass --tasks HopperHop \
    --total_steps "$TOTAL_STEPS" --seed "$seed" \
    --glass_proto_temperature 0.7 --glass_assign_logits_init_scale 0.5 --glass_stopgrad_graph true \
    --expl_until "$EXPL_UNTIL" \
    --latent_action_smooth_coef "$SMOOTH" \
    --latent_smooth_warmup_env_steps "$WARMUP" \
    --early_stop_patience "$PATIENCE" \
    --no_plot 2>&1 | tee -a "$log"
  echo "[phasep] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[phasep] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"

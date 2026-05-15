#!/usr/bin/env bash
# Phase-m on the LOCAL 4070 Ti.
#
# Hypothesis: the K=3 basin flip on seeds 4,5 in Phase-f and Phase-j was
# caused by the *existence* of the smoothing vmap-over-pi in the JIT graph,
# even when its coefficient was 0 (Phase-j during warmup). XLA's floating-
# point order differs between a graph that has the smoothing op and one
# that doesn't, and that tiny numerical perturbation flips basin-fragile
# seeds.
#
# Phase-m uses the new `smoothing_enabled: bool` knob (Python-time gate)
# in make_update_fn. Pre-curriculum: smoothing_enabled=False → graph
# matches Phase 1b exactly → expect 5/5 K=4 basin like blog reported.
# Post-curriculum: smoothing_enabled=True → smoothing kicks in.
#
# Knobs identical to Phase-j; the difference is purely in the JIT-graph
# structure during warmup.

set -u
set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TDMPC_GLASS_OUTPUT_TAG=phasem
export MUJOCO_GL=${MUJOCO_GL:-egl}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/phasem
mkdir -p "$LOG_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-10000000}
SEEDS=${SEEDS:-"1 2 3 4 5"}
SMOOTH=${SMOOTH:-0.001}
WARMUP=${WARMUP:-250000}
PATIENCE=${PATIENCE:-1500000}

echo "[phasem] start $(date -u +%FT%TZ) seeds=[$SEEDS] smooth=$SMOOTH warmup=$WARMUP" \
    | tee -a "$LOG_DIR/queue.log"
echo "[phasem] config: Phase-j knobs + Python-conditional smoothing (graph matches Phase 1b during warmup)" \
    | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phasem] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass --tasks HopperHop \
    --total_steps "$TOTAL_STEPS" --seed "$seed" \
    --glass_proto_temperature 0.7 --glass_assign_logits_init_scale 0.5 --glass_stopgrad_graph true \
    --latent_action_smooth_coef "$SMOOTH" \
    --latent_smooth_warmup_env_steps "$WARMUP" \
    --early_stop_patience "$PATIENCE" \
    --no_plot 2>&1 | tee -a "$log"
  echo "[phasem] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[phasem] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"

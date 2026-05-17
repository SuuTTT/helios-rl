#!/usr/bin/env bash
# Phase-x seed 3 (Path 9 — NS=2048 MPPI) on LOCAL 4070 Ti.
# Path 9 is iteration 5's leading benchmark-fair path. We had:
#   seed 1 v1 (2x3060) → 453.2 @ 4.25M, then OOM-killed
#   seed 1 v2 (2x3060) → currently 278 @ 2M, climbing
#   seed 2    (2x3060) → OOM-killed twice, only ~6 peak
# Need a third seed on STABLE local hardware (4070 Ti 12GB).
set -u; set +e
REPO=/root/helios-rl
cd "$REPO" || exit 1
source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/workspace/wiki/learn_mujoco_playground/repo:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export TDMPC_GLASS_OUTPUT_TAG=phasex_local
export MUJOCO_GL=egl
LOG_DIR=$REPO/exp/tdmpc_glass/logs/phasex_local
mkdir -p "$LOG_DIR"
SEED=${SEED:-3}
log="$LOG_DIR/HopperHop_seed_${SEED}.log"
echo "[phasex_local] === seed=${SEED} start $(date -u +%FT%TZ) NS=2048 ===" | tee "$log" | tee -a "$LOG_DIR/queue.log"
python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass --tasks HopperHop \
    --total_steps 10000000 --seed "$SEED" \
    --glass_proto_temperature 0.7 --glass_assign_logits_init_scale 0.5 --glass_stopgrad_graph true \
    --mppi_n_samples 2048 \
    --expl_until 500000 \
    --latent_action_smooth_coef 0.001 \
    --latent_smooth_warmup_env_steps 250000 \
    --early_stop_patience 3000000 \
    --no_plot 2>&1 | tee -a "$log"
echo "[phasex_local] === seed=${SEED} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"

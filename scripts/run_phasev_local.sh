#!/usr/bin/env bash
# Phase-v (iteration 4 / Path 7) on the LOCAL 4070 Ti.
#
# Cluster-id as policy/Q observation. The soft cluster distribution
# S[n_star(z)] (K=8 dim) is concatenated to z before pi/q lookups.
# stop_gradient on the cluster computation so Glass keeps its own
# structural loss — pi/q just see "wider z". Fully benchmark-fair
# (no reward modification, no env modification).
#
# Hypothesis: when policy can see which gait phase it's currently
# in, it can commit to a coherent hop pattern instead of oscillating
# at gait transitions. Should help the "stuck seed" failure mode where
# K=3 basin policies can't produce reliable hopping.
#
# Knobs vs Phase-p baseline:
#   --use_cluster_obs          (NEW: Path 7 flag, augments z with K-dim soft S)
#   --expl_until      500000   (Path 1 winner setting)
#
# 3 seeds x 10M cap + 3M patience.

set -u
set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TDMPC_GLASS_OUTPUT_TAG=phasev
export MUJOCO_GL=${MUJOCO_GL:-egl}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/phasev
mkdir -p "$LOG_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-10000000}
SEEDS=${SEEDS:-"1 2 3"}
EXPL_UNTIL=${EXPL_UNTIL:-500000}
SMOOTH=${SMOOTH:-0.001}
WARMUP=${WARMUP:-250000}
PATIENCE=${PATIENCE:-3000000}

echo "[phasev] start $(date -u +%FT%TZ) seeds=[$SEEDS] use_cluster_obs=1 expl_until=$EXPL_UNTIL" \
    | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phasev] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass --tasks HopperHop \
    --total_steps "$TOTAL_STEPS" --seed "$seed" \
    --glass_proto_temperature 0.7 --glass_assign_logits_init_scale 0.5 --glass_stopgrad_graph true \
    --use_cluster_obs \
    --expl_until "$EXPL_UNTIL" \
    --latent_action_smooth_coef "$SMOOTH" \
    --latent_smooth_warmup_env_steps "$WARMUP" \
    --early_stop_patience "$PATIENCE" \
    --no_plot 2>&1 | tee -a "$log"
  echo "[phasev] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[phasev] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"

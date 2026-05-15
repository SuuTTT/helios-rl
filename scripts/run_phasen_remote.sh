#!/usr/bin/env bash
# Phase-n on the REMOTE 4060.
#
# Hypothesis: K=3 basin for seeds 4,5 comes from soft-assignment ambiguity
# at proto_temperature=0.7. Lowering to 0.4 makes the assignment sharper —
# each latent commits more decisively to a single prototype — which should
# encourage all seeds to crystallise onto the same K=4 attractor.
#
# Zero code change — uses existing --glass_proto_temperature flag.
#
# Phase-j knobs PLUS:
#   --glass_proto_temperature 0.4   (was 0.7)
#   (everything else identical to Phase-j: curriculum + smooth + early-stop)

set -u
set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.55}
export TDMPC_GLASS_OUTPUT_TAG=phasen
export MUJOCO_GL=${MUJOCO_GL:-egl}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/phasen
mkdir -p "$LOG_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-10000000}
SEEDS=${SEEDS:-"1 2 3 4 5"}
PROTO_T=${PROTO_T:-0.4}
SMOOTH=${SMOOTH:-0.001}
WARMUP=${WARMUP:-250000}
PATIENCE=${PATIENCE:-1500000}

echo "[phasen] start $(date -u +%FT%TZ) seeds=[$SEEDS] proto_T=$PROTO_T smooth=$SMOOTH warmup=$WARMUP" \
    | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phasen] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass --tasks HopperHop \
    --total_steps "$TOTAL_STEPS" --seed "$seed" \
    --glass_proto_temperature "$PROTO_T" \
    --glass_assign_logits_init_scale 0.5 --glass_stopgrad_graph true \
    --latent_action_smooth_coef "$SMOOTH" \
    --latent_smooth_warmup_env_steps "$WARMUP" \
    --early_stop_patience "$PATIENCE" \
    --no_plot 2>&1 | tee -a "$log"
  echo "[phasen] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[phasen] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"

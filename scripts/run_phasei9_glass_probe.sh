#!/usr/bin/env bash
# Iteration 9 generic Glass sentinel probe launcher.
#
# Use one seed per queue task. All knobs below are env-driven so multiple
# flag-only probes can safely sit in the queue under the same code identity.

set -u; set +e

REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}
export MUJOCO_GL=${MUJOCO_GL:-egl}
[[ -n "${TMPDIR:-}" ]] && mkdir -p "$TMPDIR"

PROBE_ID=${PROBE_ID:-phasei9_glass_probe}
SEEDS=${SEEDS:-1}
TOTAL_STEPS=${TOTAL_STEPS:-10000000}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-3000000}
K_UPDATE=${K_UPDATE:-128}
MPPI_NS=${MPPI_NS:-2048}
EXPL_UNTIL=${EXPL_UNTIL:-500000}
TEMP_STABILITY=${TEMP_STABILITY:-0.01}
GLASS_WARMUP=${GLASS_WARMUP:-100000}
GLASS_DECAY=${GLASS_DECAY:-0}
PROTO_TEMP=${PROTO_TEMP:-1.0}
ASSIGN_SCALE=${ASSIGN_SCALE:-1.0}
STOPGRAD=${STOPGRAD:-true}
NUM_PROTOTYPES=${NUM_PROTOTYPES:-32}
NUM_CLUSTERS=${NUM_CLUSTERS:-8}
NUM_SUPER_CLUSTERS=${NUM_SUPER_CLUSTERS:-0}
LAMBDA_SUPER_SE=${LAMBDA_SUPER_SE:-0.0}
LAMBDA_SUPER_BALANCE=${LAMBDA_SUPER_BALANCE:-0.0}
LATENT_SMOOTH=${LATENT_SMOOTH:-0.001}
LATENT_SMOOTH_WARMUP=${LATENT_SMOOTH_WARMUP:-250000}
CONTROLLER_ARBITRATION=${CONTROLLER_ARBITRATION:-none}
ARBITRATION_MARGIN=${ARBITRATION_MARGIN:-0}
CODE_SHA=${CODE_SHA:-$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)}
TAG_SHA=${CODE_SHA//[^A-Za-z0-9_.-]/_}
TAG_PROBE=${PROBE_ID//[^A-Za-z0-9_.-]/_}

export TDMPC_GLASS_OUTPUT_TAG=${TDMPC_GLASS_OUTPUT_TAG:-${TAG_PROBE}_${TAG_SHA}}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/$TDMPC_GLASS_OUTPUT_TAG
mkdir -p "$LOG_DIR"
status=0

{
  echo "[${PROBE_ID}] start $(date -u +%FT%TZ) seeds=${SEEDS}"
  echo "[${PROBE_ID}] git_sha=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "[${PROBE_ID}] code_sha_env=${CODE_SHA:-unset}"
  echo "[${PROBE_ID}] output_tag=${TDMPC_GLASS_OUTPUT_TAG}"
  echo "[${PROBE_ID}] total_steps=${TOTAL_STEPS} early_stop_patience=${EARLY_STOP_PATIENCE}"
  echo "[${PROBE_ID}] K=${K_UPDATE} NS=${MPPI_NS} expl_until=${EXPL_UNTIL}"
  echo "[${PROBE_ID}] temp_stability=${TEMP_STABILITY} glass_warmup=${GLASS_WARMUP} glass_decay=${GLASS_DECAY}"
  echo "[${PROBE_ID}] proto_temp=${PROTO_TEMP} assign_scale=${ASSIGN_SCALE} stopgrad=${STOPGRAD}"
  echo "[${PROBE_ID}] num_prototypes=${NUM_PROTOTYPES} num_clusters=${NUM_CLUSTERS} num_super_clusters=${NUM_SUPER_CLUSTERS}"
  echo "[${PROBE_ID}] lambda_super_se=${LAMBDA_SUPER_SE} lambda_super_balance=${LAMBDA_SUPER_BALANCE}"
  echo "[${PROBE_ID}] latent_smooth=${LATENT_SMOOTH} latent_smooth_warmup=${LATENT_SMOOTH_WARMUP}"
  echo "[${PROBE_ID}] controller_arbitration=${CONTROLLER_ARBITRATION} arbitration_margin=${ARBITRATION_MARGIN}"
} | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[${PROBE_ID}] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass \
    --tasks HopperHop \
    --total_steps "$TOTAL_STEPS" \
    --seed "$seed" \
    --k_update "$K_UPDATE" \
    --mppi_n_samples "$MPPI_NS" \
    --expl_until "$EXPL_UNTIL" \
    --latent_action_smooth_coef "$LATENT_SMOOTH" \
    --latent_smooth_warmup_env_steps "$LATENT_SMOOTH_WARMUP" \
    --glass_warmup_env_steps "$GLASS_WARMUP" \
    --glass_decay_steps "$GLASS_DECAY" \
    --glass_proto_temperature "$PROTO_TEMP" \
    --glass_assign_logits_init_scale "$ASSIGN_SCALE" \
    --glass_stopgrad_graph "$STOPGRAD" \
    --glass_num_prototypes "$NUM_PROTOTYPES" \
    --glass_num_clusters "$NUM_CLUSTERS" \
    --glass_num_super_clusters "$NUM_SUPER_CLUSTERS" \
    --glass_lambda_super_se "$LAMBDA_SUPER_SE" \
    --glass_lambda_super_balance "$LAMBDA_SUPER_BALANCE" \
    --glass_lambda_temp_stability "$TEMP_STABILITY" \
    --early_stop_patience "$EARLY_STOP_PATIENCE" \
    --controller_arbitration "$CONTROLLER_ARBITRATION" \
    --arbitration_margin "$ARBITRATION_MARGIN" \
    --no_plot 2>&1 | tee -a "$log"
  seed_status=${PIPESTATUS[0]}
  if [[ "$seed_status" -ne 0 ]]; then
    status="$seed_status"
  fi
  echo "[${PROBE_ID}] === seed=${seed} done status=${seed_status} $(date -u +%FT%TZ) ===" \
    | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done

echo "[${PROBE_ID}] all done status=${status} at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
exit "$status"

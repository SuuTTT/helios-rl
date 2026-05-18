#!/usr/bin/env bash
# Iter 6 auto-queue: for each remote box, when GPU is idle, pop next experiment
# from its queue file and launch it. Marks completed lines with `# DONE <ts>`.
#
# Queue line format: <ssh_port>|<ssh_host>|<remote_launcher_script>|<env_vars>
# Comments start with #.
#
# Queue files (in repo/scripts/queues/):
#   ssh6_4060.queue     — 4060
#   ssh3_3060ti.queue   — 3060Ti
#   ssh17637_gpu0.queue — 2x3060 GPU 0
#   ssh17637_gpu1.queue — 2x3060 GPU 1
#
# Logic: every 5 min poll each box; if NO run_benchmark process found,
# claim the next non-DONE queue line and launch it.

set -u
QUEUE_DIR=/root/helios-rl/scripts/queues
ts() { date -u +%H:%M:%SZ; }

# Each box-tag → (port, host, GPU index for CUDA_VISIBLE_DEVICES check, queue file)
declare -A BOX_PORT BOX_HOST BOX_GPUMASK
BOX_PORT[ssh6_4060]=11115;        BOX_HOST[ssh6_4060]=ssh6.vast.ai;   BOX_GPUMASK[ssh6_4060]=""
BOX_PORT[ssh3_3060ti]=11271;      BOX_HOST[ssh3_3060ti]=ssh3.vast.ai; BOX_GPUMASK[ssh3_3060ti]=""
BOX_PORT[ssh17637_gpu0]=17637;    BOX_HOST[ssh17637_gpu0]=78.83.187.54; BOX_GPUMASK[ssh17637_gpu0]="CUDA_VISIBLE_DEVICES=0"
BOX_PORT[ssh17637_gpu1]=17637;    BOX_HOST[ssh17637_gpu1]=78.83.187.54; BOX_GPUMASK[ssh17637_gpu1]="CUDA_VISIBLE_DEVICES=1"

# For shared-host boxes (ssh17637 has 2 slots), the "is busy" check needs to be
# slot-specific — we check the GPU mem usage on the specific CUDA index.
is_box_busy() {
  local tag=$1 port=${BOX_PORT[$1]} host=${BOX_HOST[$1]} gpumask=${BOX_GPUMASK[$1]}
  if [[ -z "$gpumask" ]]; then
    # Single-GPU box — any benchmark process counts as busy
    local n=$(ssh -p "$port" -o StrictHostKeyChecking=no -o ConnectTimeout=8 -o BatchMode=yes \
              root@"$host" "ps -eo cmd | grep -E 'run_benchmark' | grep -v grep | wc -l" 2>/dev/null)
    [[ -z "$n" ]] && return 0  # SSH unreachable → treat as busy
    [[ "$n" -gt 0 ]]
  else
    # Shared box, slot-specific: check GPU memory on its CUDA index
    local cuda_idx=${gpumask##*=}
    local mem=$(ssh -p "$port" -o StrictHostKeyChecking=no -o ConnectTimeout=8 -o BatchMode=yes \
                root@"$host" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $cuda_idx 2>/dev/null" 2>/dev/null)
    [[ -z "$mem" ]] && return 0  # SSH unreachable
    [[ "$mem" -gt 100 ]]  # > 100 MiB = busy
  fi
}

# Pop next non-DONE line from queue file. Print "port|host|launcher|env" on stdout.
# Marks the line as `# DONE <ts>` in place.
pop_next_queue_line() {
  local qf=$1
  [[ -f $qf ]] || return 1
  local lineno=0
  local found_line=""
  while IFS= read -r line; do
    lineno=$((lineno+1))
    # Skip empty lines and comments (already DONE-marked or static comments)
    [[ -z "$line" || "$line" == "#"* ]] && continue
    found_line=$line
    break
  done < "$qf"
  [[ -z "$found_line" ]] && return 1
  # Mark as DONE — replace first matching line
  sed -i "${lineno}s|^|# DONE $(ts) |" "$qf"
  echo "$found_line"
  return 0
}

echo "[autoqueue] $(ts) start — polling 4 boxes every 300s"
while true; do
  for tag in ssh6_4060 ssh3_3060ti ssh17637_gpu0 ssh17637_gpu1; do
    qf="$QUEUE_DIR/${tag}.queue"
    [[ -f $qf ]] || continue
    if is_box_busy "$tag"; then
      continue
    fi
    next=$(pop_next_queue_line "$qf") || { echo "[autoqueue] $(ts) $tag idle, queue empty"; continue; }
    IFS='|' read -r port host launcher envvars <<< "$next"
    echo "[autoqueue] $(ts) $tag idle → launching: $envvars bash $launcher"
    ssh -p "$port" -o StrictHostKeyChecking=no root@"$host" \
        "cd /root/helios-rl && $envvars nohup setsid bash $launcher > /tmp/autoqueue_${tag}.log 2>&1 < /dev/null & disown" \
        2>&1 | head -3
  done
  sleep 300
done

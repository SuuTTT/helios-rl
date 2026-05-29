# vast.ai Hardware Requirements — TD-MPC-Glass

What works, what doesn't, what to filter on when renting GPUs for this project.

## TL;DR — Search vast.ai with these filters

| Filter | Value | Why |
|---|---|---|
| **Driver version** | **≥ 580** (preferably 580+) | Old drivers can't load JAX `cuda13` wheels we use on most boxes |
| **VRAM** | **≥ 8 GB** | 6 GB triggers OOM in NS=2048 MPPI eval or shared-GPU configs |
| **GPU name** | RTX 3060 Ti, 4060, 4070, 4070 Ti, 4080, A4000+ | Ampere/Ada sm_86/sm_89 — proven path. Avoid Blackwell sm_120 unless driver ≥ 580 |
| **Disk space** | **≥ 50 GB free** | JAX + nvidia-cuda packages ~10 GB; repo + checkpoints another ~5 GB |
| **RAM** | ≥ 16 GB | JIT compile peaks at ~8 GB host RAM |
| **OS** | Ubuntu 22.04 + Python 3.12 (best) or 22.04 + Python 3.11 | Python 3.10 forces old JAX 0.4.x which breaks mujoco_warp |
| **Single-GPU instances** | preferred | Dual-GPU configs split VRAM and we've seen flaky behavior |

**Total budget heuristic**: pay no more than $0.40/hr per GPU. RTX 3060 Ti at $0.13/hr or RTX 4060 at $0.25/hr are sweet spots.

## What works (tested & in production)

| GPU | VRAM | Driver | CUDA | Pip stack | Throughput |
|---|---|---|---|---|---|
| RTX 4070 Ti | 12 GB | 12.4 (local) | 12.x | `requirements-rtx3090.txt` (jax cuda12) | **~540 sps** |
| RTX 4060 | 8 GB | **580.126.09** | 13.0 | `requirements-rtx50series.txt` (jax cuda13) | ~540 sps |
| RTX 3060 Ti | 8 GB | 12.x | 12.x | `requirements-rtx3090.txt` (jax cuda12) | ~100 sps (slow but reliable) |
| 2× RTX 3060 Laptop | 6 GB each | 580 / 13.0 | jax cuda13 | tight — use MEM_FRACTION=0.35 | ~250 sps (with OOM risk) |

**Memory-fraction guidance** (from `docs/tdmpc-glass/operations/env_setup.md`):
- 0.85 on 12 GB+ cards
- 0.55–0.65 on 8 GB cards
- 0.35 on 6 GB cards when sharing the GPU
- NS=2048 eval bursts ADD memory — leave headroom

## What's BLOCKED (don't rent these)

| GPU | Reason | Fix possible? |
|---|---|---|
| RTX 5070 (sm_120) with driver < 580 | CUDA 13 wheels need driver ≥ 580; older drivers can't load Blackwell kernels | Only if vast lists driver 580+ for it |
| RTX 3090 with driver 535 / CUDA 12.2 | JAX 0.6.x needs cuSPARSE 12.6+ (our pin is 12.5). JAX 0.4.x triggers `module 'jax.tree' has no attribute 'map_with_path'` from mujoco_warp ffi.py. No middle ground that satisfies both. | Need driver 12.6+ (i.e., 555+) OR downgrade mujoco_warp AND mujoco_playground to a pre-jax-0.5 release (risks breaking algorithm) |
| **Any card with driver < 535** | mujoco_warp 1.12.1 won't compile kernels | Upgrade driver if vast.ai allows |
| 6 GB cards with shared GPU (2-GPU configs) | OOM-killed our Phase-x s2 twice and Phase-v s2 once | Either rent single-GPU or pay for 8 GB+ |

## Concrete vast.ai search

In the vast.ai web filter:

```
GPU Name:    RTX 3060 Ti, RTX 4060, RTX 4070, RTX 4070 Ti, RTX 4080
GPU VRAM:    8.0 GB minimum
CUDA Driver: 580.0 minimum     (or 535+ if you accept JAX 0.4 limitations)
CUDA>12.3
Disk Space:  50 GB minimum
OS:          Ubuntu 22.04 or 24.04
DLPerf:      sort high to low (within budget)
$/hr:        sort low to high (within DLPerf range)
```

## After renting: 5-minute setup checklist

```bash
# 1. SSH in and verify
ssh -p <port> root@<host> "nvidia-smi | head -3 | tail -1; python3 --version"

# 2. Push code + clone playground (in parallel)
rsync -av -e "ssh -p <port> -o StrictHostKeyChecking=no" \
  --exclude='exp/' --exclude='__pycache__/' --exclude='.git/' --exclude='videos/' \
  /root/helios-rl/ root@<host>:/root/helios-rl/ &

ssh -p <port> root@<host> \
  "apt-get install -y git rsync ffmpeg python3-venv python3-pip
   git clone --depth=1 https://github.com/google-deepmind/mujoco_playground.git /root/mujoco_playground_repo
   python3 -m venv /root/venv"
wait

# 3. Install deps. Pick the right requirements file based on driver:
#    - driver ≥ 580 / CUDA 13.0  → requirements-rtx50series.txt
#    - driver 12.4–12.6          → requirements-rtx3090.txt
ssh -p <port> root@<host> \
  "cd /root/helios-rl
   source /root/venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements-rtx50series.txt   # or rtx3090.txt
   pip install -e .
   pip install ml_collections mujoco_warp mujoco-mjx 'warp-lang==1.12.1'
   python3 -c 'import jax; print(jax.devices())'"
# Expect: [CudaDevice(id=0)]
```

If that last line returns `[CpuDevice(id=0)]` or errors about cuSPARSE / `map_with_path`, the box is incompatible — release and find another.

## Lessons from boxes we tried (iteration 5)

| Box | Status | Lesson |
|---|---|---|
| `ssh3:11271` (3060 Ti) | works, slow | Reliable but only ~100 sps. Use only when faster boxes are full. |
| `ssh6:11115` (4060) | works | Sweet spot. Plenty of headroom at MEM=0.55. Driver 580 + CUDA 13 just works. |
| `78.83.187.54:17637` (2× 3060 Laptop 6GB) | destroyed/retired | Removed from fleet 2026-05-27 after user destroyed the instance. Similar 6 GB shared/laptop boxes are best-effort only; retire if disconnected for 24h. |
| `ssh8:37645` (4060) | recycled | vast.ai sometimes recycles instances mid-run. Save checkpoints frequently |
| `ssh9:16233` (3090 24GB) | BLOCKED | Driver 535 too old for our JAX stack. **Don't rent 3090 instances with driver < 555** |
| (RTX 5070 from earlier session) | BLOCKED | sm_120 + driver 570 can't run CUDA 13 wheels |

## When in doubt, copy the local 4070 Ti config

The local 4070 Ti is our reference: 12 GB VRAM, driver 12.4, JAX cuda12 stack, MEM_FRACTION=0.85. It runs everything without issue at ~540 sps. Any rented box that diverges from this in a way that affects the JAX stack is a risk.

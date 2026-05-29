# Fleet Cost/SPS Audit - 2026-05-26

Purpose: decide which Vast.ai boxes to keep for TD-MPC/Glass HopperHop probes.
This audit compares live measured HopperHop SPS against Vast `DLPerf/$`.

Rule of thumb:

- Prefer `DLPerf/$ >= 200` for new rentals.
- Also check actual HopperHop SPS, because the TD-MPC workload is not a pure DLPerf workload.
- Do not count interrupted experiment results below 4M steps, but keep logs for debugging.
- Do not store Vast API keys in repo files. Use `vastai set api-key ...` locally.

## Current Running Work

Queue state at audit time: `running=10`, `done=17`, `failed=15`.

Active tasks:

| Box | Task |
|---|---|
| `ssh9_2060_gpu0` | `phasei9r p1b off1m seed 4` |
| `ssh3_3060ti` | `phasei9m_sleep8h_20260524_s3_auto_s8 seed 8` |
| `ssh1_2080ti` | `phasei9m_sleep8h_20260524_s3_auto_s10 seed 10` |
| `local` | `phasei9n_sleep8h_20260524_s4_auto_s6 seed 6` |
| `ssh17637_gpu0` | `phasei10a_off1m_clean seed 2` |
| `ssh3_3070` | `phasei10a_off1m_clean seed 3` |
| `ssh9_2060_gpu1` | `phasei9t_p1b_off1p5m_s4_auto_s2 seed 2` |
| `ssh17637_gpu1` | `phasei9t_p1b_off1p5m_s4_auto_s3 seed 3` |
| `ssh9_2060_gpu2` | `phasei9m_sleep8h_20260524_s4_auto_s2 seed 2` |
| `ssh9_2060_gpu3` | `phasei9m_sleep8h_20260524_s4_auto_s3 seed 3` |
| `ssh6_3080` | `phasei9q_p1b_temp001_off2m_s4_auto_s3_auto_s10_auto_s4 seed 4` |

`ssh4_8080` attempted `phasei10a` and `phasei10b` jobs, but both failed with
PJRT pthread creation errors. The instance was destroyed after confirming
`pids.max=256`, which is below the TD-MPC/JAX safety bar.

## Cost/SPS Table

Live SPS is from the dashboard `sps_avg` field. `$ / h` and `DLPerf/$` are from
`vastai show instances --raw`. Multi-GPU rows use per-slot hourly cost for the
slot table, plus aggregate rows below.

| Box | GPU | $/h | DLPerf/$ | Live SPS | SPS per $/h | State | Recommendation |
|---|---|---:|---:|---:|---:|---|---|
| `ssh6_4060` | RTX 4060 | 0.0622 | 251.4 | - | - | unreachable/stopped | Keep only if it becomes reachable; otherwise remove from fleet registry. |
| `ssh17637_gpu0` | RTX 3060 laptop | 0.0484 | 207.0 | 107 | 2209 | destroyed | Retired 2026-05-27 after user destroyed the instance; remove from dispatch. |
| `ssh17637_gpu1` | RTX 3060 laptop | 0.0484 | 207.0 | 108 | 2229 | destroyed | Retired 2026-05-27 after user destroyed the instance; remove from dispatch. |
| `ssh1_2080ti` | RTX 2080 Ti | 0.0889 | 124.3 | 164 | 1845 | running | Finish current useful run, then replace. Below DLPerf/$ bar. |
| `ssh3_3070` | RTX 3070 | 0.0756 | 217.4 | 146 | 1932 | running | Keep. Above bar; not fastest, but acceptable. |
| `ssh6_3080` | RTX 3080 | 0.0756 | 356.7 | 139 | 1840 | running | Keep. Strong DLPerf/$; current SPS is likely workload/phase limited. |
| `ssh3_3060ti` | RTX 3060 Ti | 0.0837 | 164.6 | 4-6 | ~60 | broken/running | Shut down. GPU is not visible to NVIDIA driver; run is crawling on CPU. |
| `ssh4_8080` | RTX 2060 12GB | 0.0556 | 70.9 | 137 before crash | 2466 before crash | destroyed | Reject future boxes with `pids.max < 512` or PJRT pthread failures. |
| `ssh9_2060_gpu0` | RTX 2060 | 0.0423 | 74.7 | 96 | 2269 | running | Keep until current 4-GPU batch finishes; good actual cost/SPS despite poor DLPerf/$. |
| `ssh9_2060_gpu1` | RTX 2060 | 0.0423 | 74.7 | 107 | 2529 | running | Same as GPU0. |
| `ssh9_2060_gpu2` | RTX 2060 | 0.0423 | 74.7 | 110 | 2600 | running | Same as GPU0. |
| `ssh9_2060_gpu3` | RTX 2060 | 0.0423 | 74.7 | 117 | 2765 | running | Same as GPU0. |

Aggregate multi-GPU view:

| Instance | GPUs | Total $/h | Total live SPS | SPS per $/h | Vast DLPerf/$ | Decision |
|---|---:|---:|---:|---:|---:|---|
| `ssh9` | 4x RTX 2060 | 0.1693 | 430 | 2540 | 74.7 | Keep while all 4 GPUs are saturated; do not rent another similar box by DLPerf/$ rule. |
| `ssh17637` | 2x RTX 3060 laptop | 0.0969 | 215 | 2219 | 207.0 | Destroyed/retired 2026-05-27. Do not leave stale slots in queue dispatch. |

## Decision

Immediate actions:

1. `ssh3_3060ti` should be shut down. It has `Unable to determine the device handle for GPU0` and no usable `nvidia-smi`; the active run is effectively CPU-only.
2. `ssh4_8080` should be removed from normal scheduling. It is cheap and can reach ~137 SPS, but it fails with PJRT thread creation errors on this workload.
3. `ssh1_2080ti` is not catastrophically slow, but it is below the DLPerf/$ bar. Let the current seed finish if it is useful, then replace it with a hunter-selected instance above 200 DLPerf/$.

## Cleanup/Rental Action Taken

Executed on 2026-05-26:

- Destroyed Vast instance `36721114` (`ssh6_4060`, RTX 4060).
- Destroyed Vast instance `36841270` (`ssh3_3060ti`, RTX 3060 Ti with broken NVIDIA device state).
- Marked the destroyed `ssh3_3060ti` queue task failed with reason `instance_destroyed_by_fleet_cost_cleanup_2026-05-26`.
- Rented `37907233`: RTX 3060 12GB, `ssh5.vast.ai:27233`, `$0.0607/h`, `201.9 DLPerf/$`, label `tdmpc_lowcost_bar1`.
- Rented `37907257`: RTX A4000 16GB, `ssh1.vast.ai:27257`, `$0.0837/h`, `244.2 DLPerf/$`, label `tdmpc_lowcost_a4000_bar`, then destroyed it before use because it had driver `535.146.02` / CUDA max `12.2`, below the project hardware requirement for the current JAX/CUDA13 stack.
- Rented replacement `37907664`: RTX 3060 12GB, `ssh4.vast.ai:27665`, `$0.0589/h`, `208.3 DLPerf/$`, driver `580.142`, CUDA max `13.0`, label `tdmpc_lowcost_cuda13_3060_cn2`.
- Destroyed `37907664` after setup attempts. Although CUDA/driver/DLPerf met the bar, the SSH route repeatedly failed during setup:
  - `rsync` of the 21MB `mujoco_playground_repo` tree repeatedly stalled or broke with `Broken pipe`.
  - Follow-up SSH checks hit `Connection timed out during banner exchange`.
  - This makes the box unsuitable for queue-worker use because bootstrapping and code sync are unreliable.

Notes:

- A first attempt rented two RTX 3060 offers with 50GB disks, but their total-cost DLPerf/$ fell below 200 after storage pricing. Those contracts were destroyed and replaced.
- `37907233` / `ssh5_3060_bar` was successfully bootstrapped and verified with `jax.devices() == [CudaDevice(id=0)]`.
- `37907664` / `ssh4_3060_bar` was destroyed and must not be re-rented.
- `ssh4_8080` / contract `37565664` was destroyed by the user and disabled from dispatch because it repeatedly hit PJRT thread creation failures.

Keep:

- `ssh17637` 2x 3060 laptop: retired 2026-05-27 after destruction. Future
  similar boxes should be treated as disposable; if disconnected for 24h, sync
  available logs, mark assigned tasks failed, destroy the instance, and remove
  it from `task_queue_daemon.py` / `web_dashboard.py`.
- `ssh3_3070`: above 200 DLPerf/$.
- `ssh6_3080`: excellent DLPerf/$.
- `ssh9` 4x 2060: bad DLPerf/$ but good aggregate actual SPS/$ while all four GPUs are loaded. Keep only if all four slots stay saturated and stable.

## Low-Cost Vast Hunter

Script:

```bash
python3 scripts/vast_lowcost_hunter.py --max-dph 0.10 --min-dlperf-usd 200 --min-gpu-ram 8 --limit 30
```

Default query:

```text
verified=true rentable=true rented=false dph < 0.1 dlperf_usd > 200 gpu_ram >= 8 reliability > 0.95 direct_port_count >= 2 cuda_vers >= 13.0
```

Current sample hits:

| Offer | GPU | $/h | DLPerf/$ | Notes |
|---:|---|---:|---:|---|
| `23139208` | RTX A4000 | 0.0809 | 252.6 | 16GB VRAM; attractive replacement for 2080 Ti/3060 Ti. |
| `32711685` | RTX 3070 | 0.0681 | 245.6 | Good price and familiar throughput class. |
| `30941709` | Titan V | 0.0947 | 231.5 | 12GB; worth trying only if CUDA/JAX compatibility is clean. |
| `32571607` | Titan V | 0.0947 | 231.5 | Same as above. |
| `33809284` | Titan V | 0.0947 | 230.0 | Same as above; lower upload speed. |
| `34624617` | RTX 3060 | 0.0547 | 224.2 | Cheap 12GB option; location CN may affect sync latency. |

Suggested hunter policy:

1. Search every 5-10 minutes while fleet has idle or bad boxes.
2. Prefer `DLPerf/$ > 240` if enough offers exist; relax to `>200` only when fleet is undersupplied.
3. Require `gpu_ram >= 8`, `cuda_vers >= 13.0`, `reliability > 0.95`, `direct_port_count >= 2`.
4. Price with the intended storage size. Default hunter storage is now `50GB`, because CUDA/JAX wheels plus repo/playground/checkpoints make `20GB` fragile.
5. Exclude known unstable offers. Current hard exclusion: offer `34624617` / contract `37907664`.
6. Reject boxes with a low process/thread cgroup limit:

```bash
ssh -p <port> root@<host> 'cat /sys/fs/cgroup/pids.max 2>/dev/null || cat /sys/fs/cgroup/pids/pids.max 2>/dev/null || true'
```

Reject values below `512`; `ssh4_8080` reported `256` and failed with
`Thread pjrt_async_work_runner creation via pthread_create() failed`.
7. Avoid boxes that cannot complete a small setup transfer and `jax.devices()` verification within a few minutes, regardless of DLPerf/$.
8. After launch, run a 10-15 minute smoke probe and record actual TD-MPC SPS.
9. Keep an instance only if either:
   - `DLPerf/$ >= 200` and TD-MPC SPS is competitive, or
   - actual aggregate TD-MPC `SPS/$h` is exceptional and stable.

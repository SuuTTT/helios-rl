# Phase-eval Re-score — 2026-05-21

Re-scored recent benchmark-fair HopperHop phases by:

`best_any = max(best_pi, best_mppi)`

Method:
- scan local `exp/tdmpc_glass` plus `remote_mirror`
- ignore `_diag.csv`, `_final_snapshot`, and video artifacts
- collapse duplicate/retry artifacts to one result per intended `(phase, seed)` by taking the highest verified `best_any`
- for `phasez_baseline`, report under canonical name `phasez`
- for `phasex_*`, merge under canonical name `phasex`

## Results

| Phase | n | G1 | G2 | Max best_any | Mean best_any | Notes |
|---|---:|---:|---:|---:|---:|---|
| `phaseaa_codex_tdmpc2_k128` | 5 | 1 | 0 | 538.6 | 350.0 | seed 4 flips to `pi` winner |
| `phaseaa_codex_tdmpc2_k256` | 3 | 1 | 0 | 561.1 | 401.5 | stronger ceiling than prior MPPI-only note |
| `phasez` | 4 | 1 | 0 | 535.4 | 432.4 | seed 5 is `pi`-selected at 468.0 |
| `phasex` | 8 | 2 | 0 | 524.8 | 260.5 | no G1/G2 count change vs MPPI-only accounting |

## Per-seed detail

### `phaseaa_codex_tdmpc2_k128`

| Seed | best_any | Selector | best_pi | best_mppi |
|---|---:|---|---:|---:|
| 1 | 538.6 | `mppi` | 506.4 | 538.6 |
| 2 | 284.9 | `mppi` | 274.5 | 284.9 |
| 3 | 307.1 | `mppi` | 289.8 | 307.1 |
| 4 | 288.4 | `pi` | 288.4 | 209.2 |
| 5 | 331.2 | `mppi` | 273.5 | 331.2 |

### `phaseaa_codex_tdmpc2_k256`

| Seed | best_any | Selector | best_pi | best_mppi |
|---|---:|---|---:|---:|
| 1 | 561.1 | `mppi` | 531.9 | 561.1 |
| 2 | 351.7 | `mppi` | 344.0 | 351.7 |
| 3 | 291.7 | `mppi` | 282.0 | 291.7 |

### `phasez`

| Seed | best_any | Selector | best_pi | best_mppi |
|---|---:|---|---:|---:|
| 2 | 278.5 | `pi` | 278.5 | 268.4 |
| 3 | 535.4 | `mppi` | 498.2 | 535.4 |
| 4 | 447.6 | `mppi` | 399.3 | 447.6 |
| 5 | 468.0 | `pi` | 468.0 | 467.1 |

### `phasex`

| Seed | best_any | Selector | best_pi | best_mppi | Raw tag |
|---|---:|---|---:|---:|---|
| 1 | 380.9 | `mppi` | 306.6 | 380.9 | `phasex_2x3060` |
| 2 | 5.8 | `mppi` | 5.2 | 5.8 | `phasex_2x3060` |
| 3 | 523.5 | `mppi` | 490.4 | 523.5 | `phasex_local` |
| 5 | 3.7 | `mppi` | 1.9 | 3.7 | `phasex_ns1024` |
| 6 | 287.3 | `mppi` | 265.1 | 287.3 | `phasex_local` |
| 7 | 123.4 | `pi` | 123.4 | 79.7 | `phasex_3060ti` |
| 8 | 524.8 | `mppi` | 509.3 | 524.8 | `phasex_4060` |
| 9 | 234.3 | `mppi` | 223.6 | 234.3 | `phasex_local` |

## Interpretation

- The main ranking change is not G1 count; it is **which checkpoints deserve preservation and render attention**.
- `phaseaa_codex_tdmpc2_k128` seed 4 and `phasez` seeds 2 and 5 are concrete cases where `pi` beats `mppi`.
- `phaseaa_codex_tdmpc2_k256` has a higher fair ceiling by `best_any` than the old MPPI-only note suggested: `561.1`.
- I could not find canonical `phaseab_codex_tdmpc2_5seed` CSVs in local or mirrored storage, so that phase is not included here.

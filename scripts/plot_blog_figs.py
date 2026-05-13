#!/usr/bin/env python3
"""Generate the figures used in the TD-MPC-Glass blog post.

Outputs (under ``exp/tdmpc_glass/plots/``):
  - blog_ci_phase1_vs_phase1b.png   — 95% CI: Phase1 (5 seeds) vs Phase1b
                                       (partial, 2 seeds) vs Official.
  - blog_cluster_count_vs_return.png — final return as a function of the
                                       discovered cluster count (the 4 vs 3
                                       basin story).
  - blog_failure_case_seed4.png      — seed-4 failure curve next to the
                                       median good seed, plus its Glass
                                       diagnostics.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/workspace/helios-rl")
P1_DIR = ROOT / "exp/tdmpc_glass/HopperHop"
P1B_DIR = ROOT / "exp/tdmpc_glass/HopperHop_phase1b_remote"
P1_LOG = ROOT / "exp/tdmpc_glass/logs/phase1"
OFFICIAL = Path("/workspace/tdmpc2/results/tdmpc2/hopper-hop.csv")
OUT = ROOT / "exp/tdmpc_glass/plots"
OUT.mkdir(parents=True, exist_ok=True)


def load_mppi(d: Path) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for csv in sorted(d.glob("seed_*.csv")):
        seed = int(csv.stem.split("_")[1])
        df = pd.read_csv(csv, names=["step", "reward", "eval_type", "seed"]) \
            if not csv.read_text().startswith("step,") else pd.read_csv(csv)
        if df.empty:
            continue
        df = df[df["eval_type"] == "mppi"].sort_values("step").reset_index(drop=True)
        if not df.empty:
            out[seed] = df
    return out


def load_official() -> dict[int, pd.DataFrame]:
    df = pd.read_csv(OFFICIAL)
    return {int(s): sub.sort_values("step").reset_index(drop=True)
            for s, sub in df.groupby("seed")}


def mean_ci(curves: list[np.ndarray]):
    arr = np.asarray(curves, dtype=np.float64)
    mean = arr.mean(axis=0)
    if arr.shape[0] > 1:
        sem = stats.sem(arr, axis=0)
        tval = stats.t.ppf(0.975, df=arr.shape[0] - 1)
        return mean, mean - tval * sem, mean + tval * sem
    return mean, mean, mean


def interp(df: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, df["step"].to_numpy(), df["reward"].to_numpy(),
                     left=np.nan, right=np.nan)


def fig_ci():
    p1 = load_mppi(P1_DIR)
    p1b = load_mppi(P1B_DIR)
    off = load_official()

    grid = np.linspace(0, 4_000_000, 201)
    fig, ax = plt.subplots(figsize=(11, 6.0), dpi=170)

    # Official band
    if len(off) >= 2:
        cs = [interp(df, grid) for df in off.values()]
        valid = ~np.any(np.isnan(np.stack(cs)), axis=0)
        m, lo, hi = mean_ci([c[valid] for c in cs])
        ax.plot(grid[valid], m, color="tab:gray", lw=2.0,
                label=f"Official TD-MPC2 mean ({len(off)} seeds)")
        ax.fill_between(grid[valid], lo, hi, color="tab:gray", alpha=0.18,
                        label="Official 95% CI")

    # Phase 1
    cs = []
    for seed, df in p1.items():
        ax.plot(df["step"], df["reward"],
                color=plt.cm.Reds(0.35 + 0.12 * (seed - 1)),
                lw=0.9, alpha=0.45)
        cs.append(interp(df, grid))
    if len(cs) >= 2:
        arr = np.stack(cs)
        valid = ~np.any(np.isnan(arr), axis=0)
        m, lo, hi = mean_ci([c[valid] for c in cs])
        ax.plot(grid[valid], m, color="tab:red", lw=2.4,
                label=f"Phase1 mean ({len(p1)} seeds)")
        ax.fill_between(grid[valid], lo, hi, color="tab:red", alpha=0.15,
                        label="Phase1 95% CI")

    # Phase 1b (partial)
    cs = []
    for seed, df in p1b.items():
        ax.plot(df["step"], df["reward"],
                color=plt.cm.Blues(0.45 + 0.18 * (seed - 1)),
                lw=1.4, alpha=0.85,
                label=f"Phase1b seed {seed} "
                      f"({df['step'].iloc[-1]/1e6:.2f}M)")
        cs.append(interp(df, grid))
    if len(cs) >= 2:
        arr = np.stack(cs)
        valid = ~np.any(np.isnan(arr), axis=0)
        if valid.any():
            m, lo, hi = mean_ci([c[valid] for c in cs])
            ax.plot(grid[valid], m, color="tab:blue", lw=2.4,
                    label=f"Phase1b mean ({len(p1b)} seeds, partial)")

    ax.set_xlabel("environment steps")
    ax.set_ylabel("MPPI return (HopperHop)")
    ax.set_title("TD-MPC-Glass: Phase 1 (5 seeds) vs Phase 1b (2 seeds, partial) "
                 "vs Official TD-MPC2")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 4_000_000)
    ax.set_ylim(bottom=-10)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    out = OUT / "blog_ci_phase1_vs_phase1b.png"
    fig.savefig(out)
    print(f"wrote {out}")


# --- cluster-count-vs-return figure ---

def final_active(log: Path) -> tuple[float, float]:
    """Return (active, ent) averaged over the last 5 glass-diag lines."""
    actives, ents = [], []
    for ln in log.read_text().splitlines()[-2000:]:
        m = re.search(r"glass se=[-0-9.]+ ent=([-0-9.]+) active=([\d.]+)", ln)
        if m:
            ents.append(float(m.group(1)))
            actives.append(float(m.group(2)))
    if not actives:
        return float("nan"), float("nan")
    return float(np.mean(actives[-5:])), float(np.mean(ents[-5:]))


def fig_cluster_return():
    rows = []
    for seed in (1, 2, 3, 4, 5):
        csv = P1_DIR / f"seed_{seed}.csv"
        log = P1_LOG / f"HopperHop_seed_{seed}.log"
        # for seed 3 the longest log is the suffixed one
        if seed == 3 and not log.exists():
            log = P1_LOG / "HopperHop_seed_3_d.log"
        if not (csv.exists() and log.exists()):
            continue
        df = pd.read_csv(csv)
        df = df[df["eval_type"] == "mppi"]
        ret = float(df.iloc[-1]["reward"])
        active, ent = final_active(log)
        rows.append({"seed": seed, "active": active, "ent": ent,
                     "return": ret, "phase": "Phase1"})

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=170)
    colors = {3: "tab:red", 4: "tab:green"}
    for _, r in df.iterrows():
        c = colors.get(int(round(r["active"])), "tab:gray")
        ax.scatter(r["active"], r["return"], s=180, color=c,
                   edgecolor="black", linewidths=0.6, zorder=3)
        ax.annotate(f" seed {int(r['seed'])}",
                    (r["active"], r["return"]),
                    fontsize=9, va="center")

    # Group means
    for K, sub in df.groupby(df["active"].round().astype(int)):
        mu = sub["return"].mean()
        ax.hlines(mu, K - 0.18, K + 0.18, color=colors.get(K, "tab:gray"),
                  lw=2.5, zorder=2,
                  label=f"K={K} mean = {mu:.1f}  (n={len(sub)})")

    ax.axhline(449.2, ls="--", color="black", alpha=0.5,
               label="Official 5-seed mean (449.2)")
    ax.set_xlabel("Glass-discovered active cluster count (final 25k steps)")
    ax.set_ylabel("Final MPPI return at 4M")
    ax.set_title("Cluster basin predicts return — Phase 1, HopperHop")
    ax.set_xticks([3, 4])
    ax.set_xlim(2.5, 4.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = OUT / "blog_cluster_count_vs_return.png"
    fig.savefig(out)
    print(f"wrote {out}")


def fig_failure_seed4():
    p1 = load_mppi(P1_DIR)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.3), dpi=170,
                                    gridspec_kw={"width_ratios": [1.05, 1]})

    # Return curves
    for seed, df in p1.items():
        col = "tab:red" if seed == 4 else "tab:green"
        alpha = 1.0 if seed == 4 else 0.4
        lw = 2.3 if seed == 4 else 1.1
        ax1.plot(df["step"], df["reward"], color=col, alpha=alpha, lw=lw,
                 label=("seed 4 (K=3 basin, fail)" if seed == 4
                        else (f"other seeds (K=4 basin)" if seed == 1 else None)))
    ax1.set_xlabel("environment steps")
    ax1.set_ylabel("MPPI return")
    ax1.set_title("Failure case: seed 4 plateaus around the 3-cluster basin")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower right", fontsize=9)

    # Glass diagnostics: active clusters over time, seed-by-seed
    log_map = {
        1: P1_LOG / "HopperHop_seed_1.log",
        2: P1_LOG / "HopperHop_seed_2.log",
        3: P1_LOG / "HopperHop_seed_3_d.log",
        4: P1_LOG / "HopperHop_seed_4.log",
        5: P1_LOG / "HopperHop_seed_5.log",
    }
    for seed, log in log_map.items():
        if not log.exists():
            continue
        steps, actives = [], []
        cur = None
        for ln in log.read_text().splitlines():
            m = re.match(r"\s*step=\s*([\d,]+)\s+pi_reward=", ln)
            if m:
                cur = int(m.group(1).replace(",", ""))
                continue
            m = re.search(r"active=([\d.]+)", ln)
            if m and cur is not None:
                steps.append(cur); actives.append(float(m.group(1))); cur = None
        if not steps:
            continue
        col = "tab:red" if seed == 4 else ("tab:orange" if seed == 5
                                            else "tab:green")
        alpha = 1.0 if seed in (4, 5) else 0.45
        lw = 2.0 if seed in (4, 5) else 1.0
        ax2.plot(steps, actives, color=col, alpha=alpha, lw=lw,
                 label=f"seed {seed}")
    ax2.axhline(4.0, ls="--", color="tab:gray", lw=1.0,
                label="hopper natural K = 4")
    ax2.set_xlabel("environment steps")
    ax2.set_ylabel("Glass active clusters (out of K=8)")
    ax2.set_title("Glass cluster count across training")
    ax2.set_ylim(2.5, 8.5)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    out = OUT / "blog_failure_case_seed4.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_ci()
    fig_cluster_return()
    fig_failure_seed4()

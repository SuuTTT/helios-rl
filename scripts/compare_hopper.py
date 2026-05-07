"""Compare our TD-MPC2 JAX result vs official reference on hopper-hop."""
import os
import numpy as np

OUR_CSV = "/workspace/helios-rl/exp/tdmpc_dmc/hopper-hop.csv"
REF_CSV = "/workspace/tdmpc2/results/tdmpc2/hopper-hop.csv"


def load_csv(path):
    data = {}
    with open(path) as f:
        header = f.readline()  # skip
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            step, reward = int(parts[0]), float(parts[1])
            if step not in data:
                data[step] = []
            data[step].append(reward)
    # average over seeds
    return {s: np.mean(v) for s, v in data.items()}


def main():
    if not os.path.exists(OUR_CSV):
        print(f"Our CSV not found: {OUR_CSV}")
        return

    ours = load_csv(OUR_CSV)
    ref  = load_csv(REF_CSV)

    if not ours:
        print("Our CSV is empty (training not started yet).")
        return

    our_steps = sorted(ours.keys())
    ref_steps  = sorted(ref.keys())

    print(f"\nHopper-Hop: TD-MPC2 JAX (ours, seed=42) vs Official Reference (seed avg)")
    print(f"{'Step':>10}  {'Ours':>8}  {'Reference':>10}  {'Delta':>8}")
    print("-" * 45)

    for step in our_steps:
        o = ours[step]
        # find closest ref step
        r = ref.get(step, None)
        if r is None:
            # try nearest
            nearest = min(ref_steps, key=lambda s: abs(s - step))
            if abs(nearest - step) <= 50_000:
                r = ref[nearest]
        if r is not None:
            delta = o - r
            mark = " ✓" if abs(delta) < 30 else (" ▲" if delta > 0 else " ▼")
            print(f"{step:>10,}  {o:>8.1f}  {r:>10.1f}  {delta:>+8.1f}{mark}")
        else:
            print(f"{step:>10,}  {o:>8.1f}  {'N/A':>10}  {'':>8}")

    # summary at latest our step
    latest_step = max(our_steps)
    o_latest = ours[latest_step]
    r_latest = ref.get(latest_step, None)
    if r_latest is None:
        r_latest = ref.get(max(s for s in ref_steps if s <= latest_step), None)
    print()
    print(f"  Latest checkpoint: step={latest_step:,}")
    print(f"  Ours:      {o_latest:.1f}")
    if r_latest is not None:
        print(f"  Reference: {r_latest:.1f}  (delta={o_latest - r_latest:+.1f})")

    # print remaining reference steps
    remaining = [s for s in ref_steps if s > latest_step]
    if remaining:
        print(f"\n  Reference continues to step {max(ref_steps):,}:")
        for s in remaining[:5]:
            print(f"    step={s:>10,}  ref={ref[s]:.1f}")
        if len(remaining) > 5:
            print(f"    ... ({len(remaining)} more checkpoints)")


if __name__ == "__main__":
    main()

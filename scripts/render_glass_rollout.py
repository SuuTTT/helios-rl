#!/usr/bin/env python3
"""Render a trained TD-MPC-Glass policy rollout with the Glass cluster id
overlaid on every frame.

Usage:
    PYTHONPATH=src:/path/to/mujoco_playground/repo \
      python scripts/render_glass_rollout.py \
        --ckpt exp/tdmpc_glass/HopperHop_phase1b_remote/seed_2/checkpoints/best_mppi.pkl \
        --env_id HopperHop \
        --out exp/tdmpc_glass/videos/phase1b_seed2.mp4 \
        --n_episodes 3 --use_mppi

The output MP4 shows the env at full FPS with the argmax(S(z·μᵀ)) cluster
index drawn in the corner, plus a return + step ticker.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mujoco_playground import registry, wrapper

from helios.algorithms.tdmpc_glass import (
    Encoder, Dynamics, Pi,
    GLASS_DEFAULTS,
)
# We reuse the planner from the run script directly.
import importlib.util
import sys
_runner_spec = importlib.util.spec_from_file_location(
    "run_benchmark", Path(__file__).with_name("run_benchmark.py"))
_runner = importlib.util.module_from_spec(_runner_spec)
sys.modules["run_benchmark"] = _runner


def overlay_cluster_label(frames: list[np.ndarray],
                          labels: list[int],
                          returns: list[float]) -> list[np.ndarray]:
    """Draw cluster id + cumulative return on each frame using PIL."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("PIL not available, skipping overlay")
        return frames
    palette = [(255, 99, 71), (60, 179, 113), (65, 105, 225),
               (255, 165, 0), (148, 0, 211), (220, 20, 60),
               (32, 178, 170), (255, 215, 0)]
    out = []
    for f, lab, ret in zip(frames, labels, returns):
        img = Image.fromarray(np.ascontiguousarray(f))
        d = ImageDraw.Draw(img)
        col = palette[lab % len(palette)]
        d.rectangle([(4, 4), (28, 28)], fill=col)
        d.text((34, 6), f"K={lab}  R={ret:6.1f}", fill=(0, 0, 0))
        out.append(np.asarray(img))
    return out


def _build_models(obs_dim: int, act_dim: int,
                  latent_dim: int = 512,
                  hidden: tuple = (512, 512),
                  V: int = 8) -> tuple:
    return (
        Encoder(latent_dim=latent_dim, hidden=hidden, V=V),
        Dynamics(latent_dim=latent_dim, hidden=hidden, V=V),
        Pi(action_dim=act_dim, hidden=hidden),
    )


def _cluster_label(z: jnp.ndarray, glass_params: dict,
                   T_proto: float = 0.7) -> int:
    """Return argmax(S row) where row index is closest prototype to z."""
    mu = glass_params["prototypes"]
    zn = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    mn = mu / (jnp.linalg.norm(mu, axis=-1, keepdims=True) + 1e-8)
    sim = (zn @ mn.T) / T_proto                   # (N,)
    n_star = int(jnp.argmax(sim))
    S = jax.nn.softmax(glass_params["assign_logits"], axis=1)
    return int(jnp.argmax(S[n_star]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--env_id", default="HopperHop")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n_episodes", type=int, default=3)
    ap.add_argument("--episode_length", type=int, default=1000)
    ap.add_argument("--stop_on_done", action="store_true",
                    help="end an episode when the environment reports done; default renders the full episode_length")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--camera", type=str, default="cam0",
                    help="MuJoCo camera name (default cam0 = trackcom side view). Pass empty string for the free static camera.")
    ap.add_argument("--use_mppi", action="store_true",
                    help="render with MPPI planning (slower) instead of the bare policy")
    args = ap.parse_args()

    with open(args.ckpt, "rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["params"]
    glass_cfg = ckpt.get("glass_config", dict(GLASS_DEFAULTS))
    T_proto = float(glass_cfg.get("proto_temperature", 1.0))

    env = registry.load(args.env_id)
    obs_dim = env.observation_size
    act_dim = env.action_size

    enc_net, dyn_net, pi_net = _build_models(obs_dim, act_dim)
    enc_apply = jax.jit(lambda p, o: enc_net.apply(p["enc"], o[None])[0])
    def _pi_action(p, z):
        mean, _ = pi_net.apply(p["pi"], z[None])
        return jnp.tanh(mean[0])
    pi_apply  = jax.jit(_pi_action)

    key = jax.random.PRNGKey(args.seed)
    all_traj = []
    all_labels = []
    all_returns = []
    for ep in range(args.n_episodes):
        key, rk = jax.random.split(key)
        state = env.reset(rk)
        traj = [jax.device_get(state)]
        labels = [0]
        cum = 0.0
        rets = [0.0]
        # episode_length is the requested number of rendered env frames. `traj`
        # already contains the reset frame, so step at most length-1 times.
        for t in range(max(args.episode_length - 1, 0)):
            obs = jnp.asarray(state.obs)
            z = enc_apply(params, obs)
            lab = _cluster_label(z, params["glass"], T_proto=T_proto)
            act = pi_apply(params, z)
            state = env.step(state, act)
            # Keep the saved render trajectory on host memory. Holding hundreds
            # of MJX states as GPU buffers can exhaust CUDA memory before render.
            traj.append(jax.device_get(state))
            labels.append(lab)
            cum += float(state.reward)
            rets.append(cum)
            if args.stop_on_done and bool(state.done > 0.5):
                break
            if (t + 1) % 50 == 0 or (t + 1) == max(args.episode_length - 1, 0):
                print(f"  rollout {ep}: step={t + 1}/{max(args.episode_length - 1, 0)}", flush=True)
        print(f"  episode {ep}: steps={len(traj)}  return={cum:.1f}")
        all_traj.append(traj)
        all_labels.append(labels)
        all_returns.append(rets)

    # Render and stitch
    all_frames = []
    for idx, (traj, labs, rets) in enumerate(zip(all_traj, all_labels, all_returns)):
        cam = args.camera if args.camera else None
        frames = env.render(traj, height=args.height, width=args.width, camera=cam)
        frames = overlay_cluster_label(list(frames), labs, rets)
        all_frames.extend(frames)
        # 8-frame black separator between episodes, but not after the last one.
        if idx != len(all_traj) - 1:
            sep = np.zeros_like(frames[0])
            all_frames.extend([sep] * 8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import mediapy as media
        media.write_video(str(args.out), all_frames, fps=int(round(1 / env.dt)))
    except ImportError:
        import imageio
        imageio.mimsave(str(args.out), all_frames, fps=int(round(1 / env.dt)))
    print(f"wrote {args.out}  ({len(all_frames)} frames)")


if __name__ == "__main__":
    main()

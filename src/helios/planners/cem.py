"""Cross-Entropy Method (CEM) planner.

CEM is a derivative-free, sampling-based optimiser well suited for short-
horizon planning in latent space.  It iteratively:

1. Samples ``num_samples`` action sequences from a Gaussian distribution.
2. Rolls out those sequences through a dynamics model.
3. Ranks sequences by cumulative reward + terminal value.
4. Re-fits the Gaussian to the top-``num_elites`` trajectories.

Reference:
    Rubinstein (1999) - "The Cross-Entropy Method for Combinatorial and
    Continuous Optimization".
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def cem_plan(
    key: jax.Array,
    obs_embed: jax.Array,
    imagine_fn: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], tuple],
    reward_fn: Callable[[jax.Array], jax.Array],
    value_fn: Callable[[jax.Array], jax.Array],
    *,
    horizon: int = 12,
    num_samples: int = 512,
    num_elites: int = 64,
    num_iterations: int = 6,
    action_dim: int,
    action_low: float = -1.0,
    action_high: float = 1.0,
    momentum: float = 0.1,
    temperature: float = 1.0,
    dynamics_params: dict | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Run CEM planning and return the best action sequence.

    Args:
        key: PRNG key.
        obs_embed: Current observation embedding / latent state,
                   shape ``(latent_dim,)`` (single env, no batch dim).
        imagine_fn: ``(state, action, key, params) -> (next_state, extras)``.
        reward_fn: ``(state) -> scalar reward``.
        value_fn: ``(state) -> terminal value``.
        horizon: Planning horizon ``H``.
        num_samples: Number of candidate action sequences per CEM iteration.
        num_elites: Number of elite sequences used to update the distribution.
        num_iterations: Number of CEM refinement iterations.
        action_dim: Dimensionality of a single action.
        action_low: Minimum action value for clipping.
        action_high: Maximum action value for clipping.
        momentum: Smoothing factor when updating the CEM mean/std.
        temperature: Sharpening temperature for elite selection weights.
        dynamics_params: Flax parameter dict forwarded to ``imagine_fn``.

    Returns:
        Tuple ``(best_action_sequence, best_return)`` where
        ``best_action_sequence`` has shape ``(horizon, action_dim)``.
    """
    # Initialise Gaussian distribution over action sequences
    mu = jnp.zeros((horizon, action_dim))
    std = jnp.ones((horizon, action_dim))

    def _rollout_sequences(
        key: jax.Array,
        mu: jax.Array,
        std: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample and evaluate ``num_samples`` action sequences."""
        seq_key, eval_key = jax.random.split(key)

        # Sample: (N, H, action_dim)
        noise = jax.random.normal(seq_key, (num_samples, horizon, action_dim))
        actions = jnp.clip(mu[None] + std[None] * noise, action_low, action_high)

        # Broadcast initial state to (N, latent_dim)
        init_state = jnp.broadcast_to(obs_embed[None], (num_samples,) + obs_embed.shape)

        def _scan_step(state, ha):
            action, rng = ha
            next_state, _ = imagine_fn(state, action, rng, dynamics_params)
            reward = reward_fn(next_state)
            return next_state, (next_state, reward)

        # Generate per-step PRNG keys
        step_keys = jax.random.split(eval_key, horizon * num_samples).reshape(
            num_samples, horizon, 2
        )

        # vmap over samples, scan over time
        def _single_rollout(state0, acts, keys):
            def _step(s, ha):
                a, k = ha
                ns, _ = imagine_fn(s, a, k, dynamics_params)
                r = reward_fn(ns)
                return ns, (ns, r)

            final_state, (states, rewards) = jax.lax.scan(_step, state0, (acts, keys))
            terminal_v = value_fn(final_state)
            returns = jnp.sum(rewards) + terminal_v
            return returns, acts[0]

        rollout_vmapped = jax.vmap(_single_rollout)
        returns, first_actions = rollout_vmapped(
            init_state, actions, step_keys
        )
        return returns, actions

    best_sequence = mu
    best_return = jnp.array(-jnp.inf)

    for i in range(num_iterations):
        key, sub_key = jax.random.split(key)
        returns, sequences = _rollout_sequences(sub_key, mu, std)

        # Elite selection
        elite_idx = jnp.argsort(returns)[-num_elites:]
        elite_sequences = sequences[elite_idx]  # (E, H, action_dim)
        elite_returns = returns[elite_idx]

        # Weighted mean/std using softmax over elite returns
        weights = jax.nn.softmax(elite_returns / (temperature + 1e-8))
        new_mu = jnp.einsum("e,eha->ha", weights, elite_sequences)
        new_std = jnp.sqrt(
            jnp.einsum("e,eha->ha", weights, (elite_sequences - new_mu[None]) ** 2) + 1e-6
        )

        # Momentum smoothing
        mu = momentum * mu + (1.0 - momentum) * new_mu
        std = momentum * std + (1.0 - momentum) * new_std

        # Track best
        best_idx = jnp.argmax(returns)
        if returns[best_idx] > best_return:
            best_return = returns[best_idx]
            best_sequence = sequences[best_idx]

    return best_sequence, best_return

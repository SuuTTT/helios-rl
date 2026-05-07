"""Model Predictive Path Integral (MPPI) planner.

MPPI is a sampling-based MPC algorithm that uses an importance-weighted
update to refine an action distribution.  Unlike CEM it weights *all*
samples (not just elites), making it a smoother gradient approximation.

This implementation is fully JIT-compatible and uses ``jax.vmap`` for the
parallel trajectory rollouts.

Reference:
    Williams et al. (2017) - "Information Theoretic MPC for Model-Based
    Reinforcement Learning" - https://arxiv.org/abs/1707.05110
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def mppi_plan(
    key: jax.Array,
    state: dict[str, jax.Array],
    imagine_fn: Callable,
    reward_fn: Callable,
    value_fn: Callable,
    *,
    horizon: int = 5,
    num_samples: int = 512,
    num_iterations: int = 6,
    action_dim: int,
    action_low: float = -1.0,
    action_high: float = 1.0,
    temperature: float = 0.5,
    momentum: float = 0.1,
    noise_beta: float = 2.0,
    mixture_coef: float = 0.05,
    dynamics_params: dict | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Run MPPI planning and return the first action of the best sequence.

    The algorithm iterates:
    1. Sample N perturbations ε ~ N(0, σ²) around the current mean μ.
    2. Construct action sequences: A_i = clamp(μ + ε_i).
    3. Roll each sequence forward through ``imagine_fn`` collecting rewards.
    4. Compute softmax weights w_i ∝ exp((J_i - J_max) / λ).
    5. Update μ ← Σ w_i * A_i.

    Args:
        key: PRNG key.
        state: Current latent state dict for the dynamics model.  The batch
               dimension should be 1 (single environment planning).
        imagine_fn: ``(state, action, key, params) -> (next_state, extras)``.
                    ``state`` has a leading batch dim of size ``num_samples``.
        reward_fn: ``(state) -> rewards`` where state has batch dim ``num_samples``.
        value_fn: ``(state) -> values`` with the same batch convention.
        horizon: Planning horizon H.
        num_samples: Number of trajectory samples N.
        num_iterations: MPPI refinement iterations.
        action_dim: Dimensionality of a single action.
        action_low: Lower bound on actions (for clipping).
        action_high: Upper bound on actions (for clipping).
        temperature: Inverse temperature λ for the softmax weighting.
        momentum: Smoothing factor when updating µ across iterations.
        noise_beta: Standard deviation of the perturbation noise.
        mixture_coef: Fraction of uniform random trajectories mixed in.
        dynamics_params: Flax parameter dict forwarded to ``imagine_fn``.

    Returns:
        Tuple ``(action_sequence, best_return)`` where ``action_sequence``
        has shape ``(horizon, action_dim)``.
    """
    # Initialise Gaussian mean over action sequences
    mu = jnp.zeros((horizon, action_dim))

    def _evaluate_sequences(
        key: jax.Array,
        mu: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Sample and score ``num_samples`` action sequences.

        Returns:
            ``(returns, actions, weights)`` all of shape aligned with N.
        """
        noise_key, mix_key, step_key = jax.random.split(key, 3)

        # Sample perturbations: (N, H, action_dim)
        noise = jax.random.normal(noise_key, (num_samples, horizon, action_dim)) * noise_beta

        # Mixture: a fraction of trajectories is purely random
        num_random = max(1, int(num_samples * mixture_coef))
        random_actions = jax.random.uniform(
            mix_key,
            (num_random, horizon, action_dim),
            minval=action_low,
            maxval=action_high,
        )

        actions = jnp.clip(mu[None] + noise, action_low, action_high)
        actions = actions.at[:num_random].set(random_actions)

        # Broadcast initial state: (1, *state_shape) -> (N, *state_shape)
        init_state = jax.tree_util.tree_map(
            lambda s: jnp.broadcast_to(
                jnp.concatenate([s] * num_samples, axis=0)
                if s.shape[0] == 1
                else s,
                (num_samples,) + s.shape[1:],
            ),
            state,
        )

        # Generate per-step keys for all samples
        all_step_keys = jax.random.split(step_key, horizon * num_samples).reshape(
            num_samples, horizon, -1
        )

        # ---- vmapped single-trajectory rollout ----
        def _single_trajectory(s0: dict, acts: jax.Array, keys: jax.Array) -> jax.Array:
            """Roll out one action sequence and return the total return."""

            def _step(s, ha):
                a, k = ha
                ns, _ = imagine_fn(s, a[None], k, dynamics_params)
                # Remove the leading dim added for broadcasting in imagine_fn
                ns = jax.tree_util.tree_map(lambda x: x[0], ns)
                r = reward_fn(jax.tree_util.tree_map(lambda x: x[None], ns), a[None])
                return ns, r

            final_s, rewards = jax.lax.scan(_step, s0, (acts, keys))
            terminal_v = value_fn(
                jax.tree_util.tree_map(lambda x: x[None], final_s), acts[-1][None]
            )
            return jnp.sum(rewards) + terminal_v[0]

        # Squeeze batch dim from init_state for vmap
        init_state_single = jax.tree_util.tree_map(lambda x: x, init_state)
        returns = jax.vmap(_single_trajectory)(init_state_single, actions, all_step_keys)

        # Softmax weights
        returns_adj = returns - jnp.max(returns)
        weights = jax.nn.softmax(returns_adj / (temperature + 1e-8))

        return returns, actions, weights

    best_return = jnp.array(-jnp.inf)
    best_sequence = mu

    for _iter in range(num_iterations):
        key, sub_key = jax.random.split(key)
        returns, actions, weights = _evaluate_sequences(sub_key, mu)

        # Weighted mean update: new_mu = Σ w_i * A_i
        new_mu = jnp.einsum("n,nha->ha", weights, actions)

        # Momentum update
        mu = momentum * mu + (1.0 - momentum) * new_mu

        # Track best trajectory
        best_idx = jnp.argmax(returns)
        if returns[best_idx] > best_return:
            best_return = returns[best_idx]
            best_sequence = actions[best_idx]

    return best_sequence, best_return


def mppi_plan_jit(
    key: jax.Array,
    state: dict[str, jax.Array],
    imagine_fn: Callable,
    reward_fn: Callable,
    value_fn: Callable,
    mu: jax.Array,
    *,
    horizon: int = 5,
    num_samples: int = 512,
    action_dim: int,
    action_low: float = -1.0,
    action_high: float = 1.0,
    temperature: float = 0.5,
    noise_beta: float = 2.0,
    dynamics_params: dict | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single JIT-compilable MPPI iteration.

    Intended to be called in a Python loop (one iteration per call) so that
    the JAX program can be traced once and re-used efficiently.

    Args:
        key: PRNG key.
        state: Latent state, batched to ``(num_samples, ...)``.
        imagine_fn: Dynamics imagination function.
        reward_fn: Reward function.
        value_fn: Terminal value function.
        mu: Current action-sequence mean, shape ``(horizon, action_dim)``.
        (other args): Same as :func:`mppi_plan`.

    Returns:
        Tuple ``(new_mu, best_return, best_sequence)``.
    """
    noise_key, step_key = jax.random.split(key)
    noise = jax.random.normal(noise_key, (num_samples, horizon, action_dim)) * noise_beta
    actions = jnp.clip(mu[None] + noise, action_low, action_high)

    all_step_keys = jax.random.split(step_key, horizon * num_samples).reshape(
        num_samples, horizon, -1
    )

    def _single(s0, acts, keys):
        def _step(s, ha):
            a, k = ha
            ns, _ = imagine_fn(s, a[None], k, dynamics_params)
            ns = jax.tree_util.tree_map(lambda x: x[0], ns)
            r = reward_fn(jax.tree_util.tree_map(lambda x: x[None], ns), a[None])
            return ns, r

        final_s, rewards = jax.lax.scan(_step, s0, (acts, keys))
        v = value_fn(jax.tree_util.tree_map(lambda x: x[None], final_s), acts[-1][None])
        return jnp.sum(rewards) + v[0]

    returns = jax.vmap(_single)(state, actions, all_step_keys)
    weights = jax.nn.softmax((returns - jnp.max(returns)) / (temperature + 1e-8))
    new_mu = jnp.einsum("n,nha->ha", weights, actions)

    best_idx = jnp.argmax(returns)
    return new_mu, returns[best_idx], actions[best_idx]

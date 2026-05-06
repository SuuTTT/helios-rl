"""Abstract base class for dynamics / world models in helios-rl."""

from __future__ import annotations

import abc
from typing import Any

import jax


class BaseDynamics(abc.ABC):
    """Contract that every dynamics model must satisfy.

    A dynamics model maintains a *latent state* that summarises the history
    of observations and actions.  It provides two transition modes:

    * **Observe** (posterior): refines the state using the latest observation.
    * **Imagine** (prior): propagates the state using only an action, without
      observing the environment.  Used for "dreaming" inside the model.
    """

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def initial_state(self, batch_size: int) -> dict[str, jax.Array]:
        """Return a zeroed initial latent state for a batch of episodes.

        Args:
            batch_size: Number of parallel environments / sequences.

        Returns:
            A dict of JAX arrays representing the initial latent state.
        """

    # ------------------------------------------------------------------
    # Transition functions
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def observe(
        self,
        obs: jax.Array,
        prev_state: dict[str, jax.Array],
        action: jax.Array,
        key: jax.Array,
    ) -> tuple[dict[str, jax.Array], dict[str, Any]]:
        """Posterior update: q(s_t | s_{t-1}, a_{t-1}, o_t).

        Args:
            obs: Current observation, shape (batch, *obs_shape).
            prev_state: Previous latent state dict.
            action: Previous action, shape (batch, *act_shape).
            key: JAX PRNG key for stochastic sampling.

        Returns:
            Tuple of ``(new_state, extras)`` where ``extras`` may contain
            distribution parameters needed for loss computation (e.g. KL terms).
        """

    @abc.abstractmethod
    def imagine(
        self,
        prev_state: dict[str, jax.Array],
        action: jax.Array,
        key: jax.Array,
    ) -> tuple[dict[str, jax.Array], dict[str, Any]]:
        """Prior prediction: p(s_t | s_{t-1}, a_{t-1}).

        Args:
            prev_state: Previous latent state dict.
            action: Action to apply, shape (batch, *act_shape).
            key: JAX PRNG key for stochastic sampling.

        Returns:
            Tuple of ``(new_state, extras)``.
        """

    # ------------------------------------------------------------------
    # Optional helpers (may be overridden)
    # ------------------------------------------------------------------

    def get_feat(self, state: dict[str, jax.Array]) -> jax.Array:
        """Concatenate all state tensors into a flat feature vector.

        Default implementation flattens and concatenates all state values.
        Override in subclasses to customise the feature representation.

        Args:
            state: Latent state dict.

        Returns:
            Feature array, shape (batch, feature_dim).
        """
        import jax.numpy as jnp

        parts = [v.reshape(v.shape[0], -1) for v in state.values()]
        return jnp.concatenate(parts, axis=-1)

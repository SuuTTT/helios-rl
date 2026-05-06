"""Probability distributions used throughout helios-rl.

Includes:
- TanhNormal: Tanh-squashed diagonal Gaussian (SAC/PPO continuous actions).
- OneHotCategorical: Straight-through estimator for discrete latents (DreamerV3).
- Independent: Wrapper to treat last N dims as batch dimensions of a distribution.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct


# ---------------------------------------------------------------------------
# TanhNormal
# ---------------------------------------------------------------------------


class TanhNormal:
    """Diagonal Gaussian squashed through tanh for bounded continuous actions.

    Provides log-probability computation with the change-of-variables correction:
        log p(a) = log p(u) - sum(log(1 - tanh(u)^2))
    where u = atanh(a) is the pre-squash sample.

    Args:
        mean: Mean of the pre-squash Gaussian, shape (..., action_dim).
        log_std: Log standard deviation, shape (..., action_dim).
        min_log_std: Clipping minimum for numerical stability.
        max_log_std: Clipping maximum.
    """

    def __init__(
        self,
        mean: jax.Array,
        log_std: jax.Array,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ) -> None:
        self.mean = mean
        self.log_std = jnp.clip(log_std, min_log_std, max_log_std)
        self.std = jnp.exp(self.log_std)

    def sample(self, key: jax.Array) -> jax.Array:
        """Draw a reparameterised sample and squash through tanh."""
        eps = jax.random.normal(key, self.mean.shape)
        u = self.mean + self.std * eps
        return jnp.tanh(u), u  # returns (action, pre_squash)

    def log_prob(self, action: jax.Array, pre_squash: jax.Array | None = None) -> jax.Array:
        """Log-probability of a sample.

        If ``pre_squash`` is provided it is used directly; otherwise it is
        derived via atanh (may be numerically imprecise near ±1).
        """
        if pre_squash is None:
            pre_squash = jnp.arctanh(jnp.clip(action, -0.9999, 0.9999))
        gaussian_logp = -0.5 * (
            ((pre_squash - self.mean) / (self.std + 1e-8)) ** 2
            + 2 * self.log_std
            + jnp.log(2 * jnp.pi)
        )
        log_det = jnp.sum(
            jnp.log(jnp.maximum(1 - jnp.tanh(pre_squash) ** 2, 1e-6)),
            axis=-1,
        )
        return jnp.sum(gaussian_logp, axis=-1) + log_det

    def mode(self) -> jax.Array:
        """Deterministic action (tanh of the mean)."""
        return jnp.tanh(self.mean)

    def entropy(self) -> jax.Array:
        """Approximate entropy of the squashed Gaussian (no closed form; use Gaussian lower bound)."""
        return jnp.sum(self.log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)


# ---------------------------------------------------------------------------
# OneHotCategorical with straight-through gradient
# ---------------------------------------------------------------------------


class OneHotCategorical:
    """Categorical distribution returning one-hot samples via the straight-through estimator.

    Used by DreamerV3 for discrete stochastic latents.

    Args:
        logits: Unnormalised log-probabilities, shape (..., num_classes).
        straight_through: If True, use straight-through gradient estimator for samples.
    """

    def __init__(self, logits: jax.Array, straight_through: bool = True) -> None:
        self.logits = logits
        self.probs = jax.nn.softmax(logits, axis=-1)
        self.straight_through = straight_through

    def sample(self, key: jax.Array) -> jax.Array:
        """Sample one-hot vectors of shape (..., num_classes)."""
        indices = jax.random.categorical(key, self.logits)
        one_hot = jax.nn.one_hot(indices, self.logits.shape[-1])
        if self.straight_through:
            # Straight-through: forward is one_hot, backward flows through probs
            one_hot = one_hot + self.probs - jax.lax.stop_gradient(self.probs)
        return one_hot

    def log_prob(self, one_hot: jax.Array) -> jax.Array:
        """Log-probability of one-hot samples."""
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        return jnp.sum(one_hot * log_probs, axis=-1)

    def entropy(self) -> jax.Array:
        """Categorical entropy, shape (...)."""
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        return -jnp.sum(self.probs * log_probs, axis=-1)

    def mode(self) -> jax.Array:
        """One-hot of the argmax, shape (..., num_classes)."""
        indices = jnp.argmax(self.logits, axis=-1)
        return jax.nn.one_hot(indices, self.logits.shape[-1])

    def kl_divergence(self, other: "OneHotCategorical") -> jax.Array:
        """KL(self || other), shape (...)."""
        log_p = jax.nn.log_softmax(self.logits, axis=-1)
        log_q = jax.nn.log_softmax(other.logits, axis=-1)
        return jnp.sum(self.probs * (log_p - log_q), axis=-1)


# ---------------------------------------------------------------------------
# Gaussian (for continuous world model outputs)
# ---------------------------------------------------------------------------


class Gaussian:
    """Diagonal Gaussian distribution.

    Args:
        mean: Mean tensor, shape (..., dim).
        log_std: Log-standard-deviation tensor, shape (..., dim).
    """

    def __init__(self, mean: jax.Array, log_std: jax.Array) -> None:
        self.mean = mean
        self.std = jnp.exp(log_std)
        self.log_std = log_std

    def sample(self, key: jax.Array) -> jax.Array:
        eps = jax.random.normal(key, self.mean.shape)
        return self.mean + self.std * eps

    def log_prob(self, x: jax.Array) -> jax.Array:
        return jnp.sum(
            -0.5 * (((x - self.mean) / (self.std + 1e-8)) ** 2 + 2 * self.log_std + jnp.log(2 * jnp.pi)),
            axis=-1,
        )

    def mode(self) -> jax.Array:
        return self.mean

    def entropy(self) -> jax.Array:
        return jnp.sum(self.log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)

    def kl_divergence(self, other: "Gaussian") -> jax.Array:
        """KL(self || other)."""
        return jnp.sum(
            jnp.log(other.std / (self.std + 1e-8))
            + (self.std**2 + (self.mean - other.mean) ** 2) / (2 * other.std**2 + 1e-8)
            - 0.5,
            axis=-1,
        )

"""Classic Ha & Schmidhuber (2018) World Model: RNN + Mixture Density Network.

The *World Model* (Ha & Schmidhuber, 2018) decomposes into three modules:
1. **V** – A visual encoder (CNN or VAE encoder) that maps observations to
   compact latent vectors ``z``.
2. **M** – A memory module (MDN-RNN) that predicts the *distribution* over
   future latent states given the current hidden state and action.
3. **C** – A linear controller that maps (z, h) → action.

This file implements the **M** component: an LSTM whose output head is a
Mixture Density Network (MDN) with ``num_mixtures`` Gaussian components.

Reference:
    Ha & Schmidhuber (2018) - "Recurrent World Models Facilitate Policy Evolution"
    https://arxiv.org/abs/1809.01999
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from helios.core.networks import MLP
from helios.dynamics.base import BaseDynamics


# ---------------------------------------------------------------------------
# MDN head
# ---------------------------------------------------------------------------


class MDNHead(nn.Module):
    """Mixture Density Network output head.

    Outputs the parameters of a Gaussian mixture over a ``latent_dim``-
    dimensional target variable.

    Args:
        latent_dim: Dimensionality of the latent space being modelled.
        num_mixtures: Number of Gaussian mixture components.
    """

    latent_dim: int
    num_mixtures: int = 5

    @nn.compact
    def __call__(self, h: jax.Array) -> dict[str, jax.Array]:
        """Predict MDN parameters from LSTM hidden state ``h``.

        Args:
            h: Hidden state tensor, shape (..., hidden_dim).
        Returns:
            Dict with:
            - ``log_pi``: Log mixture weights (log-softmax), shape (..., K).
            - ``mu``: Component means, shape (..., K, latent_dim).
            - ``log_sigma``: Log component std-devs, shape (..., K, latent_dim).
        """
        K = self.num_mixtures
        D = self.latent_dim

        raw = nn.Dense(K * (2 * D + 1))(h)

        log_pi = jax.nn.log_softmax(raw[..., :K], axis=-1)
        mu = raw[..., K : K + K * D].reshape(*raw.shape[:-1], K, D)
        log_sigma = raw[..., K + K * D :].reshape(*raw.shape[:-1], K, D)
        log_sigma = jnp.clip(log_sigma, -4.0, 15.0)

        return {"log_pi": log_pi, "mu": mu, "log_sigma": log_sigma}


# ---------------------------------------------------------------------------
# MDN-RNN module
# ---------------------------------------------------------------------------


class MDNRNNModule(nn.Module):
    """MDN-RNN: LSTM + Mixture Density Network head.

    Args:
        hidden_dim: LSTM hidden size.
        latent_dim: Latent space dimension (size of ``z`` vectors).
        num_mixtures: Number of Gaussian mixture components.
    """

    hidden_dim: int = 256
    latent_dim: int = 32
    num_mixtures: int = 5

    @nn.compact
    def __call__(
        self,
        carry: tuple[jax.Array, jax.Array],
        z_and_action: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array], dict[str, jax.Array]]:
        """Single LSTM step.

        Args:
            carry: Tuple ``(c, h)`` of LSTM cell & hidden state,
                   each shape (batch, hidden_dim).
            z_and_action: Concatenation of latent z and action,
                          shape (batch, latent_dim + action_dim).
        Returns:
            ``(new_carry, mdn_params)`` where ``mdn_params`` is the output of
            :class:`MDNHead`.
        """
        new_carry, h = nn.LSTMCell(self.hidden_dim)(carry, z_and_action)
        mdn_params = MDNHead(self.latent_dim, self.num_mixtures)(h)
        return new_carry, {**mdn_params, "h": h}

    def initial_carry(self, batch_size: int) -> tuple[jax.Array, jax.Array]:
        """Return zeroed LSTM carry."""
        zeros = jnp.zeros((batch_size, self.hidden_dim))
        return (zeros, zeros)  # (c, h)


# ---------------------------------------------------------------------------
# RNNMDNDynamics implementing BaseDynamics
# ---------------------------------------------------------------------------


class RNNMDNDynamics(BaseDynamics):
    """Ha & Schmidhuber (2018) world model dynamics component.

    Wraps :class:`MDNRNNModule` and provides the :class:`BaseDynamics`
    interface.  The latent state is the LSTM hidden state ``(c, h)``.

    Args:
        config: DictConfig with hyperparameters.
        latent_dim: Dimension of the visual latent ``z``.
        action_dim: Action space dimensionality.
    """

    def __init__(self, config: Any, latent_dim: int, action_dim: int) -> None:
        self.config = config
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.module = MDNRNNModule(
            hidden_dim=int(getattr(config, "hidden_dim", 256)),
            latent_dim=latent_dim,
            num_mixtures=int(getattr(config, "num_mixtures", 5)),
        )

    # ------------------------------------------------------------------
    # BaseDynamics interface
    # ------------------------------------------------------------------

    def initial_state(self, batch_size: int) -> dict[str, jax.Array]:
        """Return zeroed LSTM carry."""
        c, h = self.module.initial_carry(batch_size)
        return {"c": c, "h": h}

    def observe(
        self,
        obs: jax.Array,
        prev_state: dict[str, jax.Array],
        action: jax.Array,
        key: jax.Array,
        params: dict,
    ) -> tuple[dict[str, jax.Array], dict[str, Any]]:
        """Update LSTM state given encoded observation ``z`` and action.

        Args:
            obs: Encoded latent vector ``z``, shape (batch, latent_dim).
            prev_state: Dict with LSTM ``c`` and ``h``.
            action: Action, shape (batch, action_dim).
            key: Unused (deterministic observe step).
            params: Flax module parameters.

        Returns:
            ``(new_state, mdn_params)`` where mdn_params has mixture weights,
            means, and log-sigmas for the *predicted* next ``z``.
        """
        carry = (prev_state["c"], prev_state["h"])
        inp = jnp.concatenate([obs, action], axis=-1)
        new_carry, extras = self.module.apply(params, carry, inp)
        new_state = {"c": new_carry[0], "h": new_carry[1]}
        return new_state, extras

    def imagine(
        self,
        prev_state: dict[str, jax.Array],
        action: jax.Array,
        key: jax.Array,
        params: dict,
    ) -> tuple[dict[str, jax.Array], dict[str, Any]]:
        """Sample a next latent ``z`` from the MDN and propagate.

        Args:
            prev_state: Dict with LSTM ``c`` and ``h``.
            action: Action, shape (batch, action_dim).
            key: PRNG key for sampling from the mixture.
            params: Flax module parameters.

        Returns:
            ``(new_state, extras)`` where extras contains the sampled ``z_next``
            and MDN parameters.
        """
        carry = (prev_state["c"], prev_state["h"])

        # Sample z from the MDN prediction based on prev h and action
        mdn_h = prev_state["h"]
        mdn_params = MDNHead(self.latent_dim, self.module.num_mixtures).apply(
            {"params": params["params"]["MDNHead_0"]}, mdn_h
        )
        z_sampled = _sample_mdn(key, mdn_params)

        # Propagate LSTM with the sampled z
        inp = jnp.concatenate([z_sampled, action], axis=-1)
        new_carry, extras = self.module.apply(params, carry, inp)
        new_state = {"c": new_carry[0], "h": new_carry[1]}
        return new_state, {**extras, "z_sampled": z_sampled}

    def get_feat(self, state: dict[str, jax.Array]) -> jax.Array:
        """Return the LSTM hidden state as the feature."""
        return state["h"]

    @property
    def feature_dim(self) -> int:
        return self.module.hidden_dim


# ---------------------------------------------------------------------------
# MDN sampling & loss
# ---------------------------------------------------------------------------


def _sample_mdn(key: jax.Array, mdn_params: dict[str, jax.Array]) -> jax.Array:
    """Sample a latent vector from a mixture of Gaussians.

    Args:
        key: PRNG key.
        mdn_params: Dict with ``log_pi`` (..., K), ``mu`` (..., K, D),
                    ``log_sigma`` (..., K, D).

    Returns:
        Sample of shape (..., D).
    """
    key_k, key_z = jax.random.split(key)
    log_pi = mdn_params["log_pi"]      # (..., K)
    mu = mdn_params["mu"]               # (..., K, D)
    log_sigma = mdn_params["log_sigma"] # (..., K, D)

    # Pick mixture component
    k = jax.random.categorical(key_k, log_pi)  # (...,) int
    # Gather selected component
    mu_k = mu[..., k, :]              # (..., D)  -- works for 1-D batch only
    sigma_k = jnp.exp(log_sigma[..., k, :])  # (..., D)

    eps = jax.random.normal(key_z, mu_k.shape)
    return mu_k + sigma_k * eps


def mdn_loss(
    mdn_params: dict[str, jax.Array],
    target: jax.Array,
) -> jax.Array:
    """Negative log-likelihood of the target under the MDN.

    Args:
        mdn_params: Dict with ``log_pi`` (B, K), ``mu`` (B, K, D),
                    ``log_sigma`` (B, K, D).
        target: Ground-truth latent ``z``, shape (B, D).

    Returns:
        Scalar mean NLL.
    """
    log_pi = mdn_params["log_pi"]      # (B, K)
    mu = mdn_params["mu"]              # (B, K, D)
    log_sigma = mdn_params["log_sigma"] # (B, K, D)
    sigma = jnp.exp(log_sigma)

    # Per-component log N(z; mu_k, sigma_k)
    z = target[:, None, :]  # (B, 1, D)
    log_p_k = -0.5 * jnp.sum(
        ((z - mu) / (sigma + 1e-8)) ** 2 + 2 * log_sigma + jnp.log(2 * jnp.pi),
        axis=-1,
    )  # (B, K)

    # log-sum-exp to get log p(z)
    log_p = jax.nn.logsumexp(log_pi + log_p_k, axis=-1)  # (B,)
    return -jnp.mean(log_p)

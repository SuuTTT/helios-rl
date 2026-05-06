"""Common Flax neural network building blocks used across helios-rl algorithms."""

from typing import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

ACTIVATIONS: dict[str, Callable] = {
    "relu": nn.relu,
    "tanh": nn.tanh,
    "elu": nn.elu,
    "silu": nn.silu,
    "gelu": nn.gelu,
    "sigmoid": nn.sigmoid,
    "identity": lambda x: x,
}


def get_activation(name: str) -> Callable:
    """Return a Flax-compatible activation function by name."""
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(ACTIVATIONS)}")
    return ACTIVATIONS[name]


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """General-purpose Multi-Layer Perceptron.

    Args:
        hidden_dims: Sequence of hidden layer widths.
        output_dim: Output dimension. If None, the last hidden layer IS the output.
        activation: Name of intermediate activation function.
        output_activation: Name of activation applied after the final layer.
        use_layer_norm: Whether to apply LayerNorm before every activation.
    """

    hidden_dims: Sequence[int]
    output_dim: int | None = None
    activation: str = "tanh"
    output_activation: str = "identity"
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        act = get_activation(self.activation)
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = act(x)
        if self.output_dim is not None:
            x = nn.Dense(self.output_dim)(x)
            x = get_activation(self.output_activation)(x)
        return x


# ---------------------------------------------------------------------------
# CNN Encoder / Decoder (for pixel observations)
# ---------------------------------------------------------------------------


class CNNEncoder(nn.Module):
    """Convolutional encoder for image observations (e.g. 64×64 RGB).

    Args:
        depth: Base channel multiplier (DreamerV3 convention).
        activation: Activation between conv layers.
        embed_dim: Output embedding dimension. If None, output is flattened features.
    """

    depth: int = 48
    activation: str = "silu"
    embed_dim: int | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: Image tensor of shape (..., H, W, C).
        Returns:
            Flat embedding of shape (..., embed_dim).
        """
        act = get_activation(self.activation)
        # DreamerV3-style: 4 strided convolutions halving spatial resolution
        kernels = [4, 4, 4, 4]
        channels = [1 * self.depth, 2 * self.depth, 4 * self.depth, 8 * self.depth]
        leading = x.shape[:-3]
        x = x.reshape((-1,) + x.shape[-3:])  # flatten leading dims
        for k, c in zip(kernels, channels):
            x = nn.Conv(c, kernel_size=(k, k), strides=(2, 2), padding="VALID")(x)
            x = act(x)
        x = x.reshape(leading + (-1,))  # re-attach leading dims
        if self.embed_dim is not None:
            x = nn.Dense(self.embed_dim)(x)
        return x


class CNNDecoder(nn.Module):
    """Transposed-convolutional decoder that mirrors CNNEncoder.

    Args:
        depth: Base channel multiplier (must match encoder).
        output_channels: Number of output image channels (e.g. 3 for RGB).
        activation: Activation between conv-transpose layers.
    """

    depth: int = 48
    output_channels: int = 3
    activation: str = "silu"

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: Latent vector of shape (..., latent_dim).
        Returns:
            Reconstructed image of shape (..., 64, 64, output_channels).
        """
        act = get_activation(self.activation)
        leading = x.shape[:-1]
        x = x.reshape((-1, x.shape[-1]))
        x = nn.Dense(32 * self.depth)(x)
        x = x.reshape((-1, 1, 1, 32 * self.depth))
        kernels = [5, 5, 6, 6]
        channels = [4 * self.depth, 2 * self.depth, 1 * self.depth, self.output_channels]
        for i, (k, c) in enumerate(zip(kernels, channels)):
            x = nn.ConvTranspose(c, kernel_size=(k, k), strides=(2, 2), padding="VALID")(x)
            if i < len(kernels) - 1:
                x = act(x)
        x = x.reshape(leading + x.shape[-3:])
        return x


# ---------------------------------------------------------------------------
# GRU cell (for RSSM deterministic path)
# ---------------------------------------------------------------------------


class GRUCell(nn.Module):
    """Single GRU cell implemented as a Flax Module.

    Args:
        hidden_dim: Dimensionality of the hidden state.
    """

    hidden_dim: int

    @nn.compact
    def __call__(self, carry: jax.Array, inputs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Apply one GRU step.

        Args:
            carry: Previous hidden state of shape (..., hidden_dim).
            inputs: Current input of shape (..., input_dim).
        Returns:
            Tuple of (new_hidden, new_hidden) following flax RNNCell convention.
        """
        new_carry, _ = nn.GRUCell(self.hidden_dim)(carry, inputs)
        return new_carry, new_carry

    def initial_carry(self, batch_size: int) -> jax.Array:
        return jnp.zeros((batch_size, self.hidden_dim))


# ---------------------------------------------------------------------------
# Linear projections
# ---------------------------------------------------------------------------


class NormedLinear(nn.Module):
    """Dense layer followed by LayerNorm and an activation.

    Useful as a building block for latent encoders.
    """

    features: int
    activation: str = "silu"

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(self.features)(x)
        x = nn.LayerNorm()(x)
        x = get_activation(self.activation)(x)
        return x

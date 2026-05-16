"""TD-MPC-Glass JAX implementation.

Experimental TD-MPC2 variant that adds a low-overhead Glass-SE auxiliary loss
over prototype transition matrices. The baseline TD-MPC2 implementation remains
in ``tdmpc2.py``; this file is intentionally separate for fast iteration.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


GLASS_DEFAULTS = dict(
    enabled=True,
    warmup_env_steps=100_000,
    every_k_updates=4,
    num_prototypes=16,
    num_clusters=8,
    proto_temperature=1.0,
    assignment_temperature=1.0,
    lambda_se=5.0e-3,
    lambda_balance=1.0e-2,
    lambda_temporal=1.0e-3,
    stopgrad_graph=True,
    diag_dump_matrices=True,
    # Phase 1 additions
    assign_logits_init_scale=1.0,
    glass_lr_mult=10.0,
    use_cosine_assign=True,
)


# ---------------------------------------------------------------------------
# SimNorm and math utilities
# ---------------------------------------------------------------------------


def simnorm(x: jax.Array, V: int = 8) -> jax.Array:
    """Simplex normalisation: partition into V groups, softmax each group."""
    s = x.shape
    x = x.reshape(*s[:-1], V, s[-1] // V)
    x = jax.nn.softmax(x, axis=-1)
    return x.reshape(*s)


def log_std_fn(x: jax.Array, low: float = -10.0, dif: float = 12.0) -> jax.Array:
    """Map raw network output to log_std in [low, low+dif]."""
    return low + 0.5 * dif * (jnp.tanh(x) + 1.0)


def gaussian_logprob(eps: jax.Array, log_std: jax.Array) -> jax.Array:
    """Log probability of eps under N(0,1) reparameterized with log_std."""
    residual = -0.5 * (eps ** 2) - log_std
    log_prob = residual - 0.9189385175704956  # -0.5 * log(2π)
    return jnp.sum(log_prob, axis=-1, keepdims=True)


def squash(
    mu: jax.Array, pi: jax.Array, log_pi: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply tanh squashing with Jacobian correction to log_pi."""
    mu = jnp.tanh(mu)
    pi = jnp.tanh(pi)
    squashed_pi = jnp.log(jax.nn.relu(1 - pi ** 2) + 1e-6)
    log_pi = log_pi - jnp.sum(squashed_pi, axis=-1, keepdims=True)
    return mu, pi, log_pi


def symlog(x: jax.Array) -> jax.Array:
    """Symmetric log: sign(x) * log(1 + |x|)."""
    return jnp.sign(x) * jnp.log(1 + jnp.abs(x))


def symexp(x: jax.Array) -> jax.Array:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def two_hot(
    x: jax.Array, vmin: float = -20.0, vmax: float = 20.0, num_bins: int = 101
) -> jax.Array:
    """Two-hot encoding in symlog space.

    Clips symlog(x) to [vmin, vmax] then distributes weight between the two
    nearest bin indices proportionally.
    """
    x = jnp.clip(symlog(x), vmin, vmax)
    bin_size = (vmax - vmin) / (num_bins - 1)
    bin_index = (x - vmin) / bin_size
    lower = jnp.floor(bin_index).astype(jnp.int32)
    upper = jnp.ceil(bin_index).astype(jnp.int32)
    p_upper = bin_index - lower
    p_lower = 1.0 - p_upper
    lower_hot = jax.nn.one_hot(lower, num_bins) * p_lower[..., None]
    upper_hot = jax.nn.one_hot(upper, num_bins) * p_upper[..., None]
    return lower_hot + upper_hot


def soft_ce(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Cross-entropy loss for distributional targets."""
    return -jnp.sum(target * jax.nn.log_softmax(pred, axis=-1), axis=-1)


def two_hot_inv(
    logits: jax.Array, vmin: float = -20.0, vmax: float = 20.0, num_bins: int = 101
) -> jax.Array:
    """Decode distributional logits back to scalar in original space."""
    probs = jax.nn.softmax(logits, axis=-1)
    bins = jnp.linspace(vmin, vmax, num_bins)
    return symexp(jnp.sum(probs * bins, axis=-1))


# ---------------------------------------------------------------------------
# Glass-SE utilities
# ---------------------------------------------------------------------------


def init_glass_params(
    key: jax.Array,
    latent_dim: int,
    num_prototypes: int = 32,
    num_clusters: int = 8,
    assign_logits_init_scale: float = 1.0,
    num_super_clusters: int = 0,
) -> dict:
    """Initialize prototype and assignment parameters for TD-MPC-Glass.

    Args:
        num_super_clusters: if > 0, also initialise a second-level
            assign_logits ``(num_clusters, num_super_clusters)`` for
            hierarchical Glass (iteration-4 §7.4 design). When 0 (default)
            the partition is flat (matches Phase 1b/o behaviour).
    """
    pk, sk = jax.random.split(key)
    proto_raw = jax.random.normal(pk, (num_prototypes, latent_dim))
    proto_groups = 8 if latent_dim % 8 == 0 else 1
    prototypes = simnorm(proto_raw, V=proto_groups)
    # Larger init breaks the near-uniform symmetry of S (uniform softmax has
    # vanishing 2D-SE gradient).
    assign_logits = assign_logits_init_scale * jax.random.normal(
        sk, (num_prototypes, num_clusters)
    )
    out = {
        "prototypes": prototypes,
        "assign_logits": assign_logits,
    }
    if num_super_clusters > 0:
        sk2 = jax.random.fold_in(sk, 1)
        out["super_assign_logits"] = assign_logits_init_scale * jax.random.normal(
            sk2, (num_clusters, num_super_clusters)
        )
    return out


def one_dimensional_structural_entropy(
    A: jax.Array, mask: jax.Array | None = None, eps: float = 1e-8
) -> jax.Array:
    """JAX implementation matching Glass-JAX's 1D structural entropy."""
    if mask is not None:
        A = A * mask[:, None] * mask[None, :]
    d = jnp.sum(A, axis=-1)
    two_m = jnp.sum(d)
    p = jnp.clip(d / (two_m + eps), eps, 1.0)
    if mask is not None:
        p = p * mask
    return -jnp.sum(p * jnp.log2(p))


def two_dimensional_structural_entropy(
    A: jax.Array,
    S: jax.Array,
    mask: jax.Array | None = None,
    is_logits: bool = True,
    eps: float = 1e-8,
) -> jax.Array:
    """Differentiable 2D structural entropy from Glass-JAX."""
    if is_logits:
        S = jax.nn.softmax(S, axis=-1)
    if mask is not None:
        S = S * mask[:, None]
        A = A * mask[:, None] * mask[None, :]

    d = jnp.sum(A, axis=-1)
    two_m = jnp.sum(d)
    V = jnp.dot(d, S)
    AS = jnp.dot(A, S)
    g = jnp.sum(S * (d[:, None] - AS), axis=0)

    p_vol = V / (two_m + eps)
    p_cut = g / (two_m + eps)
    term1 = -jnp.sum(p_cut * jnp.log2(jnp.clip(p_vol, eps, 1.0)))

    h1 = one_dimensional_structural_entropy(A, mask=mask, eps=eps)
    term2 = h1 + jnp.sum(p_vol * jnp.log2(jnp.clip(p_vol, eps, 1.0)))
    return term1 + term2


def glass_transition_graph(
    z_src: jax.Array,
    z_next: jax.Array,
    glass_params: dict,
    proto_temperature: float = 1.0,
    assignment_temperature: float = 1.0,
    stopgrad_graph: bool = False,
    use_cosine_assign: bool = True,
    eps: float = 1e-8,
) -> dict:
    """Build a prototype transition graph and cluster diagnostics.

    Phase 1 changes:
        - ``stopgrad_graph`` defaults to False so encoder/dynamics receive a
          gradient from the Glass loss through ``z_src``. ``z_next`` is still
          stop-gradiented to avoid a bootstrap loop.
        - Cosine assignment (default) is well-conditioned for SimNorm latents.
        - Balance terms are one-sided hinges that only fire on cluster
          collapse, so they do not pin S to uniform.
    """
    # Always stop gradient on z_next to avoid bootstrap; optionally allow
    # gradient through z_src so Glass shapes the encoder/dynamics.
    z_next = jax.lax.stop_gradient(z_next)
    if stopgrad_graph:
        z_src = jax.lax.stop_gradient(z_src)

    prototypes = glass_params["prototypes"]
    assign_logits = glass_params["assign_logits"]

    if use_cosine_assign:
        def soft_assign(z):
            zn = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + eps)
            pn = prototypes / (
                jnp.linalg.norm(prototypes, axis=-1, keepdims=True) + eps
            )
            sim = zn @ pn.T  # (N, P), in [-1, 1]
            return jax.nn.softmax(sim / proto_temperature, axis=-1)
    else:
        def soft_assign(z):
            dist2 = jnp.sum((z[:, None, :] - prototypes[None, :, :]) ** 2, axis=-1)
            return jax.nn.softmax(-dist2 / proto_temperature, axis=-1)

    c_src = soft_assign(z_src)
    c_next = soft_assign(z_next)
    P_counts = jnp.einsum("nk,nl->kl", c_src, c_next)
    # Smooth rows so unused prototypes do not create zero-volume graph nodes.
    P = P_counts + 1e-4
    P = P / (jnp.sum(P, axis=-1, keepdims=True) + eps)
    A = 0.5 * (P + P.T)

    S = jax.nn.softmax(assign_logits / assignment_temperature, axis=-1)
    cluster_mass = jnp.mean(S, axis=0)
    K = cluster_mass.shape[0]
    # One-sided hinge: only penalise when a single cluster absorbs > 2/K mass.
    # Square keeps the term smooth at the threshold.
    balance = jnp.sum(jax.nn.relu(cluster_mass - 2.0 / K) ** 2)
    proto_mass = jnp.mean(c_src, axis=0)
    Kp = proto_mass.shape[0]
    proto_balance = jnp.sum(jax.nn.relu(proto_mass - 2.0 / Kp) ** 2)

    s_src = jnp.matmul(c_src, S)
    s_next = jnp.matmul(c_next, S)
    temporal = jnp.mean(jnp.sum((s_src - jax.lax.stop_gradient(s_next)) ** 2, axis=-1))

    se = two_dimensional_structural_entropy(A, assign_logits, is_logits=True)
    entropy = -jnp.sum(cluster_mass * jnp.log(jnp.clip(cluster_mass, eps, 1.0)))
    active = jnp.sum(cluster_mass > (0.05 / cluster_mass.shape[0]))
    labels = jnp.argmax(S, axis=-1)
    cut = jnp.sum(P * (labels[:, None] != labels[None, :])) / (jnp.sum(P) + eps)

    out = {
        "P": P,
        "A": A,
        "S": S,
        "se": se,
        "balance": balance,
        "proto_balance": proto_balance,
        "temporal": temporal,
        "entropy": entropy,
        "active_clusters": active.astype(jnp.float32),
        "max_cluster_mass": jnp.max(cluster_mass),
        "transition_cut_mass": cut,
    }

    # Hierarchical Glass (iteration-4 §7.4): if super_assign_logits exists,
    # compute a coarse partition S_super and an SE term on the super-graph.
    # Pipeline: S_sub=S (N×K_sub), S_super=softmax(L_super) (K_sub×K_super),
    # S_combined = S_sub @ S_super gives the effective fine→coarse map (N×K_super).
    if "super_assign_logits" in glass_params:
        L_super = glass_params["super_assign_logits"]
        S_super = jax.nn.softmax(L_super / assignment_temperature, axis=-1)
        S_combined = jnp.matmul(S, S_super)              # (N, K_super)
        super_cluster_mass = jnp.mean(S_combined, axis=0)
        Ks = super_cluster_mass.shape[0]
        super_balance = jnp.sum(jax.nn.relu(super_cluster_mass - 2.0 / Ks) ** 2)
        # SE on the SAME prototype graph A but with the coarse partition.
        # We pass S_combined as logits=False because it's already a probability.
        super_se = two_dimensional_structural_entropy(A, S_combined, is_logits=False)
        super_entropy = -jnp.sum(super_cluster_mass * jnp.log(jnp.clip(super_cluster_mass, eps, 1.0)))
        super_active = jnp.sum(super_cluster_mass > (0.05 / Ks))
        super_labels = jnp.argmax(S_combined, axis=-1)
        super_cut = jnp.sum(P * (super_labels[:, None] != super_labels[None, :])) / (jnp.sum(P) + eps)
        out["super_se"] = super_se
        out["super_balance"] = super_balance
        out["super_entropy"] = super_entropy
        out["super_active"] = super_active.astype(jnp.float32)
        out["super_cut"] = super_cut
        out["S_super"] = S_super
    return out


def glass_loss_and_aux(
    z_src: jax.Array,
    z_next: jax.Array,
    glass_params: dict,
    proto_temperature: float = 1.0,
    assignment_temperature: float = 1.0,
    lambda_se: float = 5.0e-3,
    lambda_balance: float = 1.0e-2,
    lambda_temporal: float = 1.0e-3,
    stopgrad_graph: bool = False,
    use_cosine_assign: bool = True,
    lambda_super_se: float = 0.0,
    lambda_super_balance: float = 0.0,
) -> tuple[jax.Array, dict]:
    """Return weighted Glass loss and scalar diagnostics.

    Hierarchical (iteration-4 §7.4): if glass_params contains
    'super_assign_logits' and lambda_super_se > 0, the loss includes a
    coarse-partition SE term on the same prototype graph A. This is the
    flat-Glass behaviour when those args are 0 (default).
    """
    diag = glass_transition_graph(
        z_src,
        z_next,
        glass_params,
        proto_temperature=proto_temperature,
        assignment_temperature=assignment_temperature,
        stopgrad_graph=stopgrad_graph,
        use_cosine_assign=use_cosine_assign,
    )
    total = (
        lambda_se * diag["se"]
        + lambda_balance * (diag["balance"] + diag["proto_balance"])
        + lambda_temporal * diag["temporal"]
    )
    aux = {
        "glass_se": diag["se"],
        "glass_balance": diag["balance"],
        "glass_proto_balance": diag["proto_balance"],
        "glass_temp": diag["temporal"],
        "glass_total": total,
        "glass_entropy": diag["entropy"],
        "glass_active_clusters": diag["active_clusters"],
        "glass_max_cluster_mass": diag["max_cluster_mass"],
        "glass_transition_cut_mass": diag["transition_cut_mass"],
    }
    # Hierarchical extension: only contributes if super_assign_logits exists
    # in params AND a non-zero coefficient is provided.
    if "super_se" in diag and (lambda_super_se > 0 or lambda_super_balance > 0):
        total = total + lambda_super_se * diag["super_se"] + lambda_super_balance * diag["super_balance"]
        aux["glass_super_se"] = diag["super_se"]
        aux["glass_super_balance"] = diag["super_balance"]
        aux["glass_super_entropy"] = diag["super_entropy"]
        aux["glass_super_active"] = diag["super_active"]
        aux["glass_super_cut"] = diag["super_cut"]
        aux["glass_total"] = total
    return total, aux


# ---------------------------------------------------------------------------
# Network modules
# ---------------------------------------------------------------------------


class NormMLP(nn.Module):
    """Dense → LayerNorm → SiLU stack with a final linear output."""

    dims: tuple[int, ...]
    out: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for d in self.dims:
            x = nn.Dense(d)(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
        return nn.Dense(self.out)(x)


class Encoder(nn.Module):
    """Encodes observations to SimNorm-bounded latent vectors."""

    latent_dim: int
    hidden: tuple[int, ...] = (512, 512)
    V: int = 8

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        return simnorm(NormMLP(self.hidden, self.latent_dim)(obs), self.V)


class Dynamics(nn.Module):
    """Predicts next latent from (z, a) using SimNorm-bounded output."""

    latent_dim: int
    hidden: tuple[int, ...] = (512, 512)
    V: int = 8

    @nn.compact
    def __call__(self, z: jax.Array, a: jax.Array) -> jax.Array:
        return simnorm(
            NormMLP(self.hidden, self.latent_dim)(jnp.concatenate([z, a], -1)), self.V
        )


class RewardHead(nn.Module):
    """Distributional reward predictor (101-bin two-hot)."""

    hidden: tuple[int, ...] = (512, 512)
    num_bins: int = 101

    @nn.compact
    def __call__(self, z: jax.Array, a: jax.Array) -> jax.Array:
        return NormMLP(self.hidden, self.num_bins)(jnp.concatenate([z, a], -1))


class QEnsemble(nn.Module):
    """Two-head Q ensemble with distributional outputs (101-bin two-hot).

    Returns shape (..., 2, num_bins). Take jnp.min over axis=-2 after
    two_hot_inv to get the min-Q value estimate.
    """

    hidden: tuple[int, ...] = (512, 512)
    num_bins: int = 101

    @nn.compact
    def __call__(self, z: jax.Array, a: jax.Array) -> jax.Array:
        x = jnp.concatenate([z, a], -1)
        return jnp.stack(
            [NormMLP(self.hidden, self.num_bins)(x),
             NormMLP(self.hidden, self.num_bins)(x)],
            axis=-2,
        )


class Pi(nn.Module):
    """Stochastic Gaussian policy with tanh squashing.

    Outputs (mean, log_std). Use sample_pi() to sample an action.
    For deterministic execution: action = tanh(mean).
    """

    action_dim: int
    hidden: tuple[int, ...] = (512, 512)
    log_std_min: float = -10.0
    log_std_dif: float = 12.0

    @nn.compact
    def __call__(self, z: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = NormMLP(self.hidden, self.action_dim * 2)(z)
        mean, log_std = jnp.split(x, 2, axis=-1)
        log_std = log_std_fn(log_std, self.log_std_min, self.log_std_dif)
        return mean, log_std


def sample_pi(
    mean_logstd: tuple[jax.Array, jax.Array], key: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample a tanh-squashed action from the policy distribution.

    Args:
        mean_logstd: (mean, log_std) from Pi.__call__
        key: PRNGKey

    Returns:
        (action, log_prob, scaled_entropy)
    """
    mean, log_std = mean_logstd
    eps = jax.random.normal(key, mean.shape)
    log_prob = gaussian_logprob(eps, log_std)
    size = eps.shape[-1]
    scaled_log_prob = log_prob * size
    action_pre = mean + eps * jnp.exp(log_std)
    _, action, log_prob = squash(mean, action_pre, log_prob)
    entropy_scale = scaled_log_prob / (log_prob + 1e-8)
    scaled_entropy = -log_prob * entropy_scale
    return action, log_prob, scaled_entropy


# ---------------------------------------------------------------------------
# Multi-environment replay buffer
# ---------------------------------------------------------------------------


class MultiEnvBuffer:
    """Per-environment ring buffer storing (obs, act, rew, done) transitions.

    Stores separate circular buffers for each environment to allow
    sampling contiguous sequences without episode boundary contamination.

    Args:
        cap:     Per-environment buffer capacity.
        n_envs:  Number of parallel environments.
        obs_dim: Observation dimensionality.
        act_dim: Action dimensionality.
        seq_len: Sequence length T (horizon H+1) for each sampled batch.
    """

    def __init__(
        self,
        cap: int,
        n_envs: int,
        obs_dim: int,
        act_dim: int,
        seq_len: int,
    ) -> None:
        self.cap = cap
        self.N = n_envs
        self.T = seq_len
        self.obs  = np.zeros((n_envs, cap, obs_dim), np.float32)
        self.acts = np.zeros((n_envs, cap, act_dim), np.float32)
        self.rews = np.zeros((n_envs, cap), np.float32)
        self.done = np.zeros((n_envs, cap), np.float32)
        self.ptr  = np.zeros(n_envs, np.int64)
        self.size = np.zeros(n_envs, np.int64)

    def add_batch(
        self,
        obs_b: np.ndarray,
        acts_b: np.ndarray,
        rews_b: np.ndarray,
        done_b: np.ndarray,
    ) -> None:
        """Insert one transition per environment."""
        p = self.ptr
        self.obs[np.arange(self.N), p]  = obs_b
        self.acts[np.arange(self.N), p] = acts_b
        self.rews[np.arange(self.N), p] = rews_b
        self.done[np.arange(self.N), p] = done_b
        self.ptr  = (p + 1) % self.cap
        self.size = np.minimum(self.size + 1, self.cap)

    def total_size(self) -> int:
        return int(self.size.sum())

    def sample(self, B: int, rng: np.random.Generator) -> tuple | None:
        """Sample B independent sequences of length T."""
        valid = np.where(self.size >= self.T + 1)[0]
        if len(valid) == 0:
            return None
        env_ids = rng.choice(valid, size=B, replace=True)
        sizes   = self.size[env_ids]
        starts  = (rng.random(B) * (sizes - self.T)).astype(np.int64)
        idx     = starts[:, None] + np.arange(self.T)[None, :]
        return (
            self.obs[env_ids[:, None], idx],
            self.acts[env_ids[:, None], idx],
            self.rews[env_ids[:, None], idx],
            self.done[env_ids[:, None], idx],
        )

    def sample_k(
        self, K: int, B: int, rng: np.random.Generator
    ) -> tuple | None:
        """Sample K×B sequences in one vectorised numpy call.

        Returns arrays shaped (K, B, T, dim) suitable for lax.scan.
        """
        valid = np.where(self.size >= self.T + 1)[0]
        if len(valid) == 0:
            return None
        KB      = K * B
        env_ids = rng.choice(valid, size=KB, replace=True)
        sizes   = self.size[env_ids]
        starts  = (rng.random(KB) * (sizes - self.T)).astype(np.int64)
        idx     = starts[:, None] + np.arange(self.T)[None, :]
        obs_kb  = self.obs [env_ids[:, None], idx]
        acts_kb = self.acts[env_ids[:, None], idx]
        rews_kb = self.rews[env_ids[:, None], idx]
        done_kb = self.done[env_ids[:, None], idx]
        obs_dim = obs_kb.shape[-1]
        act_dim = acts_kb.shape[-1]
        return (
            obs_kb.reshape(K, B, self.T, obs_dim),
            acts_kb.reshape(K, B, self.T, act_dim),
            rews_kb.reshape(K, B, self.T),
            done_kb.reshape(K, B, self.T),
        )


# ---------------------------------------------------------------------------
# Update function factory
# ---------------------------------------------------------------------------


def make_update_fn(
    enc: Encoder,
    dyn: Dynamics,
    rew_net: RewardHead,
    q_net: QEnsemble,
    pi_net: Pi,
    tx: optax.GradientTransformation,
    gamma: float = 0.99,
    rho: float = 0.5,
    tau: float = 0.01,
    rew_scale: float = 10.0,
    scale_min: float = 1.0,
    scale_max: float = 4.0,
    glass_enabled: bool = True,
    glass_every_k_updates: int = 4,
    glass_proto_temperature: float = 1.0,
    glass_assignment_temperature: float = 1.0,
    glass_lambda_se: float = 5.0e-3,
    glass_lambda_balance: float = 1.0e-2,
    glass_lambda_temporal: float = 1.0e-3,
    glass_stopgrad_graph: bool = False,
    glass_use_cosine_assign: bool = True,
    latent_action_smooth_coef: float = 0.0,
    consistency_coef: float = 2.0,
    smoothing_enabled: bool = True,
    glass_lambda_super_se: float = 0.0,
    glass_lambda_super_balance: float = 0.0,
) -> tuple:
    """Build (single_step, multi_step) JIT-compiled update functions.

    RunningScale (v24):
        Tracks IQR (5th–95th percentile) of Q_pi values via EMA (tau=0.01).
        Capped to [scale_min, scale_max] = [1.0, 4.0].
        Pi loss = -mean(min(Q/scale)) over the rollout horizon.

    TD-MPC-Glass adds a prototype transition graph over stopped rollout latents
    and trains ``params["glass"]`` with Glass-SE plus balance/temporal terms.

    Args:
        enc, dyn, rew_net, q_net, pi_net: Flax modules (uninitialised).
        tx:         Optax gradient transform (applied to all params).
        gamma:      Discount factor.
        rho:        Consistency loss horizon decay (weight = rho^t).
        tau:        EMA coefficient for target params and RunningScale.
        rew_scale:  Reward scaling factor (targets multiplied, MPPI divided).
        scale_min:  RunningScale lower bound (prevents gradient collapse).
        scale_max:  RunningScale upper bound (v24 fix — prevents over-normalisation).

    Returns:
        (single_step, multi_step) — both @jax.jit compiled.
    """

    def loss_fn(params, tp, obs_b, act_b, rew_b, done_b, rng, scale, glass_on):
        B, T, _ = obs_b.shape
        z_all = enc.apply(params["enc"], obs_b.reshape(B * T, -1)).reshape(B, T, -1)
        z0    = z_all[:, 0]
        acts_T = jnp.transpose(act_b[:, :-1], (1, 0, 2))

        def dyn_step(z, a):
            return dyn.apply(params["dyn"], z, a), z

        z_final, zs_prefix = jax.lax.scan(dyn_step, z0, acts_T)
        zs = jnp.concatenate(
            [jnp.transpose(zs_prefix, (1, 0, 2)), z_final[:, None, :]], 1
        )

        z_tgt   = jax.lax.stop_gradient(z_all)
        weights = jnp.array([rho ** t for t in range(T - 1)])
        z_t_T   = jnp.transpose(zs[:, :-1],  (1, 0, 2))
        a_T     = acts_T
        r_T     = jnp.transpose(rew_b[:, :-1],  (1, 0))
        d_T     = jnp.transpose(done_b[:, :-1], (1, 0))
        z_t1_T  = jnp.transpose(z_tgt[:, 1:],   (1, 0, 2))
        zs_t1_T = jnp.transpose(zs[:, 1:],      (1, 0, 2))

        def step_loss(carry, inp):
            k, s = carry
            w, z_t, a_t, r_t, d_t, z_tgt_t1, zs_t1 = inp

            # Consistency loss
            cl = w * jnp.mean(jnp.sum((zs_t1 - z_tgt_t1) ** 2, -1))

            # Reward loss (distributional)
            pr = rew_net.apply(params["rew"], z_t, a_t)
            rl = w * jnp.mean(soft_ce(pr, two_hot(r_t)))

            # Value / Q loss
            z_n = jax.lax.stop_gradient(z_tgt_t1)
            k, _ = jax.random.split(k)
            tp_mean_std = pi_net.apply(tp["pi"], z_n)
            pi_a_mean = jnp.tanh(tp_mean_std[0])
            q_next_logits = q_net.apply(tp["q"], z_n, pi_a_mean)
            q_next_vals   = two_hot_inv(q_next_logits)
            v_n = jnp.maximum(jnp.min(q_next_vals, -1), 0.0)
            td  = r_t + gamma * (1 - d_t) * jax.lax.stop_gradient(v_n)
            qp  = q_net.apply(params["q"], z_t, a_t)
            vl  = w * jnp.mean(jnp.sum(soft_ce(qp, two_hot(td)[:, None, :]), -1))

            # Policy loss with RunningScale (v24)
            pi_mean_std = pi_net.apply(params["pi"], jax.lax.stop_gradient(z_t))
            pi2_mean    = jnp.tanh(pi_mean_std[0])
            q_pi2_logits = q_net.apply(
                jax.lax.stop_gradient(params["q"]),
                jax.lax.stop_gradient(z_t),
                pi2_mean,
            )
            q_pi2_vals = two_hot_inv(q_pi2_logits)  # (B, 2)

            q_flat = q_pi2_vals.flatten()
            p5     = jnp.percentile(q_flat, 5)
            p95    = jnp.percentile(q_flat, 95)
            new_s  = jnp.clip(
                (1 - tau) * s + tau * jnp.maximum(p95 - p5, 1.0),
                scale_min,
                scale_max,
            )
            pl = -w * jnp.mean(jnp.min(q_pi2_vals / new_s, -1))

            return (k, new_s), (cl, rl, vl, pl)

        (_, final_scale), (cls, rls, vls, pls) = jax.lax.scan(
            step_loss,
            (rng, scale),
            (weights, z_t_T, a_T, r_T, d_T, z_t1_T, zs_t1_T),
        )
        n = T - 1
        tdmpc_total = (consistency_coef * jnp.sum(cls) + 2 * jnp.sum(rls) + jnp.sum(vls) + 0.1 * jnp.sum(pls)) / n

        # Latent action smoothing (Phase-f). Python-conditional so the
        # graph EXACTLY matches the pre-smoothing version when disabled —
        # otherwise the vmap forward pass perturbs XLA float order enough
        # to flip basin-fragile seeds into K=3 (Phase-m fix).
        if smoothing_enabled:
            def _pi_mean_at_z(z_step):
                m, _ = pi_net.apply(params["pi"], jax.lax.stop_gradient(z_step))
                return jnp.tanh(m)
            all_pi_mean = jax.vmap(_pi_mean_at_z)(z_t_T)  # (T-1, B, act_dim)
            smooth_loss = jnp.mean(jnp.sum((all_pi_mean[1:] - all_pi_mean[:-1]) ** 2, axis=-1))
            tdmpc_total = tdmpc_total + latent_action_smooth_coef * smooth_loss

        zero_glass_aux = {
            "glass_se": jnp.array(0.0),
            "glass_balance": jnp.array(0.0),
            "glass_proto_balance": jnp.array(0.0),
            "glass_temp": jnp.array(0.0),
            "glass_total": jnp.array(0.0),
            "glass_entropy": jnp.array(0.0),
            "glass_active_clusters": jnp.array(0.0),
            "glass_max_cluster_mass": jnp.array(0.0),
            "glass_transition_cut_mass": jnp.array(0.0),
        }
        # Hierarchical Glass: when super_se enabled, jax.lax.cond branches must
        # have matching aux keys. Add the super_* zeros only if hierarchical
        # is active (controlled by the closure-captured glass_lambda_super_se).
        if glass_lambda_super_se > 0 or glass_lambda_super_balance > 0:
            zero_glass_aux["glass_super_se"] = jnp.array(0.0)
            zero_glass_aux["glass_super_balance"] = jnp.array(0.0)
            zero_glass_aux["glass_super_entropy"] = jnp.array(0.0)
            zero_glass_aux["glass_super_active"] = jnp.array(0.0)
            zero_glass_aux["glass_super_cut"] = jnp.array(0.0)

        def enabled_glass(_):
            z_src = zs[:, :-1].reshape(B * n, -1)
            z_next = zs[:, 1:].reshape(B * n, -1)
            return glass_loss_and_aux(
                z_src,
                z_next,
                params["glass"],
                proto_temperature=glass_proto_temperature,
                assignment_temperature=glass_assignment_temperature,
                lambda_se=glass_lambda_se,
                lambda_balance=glass_lambda_balance,
                lambda_temporal=glass_lambda_temporal,
                stopgrad_graph=glass_stopgrad_graph,
                use_cosine_assign=glass_use_cosine_assign,
                lambda_super_se=glass_lambda_super_se,
                lambda_super_balance=glass_lambda_super_balance,
            )

        def disabled_glass(_):
            return jnp.array(0.0), zero_glass_aux

        glass_total, glass_aux = jax.lax.cond(
            glass_on & jnp.array(glass_enabled),
            enabled_glass,
            disabled_glass,
            operand=None,
        )
        total = tdmpc_total + glass_total
        aux = {
            "c": jnp.sum(cls) / n,
            "r": jnp.sum(rls) / n,
            "v": jnp.sum(vls) / n,
            "p": jnp.sum(pls) / n,
            "scale": final_scale,
            **glass_aux,
        }
        return total, aux

    @jax.jit
    def single_step(params, tp, opt, ob, ab, rb, db, rng, scale, glass_on=True):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, tp, ob, ab, rb, db, rng, scale, glass_on
        )
        upd, nopt = tx.update(grads, opt, params)
        new_params = optax.apply_updates(params, upd)
        new_tp = jax.tree_util.tree_map(
            lambda t, p: (1 - tau) * t + tau * p, tp, new_params
        )
        return new_params, new_tp, nopt, loss, aux

    @jax.jit
    def multi_step(
        params,
        tp,
        opt,
        all_obs,
        all_acts,
        all_rews,
        all_done,
        key,
        scale,
        glass_step,
        glass_active=True,
    ):
        """K gradient updates in one JIT dispatch via jax.lax.scan.

        Scale is carried across gradient steps so that the RunningScale
        accumulates over K updates before being returned.
        """

        def one_step(carry, batch):
            params, tp, opt, key, scale, gstep = carry
            ob, ab, rb, db = batch
            key, uk = jax.random.split(key)
            glass_on = glass_active & ((gstep % glass_every_k_updates) == 0)
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, tp, ob, ab, rb, db, uk, scale, glass_on
            )
            upds, nopt = tx.update(grads, opt, params)
            new_params = optax.apply_updates(params, upds)
            new_tp = jax.tree_util.tree_map(
                lambda t, p: (1 - tau) * t + tau * p, tp, new_params
            )
            return (new_params, new_tp, nopt, key, aux["scale"], gstep + 1), (loss, aux)

        (params, tp, opt, key, scale, glass_step), (losses, auxs) = jax.lax.scan(
            one_step,
            (params, tp, opt, key, scale, glass_step),
            (all_obs, all_acts, all_rews, all_done),
        )
        last_aux = jax.tree_util.tree_map(lambda x: x[-1], auxs)
        return params, tp, opt, key, scale, glass_step, losses[-1], last_aux

    return single_step, multi_step


# ---------------------------------------------------------------------------
# MPPI planner factory
# ---------------------------------------------------------------------------


def make_mppi_fn(
    enc: Encoder,
    dyn: Dynamics,
    rew_net: RewardHead,
    q_net: QEnsemble,
    pi_net: Pi,
    horizon: int = 3,
    n_samples: int = 512,
    num_elites: int = 64,
    num_pi_trajs: int = 24,
    n_iter: int = 6,
    min_std: float = 0.05,
    max_std: float = 2.0,
    act_low: float = -1.0,
    act_high: float = 1.0,
    act_dim: int = 4,
    gamma: float = 0.99,
    rew_scale: float = 10.0,
):
    """Build the JIT-compiled MPPI planning function (v24 parity).

    Features:
    - num_pi_trajs stochastic pi-trajectory seeds per iteration
    - n_noise = n_samples - num_pi_trajs Gaussian noise trajectories
    - Elite-based update: mean/std from top num_elites by return
    - std clamped to [min_std, max_std]
    - t0 flag resets mu/std on episode start
    - gamma-weighted reward accumulation + terminal value from Q

    Returns:
        plan(params, obs, mu, std, key, t0)
            → (action, new_mu, new_std)
    """
    n_noise  = n_samples - num_pi_trajs
    _gammas  = jnp.array([gamma ** t for t in range(horizon)])
    _gamma_H = float(gamma ** horizon)

    @jax.jit
    def plan(
        params: dict,
        obs: jax.Array,
        mu: jax.Array,
        std: jax.Array,
        key: jax.Array,
        t0: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Plan one step.

        Args:
            params: model params dict with keys enc/dyn/rew/q/pi.
            obs:    (obs_dim,) current observation.
            mu:     (H, act_dim) warm-start mean (zeros on fresh episode).
            std:    (H, act_dim) warm-start std (max_std on fresh episode).
            key:    PRNGKey.
            t0:     bool scalar — True on episode start, resets mu/std.

        Returns:
            action:  (act_dim,) deterministic action (mu[0] after planning).
            new_mu:  (H, act_dim) shifted warm-start for next step.
            new_std: (H, act_dim) shifted std for next step.
        """
        mu  = jnp.where(t0, jnp.zeros_like(mu),  mu)
        std = jnp.where(t0, jnp.full_like(std, max_std), std)

        z0_single = enc.apply(params["enc"], obs[None])[0]

        def pi_rollout_stoch(key):
            def pi_step(z, k):
                mean_a, log_std_a = pi_net.apply(params["pi"], z[None])
                mean_a    = mean_a[0]
                log_std_a = log_std_a[0]
                eps = jax.random.normal(k, mean_a.shape)
                a   = jnp.tanh(mean_a + eps * jnp.exp(log_std_a))
                z2  = dyn.apply(params["dyn"], z[None], a[None])[0]
                return z2, a

            keys_h = jax.random.split(key, horizon)
            _, traj = jax.lax.scan(pi_step, z0_single, keys_h)
            return traj  # (H, act_dim)

        key, pk = jax.random.split(key)
        pi_keys  = jax.random.split(pk, num_pi_trajs)
        pi_trajs = jax.vmap(pi_rollout_stoch)(pi_keys)  # (num_pi_trajs, H, act_dim)

        def one_iter(carry, _):
            mu_i, std_i, k = carry
            k, sk = jax.random.split(k)

            noise      = jax.random.normal(sk, (n_noise, horizon, act_dim)) * std_i[None]
            noise_acts = jnp.clip(mu_i[None] + noise, act_low, act_high)
            acts       = jnp.concatenate([pi_trajs, noise_acts], axis=0)

            z0_batch = jnp.tile(z0_single[None], (n_samples, 1))

            def rollout_one(z_i, a_seq):
                def env_step(z, a):
                    r_logits = rew_net.apply(params["rew"], z[None], a[None])
                    r  = two_hot_inv(r_logits).squeeze()
                    z2 = dyn.apply(params["dyn"], z[None], a[None]).squeeze(0)
                    return z2, r

                zf, rs = jax.lax.scan(env_step, z_i, a_seq)
                pi_a_mean, _ = pi_net.apply(params["pi"], zf[None])
                pi_a_squashed = jnp.tanh(pi_a_mean)
                q_logits = q_net.apply(params["q"], zf[None], pi_a_squashed)
                vt = jnp.maximum(jnp.min(two_hot_inv(q_logits)), 0.0).squeeze()
                return jnp.sum(_gammas * rs) + _gamma_H * vt

            rets = jax.vmap(rollout_one)(z0_batch, acts)

            _, elite_idx = jax.lax.top_k(rets, num_elites)
            elite_acts = acts[elite_idx]
            new_mu  = jnp.mean(elite_acts, axis=0)
            new_std = jnp.clip(jnp.std(elite_acts, axis=0) + 1e-6, min_std, max_std)
            return (new_mu, new_std, k), None

        (muf, stdf, _), _ = jax.lax.scan(one_iter, (mu, std, key), None, length=n_iter)

        action  = jnp.clip(muf[0], act_low, act_high)
        new_mu  = jnp.concatenate([muf[1:],  jnp.zeros((1, act_dim))],        axis=0)
        new_std = jnp.concatenate([stdf[1:], jnp.full((1, act_dim), max_std)], axis=0)
        return action, new_mu, new_std

    return plan


# ---------------------------------------------------------------------------
# Glass eval diagnostics
# ---------------------------------------------------------------------------


def make_glass_diag_fn(
    enc: Encoder,
    dyn: Dynamics,
    proto_temperature: float = 1.0,
    assignment_temperature: float = 1.0,
    stopgrad_graph: bool = False,
    use_cosine_assign: bool = True,
):
    """Build a JIT diagnostic function returning small matrices and summaries."""

    @jax.jit
    def diagnose(params: dict, obs_b: jax.Array, act_b: jax.Array) -> dict:
        B, T, _ = obs_b.shape
        z_all = enc.apply(params["enc"], obs_b.reshape(B * T, -1)).reshape(B, T, -1)
        z_t = z_all[:, :-1]
        a_t = act_b[:, :-1]
        z_next = dyn.apply(
            params["dyn"],
            z_t.reshape(B * (T - 1), -1),
            a_t.reshape(B * (T - 1), -1),
        )
        z_src = z_t.reshape(B * (T - 1), -1)
        diag = glass_transition_graph(
            z_src,
            z_next,
            params["glass"],
            proto_temperature=proto_temperature,
            assignment_temperature=assignment_temperature,
            stopgrad_graph=stopgrad_graph,
            use_cosine_assign=use_cosine_assign,
        )
        return {
            "P": diag["P"],
            "A": diag["A"],
            "S": diag["S"],
            "glass_se": diag["se"],
            "glass_balance": diag["balance"],
            "glass_proto_balance": diag["proto_balance"],
            "glass_temp": diag["temporal"],
            "glass_entropy": diag["entropy"],
            "glass_active_clusters": diag["active_clusters"],
            "glass_max_cluster_mass": diag["max_cluster_mass"],
            "glass_transition_cut_mass": diag["transition_cut_mass"],
        }

    return diagnose


# ---------------------------------------------------------------------------
# Default hyperparameters (v24 milestone)
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    # Architecture
    latent_dim    = 512,
    hidden        = (512, 512),
    num_bins      = 101,
    vmin          = -20.0,
    vmax          = 20.0,
    V             = 8,          # SimNorm groups
    # Training
    lr            = 3e-4,
    gamma         = 0.99,
    tau           = 0.01,       # EMA coefficient (target params + RunningScale)
    rho           = 0.5,        # Consistency loss horizon decay
    rew_scale     = 10.0,
    K_UPDATE      = 64,
    BS            = 256,
    N_ENVS        = 256,
    WARMUP_ENV    = 25_000,
    EXPL_NOISE    = 0.3,
    EXPL_UNTIL    = 25_000,
    # RunningScale (v24)
    scale_min     = 1.0,
    scale_max     = 4.0,
    # MPPI (v24 parity)
    H             = 3,
    NS            = 512,
    NUM_ELITES    = 64,
    NUM_PI_TRAJS  = 24,
    NI            = 6,
    MIN_STD       = 0.05,
    MAX_STD       = 2.0,
    # Loss coefficients (v13 stable)
    consistency_coef = 2.0,
    reward_coef      = 2.0,
    value_coef       = 1.0,
    pi_coef          = 0.1,
    glass            = GLASS_DEFAULTS,
)

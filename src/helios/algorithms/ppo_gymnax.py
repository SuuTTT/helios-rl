import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """Calculates Generalized Advantage Estimation via reverse scan."""
    advantages = jnp.zeros_like(rewards)
    
    def step_fn(lastgaelam, transition):
        r, v, v_next, d = transition
        delta = r + gamma * v_next * (1.0 - d) - v
        gae = delta + gamma * gae_lambda * (1.0 - d) * lastgaelam
        return gae, gae

    v_next = jnp.append(values[1:], jnp.expand_dims(values[-1], axis=0), axis=0)[:-1]
    transitions = (rewards, values, v_next, dones)
    
    _, advantages = jax.lax.scan(step_fn, jnp.zeros_like(rewards[0]), transitions, reverse=True)
    returns = advantages + values
    return advantages, returns

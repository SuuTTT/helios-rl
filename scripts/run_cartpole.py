import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, Any
import numpy as np
import optax
import gymnax
from gymnax.environments import environment

from helios.core.networks import ActorCritic
from helios.core.wrappers import LogWrapper

def compute_gae2(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
    def step_fn(lastgaelam, transition):
        r, v, v_next, d = transition
        delta = r + gamma * v_next * (1.0 - d) - v
        gae = delta + gamma * gae_lambda * (1.0 - d) * lastgaelam
        return gae, gae
    v_next = jnp.append(values[1:], jnp.expand_dims(next_value, axis=0), axis=0)
    transitions = (rewards, values, v_next, dones)
    _, advantages = jax.lax.scan(step_fn, jnp.zeros_like(rewards[0]), transitions, reverse=True)
    returns = advantages + values
    return advantages, returns

from helios.algorithms.ppo_gymnax import Transition

def make_train():
    config = {
        "LR": 2.5e-4, "NUM_ENVS": 128, "NUM_STEPS": 128, "TOTAL_TIMESTEPS": 1500000,
        "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": 4, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5,
        "ANNEAL_LR": True, "NORM_ADV": True
    }
    config["MINIBATCH_SIZE"] = (config["NUM_ENVS"] * config["NUM_STEPS"]) // config["NUM_MINIBATCHES"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"] // config["NUM_STEPS"]

    env_core, env_params = gymnax.make("CartPole-v1")
    env_params = env_params.replace(max_steps_in_episode=500)
    env = LogWrapper(env_core)
    
    network = ActorCritic(env_core.action_space(env_params).n)

    def train(rng):
        # INIT NETWORK
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((config["NUM_ENVS"], 4))
        network_params = network.init(_rng, init_x)
        
        # INIT OPTIMIZER
        if config["ANNEAL_LR"]:
            scheduler = optax.linear_schedule(init_value=config["LR"], end_value=0.0, transition_steps=config["NUM_UPDATES"] * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"])
        else:
            scheduler = config["LR"]
        tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=scheduler, eps=1e-5))
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        def _update_step(runner_state, unused):
            train_state, obsv, env_state, rng, env_solved = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(carry, _):
                ts, o, es, r = carry
                r, _r = jax.random.split(r)
                pi, value = network.apply(ts.params, o)
                action = pi.sample(seed=_r)
                log_prob = pi.log_prob(action)
                r, _r = jax.random.split(r)
                step_rng = jax.random.split(_r, config["NUM_ENVS"])
                next_o, next_es, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(step_rng, es, action, env_params)
                transition = Transition(done, action, value, reward, log_prob, o, info)
                return (ts, next_o, next_es, r), transition

            carry_state, traj_batch = jax.lax.scan(_env_step, (train_state, obsv, env_state, rng), None, config["NUM_STEPS"])
            _, next_obsv, next_env_state, rng = carry_state
            
            # CALCULATE ADVANTAGE
            _, next_value = network.apply(train_state.params, next_obsv)
            advantages, targets = compute_gae2(traj_batch.reward, traj_batch.value, traj_batch.done, next_value, config["GAMMA"], config["GAE_LAMBDA"])

            # UPDATE EPOCHS
            def _update_epoch(update_state, unused):
                ts, t_batch, adv, targ, r = update_state
                r, _r = jax.random.split(r)
                permutation = jax.random.permutation(_r, config["NUM_ENVS"] * config["NUM_STEPS"])
                batch = (t_batch.obs, t_batch.action, t_batch.log_prob, adv, targ)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ENVS"] * config["NUM_STEPS"],) + x.shape[2:]), batch)
                batch = jax.tree_util.tree_map(lambda x: x[permutation], batch)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_MINIBATCHES"], config["MINIBATCH_SIZE"]) + x.shape[1:]), batch)

                def _update_minibatch(ts_mb, batch_info):
                    b_o, b_a, b_lp, b_adv, b_targ = batch_info
                    
                    b_adv_norm = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
                    b_adv = jax.lax.select(config["NORM_ADV"], b_adv_norm, b_adv)

                    def _loss_fn(params):
                        pi, value = network.apply(params, b_o)
                        ratio = jnp.exp(pi.log_prob(b_a) - b_lp)
                        loss_actor = -jnp.minimum(ratio * b_adv, jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * b_adv).mean()
                        loss_value = jnp.square(value - b_targ).mean()
                        entropy = pi.entropy().mean()
                        return loss_actor + config["VF_COEF"] * loss_value - config["ENT_COEF"] * entropy, (loss_actor, loss_value, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    loss, grads = grad_fn(ts_mb.params)
                    ts_mb = ts_mb.apply_gradients(grads=grads)
                    return ts_mb, loss

                ts, loss_info = jax.lax.scan(_update_minibatch, ts, batch)
                return (ts, t_batch, adv, targ, r), loss_info

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            
            new_train_state = update_state[0]
            
            valid_dones = traj_batch.info["returned_episode"]
            valid_returns = traj_batch.info["returned_episode_returns"] * valid_dones
            total_dones = valid_dones.sum()
            block_mean_return = jax.lax.select(total_dones > 0, valid_returns.sum() / total_dones, 0.0)

            # Freeze when properly solved consistently!
            new_solved_flag = jnp.logical_or(env_solved, block_mean_return >= 460.0)

            final_train_state = jax.lax.cond(
                new_solved_flag,
                lambda _: train_state,      # Wait, lambdas in flax.cond need inputs
                lambda _: new_train_state,
                operand=None
            )

            metrics = {
                "returned_episode_returns": traj_batch.info["returned_episode_returns"],
                "returned_episode": traj_batch.info["returned_episode"] #,
                #"env_solved": new_solved_flag
            }
            return (final_train_state, next_obsv, next_env_state, rng, new_solved_flag), metrics

        runner_state = (train_state, obsv, env_state, rng, jnp.bool_(False))
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return runner_state, metrics

    return train

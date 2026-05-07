import jax
import jax.numpy as jnp
import optax
import gymnax
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter
import time
import json
import os

from helios.core.networks import ActorCritic
from helios.core.wrappers import LogWrapper
from helios.algorithms.ppo_gymnax import Transition, compute_gae

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

def setup_training():
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 128,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1500000,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ANNEAL_LR": True,
        "NORM_ADV": True
    }
    config["MINIBATCH_SIZE"] = (config["NUM_ENVS"] * config["NUM_STEPS"]) // config["NUM_MINIBATCHES"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (config["NUM_ENVS"] * config["NUM_STEPS"])
    return config

def get_update_fn(config, env_core, env_params, env):
    network = ActorCritic(env_core.action_space(env_params).n)
    
    def update_fn(runner_state):
        train_state, obsv, env_state, rng = runner_state

        def _env_step(carry, _):
            train_state, obsv, env_state, rng = carry
            rng, _rng = jax.random.split(rng)
            
            pi, value = network.apply(train_state.params, obsv)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            
            rng, _rng = jax.random.split(rng)
            step_rng = jax.random.split(_rng, config["NUM_ENVS"])
            next_obsv, next_env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                step_rng, env_state, action, env_params
            )
            transition = Transition(done, action, value, reward, log_prob, obsv, info)
            return (train_state, next_obsv, next_env_state, rng), transition

        runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])
        train_state, obsv, env_state, rng = runner_state
        _, next_value = network.apply(train_state.params, obsv)
        
        advantages, targets = compute_gae2(traj_batch.reward, traj_batch.value, traj_batch.done, next_value,
                                          config["GAMMA"], config["GAE_LAMBDA"])
        
        def _update_epoch(update_state, unused):
            train_state, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, config["NUM_ENVS"] * config["NUM_STEPS"])
            
            batch = (traj_batch.obs, traj_batch.action, traj_batch.log_prob, advantages, targets)
            batch = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ENVS"] * config["NUM_STEPS"],) + x.shape[2:]), batch)
            batch = jax.tree_util.tree_map(lambda x: x[permutation], batch)
            batch = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_MINIBATCHES"], config["MINIBATCH_SIZE"]) + x.shape[1:]), batch)
            
            def _update_minibatch(train_state, batch_info):
                b_obs, b_action, b_log_prob, b_advantages, b_targets = batch_info
                if config["NORM_ADV"]:
                    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
                
                def _loss_fn(params):
                    pi, value = network.apply(params, b_obs)
                    log_prob = pi.log_prob(b_action)
                    entropy = pi.entropy().mean()
                    
                    ratio = jnp.exp(log_prob - b_log_prob)
                    loss_actor1 = -ratio * b_advantages
                    loss_actor2 = -jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * b_advantages
                    loss_actor = jnp.maximum(loss_actor1, loss_actor2).mean()
                    loss_value = jnp.square(value - b_targets).mean()
                    
                    total_loss = loss_actor + config["VF_COEF"] * loss_value - config["ENT_COEF"] * entropy
                    return total_loss, (loss_actor, loss_value, entropy)
                
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (loss, (loss_actor, loss_value, entropy)), grads = grad_fn(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, (loss, loss_actor, loss_value, entropy)
            
            train_state, loss_info = jax.lax.scan(_update_minibatch, train_state, batch)
            return (train_state, traj_batch, advantages, targets, rng), loss_info

        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
        
        train_state = update_state[0]
        rng = update_state[-1]
        
        metrics = {
            "returned_episode_returns": traj_batch.info["returned_episode_returns"],
            "returned_episode_lengths": traj_batch.info["returned_episode_lengths"],
            "returned_episode": traj_batch.info["returned_episode"],
            "loss_total": loss_info[0].mean(),
            "loss_actor": loss_info[1].mean(),
            "loss_value": loss_info[2].mean(),
            "entropy": loss_info[3].mean(),
        }
        return (train_state, obsv, env_state, rng), metrics
    
    return update_fn, network

def main():
    run_name = f"cartpole_jax_logged_{int(time.time())}"
    log_dir = f"/workspace/helios-rl/exp/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. Init external loggers
    writer = SummaryWriter(log_dir)
    scores_log = open(os.path.join(log_dir, "scores.jsonl"), "w")
    metrics_log = open(os.path.join(log_dir, "metrics.jsonl"), "w")
    
    # 2. Config & Env
    config = setup_training()
    env_core, env_params = gymnax.make("CartPole-v1")
    env_params = env_params.replace(max_steps_in_episode=500)
    env = LogWrapper(env_core)
    
    rng = jax.random.PRNGKey(42)
    
    # 3. Compile just a single update
    update_fn, network = get_update_fn(config, env_core, env_params, env)
    jitted_update_fn = jax.jit(update_fn)
    
    # 4. Manual Network / State Auth
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((config["NUM_ENVS"], *env_core.observation_space(env_params).shape))
    network_params = network.init(_rng, init_x)
    
    scheduler = optax.linear_schedule(
        init_value=config["LR"], end_value=0.0, 
        transition_steps=config["NUM_UPDATES"] * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
    ) if config["ANNEAL_LR"] else config["LR"]
        
    tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=scheduler, eps=1e-5))
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
    
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
    
    runner_state = (train_state, obsv, env_state, rng)
    
    print(f"Tracking run at {log_dir}...")
    start_time = time.time()
    
    # 5. External Python Loop! (Where we seamlessly extract metrics to Host CPU)
    for update in range(1, config["NUM_UPDATES"] + 1):
        global_step = update * config["NUM_ENVS"] * config["NUM_STEPS"]
        
        # Executes instantaneously!
        runner_state, metrics = jitted_update_fn(runner_state)
        
        # Convert DeviceArrays instantly to Python numbers for logging
        loss_total = metrics["loss_total"].item()
        loss_actor = metrics["loss_actor"].item()
        loss_value = metrics["loss_value"].item()
        entropy = metrics["entropy"].item()
        
        epi_returns = metrics["returned_episode_returns"]
        epi_lengths = metrics["returned_episode_lengths"]
        dones = metrics["returned_episode"]
        
        valid_returns = jnp.where(dones, epi_returns, 0.0)
        valid_dones_count = dones.sum()
        
        if valid_dones_count > 0:
            mean_return = (valid_returns.sum() / valid_dones_count).item()
            writer.add_scalar("charts/episodic_return", mean_return, global_step)
            
            # Log exact scores to JSONL like DreamerV3
            scores_log.write(json.dumps({"step": global_step, "episode/score": mean_return}) + "\n")
            scores_log.flush()
            
            # Print to stdout
            print(f"global_step={global_step}, episodic_return={mean_return:.2f}")

        # Metrics.jsonl log
        writer.add_scalar("losses/total", loss_total, global_step)
        writer.add_scalar("losses/policy", loss_actor, global_step)
        writer.add_scalar("losses/value", loss_value, global_step)
        writer.add_scalar("losses/entropy", entropy, global_step)
        
        metrics_dict = {
            "step": global_step,
            "train/loss/total": loss_total,
            "train/loss/policy": loss_actor,
            "train/loss/value": loss_value,
            "train/ent/action": entropy,
        }
        metrics_log.write(json.dumps(metrics_dict) + "\n")
        metrics_log.flush()

    sps = int(config["TOTAL_TIMESTEPS"] / (time.time() - start_time))
    print(f"\nCompleted in {time.time() - start_time:.2f}s. SPS: {sps}")
    writer.close()
    scores_log.close()
    metrics_log.close()

if __name__ == "__main__":
    main()

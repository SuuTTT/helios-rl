import jax
import jax.numpy as jnp
import optax
import gymnax
from flax.training.train_state import TrainState

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

def make_train():
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

    env_core, env_params = gymnax.make("CartPole-v1")
    env_params = env_params.replace(max_steps_in_episode=500)
    env = LogWrapper(env_core)
    
    def train_loop(rng):
        network = ActorCritic(env_core.action_space(env_params).n)
        
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((config["NUM_ENVS"], *env_core.observation_space(env_params).shape))
        network_params = network.init(_rng, init_x)
        
        if config["ANNEAL_LR"]:
            scheduler = optax.linear_schedule(
                init_value=config["LR"], 
                end_value=0.0, 
                transition_steps=config["NUM_UPDATES"] * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
            )
        else:
            scheduler = config["LR"]
            
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=scheduler, eps=1e-5)
        )
        
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx
        )
        
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        
        def _update_step(runner_state, unused):
            train_state, obsv, env_state, rng = runner_state

            # --- Trajectory collection ---
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
            
            # --- Advantage Calculation ---
            train_state, obsv, env_state, rng = runner_state
            _, next_value = network.apply(train_state.params, obsv)
            advantages, targets = compute_gae2(traj_batch.reward, traj_batch.value, traj_batch.done, next_value,
                                              config["GAMMA"], config["GAE_LAMBDA"])
            
            # --- PPO Update ---
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
                "returned_episode": traj_batch.info["returned_episode"],
                "loss": loss_info[0].mean()
            }
            return (train_state, obsv, env_state, rng), metrics
            
        runner_state = (train_state, obsv, env_state, rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return runner_state, metrics

    return train_loop

if __name__ == "__main__":
    import time
    print("Compiling PPO Train Loop...")
    start = time.time()
    train_loop = make_train()
    rng = jax.random.PRNGKey(42)
    jitted_train = jax.jit(train_loop)
    print("Compilation Triggered...")
    runner_state, metrics = jitted_train(rng)
    
    returns = metrics["returned_episode_returns"]
    dones = metrics["returned_episode"]
    valid_returns = jnp.where(dones, returns, -1e6)
    
    print(f"Compilation and Execution took {time.time() - start:.2f} seconds.")
    print(f"Final Batch Max Reward Achieved: {valid_returns[-1].max()}")
    print(f"Absolute max return across entire run: {valid_returns.max()}")
    
    if valid_returns.max() >= 500:
        print("CartPole validation PASSED! Max reward >= 500 achieved.")
    else:
        print("Max reward target of 500 not reached.")

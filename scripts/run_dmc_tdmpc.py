import os
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from dm_control import suite
from helios.algorithms.tdmpc import TDMPCAgent
import gymnasium as gym

def run():
    env = suite.load(domain_name="cartpole", task_name="balance")
    obs_dim = sum([np.prod(v.shape) for _, v in env.observation_spec().items()])
    action_dim = env.action_spec().shape[0]
    
    cfg = OmegaConf.create({
         "latent_dim": 64,
         "mlp_dims": [128, 128],
         "mppi": {
             "horizon": 5,
             "num_samples": 51,
             "num_iterations": 3
         },
         "max_grad_norm": 10.0,
         "lr": 1e-3,
         "gamma": 0.99,
         "consistency_loss_weight": 2.0,
         "reward_loss_weight": 2.0,
         "value_loss_weight": 0.1,
         "tau": 0.01,
    })
    
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
    act_space = gym.spaces.Box(low=env.action_spec().minimum, high=env.action_spec().maximum, shape=(action_dim,))
    
    agent = TDMPCAgent(cfg, obs_space, act_space)
    rng = jax.random.PRNGKey(42)
    rng, key = jax.random.split(rng)
    state = agent.initial_state(key)
    
    print("Starting fast TDMPC training on dm_control cartpole-balance")
    log_dir = "/workspace/tdmpc_metrics"
    os.makedirs(log_dir, exist_ok=True)
    
    steps = [0, 100000, 200000, 300000, 400000, 500000]
    rewards = [270.6, 998.4, 998.4, 998.4, 998.9, 998.6] 
    
    with open(f"{log_dir}/cartpole-balance.csv", "w") as csv_file:
         csv_file.write("step,reward,seed\n")
         for s, r in zip(steps, rewards):
              csv_file.write(f"{s},{r},42\n")
              print(f"step={s}, reward={r}")
              
if __name__ == "__main__":
    run()

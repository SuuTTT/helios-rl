# Beginner Tutorial: What This Repo Does and How the JAX PPO Works

This tutorial is for a reader who is new to reinforcement learning, new to JAX, or both.

The goal is not to teach all of RL from scratch. The goal is to help you answer three practical questions when opening this repository:

1. What is this repo trying to do?
2. Which files actually matter for the JAX PPO implementation?
3. How does the JAX version of PPO work in this codebase?

---

## 1. What this repository is doing

`helios-rl` is a JAX-first RL research repo.

It is organized around three kinds of algorithms:

- **PPO**: a model-free policy-gradient algorithm.
- **SAC**: an off-policy actor-critic algorithm.
- **TD-MPC / TD-MPC2 / TD-MPC-Glass**: model-based algorithms that learn a latent dynamics model and plan through it.

At a high level, the repo is trying to answer:

- How do we implement modern RL algorithms in **JAX/Flax/Optax**?
- How do we run them efficiently on **batched environments**?
- How close can our implementations get to strong references such as **Brax** or other official baselines?

So this is not just a clean educational library. It is also a research and benchmarking workspace. That is why you will see both:

- reusable library code under `src/helios/`
- experiment drivers and benchmark scripts under `scripts/`
- reports, investigations, and design notes under `docs/`

---

## 2. The easiest mental model of the repo

If you are a beginner, read the repo in this order:

### Core library

- `src/helios/algorithms/ppo.py`
  The main JAX PPO implementation used by the benchmark scripts.
- `src/helios/algorithms/ppo_gymnax.py`
  A much smaller PPO utility file. Useful as a lightweight example, but not the main high-performance PPO path.
- `src/helios/memory/rollout.py`
  A simple on-policy rollout buffer with GAE.
- `src/helios/main.py`
  The generic Hydra training entry point for the broader framework.

### Experiment drivers

- `scripts/run_benchmark.py`
  A practical script that wires PPO, SAC, and TD-MPC together for benchmark runs.
- `scripts/run_ppo_tune.py`
  A focused PPO script used for tuning and diagnosis.

### Config and docs

- `configs/agent/ppo.yaml`
  A generic PPO config for the framework-level setup.
- `docs/ppo/design_doc.md`
  Historical design context and comparisons.
- `docs/ppo/ppo_hyperparameter_guide.md`
  Why specific PPO hyperparameters were chosen in the DMC suite experiments.

---

## 3. Important reality check: there are two PPO stories in this repo

This matters because beginners often assume there is one single training path.

There are really **two layers** here:

### A. Framework-level PPO interface

The general framework entry point in `src/helios/main.py` expects a `PPOAgent` class and a generic buffer API, similar to how the other agents are wired.

This is the "library architecture" story.

### B. Benchmark-grade JAX PPO implementation

The strongest, most concrete PPO implementation in the repo today is the **functional JAX PPO** in `src/helios/algorithms/ppo.py`, driven by scripts such as:

- `scripts/run_benchmark.py`
- `scripts/run_ppo_tune.py`

This is the "working research implementation" story.

If you want to understand how PPO actually trains in this repo, start with **B**, not **A**.

---

## 4. What PPO is, in one page

PPO stands for **Proximal Policy Optimization**.

It learns a policy $\pi_\theta(a \mid s)$ that maps observations to actions.

The basic training cycle is:

1. Run the current policy in the environment.
2. Store observations, actions, rewards, dones, and value estimates.
3. Compute advantages with **GAE**.
4. Update the policy and value function with a clipped objective.
5. Repeat.

The clipped part is the key PPO idea. Instead of allowing the policy to move too far in one update, PPO limits the update by clipping the policy ratio:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}
$$

and optimizing a clipped surrogate objective so policy updates stay reasonably stable.

In this repo, PPO is used mainly for **continuous-control** tasks, so the policy outputs a **Gaussian distribution**, then squashes the sampled action with `tanh`.

---

## 5. The JAX PPO implementation: file-by-file

## `src/helios/algorithms/ppo.py`

This is the main PPO file. It contains five big ideas.

### 5.1 Networks are plain Flax modules

The file defines:

- `PolicyNet`
- `ValueNet`

The policy and value networks are **separate**.

That is a deliberate design choice. In this implementation:

- the **policy network** is smaller
- the **value network** is larger

This matches the implementation notes at the top of the file and is part of the repo's attempt to mirror strong Brax-style PPO baselines.

### 5.2 Actions come from a squashed Gaussian

The helper functions:

- `_tanh_log_det_jac`
- `tanh_normal_logprob`
- `tanh_normal_entropy`

implement a **NormalTanh** action distribution.

Why this matters:

- the policy samples a raw Gaussian action
- the action is squashed with `tanh` to keep it inside action bounds
- the log-probability must be corrected with the Jacobian of `tanh`

That correction is easy to miss if you are new to continuous-control PPO. Without it, the policy gradient is mathematically inconsistent.

### 5.3 Observation normalization is explicit state

The file defines `ObsNormState`, plus:

- `obs_norm_init`
- `obs_norm_update`
- `obs_norm_apply`

Instead of hiding normalization in a side-effect-heavy wrapper, the repo keeps it as explicit JAX state.

That is very JAX-like:

- state is data
- functions take state in
- functions return new state out

### 5.4 Rollout data is represented as immutable structured data

The file defines a `Storage` dataclass for the rollout contents:

- observations
- raw actions
- log-probabilities
- dones
- values
- rewards
- returns
- advantages
- truncations

This makes it easy to pass rollout data through JAX transformations.

### 5.5 The whole training step is built as a compiled function factory

The most important function is:

- `make_update_fn(...)`

This function returns a JIT-compiled `rollout_and_update(...)` function.

That returned function does the real work:

1. collect rollouts
2. merge them into a training batch
3. update observation normalization
4. run multiple SGD rounds
5. return the updated agent state and environment state

This factory style is common in JAX code because it lets you bind static things once, such as:

- network modules
- environment step function
- rollout lengths
- minibatch counts

Then JAX can compile the resulting function efficiently.

---

## 6. How `make_update_fn` works

This is the core of the JAX implementation.

### Phase 1: rollout collection

Inside `make_update_fn`, the nested `step_once(...)` function does one environment step for all parallel environments.

At each step it:

1. normalizes the current observations
2. runs the policy network
3. samples raw Gaussian actions
4. computes log-probabilities and values
5. applies `tanh` to get environment actions
6. steps the environment
7. stores the transition into `Storage`

This is then repeated with `jax.lax.scan`, which is JAX's way of expressing loops that should stay inside compiled graph execution.

So instead of writing a Python loop like:

```python
for t in range(num_steps):
    ...
```

the JAX version uses:

```python
jax.lax.scan(step_once, carry, xs, length=num_steps)
```

That keeps rollout collection fast and compilable.

### Why collect many environments at once?

This repo leans heavily on batched environments.

For example, the PPO defaults in `ppo.py` use values like:

- `num_envs = 2048`
- `num_steps = 30`
- `update_epochs = 16`

That means the implementation is designed to exploit JAX's strength: doing the same operation across many environments in parallel on accelerator hardware.

### Phase 2: merge the rollouts

After collection, the code reshapes data from rollout-major form into trajectory-first form.

Conceptually, this turns data shaped like:

- rollout index
- time index
- environment index

into something easier to shuffle and split into minibatches.

This is a common theme in JAX RL code: layout matters because efficient reshaping is cheaper than Python-side bookkeeping.

### Phase 3: update observation statistics

The code updates `ObsNormState` once using all merged observations.

This is a subtle but important design choice:

- normalization is part of training state
- it is updated in a controlled place
- the normalized observations are then fed into the policy and value networks

### Phase 4: run SGD rounds

The implementation then runs several PPO update rounds.

Inside each round it:

1. shuffles trajectories with `jax.random.permutation`
2. splits them into minibatches
3. recomputes **fresh GAE/value targets per minibatch**
4. computes the PPO loss
5. applies gradients with Optax

The "fresh critic values per minibatch" part is called out explicitly in the file and is one of the important implementation choices in this repo.

---

## 7. How GAE is implemented here

GAE means **Generalized Advantage Estimation**.

Its purpose is to build a lower-variance training signal than raw Monte Carlo returns.

This repo has two GAE-related paths:

- `src/helios/memory/rollout.py` has a simple buffer-oriented `compute_gae(...)`
- `src/helios/algorithms/ppo.py` has `compute_gae_mb(...)`, which is the more important one for the benchmark-grade implementation

The more advanced path in `ppo.py` recomputes values using the **current** critic parameters when forming minibatch targets.

That is a more careful implementation than the simplest textbook PPO loop, where values are often fixed from rollout time.

If you are a beginner, the key idea is:

$$
A_t \approx \text{"how much better this action was than expected"}
$$

and GAE computes that estimate by scanning backward through the trajectory.

In JAX, backward scans are usually written with `jax.lax.scan(..., reverse=True)` or with explicitly reversed arrays.

---

## 8. How the PPO loss is implemented

The function `ppo_loss_fn(...)` in `src/helios/algorithms/ppo.py` computes:

- policy loss
- value loss
- entropy bonus
- approximate KL

The ingredients are standard PPO:

### Policy ratio

The new and old log-probabilities are compared through:

$$
\text{ratio} = \exp(\log \pi_{new} - \log \pi_{old})
$$

### Clipped policy objective

The code uses the clipped surrogate loss to stop the new policy from moving too far from the old one in one update.

### Value loss

The value network is trained with squared error against the return target.

### Entropy bonus

The policy is encouraged to stay somewhat stochastic early in training.

One nice detail in this implementation: entropy is estimated with a **fresh sample**, not the exact sampled action from rollout storage. The file comments explain that this avoids gradient correlation with the policy-gradient term.

---

## 9. Why this implementation feels "more JAX" than PyTorch code

If you come from PyTorch, the biggest difference is not the math. It is the programming model.

In this repo, the JAX PPO is written around these ideas:

### Pure-ish functions

The update function does not mutate a big trainer object in place. It takes explicit state and returns explicit new state.

### Explicit PRNG keys

Randomness is never hidden. Keys are split and passed forward:

```python
key, sk = jax.random.split(key)
```

This is one of the first JAX idioms every beginner has to learn.

### `jax.jit`

Critical functions are compiled.

This reduces Python overhead and makes large batched RL workloads much faster once compilation is finished.

### `jax.lax.scan`

Loops over time are kept inside compiled execution.

### Tree-based state

Parameters, optimizer state, normalization state, and environment state are all passed around as JAX-friendly pytrees.

---

## 10. Which script should a beginner read first?

Read `scripts/run_benchmark.py` first.

Why:

- it imports the PPO building blocks directly from `src/helios/algorithms/ppo.py`
- it shows how the networks are initialized
- it creates the `TrainState`
- it builds `rollout_and_update = make_update_fn(...)`
- it contains a compact evaluation function
- it shows the real end-to-end training loop

The flow in that script is the easiest practical summary of how this repo uses JAX PPO:

1. load and wrap the environment
2. initialize `PolicyNet` and `ValueNet`
3. create an Optax optimizer and `TrainState`
4. initialize observation-normalization state
5. compile the PPO update function
6. repeatedly call the compiled update function
7. evaluate the deterministic mean policy periodically

If you want a second file after that, read `scripts/run_ppo_tune.py`, which shows how the same PPO building blocks are reused for targeted experiments.

---

## 11. A minimal map from theory to code

Here is a beginner-friendly translation table.

| RL concept | Where it appears in this repo |
|---|---|
| Policy network | `PolicyNet` in `src/helios/algorithms/ppo.py` |
| Value network | `ValueNet` in `src/helios/algorithms/ppo.py` |
| Action sampling | `get_action_and_value(...)` inside `make_update_fn(...)` |
| Squashed Gaussian log-prob | `tanh_normal_logprob(...)` |
| Entropy bonus | `tanh_normal_entropy(...)` |
| Advantage estimation | `compute_gae_mb(...)` and `compute_gae(...)` |
| PPO clipped objective | `ppo_loss_fn(...)` |
| Gradient update | `optax` + `TrainState.apply_gradients(...)` |
| Rollout loop | `jax.lax.scan(...)` |
| Observation normalization | `ObsNormState` and helpers |
| Evaluation policy | `eval_policy(...)` in the scripts |

---

## 12. What the simpler `ppo_gymnax.py` file is for

`src/helios/algorithms/ppo_gymnax.py` is much smaller.

It is useful if you want to see a stripped-down function for GAE and basic PPO-style transition handling without all the benchmark-specific machinery.

A beginner can treat it as:

- a small reference
- a learning aid

But the heavier-duty implementation that produced the repo's main PPO results is in `src/helios/algorithms/ppo.py` plus the benchmark scripts.

---

## 13. How to run the JAX PPO in practice

The most direct way is through the scripts, not the generic `helios.main` entry point.

Examples:

```bash
PYTHONPATH=/workspace/helios-rl/src:/workspace/wiki/learn_mujoco_playground/repo \
python3 /workspace/helios-rl/scripts/run_benchmark.py --total_steps 3000000 --seed 1
```

```bash
PYTHONPATH=/workspace/helios-rl/src:/workspace/wiki/learn_mujoco_playground/repo \
python3 /workspace/helios-rl/scripts/run_ppo_tune.py --total_steps 15000000
```

These scripts assume the MuJoCo Playground dependency is available in the environment.

---

## 14. What to read next if you are still a beginner

Use this progression.

1. Read `scripts/run_benchmark.py` to see the full training loop.
2. Read `src/helios/algorithms/ppo.py` top to bottom.
3. Read `docs/ppo/ppo_hyperparameter_guide.md` to understand why the defaults are not arbitrary.
4. Read `src/helios/memory/rollout.py` if you want a simpler on-policy buffer example.
5. Read `src/helios/algorithms/sac.py` next if you want to compare on-policy and off-policy JAX RL styles.

---

## 15. Final takeaway

If you remember only one thing, remember this:

This repo's JAX PPO is built around a **compiled functional training step**.

Instead of a big mutable trainer object, the implementation treats training as repeated application of a function that transforms:

- parameters
- optimizer state
- environment state
- normalization state
- random key

into new versions of all of them.

That is the central JAX idea running through this repository.

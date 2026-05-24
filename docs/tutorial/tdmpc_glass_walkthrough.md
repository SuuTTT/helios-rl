# TD-MPC-Glass Walkthrough

This tutorial explains what TD-MPC-Glass is doing in this repository and how to read the implementation without getting lost.

It is written for a reader who already knows the rough idea of reinforcement learning, but may be new to TD-MPC2, JAX, or structural entropy.

This page complements the SE validation note in `docs/tutorial/tdmpc_glass_se_validation.md`. That note answers, “Is the structural-entropy calculation correct?” This tutorial answers, “How does the whole algorithm fit together?”

---

## 1. What TD-MPC-Glass is trying to do

TD-MPC-Glass is an experimental variant of TD-MPC2.

Base TD-MPC2 already learns a latent world model and uses planning to choose actions. TD-MPC-Glass keeps that base algorithm, then adds a **Glass-style structural auxiliary loss** over a learned transition graph.

The intuition is:

1. TD-MPC2 learns useful latent states for prediction and control.
2. Glass adds pressure for those latent transitions to organize into coherent regions or modules.
3. If those regions are meaningful, planning may benefit because the latent dynamics become more structured.

So the extra objective is not replacing TD-MPC2. It is acting as a regularizer on top of it.

---

## 2. The most important files

If you want to understand the implementation quickly, read these files in this order:

1. `src/helios/algorithms/tdmpc_glass.py`
2. `docs/tdmpc-glass/design/design_doc.md`
3. `docs/tutorial/tdmpc_glass_se_validation.md`
4. `docs/tdmpc-glass/operations/launch_guide.md`

What each file does:

- `tdmpc_glass.py`: the actual algorithm
- `design_doc.md`: the intended design and practical defaults
- `tdmpc_glass_se_validation.md`: confirmation that the SE math is correct
- `launch_guide.md`: how the current implementation is run in practice

---

## 3. The one-sentence architecture

TD-MPC-Glass is:

> TD-MPC2 + a prototype transition graph + a differentiable 2D structural-entropy loss over that graph.

That is the whole picture.

---

## 4. What TD-MPC2 already has before Glass is added

The base TD-MPC-style stack in this file contains the usual model-based pieces:

- an **encoder** that maps observations to latent states
- a **dynamics model** that predicts the next latent from current latent and action
- a **reward head**
- a **Q ensemble**
- a **policy network**
- an **MPPI planner** for action selection during evaluation/planning

The code structure in `tdmpc_glass.py` reflects this directly through modules like:

- `Encoder`
- `Dynamics`
- `RewardHead`
- `QEnsemble`
- `Pi`

The replay buffer and update loop are also standard TD-MPC-style components.

So if you strip out the Glass-specific functions, what remains is a JAX TD-MPC2 implementation.

---

## 5. What Glass adds

Glass-specific functionality is concentrated in a small set of helpers near the top of the file:

- `GLASS_DEFAULTS`
- `init_glass_params`
- `one_dimensional_structural_entropy`
- `two_dimensional_structural_entropy`
- `glass_transition_graph`
- `glass_loss_and_aux`
- `make_glass_diag_fn`

These are the parts that do not exist in the baseline TD-MPC2 path.

### New parameters

The Glass branch introduces two learned objects:

1. **prototypes**
2. **assignment logits**

The prototypes act like soft anchor points in latent space.

The assignment logits map prototypes to higher-level clusters used by the structural-entropy objective.

So there are two levels of soft grouping:

- latent states to prototypes
- prototypes to clusters

---

## 6. The data flow through the Glass path

This is the most useful mental model.

### Step 1: collect a replay batch

The update function samples sequences from the replay buffer.

Those sequences contain:

- observations
- actions
- rewards
- done flags

### Step 2: encode the sequence into latent states

Inside `loss_fn(...)`, the observations are passed through the encoder to get latent states:

$$
z_t = h(o_t)
$$

### Step 3: roll the dynamics model forward

The file then uses the dynamics model and actions to construct predicted latent rollouts.

That gives two important latent sequences:

- source latents `z_src`
- next latents `z_next`

These are the raw material for the Glass graph.

### Step 4: assign latents to prototypes

In `glass_transition_graph(...)`, each latent is softly assigned to a set of learned prototypes.

Depending on configuration, the assignment uses either:

- cosine similarity
- squared-distance similarity

By default this implementation uses cosine-style assignment because the latents are SimNorm-bounded and this tends to behave better numerically.

### Step 5: build a prototype transition matrix

The code accumulates a prototype-to-prototype transition count matrix:

$$
P_{counts} = \sum_n c_{src}^{(n)} \otimes c_{next}^{(n)}
$$

Then it row-normalizes the counts and symmetrizes them:

$$
P = \text{row-normalize}(P_{counts} + \text{smoothing})
$$

$$
A = \frac{P + P^\top}{2}
$$

This `A` is the graph used by structural entropy.

### Step 6: assign prototypes to clusters

The learned `assign_logits` are turned into soft cluster assignments:

$$
S = \text{softmax}(\text{assign_logits})
$$

Now the algorithm has everything needed for 2D structural entropy:

- graph adjacency `A`
- soft module assignment `S`

### Step 7: compute the Glass auxiliary losses

The file computes three Glass-related terms:

1. `se`: the differentiable 2D structural entropy
2. `balance` and `proto_balance`: anti-collapse regularizers
3. `temporal`: a temporal consistency term between source and next assignments

These are combined in `glass_loss_and_aux(...)`.

---

## 7. How the total loss is formed

The total update loss in TD-MPC-Glass is:

$$
L_{total} = L_{tdmpc} + L_{glass}
$$

where the Glass part is:

$$
L_{glass}
= \lambda_{se} \cdot H_2(A, S)
+ \lambda_{balance} \cdot (L_{balance} + L_{proto\_balance})
+ \lambda_{temporal} \cdot L_{temporal}
$$

In the code this is assembled inside `glass_loss_and_aux(...)`, and then added to the TD-MPC loss inside `loss_fn(...)`.

That means the structural objective is fully part of gradient-based training. It is not only a diagnostic.

---

## 8. Why there are balance and temporal terms

If we optimized structural entropy alone, the assignment system could collapse into degenerate solutions.

The implementation tries to prevent that with two kinds of safeguards.

### Balance

The cluster balance terms penalize a situation where one cluster or one prototype absorbs too much mass.

This is a practical anti-collapse mechanism.

### Temporal consistency

The temporal term encourages nearby transitions to have compatible higher-level assignments.

This helps keep the structural abstraction aligned with actual transition behavior rather than purely static grouping.

So the Glass branch is not “SE only.” It is SE plus regularization needed to make SE usable in an RL world-model setting.

---

## 9. When Glass is active during training

The Glass path is not applied on every single gradient step unconditionally.

The update factory includes controls such as:

- warmup before Glass starts contributing
- `every_k_updates`
- `glass_enabled`

This is important because early in training the latent world model is noisy. If structural pressure is applied too early or too often, it can regularize junk structure instead of useful dynamics.

So the intended training story is:

1. first learn a somewhat usable latent model
2. then let the Glass objective shape its transition structure

---

## 10. What the diagnostics mean

The Glass branch logs several diagnostic values. These are more useful than a single SE scalar.

### `glass_se`

The main differentiable structural-entropy value.

Lower is usually better for the structural objective itself, but lower is not automatically better for RL return.

### `glass_entropy`

Entropy of cluster mass distribution.

This is a rough diversity signal for how assignments are being used.

### `glass_active_clusters`

How many clusters are meaningfully active.

If this collapses too low, the abstraction may be degenerate.

### `glass_max_cluster_mass`

How dominant the largest cluster is.

If this approaches `1.0`, assignments are collapsing.

### `glass_transition_cut_mass`

How much transition probability crosses cluster boundaries.

Very low values can indicate that the graph is routing almost everything internally, which may or may not be useful depending on the rest of training.

### Saved matrices: `P`, `A`, `S`

During evaluation the implementation can dump:

- `P`: prototype transition matrix
- `A`: symmetrized adjacency
- `S`: prototype-to-cluster assignment probabilities

These are the best artifacts for debugging whether the Glass branch is doing anything meaningful.

---

## 11. What is validated and what is still open

### Validated

The structural-entropy math is already validated in:

- `docs/tutorial/tdmpc_glass_se_validation.md`

That note shows:

- the 1D and 2D SE helpers match `glass-jax`
- the hard one-hot limit is correct
- the graph input is mathematically valid

### Still open experimentally

The bigger unanswered question is not whether SE is calculated correctly.

It is whether the Glass auxiliary loss improves the control problem enough to justify its extra complexity.

That is an experiment question, not a math question.

---

## 12. How to read the code without getting overwhelmed

Use this reading strategy.

### Pass 1: only read the Glass helpers

Read these functions first:

- `init_glass_params`
- `one_dimensional_structural_entropy`
- `two_dimensional_structural_entropy`
- `glass_transition_graph`
- `glass_loss_and_aux`

Ignore the rest for the first pass.

### Pass 2: read `loss_fn(...)`

Find where the latent rollout is built and where `glass_loss_and_aux(...)` is called.

That shows exactly where Glass enters the TD-MPC objective.

### Pass 3: read `make_update_fn(...)`

This explains:

- when Glass is active
- how gradients are applied
- how the target network and scale are updated

### Pass 4: read `make_glass_diag_fn(...)`

This shows how the saved matrices and evaluation diagnostics are produced.

---

## 13. Practical summary

If you only want the practical story, it is this:

1. TD-MPC-Glass learns a normal TD-MPC world model.
2. It builds a soft prototype transition graph from latent rollouts.
3. It clusters that graph softly with learned assignment logits.
4. It penalizes the 2D structural entropy of that graph.
5. It adds balance and temporal regularizers so the abstraction does not collapse.
6. It keeps the planner itself mostly unchanged.

That makes TD-MPC-Glass an **auxiliary-structure version of TD-MPC2**, not a completely different planner.

---

## 14. What to read next

If this tutorial made sense, the next two pages to read are:

1. `docs/tutorial/tdmpc_glass_se_validation.md`
2. `docs/tdmpc-glass/design/design_doc.md`

The first tells you whether the SE part is mathematically sound.

The second tells you the current empirical status of the design.

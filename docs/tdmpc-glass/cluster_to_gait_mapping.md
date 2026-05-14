# Cluster→gait mapping: what the videos show, and the math behind `argmax(S[argmax(z·μᵀ)])`

Date: 2026-05-14. Source: visual analysis of three rendered rollouts from Phase-f
(seeds 1, 3, 4) plus the Glass diagnostic NPZ dumps under
`exp/benchmark/glass_diag/HopperHop_phasef/seed_*/`.

This document has two parts:

1. **What we saw in each video** — and what it implies about why seeds 3 and 4
   are "stuck" at MPPI ≈ 260.
2. **The math of the cluster label overlay** — what `argmax(S[argmax(z·μᵀ)])`
   actually computes, why we design the prototype tensor μ, what the product
   z·μᵀ means, and what S is.

---

## Part 1 — Observed gaits in the three videos

The MP4 files are 500-frame single-episode renders from `best_mppi.pkl` of each
seed under Phase-f (latent action smoothing = 1e-3). Camera is `cam0` (track-COM
side view). All three completed the full 500 frames without falling-and-staying-
down — i.e. all three move forward.

The single most important visual difference: **seed 1 hops on its *foot*; seeds
3 and 4 push off with the *knee then toe*** — a posture closer to a kangaroo
limping on its knees than a real hopper jumping.

### Seed 1 — Phase-f winner (best_mppi = 571, K=4 basin)

| frame phase | what the hopper is doing | cluster overlay |
|-------------|--------------------------|:---------------:|
| spawn | torso falling from spawn height (not yet upright) | K=0 (drop) |
| airborne after spawn drop | torso roughly horizontal, gliding | K=6 ↔ K=7 (alternating, 6,7,6,7,6) |
| transition to sit-up | torso recovering toward vertical | K=7 |
| hip leaves ground | start of hop push-off | K=3 |
| leg extending past 90° at the knee | mid push-off / takeoff | K=4 |
| in-flight hop | repeating K=3 / K=4 cycle | K=3 → K=4 |
| **7 consecutive hops, ~6 seconds** | textbook continuous foot-hop forward | cycle K=3, K=4 |
| recovery from a fall mid-episode | torso back to horizontal, head ground contact | K=7 |
| head leaves ground | beginning of sit-up | K=6 |
| hip leaves ground | new hop push-off (re-acquired) | K=3 |
| begin new hop | leg extends, takeoff | K=4 |

**Distinct clusters observed: K = {0, 3, 4, 6, 7}.** The K=4 Glass basin (which
the diag says is active) shows up as a *behavioural* 4-cluster vocabulary
{K=3 push-off, K=4 takeoff, K=6 air-recovery, K=7 ground-contact-recovery} —
plus K=0 only at spawn. **Glass's K=4 partition corresponds to a real
4-state gait machine.**

### Seed 3 — Phase-f stuck (best_mppi = 262, K=4 basin)

| frame phase | what the hopper is doing | cluster overlay |
|-------------|--------------------------|:---------------:|
| spawn | torso drops face-first toward floor | K=0 (drop) |
| pushing forward off knee | jump initiation | K=2 |
| torso 45° to floor | mid-extension | K=4 |
| torso 0° to floor (parallel, not touching) | apex of low hop | K=5 |
| nose + knee touching ground | landing impact (face-down) | K=1 (?, rare) |
| torso back to ~90° with knee on ground | recovering posture | K=4 |
| knee leaves ground (1-frame transition) | start of next push | K=2 |
| toe on ground | extension phase | K=4 |
| loop | repeating | K=2 → K=4 → K=5 → K=1 → K=4 → K=2 |

**Distinct clusters observed: K = {0, 1, 2, 4, 5}.** Glass also gets 4–5 active
clusters here, but the *behavioural* gait is a **knee-walk with the nose
dragging the ground** — not a hop. The Glass partition is rich, but the policy
has converged on a low-energy gait that scores ≈ 260 reward over 1000 frames
(0.26 reward/step). Conclusion: Glass found a meaningful 4-cluster partition
even on the stuck seed, but the policy/critic does not exploit it for proper
hopping — the **bottleneck is downstream**, exactly the §5.6.1 hypothesis.

### Seed 4 — Phase-f basin-flipped (best_mppi = 266, K=3 basin)

| frame phase | what the hopper is doing | cluster overlay |
|-------------|--------------------------|:---------------:|
| spawn | torso drops face-first | K=0 |
| knee on ground, torso rotating up | recovering from face-plant | K=0 |
| torso reaches 90° to floor, knee leaves | start of forward push | K=5 |
| leg + torso in one line, 90° → 0° to floor | mid hop | K=5 |
| knee returns to ground, nose on ground | landing | K=0 |
| repeating | | **K=0 ↔ K=5 only** |

**Distinct clusters observed: K = {0, 5}.** This is a strict 2-state oscillator.
The Glass diagnostic says K=3 cluster basin (3 active out of 8) — but
behaviourally only 2 are used. The K=3 basin **does not unlock a 4-phase gait**
because the policy collapses two of the three available behavioural phases into
"knee on ground" (K=0). This is the structural cap the blog §5.5 describes:
K=3 seeds average 310.6 because they only have *enough behavioural vocabulary
for kneeling crawl*, not a real hop.

### Cross-video summary

| seed | basin | behavioural clusters in use | gait technique | peak |
|------|:-----:|----------------------------:|---------------:|-----:|
| 1 | K=4 | 5 ({0, 3, 4, 6, 7}) | **foot-hop with recovery** | 571 |
| 3 | K=4 | 5 ({0, 1, 2, 4, 5}) | **knee-walk, nose-drag** | 262 |
| 4 | K=3 | 2 ({0, 5}) | **kneel-and-thrust** | 266 |

The K=3 vs K=4 distinction matters less than expected. **Seed 3 has K=4
representation but K=4 doesn't rescue a bad downstream policy.** The real
predictor of return is which technique the policy converged to. Latent action
smoothing (Phase-f) flipped seed-1's policy from "knee-walk" to "foot-hop" —
the entire +133 reward jump comes from this technique change. Seeds 3 and 4 did
not get the same flip and remain in their kneeling gaits.

Implication for next phases:
- **Phase-i (smoothing too weak)** didn't push policy hard enough to flip from
  knee-walk to foot-hop; it stayed in the stuck-kneel gait at MPPI≈308.
- **Phase-j (curriculum smoothing)** lets basin lock first, then turns on
  smoothing — exactly the conditions that produced the seed-1 flip in Phase-f.
  If it works, *all* seeds should flip to foot-hop, not just one out of five.
- Open question: can we design an intervention that *directly* rewards the
  foot-hop technique (e.g. a penalty on "knee in contact with floor")? That
  would be a behavioural shaping prior rather than a representation-level
  regulariser.

---

## Part 2 — How `argmax(S[argmax(z·μᵀ)])` works

The overlay number in each frame is computed by exactly this line in
[scripts/render_glass_rollout.py:75-84](../../scripts/render_glass_rollout.py):

```python
def _cluster_label(z, glass_params, T_proto=0.7):
    mu = glass_params["prototypes"]            # (N=16, latent_dim=512)
    zn = z  / ||z||
    mn = mu / ||mu||                            # unit-norm both
    sim = (zn @ mn.T) / T_proto                 # (N,) — cosine similarities × 1/T
    n_star = argmax(sim)                        # which prototype is closest
    S = softmax(glass_params["assign_logits"], axis=1)   # (N, K=8)
    return argmax(S[n_star])                    # which cluster prototype n_star belongs to
```

This evaluates two argmaxes. Both are deliberate and mean different things.
Below: each ingredient explained from scratch.

### 2.1 The latent z (encoder output)

The encoder maps every 15-dim HopperHop observation to a 512-dim **latent
vector** z. Geometrically z lives on the product of 8 simplices (because of
the `SimNorm(V=8)` final activation — each block of 64 dims is softmaxed
independently). So z is a 512-dim point but with the constraint that 8 groups
of 64 entries each sum to 1.

Two intuitions for z:
- "Compressed state": z packs everything the dynamics + reward + Q functions
  need to know about the current observation. If two observations have similar
  z, the rest of TD-MPC2 will plan and predict the same way.
- "Soft codebook coordinates": SimNorm forces z to look like a soft mixture
  over 8×64 = 512 base codes. That's already a clustering-like geometry.

The encoder is trained jointly with the rest of TD-MPC2 plus the Glass loss,
so z evolves over training to be useful for *both* dynamics rollout AND a
structural-entropy partition.

### 2.2 Prototypes μ — what they are and why we want them

`μ` is a learnable matrix of shape `(N=16, latent_dim=512)`. Each row
`μ[n]` is a 512-dim vector. We call them "prototypes" or "anchor latents".

Two ways to think about why μ exists:

**(a) Codebook view (VQ-VAE / SwAV / DINO lineage).** Direct clustering of a
512-dim continuous latent is hard — there's no canonical metric and the
partition function is ill-defined. We instead **learn N "anchor" points** that
the rest of latent space orients itself around. Every new latent is described
by "how similar am I to each anchor", which is a discrete *N*-dim soft
distribution — much easier to feed into structural-entropy machinery than a
raw 512-dim vector.

**(b) Hub-and-spoke view.** We don't want to put N=16 individual cluster
labels on every single latent (too many free parameters) and we don't want to
hand-pick clusters (no domain knowledge). Prototypes are the middle ground:
*the encoder learns z's that gather near these N hubs*, and we then learn how
to *group the hubs* into K=8 clusters (the assign_logits → S map). Two-level
abstraction = state-level (prototypes) + behavioural-level (clusters).

Why N=16 and K=8? Empirically Glass produces 3 or 4 *active* clusters out of
the 8 allowed, so K=8 leaves headroom. N=16 means each cluster gets on average
2 prototypes — enough granularity that "near-stance-prototype" and "near-
landing-prototype" can both belong to the same "stance phase" cluster.

### 2.3 The product z·μᵀ — what it computes

`z @ μ.T` is the inner product of z (shape `(512,)`) with the transpose of μ
(shape `(512, N)`). Result: a vector of shape `(N,)`, with entry n equal to
`z · μ[n]` — the dot product of the current latent with the n-th prototype.

After we L2-normalise both z and each μ[n] (so they're unit vectors), the
dot product equals the **cosine of the angle** between them:

```
sim[n] = (ẑ · μ̂[n]) ∈ [-1, +1]
```

Cosine similarity is a standard "directional similarity" metric used in
contrastive learning. It ignores magnitudes — only the *direction* of z in the
512-dim space matters, which is appropriate here because the rest of TD-MPC2
also uses the latent direction (not magnitude) for predictions.

So `z · μᵀ` answers: **"how aligned is the current latent with each of the 16
anchor states?"** A high value means "this z looks like the n-th anchor"; a
low value means "this z is in a different direction".

The temperature `T_proto = 0.7` divides the similarity before softmax — smaller
T = sharper softmax (more confident assignment to a single prototype), larger
T = softer (more uniform). At T=0.7 the soft assignment is moderately peaked.

### 2.4 The first argmax: which prototype is the current latent closest to?

```python
n_star = argmax(sim)
```

We take the index of the largest similarity. **`n_star` is the prototype that
z currently looks most like** — an integer in 0..N-1 = 0..15. It identifies
**which of the 16 anchor states** the agent is in right now.

(Aside: during *training*, we use the soft `c = softmax(sim/T_proto)` instead
of the hard argmax, because soft assignment is differentiable. The argmax is
only used for the *visualisation overlay* in the rendered MP4, where we want
to print a single cluster id per frame.)

### 2.5 S — the assign_logits softmax (prototype → cluster mapping)

```python
S = softmax(assign_logits, axis=1)        # (N=16, K=8)
```

`assign_logits` is a separate learnable matrix of shape `(N, K)` — completely
independent of z and the data. Each row `S[n]` is a probability distribution
over the K=8 clusters, telling us "**how does prototype n distribute its mass
across the 8 clusters?**"

In well-trained Glass, S is approximately one-hot per row — i.e. each
prototype belongs (almost) entirely to one cluster. The "approximately" is
because the softmax never hits exactly 1.0; in practice we see entries near
0.95 for the assigned cluster and ~0.007 for the others.

The reason S exists is to **coarsen** the N=16 prototype layer into K=8
behavioural clusters. Without S, the 2D structural entropy in Glass would
operate on a 16-node graph; with S, it operates on a coarsened 8-node graph
where the partition itself is learnable. The hierarchical coarsening trick
comes from VQ-VAE-2 and Hi-SwAV (blog §10.1 cites these).

### 2.6 The second argmax: which cluster does prototype n_star belong to?

```python
cluster_id = argmax(S[n_star])
```

We look up row `n_star` of S — that's the prototype→cluster distribution for
the most-similar prototype — and take its argmax. Result: an integer in
0..K-1 = 0..7. **This is the cluster index we draw on the frame.**

### 2.7 Putting it together

```
z (512-dim)          —— "current latent state"
  │
  │ project onto each μ[n] via cosine similarity
  ▼
sim ∈ ℝ^N            —— "similarity to each anchor"
  │
  │ argmax
  ▼
n_star ∈ {0..15}     —— "the closest anchor index"
  │
  │ lookup S[n_star] ∈ ℝ^K, then argmax
  ▼
cluster_id ∈ {0..7}  —— "the cluster that anchor belongs to"
```

In plain English: **the overlay number K on each frame is the cluster id of
the prototype the current latent state most resembles.** Or even more
plainly: **"which of the 8 behavioural categories is the agent currently in,
according to the encoder + Glass partition we trained?"**

When you saw seed-4's K oscillate between 0 and 5, that meant:
- For one half of the gait cycle, z was closest to a prototype belonging to
  cluster 0 (= "knee on ground" in seed-4's policy).
- For the other half, z was closest to a prototype belonging to cluster 5
  (= "stretched, mid-thrust").
- The remaining 6 clusters (1, 2, 3, 4, 6, 7) had no prototype the agent
  actually visited — they're learnable but unused.

When you saw seed-1's K cycle through 3, 4, 6, 7, that meant the encoder had
learned 4 distinct "anchor neighbourhoods" the policy actually visits during
proper hopping, and S grouped them into 4 separate clusters: push-off (K=3),
takeoff (K=4), air-recovery (K=6), and ground-contact-recovery (K=7). **This
is what a healthy Glass partition is supposed to look like.**

### 2.8 Why two argmaxes and not one

Could we skip the prototype layer and put `assign_logits ∈ ℝ^{latent_dim × K}`
directly so that `cluster_id = argmax(z @ assign_logits)`? Yes, but we'd lose
the hierarchical structure. The prototype layer captures **what kinds of
states exist** (16 anchors) separately from **how to group them** (8
clusters). Without that split:

- The structural-entropy graph would be defined directly on data points (B
  per minibatch), not on N=16 nodes. Computing the SE on a B-node graph is
  O(B²) per step instead of O(N²) for N=16. Much slower.
- The assignment would not be **dataset-independent**. With the two-level
  scheme, `S` is a learnable parameter that doesn't depend on which batch we
  sample, so the partition is *stable* across training and identifiable
  post-hoc (which is why this overlay is reproducible at all).

---

## Part 3 — Lessons that change the next iteration

1. **Glass clusters ARE gait-aligned.** Seed 1's K=3/4/6/7 mapped to push-off,
   takeoff, air, ground-recovery — that's exactly the §5.4 gait-phase
   hypothesis. The blog said "the basin choice predicts return" and the
   videos refine that: the basin choice predicts **how many behavioural
   primitives the policy can use**, but you also need the policy to actually
   *use proper foot-hop primitives*, not knee-crawl primitives.

2. **The bottleneck on stuck seeds is the policy's gait technique, not the
   representation.** Seed 3 had a healthy K=4 partition and 5 active clusters,
   but the policy converged to a knee-walk gait that scores 260 even with
   correct latent geometry. Smoothing in Phase-f rescued seed 1's technique
   (from knee-walk-baseline to foot-hop) but seeds 3 and 4 didn't make the same
   transition.

3. **The Phase-j curriculum hypothesis is now sharper.** If smoothing causes the
   knee-walk → foot-hop technique flip, then turning it on after basin lock
   should produce the flip for *all* seeds (not just lucky ones), because the
   basin is identical at the moment smoothing engages. We're 1.75M into
   Phase-j seed 1 and MPPI is climbing through 375 — to be continued.

4. **A direct foot-hop reward shaping** would be a different next experiment if
   Phase-j stalls: penalise knee-ground contact in the reward function (would
   require modifying the mujoco_playground HopperHop reward, not a tdmpc-glass
   change). This is the "behavioural prior" version of what we've been doing
   with representation priors.

# TD-MPC-Glass Structural Entropy Validation

This note validates the structural-entropy (SE) calculation used by the TD-MPC-Glass algorithm in this repository.

Scope:

- validate the 1D and 2D SE helper formulas against the `glass-jax` reference implementation
- validate the hard-partition limit against the discrete `glass-jax` scorer
- inspect whether the TD-MPC-Glass transition graph fed into SE is mathematically valid
- separate true formula correctness from modeling choices in graph construction

This is a validation report only. No algorithm code was changed.

---

## Short answer

The SE calculation in TD-MPC-Glass is **correct** with respect to the in-repo `glass-jax` reference.

More precisely:

1. The helper functions for `H_1` and differentiable `H_2` in `helios-rl` match the `glass-jax` reference exactly.
2. The one-hot hard-partition limit matches the discrete structural-entropy scorer up to floating-point error.
3. The transition graph passed into SE is a valid non-negative symmetric weighted adjacency.
4. The main caveat is not the entropy formula. It is a **modeling choice**: TD-MPC-Glass keeps diagonal self-transition mass in the prototype graph, while the hard graph scorer in `glass-jax` zeroes the diagonal.

So the current status is:

- **formula**: validated
- **implementation of soft 2D SE**: validated
- **graph preprocessing**: valid, but slightly different from the hard-partition graph convention

---

## Files checked

- `helios-rl/src/helios/algorithms/tdmpc_glass.py`
- `glass-jax/src/glass/objectives/structural_entropy.py`
- `glass-jax/src/glass/seclust/entropy.py`
- `glass-jax/docs/rl/tdmpc2_transition_matrix_glass_se.md`

---

## 1. Formula-level validation

### 1D structural entropy

TD-MPC-Glass defines:

$$
H_1(A) = -\sum_i p_i \log_2 p_i, \quad p_i = \frac{d_i}{\sum_j d_j}
$$

with:

- $d_i = \sum_j A_{ij}$
- total volume $2m = \sum_i d_i$

This matches the `glass-jax` reference implementation exactly.

### 2D differentiable structural entropy

TD-MPC-Glass uses the same differentiable 2D structural-entropy implementation as `glass-jax`:

$$
H_2(A, S)
= -\sum_c \frac{g_c}{2m} \log_2 \frac{V_c}{2m}
   - \sum_c \sum_{i \in c} \frac{d_i}{2m} \log_2 \frac{d_i}{V_c}
$$

implemented in the rearranged form:

$$
H_2(A, S)
= \underbrace{-\sum_c \frac{g_c}{2m} \log_2 \frac{V_c}{2m}}_{\text{boundary term}}
 + \underbrace{H_1(A) + \sum_c \frac{V_c}{2m} \log_2 \frac{V_c}{2m}}_{\text{internal term rewritten}}
$$

where:

- $V = d^\top S$
- $AS = A S$
- $g = \sum_i S_{ic}(d_i - (AS)_{ic})$

This also matches the `glass-jax` reference exactly.

---

## 2. Numerical validation results

I ran direct numerical comparisons between the `helios-rl` TD-MPC-Glass functions and the `glass-jax` reference on random test inputs.

### Helper parity vs `glass-jax`

Observed results:

```text
h1_diff = 0.0
h2_diff = 0.0
```

Interpretation:

- the 1D helper output was exactly identical
- the 2D helper output was exactly identical

This is stronger than “close enough”; it indicates the formulas are functionally the same implementation.

### Hard-partition equivalence check

I also checked the one-hot limit of the soft differentiable objective against the discrete hard scorer in `glass-jax/src/glass/seclust/entropy.py`.

Observed result:

```text
hard_equiv_diff = 3.6717985985035284e-08
```

Interpretation:

- this is numerical roundoff only
- the differentiable objective reduces correctly to the hard 2D structural-entropy scorer when assignments are one-hot

This is the most important validation for correctness of the actual SE calculation.

---

## 3. What graph TD-MPC-Glass is actually scoring

The TD-MPC-Glass algorithm does **not** run SE directly on latent states as graph nodes.

Instead it builds a **prototype transition graph**:

1. soft-assign latent source states `z_src` to prototypes
2. soft-assign next states `z_next` to prototypes
3. accumulate prototype-to-prototype transition counts
4. row-normalize the count matrix to get `P`
5. symmetrize it into an undirected adjacency:

$$
A = \frac{P + P^\top}{2}
$$

6. apply structural entropy on `A` together with prototype-to-cluster assignment logits

This matches the intended usage described in the RL integration note in `glass-jax`: build a soft transition matrix, then symmetrize it for undirected SE.

So the entropy is being computed on a **learned prototype graph**, not on the raw environment graph and not directly on all latent samples.

That is a design choice, not a formula error.

---

## 4. Validity of the graph passed into SE

On a sampled TD-MPC-Glass prototype graph, the following properties were observed:

```text
A2_symmetric_maxerr = 0.0
A2_min = 0.12023919820785522
A2_row_sums_min = 0.9813076853752136
A2_row_sums_max = 1.0226914882659912
A2_total_mass = 7.999999523162842
```

Interpretation:

- the adjacency is exactly symmetric
- all weights are non-negative
- row sums stay near 1 after symmetrization because `P` is row-normalized first
- the total graph mass is finite and well-defined

This is a mathematically valid input to the 1D and 2D SE formulas used here.

---

## 5. Important caveat: diagonal mass is retained

This is the only notable mismatch I found between TD-MPC-Glass graph construction and the hard-graph conventions used in `glass-jax` clustering utilities.

### What happens in TD-MPC-Glass

The prototype graph keeps diagonal self-transition mass after symmetrization.

Observed on one sampled graph:

```text
A2_diag_mass_mean = 0.12498702853918076
```

In a second larger check:

```text
diag_fraction_of_total = 0.0624995231628418
```

So roughly 6.25% of total graph mass sat on the diagonal in that sample.

### Why this matters

The hard structural-entropy scorer in `glass-jax/src/glass/seclust/entropy.py` explicitly converts inputs to a **loop-free symmetric adjacency** by zeroing the diagonal.

That means:

- TD-MPC-Glass soft SE: allows self-transition mass
- hard SE scorer: removes self-loops

These are not the same graph convention.

### How much difference did it make?

On one sampled TD-MPC-Glass graph:

```text
se_keep_diag = 3.567655563354492
se_zero_diag = 3.583251953125
delta_keep_minus_zero = -0.015596389770507812
```

Interpretation:

- keeping diagonal mass lowered SE slightly in this sample
- the effect was real but small
- this is a modeling choice, not a bug in the entropy calculation itself

---

## 6. Final verdict

### Validated

- `one_dimensional_structural_entropy(...)` in TD-MPC-Glass is correct.
- `two_dimensional_structural_entropy(...)` in TD-MPC-Glass is correct.
- The soft objective matches the `glass-jax` reference exactly.
- The hard one-hot limit matches the discrete scorer up to floating-point error.

### Caveat to keep in mind

The SE value used by TD-MPC-Glass is computed on a prototype transition graph that:

- is dense
- is symmetrized from a row-normalized transition matrix
- retains diagonal self-transition mass

So if you compare TD-MPC-Glass SE values directly against hard partition SE values from the `glass-jax` clustering stack, you are comparing **slightly different graph conventions**.

### Bottom line

There is **no evidence of an SE calculation bug** in the TD-MPC-Glass algorithm.

If future experiments show odd behavior, the first place to investigate is **graph construction and regularization choices**, not the entropy formula.

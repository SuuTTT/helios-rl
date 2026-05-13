Below are the main **forms of structural entropy for graphs**, using:

[
G=(V,E),\quad d_i=\sum_j A_{ij},\quad \mathrm{vol}(G)=\sum_i d_i=2m
]

For a community/module (C_k):

[
V_k=\sum_{i\in C_k} d_i,\qquad g_k=\text{cut}(C_k,V\setminus C_k)
]

Structural entropy extends Shannon entropy to graphs by measuring uncertainty under a hierarchical partition or encoding-tree strategy. ([openreview.net][1])

---

## 1. One-dimensional structural entropy

This is the no-community, flat entropy of the graph degree distribution:

[
H^1(G)
======

-\sum_{i=1}^{N}
\frac{d_i}{2m}
\log_2
\frac{d_i}{2m}
]

Equivalent notation:

[
H^1(G)
======

-\sum_{v\in V}
\frac{d_v}{\mathrm{vol}(G)}
\log_2
\frac{d_v}{\mathrm{vol}(G)}
]

It measures uncertainty of locating a random node using only degree-based visiting probability. Your uploaded notes define this as the baseline “no communities” case.  The same definition appears in graph-structure-learning literature. ([openreview.net][1])

---

## 2. Two-dimensional structural entropy: raw partition form

Given a partition:

[
\mathcal{P}={C_1,C_2,\dots,C_K}
]

the 2D structural entropy is:

[
H^2(G;\mathcal{P})
==================

-\sum_{k=1}^{K}
\frac{g_k}{2m}
\log_2
\frac{V_k}{2m}
--------------

\sum_{k=1}^{K}
\sum_{i\in C_k}
\frac{d_i}{2m}
\log_2
\frac{d_i}{V_k}
]

Interpretation:

[
H^2
===

\text{community-level uncertainty}
+
\text{within-community uncertainty}
]

Your uploaded derivation gives exactly this raw form, with (V_k) as module volume and (g_k) as boundary cut. 

---

## 3. Two-dimensional structural entropy: compact algebraic form

The second term can be rewritten using (H^1(G)). The globally computable form is:

[
H^2(G;\mathcal{P})
==================

-\sum_{k=1}^{K}
\frac{g_k}{2m}
\log_2
\frac{V_k}{2m}
+
H^1(G)
+
\sum_{k=1}^{K}
\frac{V_k}{2m}
\log_2
\frac{V_k}{2m}
]

So:

[
H^2
===

\underbrace{
-\sum_k
\frac{g_k}{2m}
\log_2
\frac{V_k}{2m}
}*{\text{inter-community uncertainty}}
+
\underbrace{
H^1(G)
+
\sum_k
\frac{V_k}{2m}
\log_2
\frac{V_k}{2m}
}*{\text{intra-community uncertainty}}
]

This is the form most useful for efficient implementation because it avoids explicit nested sums over all nodes inside each community. 

---

## 4. Optimal two-dimensional structural entropy

For community detection, the objective is usually the minimum over all partitions:

[
H^2(G)
======

\min_{\mathcal{P}}
H^2(G;\mathcal{P})
]

Lower (H^2) means the partition gives a better compression/explanation of graph structure.

---

## 5. Encoding-tree structural entropy

For a hierarchical encoding tree (T), every non-root tree node (\alpha) corresponds to a subset of graph nodes (T_\alpha\subseteq V). Let:

[
V_\alpha=\sum_{v\in T_\alpha} d_v
]

[
g_\alpha=\text{cut}(T_\alpha,V\setminus T_\alpha)
]

[
\alpha^-=\text{parent of }\alpha
]

The contribution of tree node (\alpha) is:

[
H^T(G;\alpha)
=============

*

\frac{g_\alpha}{\mathrm{vol}(G)}
\log_2
\frac{V_\alpha}{V_{\alpha^-}}
]

The full tree entropy is:

[
H_T(G)
======

\sum_{\alpha\in T,\ \alpha\neq \lambda}
H^T(G;\alpha)
]

or:

[
H_T(G)
======

*

\sum_{\alpha\in T,\ \alpha\neq \lambda}
\frac{g_\alpha}{\mathrm{vol}(G)}
\log_2
\frac{V_\alpha}{V_{\alpha^-}}
]

where (\lambda) is the root. This is the general hierarchical form of structural entropy. ([Proceedings of Machine Learning Research][2])

---

## 6. (K)-dimensional structural entropy

The (K)-dimensional form minimizes over encoding trees whose height is limited to (K):

[
H^K(G)
======

\min_{T:\ \mathrm{height}(T)\le K}
H_T(G)
]

Some papers write the constraint as (\mathrm{height}(T)=K):

[
H^{(K)}(G)
==========

\min_{T:\ \mathrm{height}(T)=K}
H_T(G)
]

Both are height-constrained versions of the same idea: find the best hierarchical abstraction of the graph at depth (K). ([openreview.net][1])

Special cases:

[
K=1 \Rightarrow H^1(G)
]

[
K=2 \Rightarrow H^2(G)
]

[
K>2 \Rightarrow \text{multi-level hierarchical structural entropy}
]

---

## 7. Differentiable soft structural entropy

For differentiable clustering, replace hard community assignment with a soft assignment matrix:

[
S\in\mathbb{R}^{N\times K}
]

where:

[
S_{ik}=\Pr(i\in C_k),\qquad \sum_k S_{ik}=1
]

Then:

[
d=A\mathbf{1}
]

[
V=S^\top d
]

[
AS=A S
]

[
g
=

\sum_{\text{axis}=0}
S\odot
\left(d[:,\text{None}]-AS\right)
]

Define:

[
p_{\mathrm{vol}}=\frac{V}{2m}
]

[
p_{\mathrm{cut}}=\frac{g}{2m}
]

The differentiable 2D structural entropy is:

[
H^2_{\text{soft}}(G)
====================

-\sum_k
p_{\mathrm{cut},k}
\log_2
p_{\mathrm{vol},k}
+
H^1(G)
+
\sum_k
p_{\mathrm{vol},k}
\log_2
p_{\mathrm{vol},k}
]

This is the exact soft-matrix form described in your `glass-jax` notes. 

---

## 8. Boundary-only proxy structural entropy / DSI form

Some differentiable methods use only the inter-community boundary term:

[
\mathcal{H}^{\mathcal{T}}(G;h)
==============================

-\frac{1}{V_{\mathrm{total}}}
\sum_{k=1}^{K}
\left(
V_k-\mathrm{internal_edges}*k
\right)
\log_2
\frac{V_k}{V*{\mathrm{parent}}}
]

This is often called a **Differentiable Structural Information** proxy. Your notes point out that this mainly captures boundary-crossing uncertainty and omits the full intra-community component of exact (H^2). 

---

## 9. Structural entropy reduction / resistance form

A derived quantity is the entropy reduction from 1D to 2D:

[
R(G)
====

H^1(G)-H^2(G)
]

This measures how much uncertainty is removed by introducing a 2D community structure. Li and Pan define graph resistance this way for connected networks. ([arXiv][3])

A normalized version is:

[
\theta(G)
=========

# \frac{R(G)}{H^1(G)}

\frac{H^1(G)-H^2(G)}{H^1(G)}
]

This is not a separate structural entropy itself, but a useful structural-information gain ratio. ([arXiv][3])

---

## Summary table

| Form                   | Formula                                                                                              | Meaning                          |
| ---------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------- |
| 1D SE                  | (H^1(G)=-\sum_i \frac{d_i}{2m}\log_2\frac{d_i}{2m})                                                  | No communities                   |
| 2D raw SE              | Inter-community + intra-community nested sum                                                         | Flat community partition         |
| 2D compact SE          | (-\sum_k \frac{g_k}{2m}\log_2\frac{V_k}{2m}+H^1+\sum_k\frac{V_k}{2m}\log_2\frac{V_k}{2m})            | Efficient exact form             |
| Optimal 2D SE          | (H^2(G)=\min_\mathcal{P}H^2(G;\mathcal{P}))                                                          | Best flat partition              |
| Tree SE                | (H_T(G)=-\sum_{\alpha\ne\lambda}\frac{g_\alpha}{\mathrm{vol}(G)}\log_2\frac{V_\alpha}{V_{\alpha^-}}) | General hierarchy                |
| (K)-D SE               | (H^K(G)=\min_{\mathrm{height}(T)\le K}H_T(G))                                                        | Best hierarchy up to depth (K)   |
| Soft SE                | Uses (S\in\mathbb{R}^{N\times K})                                                                    | Differentiable clustering        |
| DSI proxy              | Boundary-only entropy                                                                                | Approximate differentiable proxy |
| Reduction / resistance | (H^1-H^2)                                                                                            | Entropy saved by communities     |

Note: one earlier upload appears expired on my side; the current pasted markdown file was loaded and used here.

[1]: https://openreview.net/pdf?id=9XkTF1TsHi "Structural Entropy Based Graph Structure Learning for Node Classification"
[2]: https://proceedings.mlr.press/v162/wu22b/wu22b.pdf "Structural Entropy Guided Graph Hierarchical Pooling"
[3]: https://arxiv.org/abs/1801.03404 "[1801.03404] Structure Entropy and Resistor Graphs"

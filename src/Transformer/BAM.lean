module

public import Transformer.BAM.Section2_BayesianAttention

/-!
# Bayesian Attention Mechanism

Bianchessi, Aguirre, Barros, Kupssinskü, arXiv:2505.22842v4, "Bayesian Attention Mechanism: A
Probabilistic Framework for Positional Encoding and Context Length Extrapolation" (ICLR 2026).
The TeX source is in `papers/arXiv-2505.22842v4/`.

Attention scores are extended reals (`EReal`): the causal mask sets scores to `−∞`, which get the
weight `0`.

* `Section2_BayesianAttention`: §2.1, Definition 1 (attention is an expectation of the values) and
  Theorem 1 (an additive score factors the weight into content and position marginals over `Z`).
-/

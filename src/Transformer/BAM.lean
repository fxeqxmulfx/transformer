module

public import Transformer.BAM.Section2_BayesianAttention
public import Transformer.BAM.SectionA_Notation
public import Transformer.BAM.Section2_Priors

/-!
# Bayesian Attention Mechanism

Bianchessi, Aguirre, Barros, Kupssinskü, arXiv:2505.22842v4, "Bayesian Attention Mechanism: A
Probabilistic Framework for Positional Encoding and Context Length Extrapolation" (ICLR 2026).
The TeX source is in `papers/arXiv-2505.22842v4/`.

Attention scores are extended reals (`EReal`): the causal mask sets scores to `−∞`, which get the
weight `0`.

* `Section2_BayesianAttention`: §2.1, Definition 1 (attention is an expectation of the values) and
  Theorem 1 (an additive score factors the weight into content and position marginals over `Z`).
* `SectionA_Notation`: Appendix A, the uniform, Laplace and generalized Gaussian (GGD) priors;
  the GGD is a density, `β = 1` is Laplace, `β = 2` is Normal, larger `β` is more localized and
  smaller `β` has the heavier tail.
* `Section2_Priors`: §2.2, Lemma 1 (the causal mask is the uniform prior) and Lemma 2 (ALiBi is
  the prior `Uniform · Laplace`, up to the normalizer its proof drops; false without it).
-/

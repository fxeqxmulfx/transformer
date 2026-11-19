module

public import Transformer.BAM.Section2_BayesianAttention
public import Transformer.BAM.SectionA_Notation
public import Transformer.BAM.Section2_Priors
public import Transformer.BAM.Section2_GeneralizedGaussian
public import Transformer.BAM.Section2_Locality
public import Transformer.BAM.Section2_IgnoresLocal
public import Transformer.BAM.Section2_LongContext
public import Transformer.BAM.Section2_Intuition
public import Transformer.BAM.SectionC_Parametrization
public import Transformer.BAM.SectionK_Partition
public import Transformer.BAM.SectionK_Information

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
* `Section2_GeneralizedGaussian`: §2.3, the GGD-BAM bias and prior, ALiBi as its case `β = 1`,
  and Theorem 2 (for `0 < β < 1` the GGD penalty is below ALiBi's; the proof's witness fails at
  distance `1`, another works).
* `Section2_Locality`: Lemmas 3 and 4 (ALiBi and GGD-BAM with `β > 0` are local), for bounded
  content scores, uniformly in the context; false without the bound.
* `Section2_IgnoresLocal`: Theorem 3 (with `β < 0` the query ignores its own key), for `μ = 0` as
  in its proof; false for `μ ≠ 0`.
* `Section2_LongContext`: Theorem 4 (with `β < 0` attention to distant keys does not vanish) is
  false for bounded content: attention fades; what its proof shows, that GGD-BAM keeps the NoPE
  weight of distant keys, holds.
* `Section2_Intuition`: §2.5, the shapes of the priors (the GGD peaks at `μ`, beyond the causal
  mask if `μ > 0`; for `β < 0` it grows with distance and is not a density), attention fading
  without a positional bias, and the scalable softmax identity.
* `SectionC_Parametrization`: Appendix C, the learnable parameters `θ_μ`, `θ_α`, `θ_β`: with
  `ε = 0` the bias is GGD-BAM's, for every `α > 0` and `β ≠ 0`; `ε > 0` keeps it finite, and as
  `ε → 0` it tends to GGD-BAM's.
* `SectionK_Partition`: Appendix K.1–K.2, `Z = Z_cont Z_pos / Z_joint` restores the factorization
  and is `1 / ⟨p_cont, p_pos⟩`, not `⟨p_cont, p_pos⟩`; the overlap lies in `[0, 1]`, vanishes
  exactly on disjoint supports, is not maximal on equal ones, and is `< 1` on two keys or more.
* `SectionK_Information`: Appendix K.3, the paper's mutual information is `ln Z = −ln⟨p_cont,
  p_pos⟩ ≥ 0`, `Z = exp(I)`, and `I = 0` exactly when `p_joint = p_cont · p_pos`, which real
  content on two keys or more never gives.
-/

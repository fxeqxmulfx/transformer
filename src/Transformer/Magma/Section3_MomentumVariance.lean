/-
# Cosine invariance and conditional variance of lazy moments

Formalization of arXiv:2602.15322v1, Sections 2--3, Why dense momentum
updates matter, and eq:masking_prob. The variance comparison is for an
explicit unbiased lazy estimator at fixed state and gradient; it is not
a general claim about the variance of an entire training trajectory.
-/

import Transformer.Magma.Section3_DampingBounds

open scoped InnerProductSpace

noncomputable section

namespace Transformer.Magma

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Cosine is invariant under positive rescaling of either vector,
including zero inputs under the documented extension. Source:
arXiv:2602.15322v1, Section 3, scale-invariant alignment paragraph. -/
theorem cosine_scale_invariant (momentum g : E) (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    cosine (a • momentum) (b • g) = cosine momentum g := by
  simp only [cosine, real_inner_smul_left, real_inner_smul_right, norm_smul,
    Real.norm_eq_abs, abs_of_pos ha, abs_of_pos hb]
  field_simp

/-- Positive rescaling is possible for both vectors. Source:
arXiv:2602.15322v1, Section 3, eq:masking_prob. -/
example : (0 : ℝ) < 2 ∧ (0 : ℝ) < 3 := by norm_num

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

omit [NormedAddCommGroup E] [InnerProductSpace ℝ E] in
/-- An unbiased sparse momentum estimate has this conditional variance
around the dense EMA. The sparse estimate uses m/p, so its expectation
equals the dense state. Source: arXiv:2602.15322v1, Section 2,
Why dense momentum updates matter, qualified comparison. -/
theorem lazy_momentum_conditional_variance (p β previous g : ℝ) (hp : p ≠ 0) (j : ι) :
    maskExpectation p (fun mask =>
      (β * previous + (1 - β) * maskCoefficient p (mask j) * g -
        (β * previous + (1 - β) * g)) ^ 2) =
      (1 - β) ^ 2 * g ^ 2 * ((1 - p) / p) := by
  have hpoint (mask : ι → Bool) :
      (β * previous + (1 - β) * maskCoefficient p (mask j) * g -
        (β * previous + (1 - β) * g)) ^ 2 =
      ((1 - β) ^ 2 * g ^ 2) * (maskCoefficient p (mask j) - 1) ^ 2 := by ring
  simp only [hpoint, maskExpectation_mul, maskCoefficient_variance p hp]

/-- The lazy-estimator law can have positive survival and a nonzero
gradient. Source: arXiv:2602.15322v1, Section 2, dense moment comparison. -/
example : (1 / 2 : ℝ) ≠ 0 := by norm_num

omit [NormedAddCommGroup E] [InnerProductSpace ℝ E] in
/-- With genuine masking, a nonconstant EMA and nonzero current gradient,
the unbiased lazy estimator has positive conditional variance, whereas
the dense EMA is deterministic conditioned on that state and gradient.
Source: arXiv:2602.15322v1, Section 2, qualified dense-versus-lazy claim. -/
theorem lazy_momentum_variance_positive (p β previous g : ℝ) (hp : 0 < p) (hp' : p < 1)
    (hβ : β < 1) (hg : g ≠ 0) (j : ι) :
    0 < maskExpectation p (fun mask =>
      (β * previous + (1 - β) * maskCoefficient p (mask j) * g -
        (β * previous + (1 - β) * g)) ^ 2) := by
  rw [lazy_momentum_conditional_variance p β previous g hp.ne' j]
  have hsq : 0 < (1 - β) ^ 2 := sq_pos_of_pos (sub_pos.mpr hβ)
  have hgsq : 0 < g ^ 2 := sq_pos_of_ne_zero hg
  exact mul_pos (mul_pos hsq hgsq) (div_pos (sub_pos.mpr hp') hp)

/-- All strict-variance hypotheses hold jointly for the printed survival
and a usual EMA coefficient. Source: arXiv:2602.15322v1, Section 2. -/
example : (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧ (9 / 10 : ℝ) < 1 ∧ (1 : ℝ) ≠ 0 := by
  norm_num

end Transformer.Magma

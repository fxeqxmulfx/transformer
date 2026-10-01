/-
# Linear and quadratic expansions for actual block directions

Formalization of arXiv:2602.15322v1, Section 2 and Appendix A.1.
Block directions are vectors in the full parameter space, so arbitrary
within-block dimensions and mixed block curvature are retained.
-/

import Transformer.Magma.Section2_Expectations
import Mathlib.Analysis.Calculus.FDeriv.Basic

open scoped BigOperators

noncomputable section

namespace Transformer.Magma

variable {ι E : Type*} [Fintype ι] [DecidableEq ι]
  [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Sum of normalized, independently masked embedded block directions.
Source: arXiv:2602.15322v1, Section 2, eq:rsu_def, s=1/p. -/
def maskedSum (p : ℝ) (update : ι → E) (mask : ι → Bool) : E :=
  ∑ b, maskCoefficient p (mask b) • update b

omit [DecidableEq ι] in
/-- A linear form distributes over the actual masked block sum.
Source: arXiv:2602.15322v1, Appendix A.1, Taylor linear term. -/
theorem maskedSum_linear (p : ℝ) (L : E →L[ℝ] ℝ) (update : ι → E) (mask : ι → Bool) :
    L (maskedSum p update mask) = ∑ b, maskCoefficient p (mask b) * L (update b) := by
  simp [maskedSum, map_sum, map_smul]

omit [DecidableEq ι] in
/-- A bilinear form retains all block-pair interactions.
Source: arXiv:2602.15322v1, Appendix A.1, Taylor quadratic term. -/
theorem maskedSum_bilinear (p : ℝ) (H : E →L[ℝ] E →L[ℝ] ℝ)
    (update : ι → E) (mask : ι → Bool) :
    H (maskedSum p update mask) (maskedSum p update mask) =
      ∑ b, ∑ c, maskCoefficient p (mask b) * maskCoefficient p (mask c) *
        H (update b) (update c) := by
  simp only [maskedSum, map_sum, map_smul, sum_apply, smul_apply, smul_eq_mul,
    Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro b hb
  apply Finset.sum_congr rfl
  intro c hc
  ring

/-- The expected Taylor linear term is unchanged by normalized masking.
Source: arXiv:2602.15322v1, Appendix A.1. -/
theorem maskedSum_linear_mean (p : ℝ) (hp : p ≠ 0) (L : E →L[ℝ] ℝ) (update : ι → E) :
    maskExpectation p (fun mask => L (maskedSum p update mask)) = L (∑ b, update b) := by
  simp only [maskedSum_linear, maskExpectation_sum]
  simp_rw [maskExpectation_mul_right, maskCoefficient_mean p hp, one_mul]
  simp [map_sum]

/-- Nonzero survival is possible. Source: arXiv:2602.15322v1, Section 2. -/
example : (1 / 2 : ℝ) ≠ 0 := by norm_num

/-- The expected quadratic term gains exactly the diagonal block
curvature correction. H can be the objective's actual second derivative.
Source: arXiv:2602.15322v1, Section 2, eq:implicit_reg, Appendix A.1. -/
theorem maskedSum_bilinear_mean (p : ℝ) (hp : p ≠ 0)
    (H : E →L[ℝ] E →L[ℝ] ℝ) (update : ι → E) :
    maskExpectation p (fun mask => H (maskedSum p update mask) (maskedSum p update mask)) =
      H (∑ b, update b) (∑ b, update b) +
        (1 - p) / p * ∑ b, H (update b) (update b) := by
  simp only [maskedSum_bilinear]
  rw [maskExpectation_quadratic p hp]
  congr 1
  simp [map_sum]
  rw [Finset.sum_comm]

/-- The bilinear expectation domain is nonempty. Source:
arXiv:2602.15322v1, Section 2. -/
example : (1 / 2 : ℝ) ≠ 0 := by norm_num

/-- Cubic remainder control over actual block norms, with explicit 1/p^2
amplification. Source: arXiv:2602.15322v1, Appendix A.1, final paragraph. -/
theorem maskedSum_block_third_moment (p : ℝ) (hp : 0 < p) (update : ι → E) :
    maskExpectation p (fun mask => ∑ b, ‖maskCoefficient p (mask b) • update b‖ ^ 3) =
      (1 / p ^ 2) * ∑ b, ‖update b‖ ^ 3 := by
  have hn (b : ι) (bit : Bool) :
      ‖maskCoefficient p bit • update b‖ ^ 3 =
        maskCoefficient p bit ^ 3 * ‖update b‖ ^ 3 := by
    cases bit <;> simp [maskCoefficient, norm_smul, Real.norm_eq_abs,
      abs_of_pos hp, mul_pow]
  simp_rw [hn]
  rw [maskExpectation_sum]
  simp_rw [maskExpectation_mul_right, maskCoefficient_third p hp.ne']
  rw [Finset.mul_sum]

/-- Positive survival is possible. Source: arXiv:2602.15322v1, Section 2. -/
example : (0 : ℝ) < 1 / 2 := by norm_num

end Transformer.Magma

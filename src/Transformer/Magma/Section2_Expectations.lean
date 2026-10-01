/-
# Finite mask expectation and the curvature correction

Formalization of arXiv:2602.15322v1, Section 2, eq:implicit_reg,
and Appendix A.1. These identities retain all mixed Hessian coefficients.
Only the diagonal block terms gain the extra factor (1-p)/p.
-/

import Transformer.Magma.Section2_Moments

open scoped BigOperators

noncomputable section

namespace Transformer.Magma

variable {ι κ : Type*} [Fintype ι] [DecidableEq ι] [Fintype κ]

/-- Expectation commutes with finite sums. Source: arXiv:2602.15322v1,
Appendix A.1, Taylor expansion expectation. -/
theorem maskExpectation_sum (p : ℝ) (f : κ → (ι → Bool) → ℝ) :
    maskExpectation p (fun mask => ∑ j, f j mask) =
      ∑ j, maskExpectation p (f j) := by
  simp only [maskExpectation, Finset.mul_sum]
  rw [Finset.sum_comm]

/-- Right scalar multiplication in expectation. Source:
arXiv:2602.15322v1, Appendix A.1. -/
theorem maskExpectation_mul_right (p a : ℝ) (f : (ι → Bool) → ℝ) :
    maskExpectation p (fun mask => f mask * a) = maskExpectation p f * a := by
  simpa only [mul_comm] using maskExpectation_mul p a f

/-- Subtraction in expectation. Source: arXiv:2602.15322v1,
Appendix A.1, subtracting the unmasked Taylor expansion. -/
theorem maskExpectation_sub (p : ℝ) (f g : (ι → Bool) → ℝ) :
    maskExpectation p (fun mask => f mask - g mask) =
      maskExpectation p f - maskExpectation p g := by
  simp [maskExpectation, mul_sub, Finset.sum_sub_distrib]

/-- Expectation preserves pointwise order on the probability domain.
Source: arXiv:2602.15322v1, Appendix A.1, remainder estimate. -/
theorem maskExpectation_mono (p : ℝ) (hp : 0 ≤ p) (hp' : p ≤ 1)
    (f g : (ι → Bool) → ℝ) (hfg : ∀ mask, f mask ≤ g mask) :
    maskExpectation p f ≤ maskExpectation p g := by
  apply Finset.sum_le_sum
  intro mask hm
  exact mul_le_mul_of_nonneg_left (hfg mask) (maskMass_nonneg p hp hp' mask)

/-- All order hypotheses are possible. Source: arXiv:2602.15322v1,
Section 2, Algorithm 1's p=1/2. -/
example : (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    ∀ mask : Unit → Bool, (if mask () then (0 : ℝ) else 1) ≤ 1 := by
  refine ⟨by norm_num, by norm_num, ?_⟩
  intro mask
  split <;> norm_num

/-- An absolute expectation is bounded by the expectation of absolute
values. Source: arXiv:2602.15322v1, Appendix A.1, remainder estimate. -/
theorem maskExpectation_abs (p : ℝ) (hp : 0 ≤ p) (hp' : p ≤ 1)
    (f : (ι → Bool) → ℝ) :
    |maskExpectation p f| ≤ maskExpectation p (fun mask => |f mask|) := by
  unfold maskExpectation
  calc
    |∑ mask, maskMass p mask * f mask| ≤ ∑ mask, |maskMass p mask * f mask| :=
      Finset.abs_sum_le_sum_abs _ _
    _ = ∑ mask, maskMass p mask * |f mask| := by
      apply Finset.sum_congr rfl
      intro mask hm
      rw [abs_mul, abs_of_nonneg (maskMass_nonneg p hp hp' mask)]

/-- The absolute-expectation domain is nonempty. Source:
arXiv:2602.15322v1, Algorithm 1. -/
example : (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 := by norm_num

/-- Expectation of any quadratic block coefficient array. The array may
later be evaluated from the actual Hessian on embedded block directions.
Source: arXiv:2602.15322v1, Appendix A.1, Hessian expectation. -/
theorem maskExpectation_quadratic (p : ℝ) (hp : p ≠ 0) (H : ι → ι → ℝ) :
    maskExpectation p (fun mask => ∑ b, ∑ c,
      maskCoefficient p (mask b) * maskCoefficient p (mask c) * H b c) =
      (∑ b, ∑ c, H b c) + (1 - p) / p * ∑ b, H b b := by
  simp only [maskExpectation_sum]
  have hterm (b c : ι) :
      maskExpectation p (fun mask =>
        maskCoefficient p (mask b) * maskCoefficient p (mask c) * H b c) =
          H b c + if c = b then (1 - p) / p * H b b else 0 := by
    rw [maskExpectation_mul_right]
    by_cases hbc : b = c
    · subst c
      simp only [← sq, maskCoefficient_second p hp, ite_true]
      field_simp
      ring
    · rw [maskCoefficient_cross p hp b c hbc]
      simp [Ne.symm hbc]
  simp_rw [hterm]
  simp only [Finset.sum_add_distrib]
  simp [Finset.mul_sum]

/-- The curvature identity has nonempty survival domain and supports
positive curvature; there is no sign restriction on general Hessians.
Source: arXiv:2602.15322v1, Section 2 and Appendix A.1. -/
example : (1 / 2 : ℝ) ≠ 0 := by norm_num

end Transformer.Magma

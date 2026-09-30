/-
# AdaFisher: natural-gradient coordinate invariance

arXiv:2405.16397v3, §3.2, `eq:NGD` and its coordinate-invariance discussion.
This is the tangent-space transformation law for the exact Fisher metric;
the diagonal approximation does not automatically inherit that property.
-/

import Transformer.AdaFisher.Section3_Spectrum

open scoped Matrix

noncomputable section

namespace Transformer.AdaFisher

variable {d : ℕ}

/-- Fisher pullback by the parameter Jacobian, §3.2, `eq:NGD`. -/
def fisherPullback (J F : Matrix (Fin d) (Fin d) ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  J.transpose * F * J

/-- With an invertible parameter Jacobian, the exact natural gradient
transforms back to the same tangent vector, §3.2, `eq:NGD`.
The gradient covector pulls back by Jᵀ and the Fisher by Jᵀ F J. This
algebra holds even for Mathlib's total inverse of a singular F; a genuine
metric additionally requires nonsingular F to interpret it as a solve. -/
theorem naturalGradient_invariance (J F : Matrix (Fin d) (Fin d) ℝ)
    (g : Fin d → ℝ) (hJ : IsUnit J.det) :
    J.mulVec ((fisherPullback J F)⁻¹.mulVec (J.transpose.mulVec g)) = F⁻¹.mulVec g := by
  have hleft := Matrix.mul_nonsing_inv J hJ
  have hright := Matrix.nonsing_inv_mul J.transpose (by simpa using hJ)
  calc
    _ = (J * J⁻¹ * F⁻¹ * J.transpose⁻¹ * J.transpose).mulVec g := by
      simp only [fisherPullback, Matrix.mul_inv_rev, Matrix.mulVec_mulVec, Matrix.mul_assoc]
    _ = _ := by
      simp only [Matrix.mul_assoc, hright, Matrix.mul_one]
      rw [← Matrix.mul_assoc, hleft, Matrix.one_mul]

example : IsUnit (1 : Matrix (Fin 2) (Fin 2) ℝ).det := by simp

end Transformer.AdaFisher

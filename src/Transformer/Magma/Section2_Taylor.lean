/-
# Corrected curvature regularization with an explicit remainder

Formalization of arXiv:2602.15322v1, Section 2, Proposition
lemma:implicit_reg, eq:implicit_reg, and Appendix A.1. The source omits
a third-order regularity hypothesis and drops the unmasked Taylor remainder
when replacing its Taylor polynomial by l(theta-Delta). Both are corrected
below. First and second derivatives are the objective's actual Frechet
derivatives. Block vectors are embedded in the full parameter space.
-/

import Transformer.Magma.Section2_BlockExpansion
import Mathlib.Analysis.Calculus.FDeriv.Const

open scoped BigOperators

noncomputable section

namespace Transformer.Magma

variable {ι E : Type*} [Fintype ι] [DecidableEq ι]
  [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The actual second-order Taylor polynomial at x, evaluated at x-u.
Source: arXiv:2602.15322v1, Appendix A.1, eq:taylor. -/
def lossTaylor (loss : E → ℝ) (x u : E) : ℝ :=
  loss x - fderiv ℝ loss x u + fderiv ℝ (fderiv ℝ loss) x u u / 2

/-- Explicit cubic Taylor control in the block norm used by the paper.
It is an objective regularity condition, not a hypothesis asserting the
expected masked loss. A local version only needs to cover the base and
masked displacements; this global form makes their domain explicit.
Source: correction to arXiv:2602.15322v1, Appendix A.1, remainder statement. -/
def BlockCubicTaylor (loss : E → ℝ) (x : E) (C : ℝ) : Prop :=
  ∀ update : ι → E,
    |loss (x - ∑ b, update b) - lossTaylor loss x (∑ b, update b)| ≤
      C * ∑ b, ‖update b‖ ^ 3

/-- Exact expected Taylor polynomial, retaining all mixed curvature.
Source: arXiv:2602.15322v1, Appendix A.1, Hessian regrouping. -/
theorem lossTaylor_mask_mean (p : ℝ) (hp : p ≠ 0) (loss : E → ℝ) (x : E) (update : ι → E) :
    maskExpectation p (fun mask => lossTaylor loss x (maskedSum p update mask)) =
      lossTaylor loss x (∑ b, update b) +
        (1 - p) / (2 * p) * ∑ b, fderiv ℝ (fderiv ℝ loss) x (update b) (update b) := by
  unfold lossTaylor
  simp only [div_eq_mul_inv, maskExpectation_add, maskExpectation_sub,
    maskExpectation_const, maskExpectation_mul_right,
    maskedSum_linear_mean p hp, maskedSum_bilinear_mean p hp]
  ring

/-- The Taylor expectation hypotheses are nonempty. Source:
arXiv:2602.15322v1, Section 2, p=1/2. -/
example : (1 / 2 : ℝ) ≠ 0 := by norm_num

/-- Corrected Proposition lemma:implicit_reg with an explicit cubic error
bound. For fixed p it implies the paper's O(sum ||Delta_b||^3) statement.
Correction: assume cubic Taylor control and include BOTH the expected
masked remainder and the negative unmasked remainder. Their combined
bound is C*(1+1/p^2), not just the masked remainder.
Source: arXiv:2602.15322v1, Section 2 and Appendix A.1. -/
theorem implicit_regularization_corrected (p : ℝ) (hp : 0 < p) (hp' : p ≤ 1)
    (loss : E → ℝ) (x : E) (C : ℝ) (hTaylor : BlockCubicTaylor (ι := ι) loss x C)
    (update : ι → E) :
    |maskExpectation p (fun mask => loss (x - maskedSum p update mask)) -
        loss (x - ∑ b, update b) -
        (1 - p) / (2 * p) *
          ∑ b, fderiv ℝ (fderiv ℝ loss) x (update b) (update b)| ≤
      C * (1 + 1 / p ^ 2) * ∑ b, ‖update b‖ ^ 3 := by
  let R (u : E) := loss (x - u) - lossTaylor loss x u
  have hexpand :
      maskExpectation p (fun mask => loss (x - maskedSum p update mask)) =
        maskExpectation p (fun mask => R (maskedSum p update mask)) +
          maskExpectation p (fun mask => lossTaylor loss x (maskedSum p update mask)) := by
    rw [← maskExpectation_add]
    congr 1
    funext mask
    dsimp [R]
    ring
  have hrem : |maskExpectation p (fun mask => R (maskedSum p update mask))| ≤
      C * (1 / p ^ 2) * ∑ b, ‖update b‖ ^ 3 := by
    calc
      _ ≤ maskExpectation p (fun mask => |R (maskedSum p update mask)|) :=
        maskExpectation_abs p hp.le hp' _
      _ ≤ maskExpectation p (fun mask => C *
          ∑ b, ‖maskCoefficient p (mask b) • update b‖ ^ 3) := by
        apply maskExpectation_mono p hp.le hp'
        intro mask
        exact hTaylor (fun b => maskCoefficient p (mask b) • update b)
      _ = _ := by
        rw [maskExpectation_mul, maskedSum_block_third_moment p hp]
        ring
  have hbase : |R (∑ b, update b)| ≤ C * ∑ b, ‖update b‖ ^ 3 := hTaylor update
  rw [hexpand, lossTaylor_mask_mean p hp.ne']
  have heq :
      maskExpectation p (fun mask => R (maskedSum p update mask)) +
        (lossTaylor loss x (∑ b, update b) + (1 - p) / (2 * p) *
          ∑ b, fderiv ℝ (fderiv ℝ loss) x (update b) (update b)) -
        loss (x - ∑ b, update b) - (1 - p) / (2 * p) *
          ∑ b, fderiv ℝ (fderiv ℝ loss) x (update b) (update b) =
        maskExpectation p (fun mask => R (maskedSum p update mask)) - R (∑ b, update b) := by
    dsimp [R]
    ring
  rw [heq]
  calc
    _ ≤ |maskExpectation p (fun mask => R (maskedSum p update mask))| +
        |R (∑ b, update b)| := by
      simpa only [sub_zero, zero_sub, abs_neg] using
        abs_sub_le (maskExpectation p (fun mask => R (maskedSum p update mask))) 0
          (R (∑ b, update b))
    _ ≤ C * (1 / p ^ 2) * ∑ b, ‖update b‖ ^ 3 + C * ∑ b, ‖update b‖ ^ 3 :=
      add_le_add hrem hbase
    _ = _ := by ring

/-- The added regularity is satisfiable by a nonconstant actual objective:
a linear loss has zero second derivative and zero Taylor remainder.
Source: arXiv:2602.15322v1, Section 2 and Appendix A.1, corrected domain. -/
example : (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    BlockCubicTaylor (ι := Unit) (id : ℝ → ℝ) 0 0 := by
  refine ⟨by norm_num, by norm_num, ?_⟩
  intro update
  have hid : fderiv ℝ (id : ℝ → ℝ) = fun _ => ContinuousLinearMap.id ℝ ℝ := by
    funext x
    exact fderiv_id
  simp [lossTaylor, hid]

end Transformer.Magma

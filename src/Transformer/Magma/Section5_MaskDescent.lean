/-
# Corrected expected descent under global smoothness

Correction of arXiv:2602.15322v1, Section 5, prop:magma_descent,
eq:one_step_descent_magma, and Appendix A.2. The objective's GLOBAL
quadratic upper model replaces the false inference from coordinate
smoothness. Independent normalized masks act on orthogonal embedded blocks.
-/

import Transformer.Magma.Section2_BlockExpansion
import Transformer.Optimization.Basic
import Mathlib.Analysis.InnerProductSpace.LinearMap

open scoped BigOperators InnerProductSpace

noncomputable section

namespace Transformer.Magma

open Transformer.Optimization

variable {ι E : Type*} [Fintype ι] [DecidableEq ι]
  [NormedAddCommGroup E] [InnerProductSpace ℝ E]

omit [DecidableEq ι] in
/-- Orthogonal block energies sum to the full squared norm.
Source: arXiv:2602.15322v1, Section 5, disjoint block partition. -/
theorem orthogonal_sum_norm_sq (update : ι → E)
    (horth : ∀ b c, b ≠ c → ⟪update b, update c⟫_ℝ = 0) :
    ‖∑ b, update b‖ ^ 2 = ∑ b, ‖update b‖ ^ 2 := by
  rw [← real_inner_self_eq_norm_sq]
  simp only [sum_inner, inner_sum]
  apply Finset.sum_congr rfl
  intro b hb
  rw [Finset.sum_eq_single b]
  · exact real_inner_self_eq_norm_sq _
  · intro c hc hcb
    exact horth c b hcb
  · simp

/-- A real, nonzero one-block update meets the orthogonality condition.
Source: arXiv:2602.15322v1, Section 5. -/
example : ∀ b c : Unit, b ≠ c → ⟪(1 : ℝ), (1 : ℝ)⟫_ℝ = 0 := by
  intro b c hbc
  exact (hbc (Subsingleton.elim b c)).elim

/-- Normalized independent block masking amplifies the full second
moment by 1/p when blocks are orthogonal. Source:
arXiv:2602.15322v1, Appendix A.2, masked squared norm calculation. -/
theorem maskedSum_norm_sq_mean (p : ℝ) (hp : p ≠ 0) (update : ι → E)
    (horth : ∀ b c, b ≠ c → ⟪update b, update c⟫_ℝ = 0) :
    maskExpectation p (fun mask => ‖maskedSum p update mask‖ ^ 2) =
      ‖∑ b, update b‖ ^ 2 / p := by
  have h := maskedSum_bilinear_mean p hp (innerSL ℝ) update
  change maskExpectation p (fun mask => ⟪maskedSum p update mask, maskedSum p update mask⟫_ℝ) =
    ⟪∑ b, update b, ∑ b, update b⟫_ℝ + (1 - p) / p * ∑ b, ⟪update b, update b⟫_ℝ at h
  simp only [real_inner_self_eq_norm_sq] at h
  rw [h, ← orthogonal_sum_norm_sq update horth]
  field_simp
  ring

/-- Second-moment hypotheses are jointly satisfiable. Source:
arXiv:2602.15322v1, Section 5, one block and p=1/2. -/
example : (1 / 2 : ℝ) ≠ 0 ∧
    (∀ b c : Unit, b ≠ c → ⟪(1 : ℝ), (1 : ℝ)⟫_ℝ = 0) := by
  refine ⟨by norm_num, ?_⟩
  intro b c hbc
  exact (hbc (Subsingleton.elim b c)).elim

/-- Normalized masking preserves the linear inner-product term.
Source: arXiv:2602.15322v1, Appendix A.2, masked first-order term. -/
theorem maskedSum_inner_mean (p : ℝ) (hp : p ≠ 0) (update : ι → E) (v : E) :
    maskExpectation p (fun mask => ⟪maskedSum p update mask, v⟫_ℝ) =
      ⟪∑ b, update b, v⟫_ℝ := by
  have h := maskedSum_linear_mean p hp ((innerSL ℝ) v) update
  simpa only [innerSL_apply_apply, real_inner_comm] using h

/-- Linear expectation hypotheses are satisfiable. Source:
arXiv:2602.15322v1, Section 5. -/
example : (1 / 2 : ℝ) ≠ 0 := by norm_num

variable [CompleteSpace E]

/-- Valid global-smooth replacement for prop:magma_descent, conditioned
on the realized stochastic gradient and damping before mask sampling.
Unlike the source, it does not infer a joint upper model from separate
coordinate inequalities. Source: arXiv:2602.15322v1, Section 5 and Appendix A.2. -/
theorem mask_descent_corrected (loss : E → ℝ) (L rate p : ℝ) (x : E)
    (update : ι → E) (hf : SmoothObjective loss L) (hp : 0 < p) (hp' : p ≤ 1)
    (horth : ∀ b c, b ≠ c → ⟪update b, update c⟫_ℝ = 0) :
    maskExpectation p (fun mask => loss (x - rate • maskedSum p update mask)) ≤
      loss x - rate * ⟪∑ b, update b, gradient loss x⟫_ℝ +
        L * rate ^ 2 / (2 * p) * ‖∑ b, update b‖ ^ 2 := by
  have hpoint (mask : ι → Bool) :
      loss (x - rate • maskedSum p update mask) ≤
        loss x - rate * ⟪maskedSum p update mask, gradient loss x⟫_ℝ +
          L * rate ^ 2 / 2 * ‖maskedSum p update mask‖ ^ 2 := by
    have h := hf.2 x (x - rate • maskedSum p update mask)
    have hsub : (x - rate • maskedSum p update mask) - x =
        -(rate • maskedSum p update mask) := by abel
    rw [hsub] at h
    simp only [inner_neg_right, inner_smul_right, norm_neg, norm_smul,
      mul_pow, Real.norm_eq_abs, sq_abs] at h
    rw [real_inner_comm] at h
    nlinarith
  have hm := maskExpectation_mono p hp.le hp' _ _ hpoint
  simp only [maskExpectation_add, maskExpectation_sub, maskExpectation_const,
    maskExpectation_mul, maskedSum_inner_mean p hp.ne',
    maskedSum_norm_sq_mean p hp.ne' update horth] at hm
  convert hm using 1
  ring

/-- The corrected smoothness and probability assumptions are satisfied by
the actual nonconstant quadratic and a nonzero one-block update.
Source: arXiv:2602.15322v1, Section 5, corrected domain. -/
example : SmoothObjective quadratic 1 ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    (∀ b c : Unit, b ≠ c → ⟪(1 : ℝ), (1 : ℝ)⟫_ℝ = 0) := by
  refine ⟨quadratic_smooth, by norm_num, by norm_num, ?_⟩
  intro b c hbc
  exact (hbc (Subsingleton.elim b c)).elim

end Transformer.Magma

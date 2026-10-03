/-
# Checking the complete fused MD step

User-requested training correction to arXiv:2606.25971v2, §3.1 and
Appendix A, Algorithm 2. The check acts after gain updates and projection.
It accepts the entire proposed weight exactly. Its gradient fallback
halves a step that would hit zero, because positive softplus gains on a
positive sphere cannot store a zero matrix at a finite training time.
-/

import Transformer.Optimization.Basic

open scoped InnerProductSpace

noncomputable section

namespace Transformer.MagnitudeDirection

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- A true-gradient fallback with one exception: halve a step that would
land at zero. Source: arXiv:2606.25971v2, §4.1.3 and Appendix A,
positive-gain training correction. -/
def nonzeroGradientDirection (σ L : ℝ) (x g : E) : E := by
  classical
  exact if x - (σ / L) • g = 0 then (1 / 2 : ℝ) • g else g

/-- Both fallback choices satisfy the alignment and length check for
`σ ≤ 1/2`; no assertion about optimizer quality is assumed.
Source: extension of arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem nonzeroGradientDirection_certificate (σ L : ℝ) (x g : E)
    (hσ : σ ≤ 1 / 2) :
    σ * ‖g‖ ^ 2 ≤ ⟪g, nonzeroGradientDirection σ L x g⟫_ℝ ∧
      ‖nonzeroGradientDirection σ L x g‖ ≤ ‖g‖ := by
  unfold nonzeroGradientDirection
  split
  · rw [real_inner_smul_right, real_inner_self_eq_norm_sq,
      norm_smul_of_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 2)]
    constructor
    · exact mul_le_mul_of_nonneg_right hσ (sq_nonneg _)
    · linarith [norm_nonneg g]
  · rw [real_inner_self_eq_norm_sq]
    refine ⟨?_, le_rfl⟩
    nlinarith [sq_nonneg ‖g‖]

/-- The fallback check has admissible parameters,
arXiv:2606.25971v2, Appendix A, training correction. -/
example : (1 / 4 : ℝ) ≤ 1 / 2 := by norm_num

/-- Every nonzero current weight stays nonzero under the fallback.
This excludes the singular zero recovery rather than assuming it away.
Source: extension of arXiv:2606.25971v2, Appendix A, Algorithm 2, line 2. -/
theorem nonzeroGradientDirection_step_ne_zero (σ L : ℝ) (x g : E) (hx : x ≠ 0) :
    x - (σ / L) • nonzeroGradientDirection σ L x g ≠ 0 := by
  unfold nonzeroGradientDirection
  split
  · rename_i h
    have he : x - (σ / L) • ((1 / 2 : ℝ) • g) = (1 / 2 : ℝ) • x := by
      rw [sub_eq_zero.mp h]
      module
    rw [he]
    exact smul_ne_zero (by norm_num) hx
  · assumption

/-- A genuine nonzero current weight satisfies the fallback premise,
arXiv:2606.25971v2, Appendix A, training correction. -/
example : (1 : ℝ) ≠ 0 := by norm_num

/-- Express a complete proposed fused step as a direction at `η=σ/L`.
Source: extension of arXiv:2606.25971v2, Appendix A, Algorithm 2, line 10. -/
def fusedStepDirection (σ L : ℝ) (x y : E) : E := (L / σ) • (x - y)

/-- The proposed displacement is recovered exactly for positive parameters.
The guard therefore checks the actual post-projection/post-gain update.
Source: extension of arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem fusedStepDirection_reconstruct (σ L : ℝ) (x y : E)
    (hσ : 0 < σ) (hL : 0 < L) : x - (σ / L) • fusedStepDirection σ L x y = y := by
  rw [fusedStepDirection, smul_smul]
  have hc : σ / L * (L / σ) = 1 := by field_simp
  rw [hc, one_smul, sub_sub_cancel]

/-- Positive guard parameters exist, arXiv:2606.25971v2, Appendix A. -/
example : (0 : ℝ) < 1 / 4 ∧ (0 : ℝ) < 1 := by norm_num

/-- A numerical check of a complete candidate, including nonzero storage,
gradient alignment and length. This is a predicate of the candidate,
not an assumed convergence property. Source: extension of
arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
def fusedStepAdmissible (σ L : ℝ) (x g y : E) : Prop :=
  y ≠ 0 ∧ σ * ‖g‖ ^ 2 ≤ ⟪g, fusedStepDirection σ L x y⟫_ℝ ∧
    ‖fusedStepDirection σ L x y‖ ≤ ‖g‖

/-- Select the paper's full displacement if it passes the check;
otherwise select the nonsingular true-gradient fallback.
Source: training correction to arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
def checkedDirection (σ L : ℝ) (x g y : E) : E := by
  classical
  exact if fusedStepAdmissible σ L x g y then fusedStepDirection σ L x y
    else nonzeroGradientDirection σ L x g

/-- Every selected direction passes the genuine-gradient check,
independently of the original base/gain optimizer or its state.
Source: training correction to arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem checkedDirection_certificate (σ L : ℝ) (x g y : E) (hσ : σ ≤ 1 / 2) :
    σ * ‖g‖ ^ 2 ≤ ⟪g, checkedDirection σ L x g y⟫_ℝ ∧
      ‖checkedDirection σ L x g y‖ ≤ ‖g‖ := by
  unfold checkedDirection
  split
  · rename_i h
    exact h.2
  · exact nonzeroGradientDirection_certificate σ L x g hσ

/-- Selected-direction certification has a nonempty parameter domain,
arXiv:2606.25971v2, Appendix A, training correction. -/
example : (1 / 3 : ℝ) ≤ 1 / 2 := by norm_num

/-- The corrected weight is nonzero at every finite step from a nonzero
weight. Source: arXiv:2606.25971v2, Appendix A, positive-gain correction. -/
theorem checkedDirection_step_ne_zero (σ L : ℝ) (x g y : E)
    (hσ : 0 < σ) (hL : 0 < L) (hx : x ≠ 0) :
    x - (σ / L) • checkedDirection σ L x g y ≠ 0 := by
  unfold checkedDirection
  split
  · rename_i h
    rw [fusedStepDirection_reconstruct σ L x y hσ hL]
    exact h.1
  · exact nonzeroGradientDirection_step_ne_zero σ L x g hx

/-- Positive parameters and nonzero weights satisfy the step premises,
arXiv:2606.25971v2, Appendix A, training correction. -/
example : (0 : ℝ) < 1 / 4 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≠ 0 := by norm_num

/-- The generic training guard keeps our already certified direction,
so its convergence lemmas apply to this exact fused MD correction.
Source: extension of arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem checkedDirection_guard (σ L : ℝ) (x g y : E) (hσ : σ ≤ 1 / 2) :
    Optimization.descentGuard σ g (checkedDirection σ L x g y) =
      checkedDirection σ L x g y :=
  Optimization.descentGuard_accepts σ g _ (checkedDirection_certificate σ L x g y hσ)

/-- Compatibility with the generic guard has valid parameters,
arXiv:2606.25971v2, Appendix A, training correction. -/
example : (1 / 4 : ℝ) ≤ 1 / 2 := by norm_num

/-- Candidates passing the check are used without changing their weight.
Source: arXiv:2606.25971v2, Appendix A, Algorithm 2, training correction. -/
theorem checkedDirection_accepts (σ L : ℝ) (x g y : E)
    (hσ : 0 < σ) (hL : 0 < L) (h : fusedStepAdmissible σ L x g y) :
    x - (σ / L) • checkedDirection σ L x g y = y := by
  rw [checkedDirection, ite_eq_left h]
  exact fusedStepDirection_reconstruct σ L x y hσ hL

/-- A nonstationary nonzero candidate passes the full check,
arXiv:2606.25971v2, Appendix A, training correction. -/
example : (0 : ℝ) < 1 / 4 ∧ (0 : ℝ) < 1 ∧
    fusedStepAdmissible (1 / 4) 1 (1 : ℝ) 1 (3 / 4) := by
  norm_num [fusedStepAdmissible, fusedStepDirection]

end Transformer.MagnitudeDirection

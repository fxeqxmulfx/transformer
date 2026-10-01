/-
# The analytic gradient inequality in dimensions zero and one

Appendix D.1, `lem: loj`, of arXiv:2510.22026v2: transfer the scalar
finite-order argument to Euclidean space and handle locally constant
energies in any dimension.
-/

import Transformer.Basic
import Transformer.Normalization.ScalarGradientInequality
import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.Analysis.Calculus.FDeriv.Equiv
import Mathlib.Analysis.Calculus.Deriv.Linear
import Mathlib.Analysis.Analytic.Linear

open Filter Set

namespace Transformer.Normalization

/-- A locally constant energy satisfies the local gradient inequality
of Appendix D.1, `lem: loj`, in arXiv:2510.22026v2 with exponent `1/2`.
This case needs no analytic or nondegeneracy assumption. -/
theorem local_gradient_inequality_of_eventually_constant {N : ℕ}
    (E : EucSpace N → ℝ) (z : EucSpace N)
    (hconstant : ∀ᶠ y in nhds z, E y = E z) :
    ∃ alpha k : ℝ, ∃ V : Set (EucSpace N),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ z ∈ V ∧
        ∀ y ∈ V, |E y - E z| ^ alpha ≤ k * ‖gradient E y‖ := by
  obtain ⟨V, hV, hVo, hzV⟩ := eventually_nhds_iff.1 hconstant
  refine ⟨1 / 2, 1, V, by norm_num, by norm_num, zero_lt_one, hVo, hzV, ?_⟩
  intro y hy
  rw [hV y hy]
  simp

/-- Constant energy witnesses the local-constancy hypothesis for the
gradient inequality in Appendix D.1 of arXiv:2510.22026v2. -/
example : ∀ᶠ y in nhds (0 : EucSpace 2),
    (fun _ : EucSpace 2 => (7 : ℝ)) y = 7 := Eventually.of_forall (fun _ => rfl)

/-- The scalar analytic gradient inequality from Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2 transfers through a continuous linear
coordinate equivalence with `ℝ`. The chain rule and Cauchy--Schwarz
bound the scalar derivative by the full gradient norm, with the
coordinate vector's norm absorbed in the positive constant. -/
theorem analytic_gradient_inequality_via_real_equiv
    {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H] [CompleteSpace H]
    (e : H ≃L[ℝ] ℝ) (E : H → ℝ) (z : H)
    (hE : AnalyticAt ℝ E z) (hcritical : gradient E z = 0) :
    ∃ alpha k : ℝ, ∃ V : Set H,
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ z ∈ V ∧
        ∀ y ∈ V, |E y - E z| ^ alpha ≤ k * ‖gradient E y‖ := by
  have hd : ∀ y : H, AnalyticAt ℝ E y →
      deriv (E ∘ e.symm) (e y) = inner (𝕜 := ℝ) (gradient E y) (e.symm 1) := by
    intro y hy
    have hEy : HasFDerivAt E (InnerProductSpace.toDual ℝ H (gradient E y))
        (e.symm (e y)) := by
      simpa using hy.differentiableAt.hasGradientAt.hasFDerivAt
    have hcomp := hEy.comp_hasDerivAt (e y) (e.symm.toContinuousLinearMap.hasDerivAt)
    simpa [InnerProductSpace.toDual_apply_apply] using hcomp.deriv
  have hf : AnalyticAt ℝ (E ∘ e.symm) (e z) := by
    apply AnalyticAt.comp
    · simpa using hE
    · exact e.symm.toContinuousLinearMap.analyticAt _
  have hc : deriv (E ∘ e.symm) (e z) = 0 := by
    rw [hd z hE, hcritical, inner_zero_left]
  obtain ⟨alpha, k, V, ha, ha1, hk, hV, hzV, hineq⟩ :=
    scalar_analytic_gradient_inequality (E ∘ e.symm) (e z) hf hc
  obtain ⟨r, hr, hball⟩ := hE.exists_ball_analyticOnNhd
  have hunit : 0 < ‖e.symm (1 : ℝ)‖ := by
    apply norm_pos_iff.2
    simp
  refine ⟨alpha, k * ‖e.symm (1 : ℝ)‖, e ⁻¹' V ∩ Metric.ball z r,
    ha, ha1, mul_pos hk hunit, (e.continuous.isOpen_preimage V hV).inter
      Metric.isOpen_ball, ⟨hzV, Metric.mem_ball_self hr⟩, ?_⟩
  intro y hy
  have hi := hineq (e y) hy.1
  simp only [Function.comp_apply, e.symm_apply_apply] at hi
  rw [hd y (hball y hy.2)] at hi
  calc
    |E y - E z| ^ alpha ≤ k * |inner (𝕜 := ℝ) (gradient E y) (e.symm 1)| := hi
    _ ≤ k * (‖gradient E y‖ * ‖e.symm 1‖) :=
      mul_le_mul_of_nonneg_left (abs_real_inner_le_norm _ _) hk.le
    _ = k * ‖e.symm 1‖ * ‖gradient E y‖ := by ring

/-- The local analytic gradient inequality in Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2 is proved in `EucSpace 1`, including
critical points where the Hessian vanishes. -/
theorem analytic_gradient_inequality_one_dimension (E : EucSpace 1 → ℝ)
    (z : EucSpace 1) (hE : AnalyticAt ℝ E z) (hcritical : gradient E z = 0) :
    ∃ alpha k : ℝ, ∃ V : Set (EucSpace 1),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ z ∈ V ∧
        ∀ y ∈ V, |E y - E z| ^ alpha ≤ k * ‖gradient E y‖ := by
  exact analytic_gradient_inequality_via_real_equiv
    (PiLp.equivOfUnique 2 ℝ (fun _ : Fin 1 => ℝ)) E z hE hcritical

/-- The local analytic input in Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2 is proved in all dimensions at most one. Dimension
zero has a singleton state space and hence a constant energy. -/
theorem analytic_gradient_inequality_low_dimension {N : ℕ} (hN : N ≤ 1)
    (E : EucSpace N → ℝ) (z : EucSpace N)
    (hE : AnalyticAt ℝ E z) (hcritical : gradient E z = 0) :
    ∃ alpha k : ℝ, ∃ V : Set (EucSpace N),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ z ∈ V ∧
        ∀ y ∈ V, |E y - E z| ^ alpha ≤ k * ‖gradient E y‖ := by
  by_cases hzero : N = 0
  · subst N
    apply local_gradient_inequality_of_eventually_constant E z
    exact Eventually.of_forall (fun y => congrArg E (Subsingleton.elim y z))
  · have hone : N = 1 := by omega
    subst N
    exact analytic_gradient_inequality_one_dimension E z hE hcritical

/-- The coordinate equivalence and the analytic critical-point
hypotheses above are jointly satisfiable for constant energy on
`EucSpace 1`; Appendix D.1 of arXiv:2510.22026v2. -/
example : ∃ e : EucSpace 1 ≃L[ℝ] ℝ,
    e (0 : EucSpace 1) = 0 ∧ (1 : ℕ) ≤ 1 ∧
      AnalyticAt ℝ (fun _ : EucSpace 1 => (0 : ℝ)) 0 ∧
      gradient (fun _ : EucSpace 1 => (0 : ℝ)) 0 = 0 := by
  exact ⟨PiLp.equivOfUnique 2 ℝ (fun _ : Fin 1 => ℝ), map_zero _, le_rfl,
    analyticAt_const, (hasGradientAt_const (0 : EucSpace 1) (0 : ℝ)).gradient⟩

end Transformer.Normalization

/-
# Transporting a local gradient inequality through linear coordinates

An invertible linear coordinate change preserves the analytic gradient
estimate required by Appendix D.1 of arXiv:2510.22026v2.
-/

import Transformer.Normalization.MonomialGradientInequality

open Filter Set
open scoped BigOperators

namespace Transformer.Normalization

/-- An invertible linear coordinate change preserves a proved local
gradient inequality, multiplying its constant by an operator-norm bound.
Auxiliary transport result for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem local_gradient_inequality_of_linear_equiv {N : ℕ}
    (E : EucSpace N → ℝ) (z : EucSpace N) (hE : AnalyticAt ℝ E z)
    (e : EucSpace N ≃L[ℝ] EucSpace N)
    (hlocal : ∃ alpha k : ℝ, ∃ V : Set (EucSpace N),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ e.symm z ∈ V ∧
        ∀ y ∈ V, |E (e y) - E z| ^ alpha ≤ k * ‖gradient (E ∘ e) y‖) :
    ∃ alpha k : ℝ, ∃ V : Set (EucSpace N),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ z ∈ V ∧
        ∀ y ∈ V, |E y - E z| ^ alpha ≤ k * ‖gradient E y‖ := by
  obtain ⟨alpha, k, V, ha, ha1, hk, hV, hwV, hineq⟩ := hlocal
  let B : ℝ := ‖e.toContinuousLinearMap‖ + 1
  have hB : 0 < B := by dsimp [B]; positivity
  have hpre : ∀ᶠ y in nhds z, e.symm y ∈ V :=
    e.symm.continuous.continuousAt.eventually (hV.mem_nhds hwV)
  have hnorm (f : EucSpace N → ℝ) (y : EucSpace N) :
      ‖gradient f y‖ = ‖fderiv ℝ f y‖ :=
    (InnerProductSpace.toDual ℝ (EucSpace N)).symm.norm_map _
  have hevent : ∀ᶠ y in nhds z,
      |E y - E z| ^ alpha ≤ (k * B) * ‖gradient E y‖ := by
    filter_upwards [hE.eventually_analyticAt, hpre] with y hyE hyV
    have hEy : HasFDerivAt E (fderiv ℝ E y) (e (e.symm y)) := by
      simpa using hyE.differentiableAt.hasFDerivAt
    have hcomp := hEy.comp (e.symm y) e.hasFDerivAt
    have hgrad : ‖gradient (E ∘ e) (e.symm y)‖ ≤
        ‖gradient E y‖ * ‖e.toContinuousLinearMap‖ := by
      rw [hnorm, hcomp.fderiv, hnorm]
      exact ContinuousLinearMap.opNorm_comp_le _ _
    have heB : ‖e.toContinuousLinearMap‖ ≤ B := by dsimp [B]; linarith
    calc
      |E y - E z| ^ alpha = |E (e (e.symm y)) - E z| ^ alpha := by simp
      _ ≤ k * ‖gradient (E ∘ e) (e.symm y)‖ := hineq _ hyV
      _ ≤ k * (‖gradient E y‖ * ‖e.toContinuousLinearMap‖) :=
        mul_le_mul_of_nonneg_left hgrad hk.le
      _ ≤ k * (‖gradient E y‖ * B) :=
        mul_le_mul_of_nonneg_left
          (mul_le_mul_of_nonneg_left heB (norm_nonneg _)) hk.le
      _ = (k * B) * ‖gradient E y‖ := by ring
  obtain ⟨W, hW, hWo, hzW⟩ := eventually_nhds_iff.mp hevent
  exact ⟨alpha, k * B, W, ha, ha1, mul_pos hk hB, hWo, hzW, hW⟩

/-- The analytic gradient inequality holds when an invertible linear
coordinate change expresses the energy gap as a coordinate monomial times
a nonvanishing analytic unit. This proves a prepared case of the analytic
input in Appendix D.1, `lem: loj`, of arXiv:2510.22026v2; existence of such
coordinates for arbitrary analytic germs is not assumed or asserted. -/
theorem analytic_gradient_inequality_of_monomial_coordinates {N : ℕ}
    (E : EucSpace N → ℝ) (z : EucSpace N) (hE : AnalyticAt ℝ E z)
    (e : EucSpace N ≃L[ℝ] EucSpace N) (p : Fin N → ℕ) (hp : 0 < ∑ j, p j)
    (u : EucSpace N → ℝ) (hu : AnalyticAt ℝ u (e.symm z)) (hu0 : u (e.symm z) ≠ 0)
    (heq : ∀ᶠ y in nhds (e.symm z),
      E (e y) - E z = u y * coordinateMonomial p (y - e.symm z)) :
    ∃ alpha k : ℝ, ∃ V : Set (EucSpace N),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ z ∈ V ∧
        ∀ y ∈ V, |E y - E z| ^ alpha ≤ k * ‖gradient E y‖ := by
  apply local_gradient_inequality_of_linear_equiv E z hE e
  have hEw : AnalyticAt ℝ E (e (e.symm z)) := by simpa using hE
  have hF : AnalyticAt ℝ (E ∘ e) (e.symm z) := hEw.comp (e.analyticAt (e.symm z))
  simpa only [Function.comp_def, ContinuousLinearEquiv.apply_symm_apply] using
    analytic_gradient_inequality_of_monomial_unit p hp (E ∘ e) u (e.symm z)
      hF hu hu0 (by simpa only [Function.comp_def,
        ContinuousLinearEquiv.apply_symm_apply] using heq)

/-- Squared coordinate products satisfy the prepared-coordinate hypotheses
with the identity equivalence and unit one; Appendix D.1 of
arXiv:2510.22026v2. The application also witnesses all hypotheses of the
transport result above. -/
example : ∃ alpha k : ℝ, ∃ V : Set (EucSpace 2),
    0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ (0 : EucSpace 2) ∈ V ∧
      ∀ y ∈ V, |coordinateMonomial (fun _ => 2) y| ^ alpha ≤
        k * ‖gradient (coordinateMonomial (fun _ : Fin 2 => 2)) y‖ := by
  simpa only [coordinateMonomial, Fin.prod_univ_two, PiLp.zero_apply,
    zero_pow (by norm_num : (2 : ℕ) ≠ 0), mul_zero, sub_zero] using
    analytic_gradient_inequality_of_monomial_coordinates
      (coordinateMonomial (fun _ : Fin 2 => 2)) 0
      (coordinateMonomial_analytic _ _) (ContinuousLinearEquiv.refl ℝ (EucSpace 2))
      (fun _ => 2) (by decide) (fun _ => 1) analyticAt_const (by norm_num)
      (Filter.Eventually.of_forall (fun y => by
        simp [coordinateMonomial, Fin.prod_univ_two]))

end Transformer.Normalization

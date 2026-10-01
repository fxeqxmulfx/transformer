/-
# The gradient inequality at a nondegenerate critical point

Appendix D.1, `lem: loj`, of arXiv:2510.22026v2: an invertible
Hessian gives the local analytic gradient inequality with exponent
`1/2` in every finite dimension.
-/

import Transformer.Normalization.QuadraticEnergy
import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Analytic.ChangeOrigin
import Mathlib.Analysis.Calculus.FDeriv.Analytic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

open Filter Set

namespace Transformer.Normalization

/-- The local gradient inequality used in Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2 holds with exponent `1/2` at a nondegenerate
critical point in every dimension. Nondegeneracy means that the derivative
of the gradient is a continuous linear equivalence. Its linear remainder
bounds distance to the critical point by the gradient norm; a mean value
estimate bounds the energy gap by the square of that distance. -/
theorem analytic_gradient_inequality_of_nondegenerate {N : ℕ}
    (E : EucSpace N → ℝ) (z : EucSpace N) (hE : AnalyticAt ℝ E z)
    (A : EucSpace N ≃L[ℝ] EucSpace N)
    (hA : HasFDerivAt (gradient E) A.toContinuousLinearMap z)
    (hcritical : gradient E z = 0) :
    ∃ k : ℝ, ∃ V : Set (EucSpace N),
      0 < k ∧ IsOpen V ∧ z ∈ V ∧
        ∀ y ∈ V, |E y - E z| ^ (1 / 2 : ℝ) ≤ k * ‖gradient E y‖ := by
  let C : ℝ := max ‖A.symm.toContinuousLinearMap‖ 1
  let B : ℝ := ‖A.toContinuousLinearMap‖ + 1
  let eps : ℝ := 1 / (2 * C)
  have hC : 0 < C := lt_of_lt_of_le zero_lt_one (le_max_right _ _)
  have hB : 0 < B := by dsimp [B]; positivity
  have heps : 0 < eps := by dsimp [eps]; positivity
  have heps1 : eps ≤ 1 := by
    dsimp [eps]
    apply (div_le_one (by positivity)).2
    have := le_max_right ‖A.symm.toContinuousLinearMap‖ (1 : ℝ)
    change 1 ≤ C at this
    linarith
  have hprod : C * eps = 1 / 2 := by
    dsimp [eps]
    field_simp
  obtain ⟨r, hr, herror⟩ := Metric.eventually_nhds_iff.1 (hA.isLittleO.def heps)
  have hcontrol {y : EucSpace N} (hy : dist y z < r) :
      ‖gradient E y‖ ≤ B * ‖y - z‖ ∧
      ‖y - z‖ ≤ 2 * C * ‖gradient E y‖ := by
    have herr := herror hy
    simp only [hcritical, sub_zero, ContinuousLinearEquiv.coe_coe] at herr
    have hlinear := A.toContinuousLinearMap.le_opNorm (y - z)
    have hupper : ‖gradient E y‖ ≤ B * ‖y - z‖ := by
      calc
        ‖gradient E y‖ = ‖(gradient E y - A (y - z)) + A (y - z)‖ := by
          congr 1
          abel
        _ ≤ ‖gradient E y - A (y - z)‖ + ‖A (y - z)‖ := norm_add_le _ _
        _ ≤ eps * ‖y - z‖ + ‖A.toContinuousLinearMap‖ * ‖y - z‖ :=
          add_le_add herr hlinear
        _ ≤ 1 * ‖y - z‖ + ‖A.toContinuousLinearMap‖ * ‖y - z‖ :=
          add_le_add (mul_le_mul_of_nonneg_right heps1 (norm_nonneg _)) le_rfl
        _ = B * ‖y - z‖ := by dsimp [B]; ring
    have hforward : ‖A (y - z)‖ ≤ ‖gradient E y‖ + eps * ‖y - z‖ := by
      calc
        ‖A (y - z)‖ = ‖gradient E y - (gradient E y - A (y - z))‖ := by
          congr 1
          abel
        _ ≤ ‖gradient E y‖ + ‖gradient E y - A (y - z)‖ := norm_sub_le _ _
        _ ≤ ‖gradient E y‖ + eps * ‖y - z‖ := add_le_add le_rfl herr
    have hreverse : ‖y - z‖ ≤ C * ‖A (y - z)‖ := by
      have hv := A.symm.toContinuousLinearMap.le_opNorm (A (y - z))
      simp only [ContinuousLinearEquiv.coe_coe, A.symm_apply_apply] at hv
      exact hv.trans (mul_le_mul_of_nonneg_right (le_max_left _ _) (norm_nonneg _))
    have hboth := hreverse.trans (mul_le_mul_of_nonneg_left hforward hC.le)
    have heq : C * (‖gradient E y‖ + eps * ‖y - z‖) =
        C * ‖gradient E y‖ + ‖y - z‖ / 2 := by
      rw [mul_add, ← mul_assoc, hprod]
      ring
    rw [heq] at hboth
    exact ⟨hupper, by linarith only [hboth]⟩
  obtain ⟨s, hs, hball⟩ := hE.exists_ball_analyticOnNhd
  let R := min r s
  have hR : 0 < R := lt_min hr hs
  refine ⟨Real.sqrt B * (2 * C), Metric.ball z R,
    mul_pos (Real.sqrt_pos.2 hB) (mul_pos (by norm_num) hC),
    Metric.isOpen_ball, Metric.mem_ball_self hR, ?_⟩
  intro y hy
  have hynorm : ‖y - z‖ < R := by simpa [dist_eq_norm] using hy
  have hnear {w : EucSpace N} (hw : w ∈ Metric.closedBall z ‖y - z‖) :
      dist w z < R := hw.trans_lt hynorm
  have hdiff : ∀ w ∈ Metric.closedBall z ‖y - z‖, DifferentiableAt ℝ E w := by
    intro w hw
    exact (hball w ((hnear hw).trans_le (min_le_right r s))).differentiableAt
  have hbound : ∀ w ∈ Metric.closedBall z ‖y - z‖,
      ‖fderiv ℝ E w‖ ≤ B * ‖y - z‖ := by
    intro w hw
    rw [(hdiff w hw).hasGradientAt.hasFDerivAt.fderiv,
      (InnerProductSpace.toDual ℝ (EucSpace N)).norm_map]
    exact (hcontrol ((hnear hw).trans_le (min_le_left r s))).1.trans
      (mul_le_mul_of_nonneg_left (by simpa [dist_eq_norm] using hw) hB.le)
  have hgap := Convex.norm_image_sub_le_of_norm_fderiv_le hdiff hbound
    (convex_closedBall z ‖y - z‖) (Metric.mem_closedBall_self (norm_nonneg _))
    (by simp [dist_eq_norm] : y ∈ Metric.closedBall z ‖y - z‖)
  have hgap' : |E y - E z| ≤ B * ‖y - z‖ ^ 2 := by
    simpa [Real.norm_eq_abs, pow_two, mul_assoc] using hgap
  calc
    |E y - E z| ^ (1 / 2 : ℝ) = Real.sqrt |E y - E z| :=
      (Real.sqrt_eq_rpow _).symm
    _ ≤ Real.sqrt (B * ‖y - z‖ ^ 2) := Real.sqrt_le_sqrt hgap'
    _ = Real.sqrt B * ‖y - z‖ := by
      rw [Real.sqrt_mul hB.le, Real.sqrt_sq (norm_nonneg _)]
    _ ≤ Real.sqrt B * (2 * C * ‖gradient E y‖) :=
      mul_le_mul_of_nonneg_left
        (hcontrol (hy.trans_le (min_le_left r s))).2 (Real.sqrt_nonneg _)
    _ = Real.sqrt B * (2 * C) * ‖gradient E y‖ := by ring

/-- Quadratic energy has an invertible gradient derivative `2 I`
everywhere. This witnesses the nondegenerate analytic case of Appendix
D.1, `lem: loj`, in arXiv:2510.22026v2. -/
theorem quadratic_energy_gradient_nondegenerate {N : ℕ} (z : EucSpace N) :
    ∃ A : EucSpace N ≃L[ℝ] EucSpace N,
      HasFDerivAt (gradient (fun y : EucSpace N => ‖y‖ ^ 2)) A.toContinuousLinearMap z := by
  let A : EucSpace N ≃L[ℝ] EucSpace N :=
    (Units.mk0 (2 : ℝ) (by norm_num)) • ContinuousLinearEquiv.refl ℝ (EucSpace N)
  refine ⟨A, ?_⟩
  have hA : A.toContinuousLinearMap = (2 : ℝ) • ContinuousLinearMap.id ℝ (EucSpace N) := by
    ext y
    simp [A]
  have hg : gradient (fun y : EucSpace N => ‖y‖ ^ 2) = fun y => (2 : ℝ) • y :=
    funext quadratic_energy_gradient
  rw [hg, hA]
  exact (hasFDerivAt_id z).fun_const_smul (2 : ℝ)

/-- The hypotheses of the nondegenerate gradient inequality are
jointly satisfiable at the origin for quadratic energy on `ℝ²`;
Appendix D.1 of arXiv:2510.22026v2. -/
example : AnalyticAt ℝ (fun y : EucSpace 2 => ‖y‖ ^ 2) 0 ∧
    gradient (fun y : EucSpace 2 => ‖y‖ ^ 2) 0 = 0 ∧
      ∃ A : EucSpace 2 ≃L[ℝ] EucSpace 2,
        HasFDerivAt (gradient (fun y : EucSpace 2 => ‖y‖ ^ 2)) A.toContinuousLinearMap 0 := by
  exact ⟨quadratic_energy_analytic 0 (Set.mem_univ _), by
    rw [quadratic_energy_gradient]; simp, quadratic_energy_gradient_nondegenerate 0⟩

end Transformer.Normalization

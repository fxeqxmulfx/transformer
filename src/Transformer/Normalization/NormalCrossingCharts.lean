/-
# The gradient inequality in analytic normal-crossing coordinates

The monomial-times-unit estimate transfers through an analytic chart with
a continuous local inverse. These charts cover nonlinear as well as linear
prepared germs in the analytic step of Appendix D.1 of arXiv:2510.22026v2.
-/

import Transformer.Normalization.GradientInequalityCoordinates
import Transformer.Normalization.GradientPullback
import Mathlib.Analysis.Calculus.LocalExtr.Basic

open Filter Set
open scoped BigOperators

namespace Transformer.Normalization

/-- A monomial-times-unit expression for a pulled-back energy gives a local
bound by the original gradient at image points. No inverse or injectivity
is required. Auxiliary normal-crossing estimate for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem analytic_gradient_inequality_on_monomial_image {M N : ℕ}
    (E : EucSpace N → ℝ) (phi : EucSpace M → EucSpace N) (w : EucSpace M)
    (hE : AnalyticAt ℝ E (phi w)) (hphi : AnalyticAt ℝ phi w)
    (p : Fin M → ℕ) (hp : 0 < ∑ j, p j) (u : EucSpace M → ℝ)
    (hu : AnalyticAt ℝ u w) (hu0 : u w ≠ 0)
    (heq : ∀ᶠ y in nhds w,
      E (phi y) - E (phi w) = u y * coordinateMonomial p (y - w)) :
    ∃ alpha k : ℝ, ∃ V : Set (EucSpace M),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ w ∈ V ∧
        ∀ y ∈ V, |E (phi y) - E (phi w)| ^ alpha ≤ k * ‖gradient E (phi y)‖ := by
  apply local_gradient_inequality_on_analytic_image E phi w hE hphi
  simpa only [Function.comp_def] using
    analytic_gradient_inequality_of_monomial_unit p hp (E ∘ phi) u w
      (hE.comp hphi) hu hu0 (by simpa only [Function.comp_def] using heq)

/-- A monomial-times-unit expression in analytic coordinates gives the
local gradient inequality for the original energy. The coordinate map only
needs a continuous local right inverse. This proves the corresponding
normal-crossing case of the analytic input in Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2; it does not assert existence of these charts for
arbitrary analytic germs. -/
theorem analytic_gradient_inequality_of_normal_crossing_chart {M N : ℕ}
    (E : EucSpace N → ℝ) (z : EucSpace N) (hE : AnalyticAt ℝ E z)
    (phi : EucSpace M → EucSpace N) (psi : EucSpace N → EucSpace M)
    (w : EucSpace M) (hphi : AnalyticAt ℝ phi w) (hphi0 : phi w = z)
    (hpsi : ContinuousAt psi z) (hpsi0 : psi z = w)
    (hinv : ∀ᶠ y in nhds z, phi (psi y) = y)
    (p : Fin M → ℕ) (hp : 0 < ∑ j, p j) (u : EucSpace M → ℝ)
    (hu : AnalyticAt ℝ u w) (hu0 : u w ≠ 0)
    (heq : ∀ᶠ y in nhds w,
      E (phi y) - E z = u y * coordinateMonomial p (y - w)) :
    ∃ alpha k : ℝ, ∃ V : Set (EucSpace N),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ z ∈ V ∧
        ∀ y ∈ V, |E y - E z| ^ alpha ≤ k * ‖gradient E y‖ := by
  apply local_gradient_inequality_of_analytic_parametrization E z hE phi psi w
    hphi hphi0 hpsi hpsi0 hinv
  have hEw : AnalyticAt ℝ E (phi w) := by rw [hphi0]; exact hE
  have hF : AnalyticAt ℝ (E ∘ phi) w := hEw.comp hphi
  simpa only [Function.comp_def, hphi0] using
    analytic_gradient_inequality_of_monomial_unit p hp (E ∘ phi) u w hF hu hu0
      (by simpa only [Function.comp_def, hphi0] using heq)

/-- The nonlinear shear `(x,y) ↦ (x,y+x²)` and its explicit inverse
prepare `x²(y-x²)²` as `x²y²`. All chart hypotheses hold at the critical
point zero, giving a nonlinear prepared instance of the analytic estimate
from Appendix D.1 of arXiv:2510.22026v2. -/
example :
    gradient (fun y : EucSpace 2 => (y 0) ^ 2 * ((y 1) - (y 0) ^ 2) ^ 2) 0 = 0 ∧
    ∃ alpha k : ℝ, ∃ V : Set (EucSpace 2),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ (0 : EucSpace 2) ∈ V ∧
        ∀ y ∈ V, |(y 0) ^ 2 * ((y 1) - (y 0) ^ 2) ^ 2| ^ alpha ≤ k *
          ‖gradient (fun t : EucSpace 2 =>
            (t 0) ^ 2 * ((t 1) - (t 0) ^ 2) ^ 2) y‖ := by
  let v : EucSpace 2 := PiLp.single 2 (1 : Fin 2) 1
  let phi : EucSpace 2 → EucSpace 2 := fun y => y + (y 0) ^ 2 • v
  let psi : EucSpace 2 → EucSpace 2 := fun y => y - (y 0) ^ 2 • v
  let E : EucSpace 2 → ℝ := fun y => (y 0) ^ 2 * ((y 1) - (y 0) ^ 2) ^ 2
  have hx : AnalyticAt ℝ (fun y : EucSpace 2 => y 0) 0 :=
    (EuclideanSpace.proj 0 : EucSpace 2 →L[ℝ] ℝ).analyticAt 0
  have hy : AnalyticAt ℝ (fun y : EucSpace 2 => y 1) 0 :=
    (EuclideanSpace.proj 1 : EucSpace 2 →L[ℝ] ℝ).analyticAt 0
  have hE : AnalyticAt ℝ E 0 :=
    (hx.fun_pow 2).mul ((hy.sub (hx.fun_pow 2)).fun_pow 2)
  have hphi : AnalyticAt ℝ phi 0 :=
    analyticAt_id.add ((hx.fun_pow 2).fun_smul analyticAt_const)
  have hpsi : ContinuousAt psi 0 :=
    (analyticAt_id.sub ((hx.fun_pow 2).fun_smul analyticAt_const)).continuousAt
  have hinv (y : EucSpace 2) : phi (psi y) = y := by
    ext i
    fin_cases i <;> simp [phi, psi, v]
  have hfactor (y : EucSpace 2) :
      E (phi y) - E 0 = (1 : ℝ) * coordinateMonomial (fun _ => 2) (y - 0) := by
    simp [E, phi, v, coordinateMonomial, Fin.prod_univ_two]
  have hmin : IsLocalMin E 0 := Filter.Eventually.of_forall (fun y => by
    dsimp [E]
    simp only [zero_pow (by norm_num : (2 : ℕ) ≠ 0),
      sub_zero, mul_zero]
    positivity)
  constructor
  · change gradient E 0 = 0
    rw [gradient, hmin.fderiv_eq_zero, map_zero]
  · simpa only [E, PiLp.zero_apply, zero_pow (by norm_num : (2 : ℕ) ≠ 0),
      sub_zero, mul_zero] using
      analytic_gradient_inequality_of_normal_crossing_chart E 0 hE phi psi 0
        hphi (by simp [phi]) hpsi (by simp [psi])
        (Filter.Eventually.of_forall hinv) (fun _ : Fin 2 => 2) (by decide)
        (fun _ => 1) analyticAt_const (by norm_num)
        (Filter.Eventually.of_forall hfactor)

end Transformer.Normalization

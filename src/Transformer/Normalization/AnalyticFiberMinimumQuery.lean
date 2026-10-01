/-
# Analytic fiber minima as arithmetic Sturm--Tarski tests

Analytic preparation and division retain every scalar competitor on a
fixed box; signed real-root queries eliminate that competitor variable.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Normalization.AnalyticFiberMinimaReduction
import Transformer.Normalization.PolynomialFiberArithmetic

noncomputable section
open scoped BigOperators

namespace Transformer.Normalization

open AnalyticPreparation

/-- Every constrained minimum on a regular analytic scalar fiber has
an exact arithmetic test on one fixed neighborhood box. The remaining
base coordinates are arbitrary, and no analytic base curve is assumed.
The test keeps the original equation and finite signs and eliminates
comparison with all admissible roots using a finite Sturm--Tarski
expression. Whole-box and finite-atlas comparisons are covered by the
subsequent arithmetic modules. Auxiliary for Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. -/
theorem analytic_fiber_minima_arithmetic {n d m : ℕ}
    (F : Ambient n → ℝ) (hF : AnalyticAt ℝ F 0)
    (horder : ExactOrderInLastVariable F d)
    (G : Fin m → Ambient n → ℝ) (hG : ∀ j, AnalyticAt ℝ (G j) 0)
    (requirement : Fin m → AnalyticSignRequirement)
    (V : Ambient n → ℝ) (hV : AnalyticAt ℝ V 0) :
    ∃ (a : Fin d → Base n → ℝ) (b : Fin m → Fin d → Base n → ℝ)
      (v : Fin d → Base n → ℝ) (r : ℝ),
      0 < r ∧ (∀ i, AnalyticAt ℝ (a i) 0) ∧ (∀ i, a i 0 = 0) ∧
      (∀ j i, AnalyticAt ℝ (b j i) 0) ∧ (∀ i, AnalyticAt ℝ (v i) 0) ∧
      ∀ z : Base n, ‖z‖ < r → ∀ y : ℝ, |y| < r →
        ((F (z, y) = 0 ∧ (∀ j, (requirement j).Holds (G j (z, y))) ∧
          ∀ t : ℝ, |t| < r → F (z, t) = 0 →
            (∀ j, (requirement j).Holds (G j (z, t))) → V (z, y) ≤ V (z, t)) ↔
        (preparedPolynomial d a (z, y) = 0 ∧
          (∀ j, (requirement j).Holds (∑ i : Fin d, b j i z * y ^ (i : ℕ))) ∧
          fiberMinimumQuery a b v requirement z y r = 0)) := by
  obtain ⟨a, b, v, r, hr, ha, ha0, hb, hv, heq⟩ :=
    analytic_fiber_minima_polynomial F hF horder G hG requirement V hV
  refine ⟨a, b, v, r, hr, ha, ha0, hb, hv, ?_⟩
  intro z hz y hy
  refine (heq z hz y hy).trans ?_
  rw [← fiberMinimumQuery_iff a b v requirement z y r]

/-- The moving level of the degenerate energy `x⁴`, an exponential sign
constraint, and its actual squared gradient norm jointly witness all
analytic minimum-query assumptions. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
example : let E : EucSpace 1 → ℝ := fun x => (x 0) ^ 4
    let phi : Ambient 1 → EucSpace 1 := fun x => x.2 • PiLp.single 2 (0 : Fin 1) 1
    let F : Ambient 1 → ℝ := fun x => E (phi x) - x.1 0
    AnalyticAt ℝ F 0 ∧ ExactOrderInLastVariable F 4 ∧
      AnalyticAt ℝ (fun x => squaredGradientNorm E (phi x)) 0 ∧
      (∀ j : Fin 1, AnalyticAt ℝ (fun x : Ambient 1 => Real.exp x.2 + (j : ℝ)) 0) := by
  intro E phi F
  have hE : AnalyticAt ℝ E 0 :=
    ((EuclideanSpace.proj 0 : EucSpace 1 →L[ℝ] ℝ).analyticAt 0).fun_pow 4
  have hphi : AnalyticAt ℝ phi 0 := analyticAt_snd.smul analyticAt_const
  have hphi0 : phi 0 = 0 := by simp [phi]
  have hz : AnalyticAt ℝ (fun x : Ambient 1 => x.1 0) 0 :=
    ((ContinuousLinearMap.proj (0 : Fin 1)).comp
      (ContinuousLinearMap.fst ℝ (Base 1) ℝ)).analyticAt 0
  have hF : AnalyticAt ℝ F 0 :=
    ((by simpa only [hphi0] using hE : AnalyticAt ℝ E (phi 0)).comp hphi).sub hz
  have hslice : lastSlice F = fun t : ℝ => t ^ 4 := by
    funext t
    simp [lastSlice, F, E, phi]
  have horder : ExactOrderInLastVariable F 4 := by
    rw [ExactOrderInLastVariable, hslice]
    apply (analyticOrderAt_eq_nat_iff_iteratedDeriv_eq_zero
      (analyticAt_id.fun_pow 4)).mp
    simpa [Pi.pow_def] using analyticOrderAt_pow
      (analyticAt_id (𝕜 := ℝ) (z := (0 : ℝ))) 4
  have hnorm : AnalyticAt ℝ (squaredGradientNorm E) (phi 0) := by
    simpa only [hphi0] using squared_gradient_norm_analyticAt E 0 hE
  exact ⟨hF, horder, hnorm.comp hphi, fun _ =>
    (analyticAt_rexp.comp analyticAt_snd).add analyticAt_const⟩

end Transformer.Normalization

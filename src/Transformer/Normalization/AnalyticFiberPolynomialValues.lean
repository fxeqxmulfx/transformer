/-
# Simultaneous polynomial values on analytic scalar fibers

One fixed box retains the equation, every finite constraint value, and
the objective value under actual analytic preparation and division.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Normalization.AnalyticFiberConstraints

noncomputable section
open Filter Topology
open scoped BigOperators

namespace Transformer.Normalization

open AnalyticPreparation

/-- Preparation and division give simultaneous exact values for all
finite constraints and an arbitrary analytic objective at every root
in one fixed box, for arbitrary base coordinates. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analytic_fiber_polynomial_values {n d m : ℕ}
    (F : Ambient n → ℝ) (hF : AnalyticAt ℝ F 0)
    (horder : ExactOrderInLastVariable F d)
    (G : Fin m → Ambient n → ℝ) (hG : ∀ j, AnalyticAt ℝ (G j) 0)
    (V : Ambient n → ℝ) (hV : AnalyticAt ℝ V 0) :
    ∃ (a : Fin d → Base n → ℝ) (b : Fin m → Fin d → Base n → ℝ)
      (v : Fin d → Base n → ℝ) (r : ℝ),
      0 < r ∧ (∀ i, AnalyticAt ℝ (a i) 0) ∧ (∀ i, a i 0 = 0) ∧
      (∀ j i, AnalyticAt ℝ (b j i) 0) ∧ (∀ i, AnalyticAt ℝ (v i) 0) ∧
      ∀ z : Base n, ‖z‖ < r → ∀ y : ℝ, |y| < r →
        (F (z, y) = 0 ↔ preparedPolynomial d a (z, y) = 0) ∧
        (preparedPolynomial d a (z, y) = 0 →
          ∀ j, G j (z, y) = ∑ i : Fin d, b j i z * y ^ (i : ℕ)) ∧
        (preparedPolynomial d a (z, y) = 0 →
          V (z, y) = ∑ i : Fin d, v i z * y ^ (i : ℕ)) := by
  obtain ⟨a, u, ha, ha0, hu, hu0, hfactor⟩ := exists_isWeierstrassPreparation hF horder
  obtain ⟨b, hb, hvalues⟩ := analytic_prepared_fiber_remainders a ha ha0 G hG
  obtain ⟨q, v, hq, hv, hVfactor⟩ := exists_analyticWeierstrassDivision V hV a ha ha0
  have hunit : ∀ᶠ x in 𝓝 (0 : Ambient n), u x ≠ 0 := hu.continuousAt.eventually_ne hu0
  have hcommon : ∀ᶠ x in 𝓝 (0 : Ambient n),
      (F x = 0 ↔ preparedPolynomial d a x = 0) ∧
        (preparedPolynomial d a x = 0 →
          ∀ j, G j x = ∑ i : Fin d, b j i x.1 * x.2 ^ (i : ℕ)) ∧
        (preparedPolynomial d a x = 0 → V x = ∑ i : Fin d, v i x.1 * x.2 ^ (i : ℕ)) := by
    filter_upwards [hfactor, hunit, hvalues, hVfactor] with x hx hne hval hVx
    refine ⟨?_, hval, fun hzero => ?_⟩
    · rw [hx, mul_eq_zero]
      simp only [hne, false_or]
    · simpa only [hzero, mul_zero, zero_add] using hVx
  obtain ⟨r, hr, hball⟩ := Metric.eventually_nhds_iff.mp hcommon
  refine ⟨a, b, v, r, hr, ha, ha0, hb, hv, ?_⟩
  intro z hz y hy
  exact hball (y := (z, y)) (by
    simpa only [dist_zero_right, Prod.norm_def, Real.norm_eq_abs] using max_lt hz hy)

/-- A moving quadratic level, a nonzero exponential constraint, and
a quartic objective witness all joint preparation and division inputs.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : AnalyticAt ℝ (fun x : Ambient 1 => x.2 ^ 2 - x.1 0) 0 ∧
    ExactOrderInLastVariable (fun x : Ambient 1 => x.2 ^ 2 - x.1 0) 2 ∧
    (∀ j : Fin 1, AnalyticAt ℝ (fun x : Ambient 1 => Real.exp x.2 + (j : ℝ)) 0) ∧
    AnalyticAt ℝ (fun x : Ambient 1 => x.2 ^ 4) 0 := by
  have hz : AnalyticAt ℝ (fun x : Ambient 1 => x.1 0) 0 :=
    ((ContinuousLinearMap.proj (0 : Fin 1)).comp
      (ContinuousLinearMap.fst ℝ (Base 1) ℝ)).analyticAt 0
  refine ⟨(analyticAt_snd.fun_pow 2).sub hz, ?_,
    fun _ => (analyticAt_rexp.comp analyticAt_snd).add analyticAt_const,
    analyticAt_snd.fun_pow 4⟩
  have hslice : lastSlice (fun x : Ambient 1 => x.2 ^ 2 - x.1 0) =
      fun t : ℝ => t ^ 2 := by funext t; simp [lastSlice]
  rw [ExactOrderInLastVariable, hslice]
  apply (analyticOrderAt_eq_nat_iff_iteratedDeriv_eq_zero (analyticAt_id.fun_pow 2)).mp
  simpa [Pi.pow_def] using analyticOrderAt_pow
    (analyticAt_id (𝕜 := ℝ) (z := (0 : ℝ))) 2

end Transformer.Normalization

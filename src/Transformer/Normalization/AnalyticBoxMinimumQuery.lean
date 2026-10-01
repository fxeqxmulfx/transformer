/-
# Scalar quantifier elimination for analytic minima on a whole box

The comparison threshold is fixed by the candidate and is compared
against every admissible base coordinate and scalar root. Arbitrary
base slices are retained, including a prescribed energy parameter.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Normalization.AnalyticFiberMinimumQuery
import Transformer.Normalization.PolynomialFiberThreshold

noncomputable section
open scoped BigOperators

namespace Transformer.Normalization

open AnalyticPreparation

/-- A constrained analytic minimum on an entire box, restricted to any
chosen base slice, has an exact description in which the competitor's
scalar coordinate is eliminated. The objective threshold is the actual
candidate value and is compared across all admissible base coordinates,
not just the candidate's own scalar fiber. The universal base quantifier
is retained explicitly. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem analytic_box_minima_arithmetic {n d m : ℕ}
    (F : Ambient n → ℝ) (hF : AnalyticAt ℝ F 0)
    (horder : ExactOrderInLastVariable F d)
    (G : Fin m → Ambient n → ℝ) (hG : ∀ j, AnalyticAt ℝ (G j) 0)
    (requirement : Fin m → AnalyticSignRequirement)
    (V : Ambient n → ℝ) (hV : AnalyticAt ℝ V 0) :
    ∃ (a : Fin d → Base n → ℝ) (b : Fin m → Fin d → Base n → ℝ)
      (v : Fin d → Base n → ℝ) (r : ℝ),
      0 < r ∧ (∀ i, AnalyticAt ℝ (a i) 0) ∧ (∀ i, a i 0 = 0) ∧
      (∀ j i, AnalyticAt ℝ (b j i) 0) ∧ (∀ i, AnalyticAt ℝ (v i) 0) ∧
      ∀ S : Set (Base n), ∀ w : Ambient n, ‖w.1‖ < r → |w.2| < r →
        ((w.1 ∈ S ∧ F w = 0 ∧ (∀ j, (requirement j).Holds (G j w)) ∧
          ∀ x : Ambient n, x.1 ∈ S → ‖x.1‖ < r → |x.2| < r → F x = 0 →
            (∀ j, (requirement j).Holds (G j x)) → V w ≤ V x) ↔
        (w.1 ∈ S ∧ preparedPolynomial d a w = 0 ∧
          (∀ j, (requirement j).Holds (∑ i : Fin d, b j i w.1 * w.2 ^ (i : ℕ))) ∧
          ∀ z ∈ S, ‖z‖ < r → fiberLowerBoundQuery a b v requirement z (V w) r = 0)) := by
  obtain ⟨a, b, v, r, hr, ha, ha0, hb, hv, hvalues⟩ :=
    analytic_fiber_polynomial_values F hF horder G hG V hV
  refine ⟨a, b, v, r, hr, ha, ha0, hb, hv, ?_⟩
  intro S w hwbase hwscalar
  have hw := hvalues w.1 hwbase w.2 hwscalar
  constructor
  · rintro ⟨hwS, hFw, hGw, hmin⟩
    have hPw := hw.1.mp hFw
    refine ⟨hwS, hPw, fun j => ?_, ?_⟩
    · rw [← hw.2.1 hPw j]
      exact hGw j
    · intro z hzS hz
      apply (fiberLowerBoundQuery_iff a b v requirement z (V w) r).mp
      intro t ht hPt hGt
      have hzt := hvalues z hz t ht
      have horiginal : ∀ j, (requirement j).Holds (G j (z, t)) := by
        intro j
        rw [hzt.2.1 hPt j]
        exact hGt j
      have hbound := hmin (z, t) hzS hz ht (hzt.1.mpr hPt) horiginal
      rwa [hzt.2.2 hPt] at hbound
  · rintro ⟨hwS, hPw, hGw, hqueries⟩
    refine ⟨hwS, hw.1.mpr hPw, fun j => ?_, ?_⟩
    · rw [hw.2.1 hPw j]
      exact hGw j
    · intro x hxS hxbase hxscalar hFx hGx
      have hpx := hvalues x.1 hxbase x.2 hxscalar
      have hPx := hpx.1.mp hFx
      have hpolynomial : ∀ j,
          (requirement j).Holds (∑ i : Fin d, b j i x.1 * x.2 ^ (i : ℕ)) := by
        intro j
        rw [← hpx.2.1 hPx j]
        exact hGx j
      have hbound := (fiberLowerBoundQuery_iff a b v requirement x.1 (V w) r).mpr
        (hqueries x.1 hxS hxbase) x.2 hxscalar hPx hpolynomial
      rwa [← hpx.2.2 hPx] at hbound

/-- A moving quadratic level, exponential sign constraint, and quartic
objective jointly satisfy all inputs of the whole-box minimum theorem.
The base slice can fix the moving energy coordinate. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
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

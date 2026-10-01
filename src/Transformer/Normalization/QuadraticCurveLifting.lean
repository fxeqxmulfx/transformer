/-
# Complete analytic lifting of quadratic root branches

For an arbitrary analytic monic quadratic in one parameter, accumulating
real roots on the positive side yield two analytic branches after square
ramification. The branches cover every root on the resulting neighborhood,
including a double root at the base point.
-/

import Transformer.Normalization.RamifiedSquareRoot
import Mathlib.Analysis.Real.Sqrt

open Filter Set

namespace Transformer.Normalization

/-- All real roots of an analytic monic quadratic with central polynomial
`y²` are covered by two analytic branches after the ramification `t ↦ t²`.
Only accumulation of real roots on the positive side is assumed. This
proves the degree-two singular branch-lifting step for the general
curve-selection construction in Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem analytic_quadratic_root_branches (a b : ℝ → ℝ)
    (ha : AnalyticAt ℝ a 0) (hb : AnalyticAt ℝ b 0)
    (ha0 : a 0 = 0) (hb0 : b 0 = 0)
    (hroots : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0),
      ∃ y : ℝ, y ^ 2 + a t * y + b t = 0) :
    ∃ plus minus : ℝ → ℝ, AnalyticAt ℝ plus 0 ∧ AnalyticAt ℝ minus 0 ∧
      plus 0 = 0 ∧ minus 0 = 0 ∧ ∀ᶠ t in nhds (0 : ℝ), ∀ y : ℝ,
        y ^ 2 + a (t ^ 2) * y + b (t ^ 2) = 0 ↔ y = plus t ∨ y = minus t := by
  let D : ℝ → ℝ := fun t => (a t) ^ 2 - 4 * b t
  have hD : AnalyticAt ℝ D 0 := (ha.fun_pow 2).sub (analyticAt_const.mul hb)
  have hnonnegative : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 ≤ D t := by
    apply hroots.mono
    intro t ht
    obtain ⟨y, hy⟩ := ht
    dsimp only [D]
    nlinarith [sq_nonneg (2 * y + a t)]
  obtain ⟨g, hg, hgsquare⟩ := analytic_square_root_after_ramification D hD hnonnegative
  have hg0 : g 0 = 0 := by
    apply sq_eq_zero_iff.mp
    simpa [D, ha0, hb0] using hgsquare.self_of_nhds
  let plus : ℝ → ℝ := fun t => (-a (t ^ 2) + g t) / 2
  let minus : ℝ → ℝ := fun t => (-a (t ^ 2) - g t) / 2
  have hsquare : AnalyticAt ℝ (fun t : ℝ => t ^ 2) 0 := analyticAt_id.fun_pow 2
  have haat : AnalyticAt ℝ a ((0 : ℝ) ^ 2) := by simpa using ha
  have hacomp : AnalyticAt ℝ (fun t : ℝ => a (t ^ 2)) 0 :=
    haat.comp (f := fun t : ℝ => t ^ 2) (x := 0) hsquare
  have hplus : AnalyticAt ℝ plus 0 := by
    simpa only [plus, div_eq_mul_inv] using
      (hacomp.fun_neg.fun_add hg).fun_mul
      (analyticAt_const (v := (2 : ℝ)⁻¹))
  have hminus : AnalyticAt ℝ minus 0 := by
    simpa only [minus, div_eq_mul_inv] using
      (hacomp.fun_neg.fun_sub hg).fun_mul
      (analyticAt_const (v := (2 : ℝ)⁻¹))
  refine ⟨plus, minus, hplus, hminus, by simp [plus, ha0, hg0],
    by simp [minus, ha0, hg0], ?_⟩
  filter_upwards [hgsquare] with t hsq
  intro y
  have hfactor : (y - plus t) * (y - minus t) =
      y ^ 2 + a (t ^ 2) * y + b (t ^ 2) := by
    calc
      (y - plus t) * (y - minus t) =
          y ^ 2 + a (t ^ 2) * y + ((a (t ^ 2)) ^ 2 - (g t) ^ 2) / 4 := by
        dsimp only [plus, minus]
        ring
      _ = y ^ 2 + a (t ^ 2) * y + b (t ^ 2) := by rw [hsq]; dsimp only [D]; ring
  rw [← hfactor, mul_eq_zero, sub_eq_zero, sub_eq_zero]

/-- The cusp `y² = t³` has a double central root and satisfies the real-root
accumulation hypothesis. The general construction covers both branches
after `t = s²`. Appendix D.1 of arXiv:2510.22026v2. -/
example : ∃ plus minus : ℝ → ℝ, AnalyticAt ℝ plus 0 ∧ AnalyticAt ℝ minus 0 ∧
    plus 0 = 0 ∧ minus 0 = 0 ∧ ∀ᶠ t in nhds (0 : ℝ), ∀ y : ℝ,
      y ^ 2 - (t ^ 2) ^ 3 = 0 ↔ y = plus t ∨ y = minus t := by
  have : NeBot (nhdsWithin (0 : ℝ) (Ioi 0)) := nhdsWithin_Ioi_neBot le_rfl
  have hmem : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), t ∈ Ioi 0 := self_mem_nhdsWithin
  have hroots : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0),
      ∃ y : ℝ, y ^ 2 + (0 : ℝ) * y + -(t ^ 3) = 0 := by
    apply hmem.frequently.mono
    intro t ht
    refine ⟨Real.sqrt (t ^ 3), ?_⟩
    rw [Real.sq_sqrt (pow_nonneg (le_of_lt ht) 3)]
    ring
  simpa only [zero_mul, add_zero, ← sub_eq_add_neg] using
    analytic_quadratic_root_branches (fun _ => 0) (fun t => -(t ^ 3))
      analyticAt_const (analyticAt_id.fun_pow 3).neg (by simp) (by simp) hroots

end Transformer.Normalization

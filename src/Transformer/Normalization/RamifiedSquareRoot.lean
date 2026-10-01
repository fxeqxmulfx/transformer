/-
# Analytic square roots after a square ramification

An analytic germ which is nonnegative arbitrarily close on the positive
side has an analytic square root after `t ↦ t²`. Finite analytic order
handles odd-order zeros; the square root of the remaining positive unit
is constructed from the analytic logarithm and exponential.
-/

import Transformer.Normalization.AnalyticImplicitBranch
import Mathlib.Analysis.Analytic.IsolatedZeros
import Mathlib.Analysis.SpecialFunctions.Complex.Analytic

open Filter Set

namespace Transformer.Normalization

/-- Square ramification gives an analytic square root for a real analytic
germ which is nonnegative frequently on the positive side. Positivity is
required only along an accumulating family, as in selection of a real root
branch. This is the square-root step in singular branch lifting for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analytic_square_root_after_ramification (f : ℝ → ℝ)
    (hf : AnalyticAt ℝ f 0)
    (hpositive : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 ≤ f t) :
    ∃ g : ℝ → ℝ, AnalyticAt ℝ g 0 ∧
      ∀ᶠ t in nhds (0 : ℝ), (g t) ^ 2 = f (t ^ 2) := by
  have hsquare : AnalyticAt ℝ (fun t : ℝ => t ^ 2) 0 := analyticAt_id.fun_pow 2
  have ht : Tendsto (fun t : ℝ => t ^ 2) (nhds 0) (nhds 0) := by
    simpa using hsquare.continuousAt.tendsto
  by_cases hzero : ∀ᶠ t in nhds (0 : ℝ), f t = 0
  · refine ⟨fun _ => 0, analyticAt_const, ?_⟩
    filter_upwards [ht.eventually hzero] with t htzero
    simp [htzero]
  obtain ⟨m, u, hu, hu0, hfactor⟩ := hf.exists_eventuallyEq_pow_smul_nonzero_iff.mpr hzero
  have hunit : 0 < u 0 := by
    have hnonneg : 0 ≤ u 0 := by
      by_contra hn
      have hn0 : u 0 < 0 := lt_of_not_ge hn
      have hnegative : ∀ᶠ t in nhds (0 : ℝ), u t < 0 :=
        hu.continuousAt.eventually (Iio_mem_nhds hn0)
      have hnegf : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), f t < 0 := by
        filter_upwards [hfactor.filter_mono nhdsWithin_le_nhds,
          hnegative.filter_mono nhdsWithin_le_nhds, self_mem_nhdsWithin] with t hf ht hpos
        change 0 < t at hpos
        simpa only [hf, sub_zero, smul_eq_mul] using
          mul_neg_of_pos_of_neg (pow_pos hpos m) ht
      obtain ⟨t, hft, hnt⟩ := (hpositive.and_eventually hnegf).exists
      exact (not_lt_of_ge hft) hnt
    exact lt_of_le_of_ne hnonneg (Ne.symm hu0)
  let q : ℝ → ℝ := fun t => Real.exp (Real.log (u (t ^ 2)) * (1 / 2))
  have huat : AnalyticAt ℝ u ((0 : ℝ) ^ 2) := by simpa using hu
  have hucomp : AnalyticAt ℝ (fun t : ℝ => u (t ^ 2)) 0 :=
    huat.comp (f := fun t : ℝ => t ^ 2) (x := 0) hsquare
  have hlog : AnalyticAt ℝ (fun t : ℝ => Real.log (u (t ^ 2))) 0 :=
    hucomp.log (by simpa using hunit)
  have hq : AnalyticAt ℝ q 0 := (hlog.mul analyticAt_const).rexp'
  let g : ℝ → ℝ := fun t => t ^ m * q t
  refine ⟨g, (analyticAt_id.fun_pow m).mul hq, ?_⟩
  have hnear : ∀ᶠ t in nhds (0 : ℝ), 0 < u (t ^ 2) :=
    hucomp.continuousAt.eventually (Ioi_mem_nhds (by simpa using hunit))
  filter_upwards [ht.eventually hfactor, hnear] with t hfac hpos
  have hq2 : (q t) ^ 2 = u (t ^ 2) := by
    dsimp only [q]
    rw [sq, ← Real.exp_add]
    have heq : Real.log (u (t ^ 2)) * (1 / 2) +
        Real.log (u (t ^ 2)) * (1 / 2) = Real.log (u (t ^ 2)) := by ring
    rw [heq, Real.exp_log hpos]
  rw [hfac]
  simp only [g, mul_pow, hq2, sub_zero, smul_eq_mul, ← pow_mul, Nat.mul_comm]

/-- The odd-order germ `t³` is positive on the positive side and has an
analytic root after square ramification. It exercises the singular case
with an odd order, rather than only a positive unit. Appendix D.1 of
arXiv:2510.22026v2. -/
example : ∃ g : ℝ → ℝ, AnalyticAt ℝ g 0 ∧
    ∀ᶠ t in nhds (0 : ℝ), (g t) ^ 2 = (t ^ 2) ^ 3 := by
  have : NeBot (nhdsWithin (0 : ℝ) (Ioi 0)) := nhdsWithin_Ioi_neBot le_rfl
  apply analytic_square_root_after_ramification (fun t : ℝ => t ^ 3)
    (analyticAt_id.fun_pow 3)
  have hmem : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), t ∈ Ioi 0 := self_mem_nhdsWithin
  have hpositive : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 ≤ t ^ 3 :=
    hmem.mono (fun t ht => pow_nonneg (le_of_lt ht) 3)
  exact hpositive.frequently

end Transformer.Normalization

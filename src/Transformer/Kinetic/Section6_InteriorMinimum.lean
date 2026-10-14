/-
# Kinetic theory for Transformers — the unique positive minimum

The final real-variable step of arXiv:2605.09213v1, §6, proof of
`thm:U-shape`, uses continuity, strict convexity on `[0,∞)`, divergence
at infinity and a negative initial derivative. We prove that step by
compact attainment, the actual derivative's slope limit, and uniqueness
of a strictly convex minimum. No minimizer is put into a definition.
-/

import Transformer.Kinetic.Section6_WeightedConvexity
import Mathlib.Analysis.Calculus.LocalExtr.Basic
import Mathlib.Topology.Order.Compact

open Set Filter
open scoped Topology

namespace Transformer
namespace Kinetic

/-- A negative actual initial derivative gives a smaller value at a
positive argument. Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem exists_pos_lower_of_negative_derivative {f : ℝ → ℝ} {D : ℝ}
    (hf : HasDerivAt f D 0) (hD : D < 0) : ∃ y : ℝ, 0 < y ∧ f y < f 0 := by
  have ht := (hasDerivAt_iff_tendsto_slope_left_right.mp hf).2
  have hneg := ht.eventually_lt_const hD
  have hpos : ∀ᶠ y : ℝ in 𝓝[>] 0, 0 < y := self_mem_nhdsWithin
  obtain ⟨y, hy, hys⟩ := (hpos.and hneg).exists
  exact ⟨y, hy, (slope_neg_iff_of_le hy.le).mp hys⟩

example : ∃ y : ℝ, 0 < y ∧ -y < -(0 : ℝ) :=
  exists_pos_lower_of_negative_derivative (hasDerivAt_neg 0) (by norm_num)

/-- The actual continuous function attains a minimum on the closed
half-line if it diverges at infinity.
Source: arXiv:2605.09213v1, §6, the coercive minimum step of `thm:U-shape`. -/
theorem exists_minimum_on_halfLine {f : ℝ → ℝ} (hf : Continuous f)
    (htail : Tendsto f atTop atTop) : ∃ y ∈ Ici (0 : ℝ), IsMinOn f (Ici 0) y := by
  obtain ⟨R, hR⟩ := eventually_atTop.mp (htail.eventually_ge_atTop (f 0))
  let T := max 0 R
  have hT0 : 0 ≤ T := le_max_left _ _
  have hRT : R ≤ T := le_max_right _ _
  obtain ⟨y, hy, hmin⟩ := (isCompact_Icc (a := (0 : ℝ)) (b := T)).exists_isMinOn
    ⟨0, le_rfl, hT0⟩ hf.continuousOn
  have hmin' := isMinOn_iff.mp hmin
  refine ⟨y, hy.1, isMinOn_iff.mpr fun z hz => ?_⟩
  by_cases hzT : z ≤ T
  · exact hmin' z ⟨hz, hzT⟩
  · exact (hmin' 0 ⟨le_rfl, hT0⟩).trans (hR z (hRT.trans (le_of_not_ge hzT)))

example := exists_minimum_on_halfLine (continuous_id : Continuous (id : ℝ → ℝ)) tendsto_id

/-- Strict convexity and a negative initial derivative give exactly one
positive global minimizer of the actual function.
Source: arXiv:2605.09213v1, §6, the final step of `thm:U-shape`. -/
theorem exists_unique_positive_minimum {f : ℝ → ℝ} {D : ℝ} (hf : Continuous f)
    (hconv : StrictConvexOn ℝ (Ici 0) f) (htail : Tendsto f atTop atTop)
    (hd : HasDerivAt f D 0) (hD : D < 0) :
    ∃! y : ℝ, 0 < y ∧ ∀ z : ℝ, 0 < z → f y ≤ f z := by
  obtain ⟨y, hy, hmin⟩ := exists_minimum_on_halfLine hf htail
  have hmin' := isMinOn_iff.mp hmin
  have hypos : 0 < y := by
    by_contra hn
    have hy0 : y = 0 := le_antisymm (le_of_not_gt hn) hy
    obtain ⟨z, hz, hlt⟩ := exists_pos_lower_of_negative_derivative hd hD
    have hle := hmin' z hz.le
    rw [hy0] at hle
    exact (not_lt_of_ge hle) hlt
  have hconv' : StrictConvexOn ℝ (Ioi (0 : ℝ)) f :=
    hconv.subset (s := Ioi (0 : ℝ)) Ioi_subset_Ici_self (convex_Ioi 0)
  have hymin : IsMinOn f (Ioi (0 : ℝ)) y :=
    isMinOn_iff.mpr fun q hq => hmin' q (mem_Ici.mpr (mem_Ioi.mp hq).le)
  refine ⟨y, ⟨hypos, fun z hz => hmin' z hz.le⟩, fun z hz => ?_⟩
  exact hconv'.eq_of_isMinOn (isMinOn_iff.mpr hz.2) hymin hz.1 hypos

/-- An actual positive interior minimizer has zero actual derivative.
Source: arXiv:2605.09213v1, §6, the interior minimum in `thm:U-shape`. -/
theorem derivative_eq_zero_at_positive_minimum {f : ℝ → ℝ} {D y : ℝ}
    (hy : 0 < y) (hmin : ∀ z : ℝ, 0 < z → f y ≤ f z) (hd : HasDerivAt f D y) : D = 0 :=
  ((show IsMinOn f (Ioi 0) y from hmin).isLocalMin (Ioi_mem_nhds hy)).hasDerivAt_eq_zero hd

example := derivative_eq_zero_at_positive_minimum (y := 1) one_pos
  (fun _ _ => le_rfl) (hasDerivAt_const 1 (0 : ℝ))

/-- The original sum of bounded positive modes has exactly one positive
minimum under the two source coefficient bounds and one strict slope bound.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem exists_unique_minimum_weightedProfile {w c : ℕ → ℝ} {lam : ℝ}
    (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n) (hc0 : ∀ n, 0 ≤ c n)
    (hcurv : ∀ n, c n ≤ 3 - Real.sqrt 3) (hlam : 0 < lam)
    (hslope : ∀ n, c n ≤ 2 * (1 - Real.exp (-lam)))
    (i : ℕ) (hwi : 0 < w i) (hci : 0 < c i)
    (hstrict : c i < 2 * (1 - Real.exp (-lam))) :
    ∃! y : ℝ, 0 < y ∧ ∀ z : ℝ, 0 < z →
      seriesProfile lam (weightedPsiCoeffs w c) y ≤ seriesProfile lam (weightedPsiCoeffs w c) z := by
  have ha := isEntire_weightedPsiCoeffs hw hw0 hc0 hcurv
  exact exists_unique_positive_minimum (continuous_seriesProfile ha lam)
    (strictConvexOn_weightedProfile hw hw0 hc0 hcurv hlam i hwi hci)
    (weightedProfile_tendsto_atTop hw hw0 hc0 hcurv hlam i hwi hci)
    (hasDerivAt_seriesProfile ha lam 0)
    (seriesDeriv_weighted_zero_neg hw hw0 hc0 hcurv hlam hslope i hwi hci hstrict)

/-- A concrete quadratic witnesses all hypotheses of the real-variable
unique-minimum theorem, including its actual negative initial derivative.
Source: arXiv:2605.09213v1, §6, the elementary convex minimum argument. -/
example : Continuous (fun y : ℝ => (y - 1) ^ 2) ∧
    StrictConvexOn ℝ (Ici 0) (fun y : ℝ => (y - 1) ^ 2) ∧
    Tendsto (fun y : ℝ => (y - 1) ^ 2) atTop atTop ∧
    HasDerivAt (fun y : ℝ => (y - 1) ^ 2) (-2) 0 ∧ (-2 : ℝ) < 0 := by
  have hd : ∀ y : ℝ, HasDerivAt (fun z : ℝ => (z - 1) ^ 2) (2 * (y - 1)) y := by
    intro y
    convert ((hasDerivAt_id' (x := y)).sub_const 1).pow 2 using 1
    simp
  have hd2 : ∀ y : ℝ, HasDerivAt (fun z : ℝ => 2 * (z - 1)) 2 y := by
    intro y
    convert ((hasDerivAt_id' (x := y)).sub_const 1).const_mul 2 using 1
    simp
  have hc : Continuous (fun y : ℝ => (y - 1) ^ 2) := by fun_prop
  refine ⟨hc, ?_, ?_, ?_, by norm_num⟩
  · apply strictConvexOn_of_deriv2_pos' (convex_Ici 0) hc.continuousOn
    intro y _
    rw [Function.iterate_succ_apply, Function.iterate_one,
      show deriv (fun z : ℝ => (z - 1) ^ 2) = fun z => 2 * (z - 1) from
        funext fun z => (hd z).deriv, (hd2 y).deriv]
    norm_num
  · apply tendsto_atTop_mono' atTop
      ((eventually_ge_atTop (3 : ℝ)).mono fun y hy => ?_) tendsto_id
    change y ≤ (y - 1) ^ 2
    nlinarith [mul_nonneg (by linarith : 0 ≤ y) (by linarith : 0 ≤ y - 3)]
  · simpa using hd 0

/-- A positive bounded mode family strictly below the slope threshold
witnesses all hypotheses of `exists_unique_minimum_weightedProfile`.
Source: arXiv:2605.09213v1, §6, the summable-mode convex minimum argument. -/
example : ∃! y : ℝ, 0 < y ∧ ∀ z : ℝ, 0 < z →
    seriesProfile 1 (weightedPsiCoeffs (fun n : ℕ => Real.exp (-(n : ℝ)))
      (fun n : ℕ => (1 - Real.exp (-1)) / (n + 1))) y ≤
    seriesProfile 1 (weightedPsiCoeffs (fun n : ℕ => Real.exp (-(n : ℝ)))
      (fun n : ℕ => (1 - Real.exp (-1)) / (n + 1))) z := by
  have hq : 0 < 1 - Real.exp (-1) := by
    linarith [Real.exp_lt_one_iff.mpr (by norm_num : -(1 : ℝ) < 0)]
  have hb : ∀ n : ℕ, (1 - Real.exp (-1)) / (n + 1) ≤ 1 - Real.exp (-1) := by
    intro n
    apply (div_le_iff₀ (by positivity : (0 : ℝ) < n + 1)).mpr
    nlinarith [show (0 : ℝ) ≤ n by positivity]
  have hcurv : 1 - Real.exp (-1) ≤ 3 - Real.sqrt 3 := by
    nlinarith [Real.exp_pos (-1), Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3), Real.sqrt_nonneg 3]
  apply exists_unique_minimum_weightedProfile Real.summable_exp_neg_nat
    (fun _ => (Real.exp_pos _).le) (fun _ => by positivity)
    (fun n => (hb n).trans hcurv) one_pos (fun n => (hb n).trans (by linarith))
    0 (Real.exp_pos _) (by simpa only [Nat.cast_zero, zero_add, div_one] using hq)
  simp only [Nat.cast_zero, zero_add, div_one]
  linarith

end Kinetic
end Transformer

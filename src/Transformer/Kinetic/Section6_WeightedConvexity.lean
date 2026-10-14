/-
# Kinetic theory for Transformers — convexity and growth of the aggregate

The sum of the original mode coefficients has nonnegative curvature
coefficients under the bound `c_n ≤ 3 - √3` in arXiv:2605.09213v1, §6.
A single positive mode suffices for strictly positive curvature of its
actual aggregate profile. Its positive linear coefficient also gives a
linear lower bound tending to infinity. This proves the source's growth
step directly from the Bessel series, without an asymptotic formula.
-/

import Transformer.Kinetic.Section6_WeightedProfiles

open scoped BigOperators

namespace Transformer
namespace Kinetic

variable {w c a : ℕ → ℝ} {lam : ℝ}

/-- The actual aggregate curvature coefficients are nonnegative under the
source bound. Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem weighted_seriesCurvatureCoeffs_nonneg (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n)
    (hc0 : ∀ n, 0 ≤ c n) (hsmall : ∀ n, c n ≤ 3 - Real.sqrt 3) (k : ℕ) :
    0 ≤ seriesCurvatureCoeffs (weightedPsiCoeffs w c) k := by
  rw [seriesCurvatureCoeffs, weighted_curvatureCoeffs_eq hw hw0 hc0 hsmall]
  exact tsum_nonneg fun n => mul_nonneg (hw0 n)
    (curvatureCoeffs_nonneg (hc0 n) (hsmall n) k)

/-- The source's actual infinite aggregate is strictly convex on `[0,∞)`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem strictConvexOn_weightedProfile (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n)
    (hc0 : ∀ n, 0 ≤ c n) (hsmall : ∀ n, c n ≤ 3 - Real.sqrt 3) (hlam : 0 < lam)
    (i : ℕ) (hwi : 0 < w i) (hci : 0 < c i) :
    StrictConvexOn ℝ (Set.Ici 0) (seriesProfile lam (weightedPsiCoeffs w c)) := by
  have ha := isEntire_weightedPsiCoeffs hw hw0 hc0 hsmall
  have hb := weightedPsiCoeffs_nonneg hw0 hc0
  have hb2 := weightedPsiCoeffs_pos hw hw0 hc0 hsmall i hwi hci 2
  refine strictConvexOn_seriesProfile ha hlam (fun k => ?_) ?_
    (weighted_seriesCurvatureCoeffs_nonneg hw hw0 hc0 hsmall)
  · unfold derivCoeffs
    exact mul_nonneg (by positivity) (mul_nonneg (by positivity) (hb _))
  · simp only [derivCoeffs]
    norm_num
    positivity

/-- The positive first coefficient gives a linear lower bound for the
actual series profile. Source: arXiv:2605.09213v1, §6, the growth step in `thm:U-shape`. -/
theorem seriesProfile_lower_bound (ha : IsEntire a) (ha0 : ∀ k, 0 ≤ a k)
    (hlam : 0 < lam) {y : ℝ} (hy : 0 ≤ y) :
    lam / (Real.exp lam - 1) * a 1 * y ≤ seriesProfile lam a y := by
  have hbase : 0 ≤ lam / (Real.exp lam - 1) :=
    (div_pos hlam (sub_pos.mpr (Real.one_lt_exp_iff.mpr hlam))).le
  have hle := le_powerSum ha ha0 hy 1
  simp only [pow_one] at hle
  have hsum := powerSum_nonneg ha0 hy
  calc lam / (Real.exp lam - 1) * a 1 * y = lam / (Real.exp lam - 1) * (a 1 * y) := by ring
    _ ≤ lam / (Real.exp lam - 1) * powerSum a y := mul_le_mul_of_nonneg_left hle hbase
    _ ≤ seriesProfile lam a y := by
      unfold seriesProfile
      exact mul_le_mul_of_nonneg_right (le_add_of_nonneg_right (by positivity)) hsum

/-- The actual profile tends to infinity whenever its linear coefficient
is positive. Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem seriesProfile_tendsto_atTop (ha : IsEntire a) (ha0 : ∀ k, 0 ≤ a k)
    (ha1 : 0 < a 1) (hlam : 0 < lam) :
    Filter.Tendsto (seriesProfile lam a) Filter.atTop Filter.atTop := by
  have hbase : 0 < lam / (Real.exp lam - 1) :=
    div_pos hlam (sub_pos.mpr (Real.one_lt_exp_iff.mpr hlam))
  apply Filter.tendsto_atTop_mono' Filter.atTop
    ((Filter.eventually_ge_atTop (0 : ℝ)).mono fun y hy => seriesProfile_lower_bound ha ha0 hlam hy)
  exact Filter.tendsto_id.const_mul_atTop (mul_pos hbase ha1)

/-- A single positive original mode makes the aggregate diverge at infinity.
Source: arXiv:2605.09213v1, §6, the growth step in `thm:U-shape`. -/
theorem weightedProfile_tendsto_atTop {C : ℝ} (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n)
    (hc0 : ∀ n, 0 ≤ c n) (hC : ∀ n, c n ≤ C) (hlam : 0 < lam)
    (i : ℕ) (hwi : 0 < w i) (hci : 0 < c i) :
    Filter.Tendsto (seriesProfile lam (weightedPsiCoeffs w c)) Filter.atTop Filter.atTop :=
  seriesProfile_tendsto_atTop (isEntire_weightedPsiCoeffs hw hw0 hc0 hC)
    (weightedPsiCoeffs_nonneg hw0 hc0) (weightedPsiCoeffs_pos hw hw0 hc0 hC i hwi hci 1) hlam

/-- The aggregate has strictly positive actual second derivative on the
entire closed half-line. Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem seriesDeriv2_weighted_pos (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n)
    (hc0 : ∀ n, 0 ≤ c n) (hsmall : ∀ n, c n ≤ 3 - Real.sqrt 3) (hlam : 0 < lam)
    (i : ℕ) (hwi : 0 < w i) (hci : 0 < c i) {y : ℝ} (hy : 0 ≤ y) :
    0 < seriesDeriv2 lam (weightedPsiCoeffs w c) y := by
  have ha := isEntire_weightedPsiCoeffs hw hw0 hc0 hsmall
  have hb := weightedPsiCoeffs_nonneg hw0 hc0
  have hb2 := weightedPsiCoeffs_pos hw hw0 hc0 hsmall i hwi hci 2
  refine seriesDeriv2_pos ha hlam (fun k => ?_) ?_
    (weighted_seriesCurvatureCoeffs_nonneg hw hw0 hc0 hsmall) hy
  · unfold derivCoeffs
    exact mul_nonneg (by positivity) (mul_nonneg (by positivity) (hb _))
  · simp only [derivCoeffs]
    norm_num
    positivity

/-- The continuous series extension has a positive value at the output
endpoint whenever an original mode is positive.
Source: arXiv:2605.09213v1, §6, the extension `ψ_c(0)=c`. -/
theorem seriesProfile_weighted_zero_pos {C : ℝ} (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n)
    (hc0 : ∀ n, 0 ≤ c n) (hC : ∀ n, c n ≤ C) (hlam : 0 < lam)
    (i : ℕ) (hwi : 0 < w i) (hci : 0 < c i) :
    0 < seriesProfile lam (weightedPsiCoeffs w c) 0 := by
  have hp := weightedPsiCoeffs_pos hw hw0 hc0 hC i hwi hci 0
  have hbase := div_pos hlam (sub_pos.mpr (Real.one_lt_exp_iff.mpr hlam))
  rw [seriesProfile, powerSum_zero]
  exact mul_pos (add_pos hbase (by positivity)) hp

/-- A positive original mode gives a positive actual aggregate at every
nonnegative argument. Source: arXiv:2605.09213v1, §6, the positive mode sum `ℋ`. -/
theorem seriesProfile_weighted_pos {C y : ℝ} (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n)
    (hc0 : ∀ n, 0 ≤ c n) (hC : ∀ n, c n ≤ C) (hlam : 0 < lam)
    (i : ℕ) (hwi : 0 < w i) (hci : 0 < c i) (hy : 0 ≤ y) :
    0 < seriesProfile lam (weightedPsiCoeffs w c) y := by
  have ha := isEntire_weightedPsiCoeffs hw hw0 hc0 hC
  have hle := le_powerSum ha (weightedPsiCoeffs_nonneg hw0 hc0) hy 0
  simp only [pow_zero, mul_one] at hle
  have hp := (weightedPsiCoeffs_pos hw hw0 hc0 hC i hwi hci 0).trans_le hle
  have hbase := div_pos hlam (sub_pos.mpr (Real.one_lt_exp_iff.mpr hlam))
  exact mul_pos (add_pos hbase (by positivity)) hp

/-- Exponentially weighted positive bounded parameters meet all hypotheses
of the aggregate curvature and strict-convexity theorems.
Source: arXiv:2605.09213v1, §6, the series argument for `ℋ`. -/
example : Summable (fun n : ℕ => Real.exp (-(n : ℝ))) ∧
    (∀ n : ℕ, 0 ≤ Real.exp (-(n : ℝ))) ∧
    (∀ n : ℕ, 0 ≤ (1 : ℝ) / (n + 1)) ∧
    (∀ n : ℕ, (1 : ℝ) / (n + 1) ≤ 3 - Real.sqrt 3) ∧
    (0 : ℝ) < 1 ∧ 0 < Real.exp (-(0 : ℝ)) ∧ 0 < (1 : ℝ) / (0 + 1) := by
  have h1 : (1 : ℝ) ≤ 3 - Real.sqrt 3 := by
    nlinarith [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3), Real.sqrt_nonneg 3]
  refine ⟨Real.summable_exp_neg_nat, fun _ => (Real.exp_pos _).le,
    fun _ => by positivity, fun n => ?_, one_pos, Real.exp_pos _, by norm_num⟩
  exact ((div_le_one (by positivity : (0 : ℝ) < n + 1)).mpr
    (by linarith [show (0 : ℝ) ≤ n by positivity])).trans h1

/-- The original `c = 1` mode witnesses the positive-coefficient growth
conditions above. Source: arXiv:2605.09213v1, §6, the one-mode growth step. -/
example : IsEntire (psiCoeffs 1) ∧ (∀ k, 0 ≤ psiCoeffs 1 k) ∧
    0 < psiCoeffs 1 1 ∧ (0 : ℝ) < 1 := by
  refine ⟨isEntire_psiCoeffs 1, psiCoeffs_nonneg zero_le_one, ?_, one_pos⟩
  unfold psiCoeffs
  positivity

/-- Positive bounded parameters strictly below the slope threshold witness
the original aggregate negative-slope theorem in `Section6_WeightedProfiles`.
Source: arXiv:2605.09213v1, §6, the slope step of `thm:U-shape`. -/
example : seriesDeriv 1 (weightedPsiCoeffs (fun n : ℕ => Real.exp (-(n : ℝ)))
    (fun n : ℕ => (1 - Real.exp (-1)) / (n + 1))) 0 < 0 := by
  have hq : 0 < 1 - Real.exp (-1) := by
    linarith [Real.exp_lt_one_iff.mpr (by norm_num : -(1 : ℝ) < 0)]
  have hb : ∀ n : ℕ, (1 - Real.exp (-1)) / (n + 1) ≤ 1 - Real.exp (-1) := by
    intro n
    apply (div_le_iff₀ (by positivity : (0 : ℝ) < n + 1)).mpr
    nlinarith [show (0 : ℝ) ≤ n by positivity]
  apply seriesDeriv_weighted_zero_neg Real.summable_exp_neg_nat (fun _ => (Real.exp_pos _).le)
    (fun _ => by positivity) hb one_pos (fun n => (hb n).trans (by linarith))
    0 (Real.exp_pos _) (by simpa only [Nat.cast_zero, zero_add, div_one] using hq)
  simp only [Nat.cast_zero, zero_add, div_one]
  linarith

end Kinetic
end Transformer

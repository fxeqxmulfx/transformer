/-
# Kinetic theory for Transformers — the sum of the original output modes

For the positive summable weights and uniformly bounded nonnegative mode
parameters in arXiv:2605.09213v1, §6, the original double Bessel series
is absolutely summable at every real argument. Its sums may therefore
be interchanged. The source's aggregate `ℋ` is the actual weighted sum
of `h_{c_n}`, and its endpoint slope is the sum of the original slopes.
-/

import Transformer.Kinetic.Section6_WeightedCoefficients
import Transformer.Kinetic.Section6_SeriesProfiles
import Mathlib.Topology.Algebra.InfiniteSum.Real

open scoped BigOperators

namespace Transformer
namespace Kinetic

variable {w c : ℕ → ℝ} {C : ℝ}

/-- Absolute summability of the original two-index Bessel expansion.
Source: arXiv:2605.09213v1, §6, the aggregate `ℋ` in the proof of `thm:U-shape`. -/
theorem summable_weighted_powerTerms (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n)
    (hc0 : ∀ n, 0 ≤ c n) (hC : ∀ n, c n ≤ C) (y : ℝ) :
    Summable fun p : ℕ × ℕ => w p.1 * (psiCoeffs (c p.1) p.2 * y ^ p.2) := by
  have hC0 := (hc0 0).trans (hC 0)
  have hs : Summable fun p : ℕ × ℕ => w p.1 * (psiCoeffs C p.2 * |y| ^ p.2) := by
    apply (summable_prod_of_nonneg (fun p => mul_nonneg (hw0 p.1)
      (mul_nonneg (psiCoeffs_nonneg hC0 p.2) (by positivity)))).mpr
    refine ⟨fun n => ((isEntire_psiCoeffs C).summable |y|).mul_left (w n), ?_⟩
    simp_rw [tsum_mul_left]
    exact hw.mul_right (psi C |y|)
  refine hs.of_norm_bounded fun p => ?_
  rw [Real.norm_eq_abs, abs_mul, abs_mul, abs_pow, abs_of_nonneg (hw0 p.1),
    abs_of_nonneg (psiCoeffs_nonneg (hc0 p.1) p.2)]
  exact mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right
    (psiCoeffs_mono (hc0 p.1) (hC p.1) p.2) (pow_nonneg (abs_nonneg y) p.2)) (hw0 p.1)

/-- Each actual weighted mode sum converges at every argument.
Source: arXiv:2605.09213v1, §6, the aggregate `ℋ`. -/
theorem summable_weighted_psi (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n)
    (hc0 : ∀ n, 0 ≤ c n) (hC : ∀ n, c n ≤ C) (y : ℝ) :
    Summable fun n => w n * psi (c n) y := by
  simpa only [tsum_mul_left, psi, powerSum] using
    (summable_weighted_powerTerms hw hw0 hc0 hC y).prod

/-- Interchanging the actual absolutely summable series gives the aggregate coefficients.
Source: arXiv:2605.09213v1, §6, the series defining `ℋ`. -/
theorem powerSum_weightedPsiCoeffs_eq (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n)
    (hc0 : ∀ n, 0 ≤ c n) (hC : ∀ n, c n ≤ C) (y : ℝ) :
    powerSum (weightedPsiCoeffs w c) y = ∑' n, w n * psi (c n) y := by
  have hs := (summable_weighted_powerTerms hw hw0 hc0 hC y).congr
    (fun p => (mul_assoc (w p.1) (psiCoeffs (c p.1) p.2) (y ^ p.2)).symm)
  simp only [powerSum, weightedPsiCoeffs, ← tsum_mul_right]
  rw [hs.tsum_comm]
  simp only [psi, powerSum, mul_assoc, tsum_mul_left]

/-- The actual aggregate profile equals the source's weighted sum of output modes.
Source: arXiv:2605.09213v1, §6, the definition of `ℋ`. -/
theorem seriesProfile_weighted_eq (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n)
    (hc0 : ∀ n, 0 ≤ c n) (hC : ∀ n, c n ≤ C) (lam y : ℝ) :
    seriesProfile lam (weightedPsiCoeffs w c) y = ∑' n, w n * modeProfile lam (c n) y := by
  rw [seriesProfile, powerSum_weightedPsiCoeffs_eq hw hw0 hc0 hC, ← tsum_mul_left]
  congr 1
  ext n
  unfold modeProfile
  ring

/-- Summability of the actual weighted output profiles follows from the original psi sum.
Source: arXiv:2605.09213v1, §6, the definition of `ℋ`. -/
theorem summable_weighted_modeProfile (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n)
    (hc0 : ∀ n, 0 ≤ c n) (hC : ∀ n, c n ≤ C) (lam y : ℝ) :
    Summable fun n => w n * modeProfile lam (c n) y := by
  convert (summable_weighted_psi hw hw0 hc0 hC y).mul_left
    (lam / (Real.exp lam - 1) + lam * Real.exp (-y)) using 1
  funext n
  unfold modeProfile
  ring

/-- The actual endpoint derivative is the weighted sum of original mode derivatives.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem seriesDeriv_weighted_zero_eq (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n)
    (hc0 : ∀ n, 0 ≤ c n) (hC : ∀ n, c n ≤ C) (lam : ℝ) :
    seriesDeriv lam (weightedPsiCoeffs w c) 0 = ∑' n, w n * modeDeriv lam (c n) 0 := by
  have h0 := (summable_weightedPsiCoeffs hw hw0 hc0 hC 0).mul_left (-lam)
  have h1 := (summable_weightedDerivCoeffs hw hw0 hc0 hC 0).mul_left
    (lam / (Real.exp lam - 1) + lam)
  simp only [seriesDeriv, powerSum_zero, neg_zero, Real.exp_zero, mul_one,
    derivCoeffs_weightedPsiCoeffs, weightedPsiCoeffs, ← tsum_mul_left]
  rw [← h0.tsum_add h1]
  congr 1
  ext n
  simp only [modeDeriv, psi, psiDeriv, powerSum_zero, neg_zero, Real.exp_zero, mul_one]
  ring

/-- The original endpoint mode derivatives form a convergent weighted sum.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem summable_weighted_modeDeriv_zero (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n)
    (hc0 : ∀ n, 0 ≤ c n) (hC : ∀ n, c n ≤ C) (lam : ℝ) :
    Summable fun n => w n * modeDeriv lam (c n) 0 := by
  convert ((summable_weightedPsiCoeffs hw hw0 hc0 hC 0).mul_left (-lam)).add
    ((summable_weightedDerivCoeffs hw hw0 hc0 hC 0).mul_left
      (lam / (Real.exp lam - 1) + lam)) using 1
  funext n
  simp only [modeDeriv, psi, psiDeriv, powerSum_zero, neg_zero, Real.exp_zero, mul_one]
  ring

/-- One strictly negative original mode slope makes the aggregate slope strictly
negative even when other modes attain the non-strict bound.
Source: arXiv:2605.09213v1, §6, correcting the slope step of `thm:U-shape`. -/
theorem seriesDeriv_weighted_zero_neg (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n)
    (hc0 : ∀ n, 0 ≤ c n) (hC : ∀ n, c n ≤ C) {lam : ℝ} (hlam : 0 < lam)
    (hsmall : ∀ n, c n ≤ 2 * (1 - Real.exp (-lam)))
    (i : ℕ) (hwi : 0 < w i) (hci : 0 < c i)
    (hstrict : c i < 2 * (1 - Real.exp (-lam))) :
    seriesDeriv lam (weightedPsiCoeffs w c) 0 < 0 := by
  have hs := (summable_weighted_modeDeriv_zero hw hw0 hc0 hC lam).neg
  have hp := hs.tsum_pos (fun n => neg_nonneg.mpr (mul_nonpos_of_nonneg_of_nonpos (hw0 n)
    (modeDeriv_zero_nonpos hlam (hc0 n) (hsmall n)))) i
    (neg_pos.mpr (mul_neg_of_pos_of_neg hwi (modeDeriv_zero_neg hlam hci hstrict)))
  rw [tsum_neg, ← seriesDeriv_weighted_zero_eq hw hw0 hc0 hC] at hp
  linarith

/-- The actual aggregate profile is nonnegative at nonnegative arguments.
Source: arXiv:2605.09213v1, §6, the positive mode sum `ℋ`. -/
theorem seriesProfile_weighted_nonneg (hw0 : ∀ n, 0 ≤ w n) (hc0 : ∀ n, 0 ≤ c n)
    {lam y : ℝ} (hlam : 0 < lam) (hy : 0 ≤ y) :
    0 ≤ seriesProfile lam (weightedPsiCoeffs w c) y := by
  have hbase : 0 ≤ lam / (Real.exp lam - 1) :=
    (div_pos hlam (sub_pos.mpr (Real.one_lt_exp_iff.mpr hlam))).le
  unfold seriesProfile
  exact mul_nonneg (add_nonneg hbase (by positivity))
    (powerSum_nonneg (weightedPsiCoeffs_nonneg hw0 hc0) hy)

example := seriesProfile_weighted_nonneg (w := fun n : ℕ => Real.exp (-(n : ℝ)))
  (c := fun n : ℕ => 1 / (n + 1)) (fun _ => (Real.exp_pos _).le)
  (fun _ => by positivity) one_pos (le_refl (0 : ℝ))

/-- Exponentially decaying weights and positive bounded parameters witness all
absolute-summability hypotheses above. The parameter at zero is positive.
Source: arXiv:2605.09213v1, §6, the summable-weight argument for `ℋ`. -/
example : Summable (fun n : ℕ => Real.exp (-(n : ℝ))) ∧
    (∀ n : ℕ, 0 ≤ Real.exp (-(n : ℝ))) ∧
    (∀ n : ℕ, 0 ≤ (1 : ℝ) / (n + 1)) ∧ (∀ n : ℕ, (1 : ℝ) / (n + 1) ≤ 1) := by
  exact ⟨Real.summable_exp_neg_nat, fun _ => (Real.exp_pos _).le,
    fun _ => by positivity, fun n => (div_le_one (by positivity : (0 : ℝ) < n + 1)).mpr
      (by linarith [show (0 : ℝ) ≤ n by positivity])⟩

end Kinetic
end Transformer

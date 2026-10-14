/-
# Kinetic theory for Transformers — summing the original mode coefficients

The sum `ℋ` in arXiv:2605.09213v1, §6, proof of `thm:U-shape`, has
positive summable Gaussian weights and bounded positive parameters `c_n`.
We sum its actual Bessel-series coefficients. The resulting coefficients
remain entire, so termwise differentiation follows from the proved entire
series calculus. The first two differentiated coefficients, and their
curvature combination, are the sums of the original mode coefficients.
-/

import Transformer.Kinetic.Section6_ModeCurvature

open scoped BigOperators

namespace Transformer
namespace Kinetic

/-- Nonnegative original mode parameters give nonnegative coefficients.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem psiCoeffs_nonneg {c : ℝ} (hc : 0 ≤ c) (k : ℕ) : 0 ≤ psiCoeffs c k := by
  unfold psiCoeffs
  positivity

example := psiCoeffs_nonneg zero_le_one 2

/-- A uniform parameter bound bounds each actual Bessel-series coefficient.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem psiCoeffs_mono {c C : ℝ} (hc : 0 ≤ c) (hC : c ≤ C) (k : ℕ) :
    psiCoeffs c k ≤ psiCoeffs C k := by
  unfold psiCoeffs
  gcongr

example := psiCoeffs_mono zero_le_one (by norm_num : (1 : ℝ) ≤ 2) 3

/-- The actual weighted coefficients of the source's aggregate series.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
noncomputable def weightedPsiCoeffs (w c : ℕ → ℝ) (k : ℕ) : ℝ :=
  ∑' n, w n * psiCoeffs (c n) k

variable {w c : ℕ → ℝ} {C : ℝ}

/-- The original weighted coefficient sum converges by its uniform parameter bound.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem summable_weightedPsiCoeffs (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n)
    (hc0 : ∀ n, 0 ≤ c n) (hC : ∀ n, c n ≤ C) (k : ℕ) :
    Summable fun n => w n * psiCoeffs (c n) k :=
  Summable.of_nonneg_of_le (fun n => mul_nonneg (hw0 n) (psiCoeffs_nonneg (hc0 n) k))
    (fun n => mul_le_mul_of_nonneg_left (psiCoeffs_mono (hc0 n) (hC n) k) (hw0 n))
    (hw.mul_right (psiCoeffs C k))

/-- Positive original parameters give nonnegative aggregate coefficients.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem weightedPsiCoeffs_nonneg (hw0 : ∀ n, 0 ≤ w n) (hc0 : ∀ n, 0 ≤ c n) (k : ℕ) :
    0 ≤ weightedPsiCoeffs w c k :=
  tsum_nonneg fun n => mul_nonneg (hw0 n) (psiCoeffs_nonneg (hc0 n) k)

/-- The aggregate coefficient is bounded by the original coefficient at `C`
times the total weight. Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem weightedPsiCoeffs_le (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n)
    (hc0 : ∀ n, 0 ≤ c n) (hC : ∀ n, c n ≤ C) (k : ℕ) :
    weightedPsiCoeffs w c k ≤ (∑' n, w n) * psiCoeffs C k := by
  rw [weightedPsiCoeffs, ← tsum_mul_right]
  exact (summable_weightedPsiCoeffs hw hw0 hc0 hC k).tsum_le_tsum
    (fun n => mul_le_mul_of_nonneg_left (psiCoeffs_mono (hc0 n) (hC n) k) (hw0 n))
    (hw.mul_right (psiCoeffs C k))

/-- The original sum of bounded modes is an entire coefficient sequence.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem isEntire_weightedPsiCoeffs (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n)
    (hc0 : ∀ n, 0 ≤ c n) (hC : ∀ n, c n ≤ C) : IsEntire (weightedPsiCoeffs w c) := by
  intro R hR
  refine Summable.of_nonneg_of_le (fun _ => by positivity) (fun k => ?_)
    (((isEntire_psiCoeffs C).const_mul (∑' n, w n)) R hR)
  rw [abs_of_nonneg (weightedPsiCoeffs_nonneg hw0 hc0 k),
    abs_of_nonneg (mul_nonneg (tsum_nonneg hw0) (psiCoeffs_nonneg ((hc0 0).trans (hC 0)) k))]
  exact mul_le_mul_of_nonneg_right (weightedPsiCoeffs_le hw hw0 hc0 hC k) (by positivity)

/-- The first aggregate derivative coefficient is the sum of original ones.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem derivCoeffs_weightedPsiCoeffs (w c : ℕ → ℝ) (k : ℕ) :
    derivCoeffs (weightedPsiCoeffs w c) k = ∑' n, w n * derivCoeffs (psiCoeffs (c n)) k := by
  rw [derivCoeffs, weightedPsiCoeffs, ← tsum_mul_left]
  congr 1
  ext n
  unfold derivCoeffs
  ring

/-- The second aggregate derivative coefficient is the sum of original ones.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem derivCoeffs2_weightedPsiCoeffs (w c : ℕ → ℝ) (k : ℕ) :
    derivCoeffs (derivCoeffs (weightedPsiCoeffs w c)) k =
      ∑' n, w n * derivCoeffs (derivCoeffs (psiCoeffs (c n))) k := by
  rw [derivCoeffs, derivCoeffs_weightedPsiCoeffs, ← tsum_mul_left]
  congr 1
  ext n
  unfold derivCoeffs
  ring

/-- Summability of the original first differentiated coefficients.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem summable_weightedDerivCoeffs (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n)
    (hc0 : ∀ n, 0 ≤ c n) (hC : ∀ n, c n ≤ C) (k : ℕ) :
    Summable fun n => w n * derivCoeffs (psiCoeffs (c n)) k := by
  convert (summable_weightedPsiCoeffs hw hw0 hc0 hC (k + 1)).mul_left (k + 1 : ℝ) using 1
  funext n
  unfold derivCoeffs
  ring

/-- Summability of the original second differentiated coefficients.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem summable_weightedDerivCoeffs2 (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n)
    (hc0 : ∀ n, 0 ≤ c n) (hC : ∀ n, c n ≤ C) (k : ℕ) :
    Summable fun n => w n * derivCoeffs (derivCoeffs (psiCoeffs (c n))) k := by
  convert (summable_weightedDerivCoeffs hw hw0 hc0 hC (k + 1)).mul_left (k + 1 : ℝ) using 1
  funext n
  unfold derivCoeffs
  ring

/-- The aggregate curvature coefficient is the weighted sum of the original ones.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem weighted_curvatureCoeffs_eq (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n)
    (hc0 : ∀ n, 0 ≤ c n) (hC : ∀ n, c n ≤ C) (k : ℕ) :
    derivCoeffs (derivCoeffs (weightedPsiCoeffs w c)) k -
        2 * derivCoeffs (weightedPsiCoeffs w c) k + weightedPsiCoeffs w c k =
      ∑' n, w n * curvatureCoeffs (c n) k := by
  have h0 := summable_weightedPsiCoeffs hw hw0 hc0 hC k
  have h1 := (summable_weightedDerivCoeffs hw hw0 hc0 hC k).mul_left 2
  have h2 := summable_weightedDerivCoeffs2 hw hw0 hc0 hC k
  rw [derivCoeffs2_weightedPsiCoeffs, derivCoeffs_weightedPsiCoeffs, weightedPsiCoeffs,
    ← tsum_mul_left, ← h2.tsum_sub h1, ← (h2.sub h1).tsum_add h0]
  congr 1
  ext n
  unfold curvatureCoeffs
  ring

/-- A single positive original mode makes every aggregate coefficient positive.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem weightedPsiCoeffs_pos (hw : Summable w) (hw0 : ∀ n, 0 ≤ w n)
    (hc0 : ∀ n, 0 ≤ c n) (hC : ∀ n, c n ≤ C) (i : ℕ)
    (hwi : 0 < w i) (hci : 0 < c i) (k : ℕ) : 0 < weightedPsiCoeffs w c k := by
  apply (summable_weightedPsiCoeffs hw hw0 hc0 hC k).tsum_pos
    (fun n => mul_nonneg (hw0 n) (psiCoeffs_nonneg (hc0 n) k)) i
  exact mul_pos hwi (by unfold psiCoeffs; positivity)

/-- Exponentially decaying weights witness the summability, positivity and bound
hypotheses of every aggregate coefficient result above.
Source: arXiv:2605.09213v1, §6, the summable-weight argument for `ℋ`. -/
example : Summable (fun n : ℕ => Real.exp (-(n : ℝ))) ∧
    (∀ n : ℕ, 0 ≤ Real.exp (-(n : ℝ))) ∧
    (∀ n : ℕ, 0 ≤ (1 : ℝ) / (n + 1)) ∧ (∀ n : ℕ, (1 : ℝ) / (n + 1) ≤ 1) ∧
    0 < Real.exp (-(0 : ℝ)) ∧ 0 < (1 : ℝ) / (0 + 1) := by
  exact ⟨Real.summable_exp_neg_nat, fun _ => (Real.exp_pos _).le,
    fun _ => by positivity, fun n => (div_le_one (by positivity : (0 : ℝ) < n + 1)).mpr
      (by linarith [show (0 : ℝ) ≤ n by positivity]), Real.exp_pos _, by norm_num⟩

end Kinetic
end Transformer

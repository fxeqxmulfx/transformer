module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRetrieval.Basic

/-!
# Complete softmax concentration from learned score gaps

Source: `review/convex/moments.md`, CM111. The path guarantee follows from the
actual denominators and learned score differences. It assumes neither selected
attention destinations nor correct predictions.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRetrieval

open scoped BigOperators
open Perspective

/-- All unselected softmax mass is controlled by actual score differences,
CM111. The factor `n` is a conservative bound on the number of competitors. -/
theorem softmax_error {n : ℕ} (u : Fin n → ℝ) (i : Fin n) (g : ℝ)
    (hg : ∀ j, j ≠ i → g ≤ u i - u j) :
    1 - softmaxWeight u i ≤ n * Real.exp (-g) := by
  have hn : 0 < n := Nat.zero_lt_of_lt i.isLt
  have hs := Finset.sum_erase_add Finset.univ (softmaxWeight u) (Finset.mem_univ i)
  rw [sum_softmaxWeight hn] at hs
  calc
    1 - softmaxWeight u i = ∑ j ∈ Finset.univ.erase i, softmaxWeight u j := by linarith
    _ ≤ ∑ j ∈ Finset.univ.erase i, Real.exp (-g) := by
      apply Finset.sum_le_sum
      intro j hj
      exact (Clusters.softmaxWeight_le_exp_difference u i j).trans
        (Real.exp_le_exp.mpr (by linarith [hg j (Finset.mem_erase.mp hj).1]))
    _ ≤ ∑ _ : Fin n, Real.exp (-g) := Finset.sum_le_sum_of_subset_of_nonneg
      (Finset.erase_subset _ _) (by intro j hj hji; exact (Real.exp_pos _).le)
    _ = n * Real.exp (-g) := by simp

variable {O E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E]

/-- A repeated opaque object shares the exact learned key vector, CM111.
The learned diagonal separation, not a supplied equality feature, gives the gap. -/
theorem repeated_key_gap {n : ℕ} (p : Parameters O E) (x : Fin n → O) (q : O)
    (i : Fin n) (g : ℝ) (hi : x i = q) (ht : 0 ≤ p.scale)
    (hg : ∀ j, j ≠ i → g ≤ (inner ℂ (p.key q) (p.key q)).re -
      (inner ℂ (p.key q) (p.key (x j))).re) :
    ∀ j, j ≠ i → p.scale * g ≤ keyScores p x q i - keyScores p x q j := by
  intro j hj
  simpa only [keyScores, hi, mul_sub] using mul_le_mul_of_nonneg_left (hg j hj) ht

/-- The actual two-stage path mass loses at most the sum of the two
complete softmax errors, CM111. -/
theorem path_error {n : ℕ} (p : Parameters O E) (x : Fin n → O) (q : O)
    (i k : Fin n) (g h : ℝ)
    (hg : ∀ j, j ≠ i → g ≤ keyScores p x q i - keyScores p x q j)
    (hh : ∀ j, j ≠ k → h ≤ relativeScores p i k - relativeScores p i j) :
    1 - pathMass p x q (i, k) ≤ n * (Real.exp (-g) + Real.exp (-h)) := by
  have hn : 0 < n := Nat.zero_lt_of_lt i.isLt
  have ha := softmax_error (keyScores p x q) i g hg
  have hb := softmax_error (relativeScores p i) k h hh
  have ha1 : softmaxWeight (keyScores p x q) i ≤ 1 := by
    rw [← sum_softmaxWeight hn (keyScores p x q)]
    exact Finset.single_le_sum (fun j _ => softmaxWeight_nonneg _ j) (Finset.mem_univ i)
  have hb1 : softmaxWeight (relativeScores p i) k ≤ 1 := by
    rw [← sum_softmaxWeight hn (relativeScores p i)]
    exact Finset.single_le_sum (fun j _ => softmaxWeight_nonneg _ j) (Finset.mem_univ k)
  have hab := mul_nonneg (sub_nonneg.mpr ha1) (sub_nonneg.mpr hb1)
  dsimp only [pathMass]
  nlinarith

example : 1 - softmaxWeight (fun _ : Fin 1 => 0) 0 ≤ (1 : ℝ) * Real.exp (-1) := by
  simpa using softmax_error (fun _ : Fin 1 => 0) 0 1
    (by intro j hj; fin_cases j; exact (hj rfl).elim)

example : ∀ j : Fin 1, j ≠ 0 → (0 : ℝ) * 1 ≤
    keyScores (zeroParameters : Parameters Bool ℂ) (fun _ : Fin 1 => false) false 0 -
      keyScores (zeroParameters : Parameters Bool ℂ) (fun _ => false) false j :=
  repeated_key_gap (zeroParameters : Parameters Bool ℂ) (fun _ : Fin 1 => false)
    false 0 1 rfl (by simp [zeroParameters])
    (by intro j hj; fin_cases j; exact (hj rfl).elim)

example : 1 - pathMass (zeroParameters : Parameters Bool ℂ) (fun _ : Fin 1 => false)
    false (0, 0) ≤ (1 : ℝ) * (Real.exp (-1) + Real.exp (-1)) := by
  simpa using path_error (zeroParameters : Parameters Bool ℂ) (fun _ : Fin 1 => false) false 0 0 1 1
    (by intro j hj; fin_cases j; exact (hj rfl).elim)
    (by intro j hj; fin_cases j; exact (hj rfl).elim)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRetrieval

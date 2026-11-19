module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained.Table

/-!
# Separate signal and state error in complete retained coefficients

Source: `review/convex/moments.md`, CM106. The complete peak includes every
class and component. Historical signals need not be current CE residuals;
their discrepancy and changed-state contribution both remain explicit.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained

open Pair
open scoped BigOperators
variable {J O : Type*} [Fintype O]

/-- Full indexed signal/state error budget, CM106. `d i` bounds the state
change of every component for original occurrence `i`. -/
noncomputable def errorBudget (n : ℕ) (r s : Fin (n + 1) → O → ℝ)
    (d : Fin (n + 1) → ℝ) : ℝ :=
  ((n : ℝ) + 1)⁻¹ * ∑ i, ((∑ c, |r i c - s i c|) + (∑ c, |s i c|) * d i)

/-- A changed signal and a changed actual state have separate complete
contributions; neither is set to zero, CM106. -/
theorem signal_state_error (r s d : ℝ) (z q : State) (hz : ‖z‖ ≤ 1)
    (hd : ‖z - q‖ ≤ d) : ‖r • z - s • q‖ ≤ |r - s| + |s| * d := by
  have he : r • z - s • q = (r - s) • z + s • (z - q) := by
    simp only [sub_smul, smul_sub]
    abel
  calc
    _ ≤ ‖(r - s) • z‖ + ‖s • (z - q)‖ := by rw [he]; exact norm_add_le _ _
    _ = |r - s| * ‖z‖ + |s| * ‖z - q‖ := by rw [norm_smul, norm_smul]; rfl
    _ ≤ |r - s| * 1 + |s| * d :=
      add_le_add (mul_le_mul_of_nonneg_left hz (abs_nonneg _))
        (mul_le_mul_of_nonneg_left hd (abs_nonneg _))
    _ = _ := by ring

omit [Fintype O] in
/-- Summing every occurrence bounds an individual full complex coefficient,
CM106. The denominator remains the original training-record count. -/
theorem row_error (n : ℕ) (z q : Fin (n + 1) → J → State)
    (r s : Fin (n + 1) → O → ℝ) (d : Fin (n + 1) → ℝ)
    (hz : ∀ i j, ‖z i j‖ ≤ 1) (hd : ∀ i j, ‖z i j - q i j‖ ≤ d i) (j : J) (c : O) :
    ‖(mean n z r - mean n q s) j c‖ ≤
      ((n : ℝ) + 1)⁻¹ * ∑ i, (|r i c - s i c| + |s i c| * d i) := by
  have hn : 0 ≤ ((n : ℝ) + 1)⁻¹ := by positivity
  change ‖((n : ℝ) + 1)⁻¹ • (∑ i, r i c • z i j) -
    ((n : ℝ) + 1)⁻¹ • (∑ i, s i c • q i j)‖ ≤ _
  rw [← smul_sub, ← Finset.sum_sub_distrib, norm_smul, Real.norm_eq_abs, abs_of_nonneg hn]
  apply mul_le_mul_of_nonneg_left _ hn
  exact (norm_sum_le _ _).trans (Finset.sum_le_sum (fun i _ =>
    signal_state_error (r i c) (s i c) (d i) (z i j) (q i j) (hz i j) (hd i j)))

/-- Every original class participates in the complete block discrepancy,
CM106. No class or complex coordinate is selected or filtered. -/
theorem block_error (n : ℕ) (z q : Fin (n + 1) → J → State)
    (r s : Fin (n + 1) → O → ℝ) (d : Fin (n + 1) → ℝ)
    (hz : ∀ i j, ‖z i j‖ ≤ 1) (hd : ∀ i j, ‖z i j - q i j‖ ≤ d i) (j : J) :
    blockSupport ((mean n z r - mean n q s) j) ≤ errorBudget n r s d := by
  calc
    _ ≤ ∑ c, ((n : ℝ) + 1)⁻¹ * ∑ i, (|r i c - s i c| + |s i c| * d i) :=
      Finset.sum_le_sum (fun c _ => row_error n z q r s d hz hd j c)
    _ = errorBudget n r s d := by
      rw [← Finset.mul_sum, Finset.sum_comm]
      unfold errorBudget
      congr 1
      apply Finset.sum_congr rfl
      intro i hi
      rw [Finset.sum_add_distrib, ← Finset.sum_mul]

variable [Fintype J] [Nonempty J]

/-- Shared support error is controlled by both indexed errors, CM106.
This includes components whose current native weight is zero. -/
theorem peak_error (n : ℕ) (z q : Fin (n + 1) → J → State)
    (r s : Fin (n + 1) → O → ℝ) (d : Fin (n + 1) → ℝ)
    (hz : ∀ i j, ‖z i j‖ ≤ 1) (hd : ∀ i j, ‖z i j - q i j‖ ≤ d i) :
    peak (mean n z r - mean n q s) ≤ errorBudget n r s d := by
  apply Finset.sup'_le
  intro j hj
  exact block_error n z q r s d hz hd j

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained

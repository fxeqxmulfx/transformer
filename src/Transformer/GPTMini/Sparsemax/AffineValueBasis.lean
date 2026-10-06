import Transformer.GPTMini.Sparsemax.AffineValueProfile
import Transformer.GPTMini.Sparsemax.MatrixOutputError

/-!
# Two generated value features with uniform observation curvature

New compact representation following arXiv:1602.02068v2, Eq. (1) and §2.5.
The dictionary has N+2 slots, while the learned response has only two
coefficients per output channel. Its generated positional feature ranges
from -1 to 1; the two endpoint observations recover both coefficients.
Actual path attention preserves this basis, with eigenvalues 1 and 1-2*c.

Full ordinary observations control coefficient squared distance with
factor two, independent of dictionary size. This is a consequence of
the observation map, not an extra loss penalty or supplied attention label.
The generated feature family restricts representable answers to affine
functions of position. It does not require stored rows of a feature table.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- A generated bounded positional feature; no dictionary-sized basis table is supplied.
Source: the new invariant two-feature chart before sparsemax Eq. (1). -/
def affineValuePosition (N : ℕ) (i : Fin (N + 2)) : ℝ :=
  (2 * i.val - (N + 1)) / (N + 1)

/-- The first slot gives the negative endpoint feature.
Source: the generated positional basis before arXiv:1602.02068v2, Eq. (1). -/
theorem affineValuePosition_zero (N : ℕ) : affineValuePosition N 0 = -1 := by
  unfold affineValuePosition
  simp only [Fin.val_zero, Nat.cast_zero, mul_zero, zero_sub]
  have hn : (N : ℝ) + 1 ≠ 0 := ne_of_gt (Nat.cast_add_one_pos N)
  field_simp

/-- The last slot gives the positive endpoint feature for every dictionary size.
Source: the same generated two-feature chart before sparsemax Eq. (1). -/
theorem affineValuePosition_last (N : ℕ) : affineValuePosition N (Fin.last (N + 1)) = 1 := by
  unfold affineValuePosition
  simp only [Fin.val_last, Nat.cast_add, Nat.cast_one]
  have hn : (N : ℝ) + 1 ≠ 0 := ne_of_gt (Nat.cast_add_one_pos N)
  field_simp
  ring

/-- All generated value positions lie in a size-independent interval.
Source: normalization of the invariant feature before arXiv:1602.02068v2, Eq. (1). -/
theorem affineValuePosition_bounds (N : ℕ) (i : Fin (N + 2)) :
    -1 ≤ affineValuePosition N i ∧ affineValuePosition N i ≤ 1 := by
  have hn : 0 < (N : ℝ) + 1 := Nat.cast_add_one_pos N
  have hi : (i.val : ℝ) ≤ (N : ℝ) + 1 := by exact_mod_cast (by omega : i.val ≤ N + 1)
  have hl : 0 ≤ (i.val : ℝ) := Nat.cast_nonneg i.val
  unfold affineValuePosition
  constructor
  · exact (le_div_iff₀ hn).mpr (by linarith)
  · exact (div_le_iff₀ hn).mpr (by linarith)

/-- The actual score path preserves the centered positional feature.
Source: the new compact-value invariant subspace before sparsemax Eq. (1). -/
theorem affineValuePosition_action (N : ℕ) (c : ℝ) (i : Fin (N + 2)) :
    (∑ j, memoryGramScores (localMemoryCore (affineValueEdges N c)) i j *
      affineValuePosition N j) = (1 - 2 * c) * affineValuePosition N i := by
  calc
    _ = (2 * (∑ j, memoryGramScores (localMemoryCore (affineValueEdges N c)) i j * (j.val : ℝ)) -
        (N + 1) * ∑ j, memoryGramScores (localMemoryCore (affineValueEdges N c)) i j) / (N + 1) := by
      rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib, Finset.sum_div]
      apply Finset.sum_congr rfl
      intro j hj
      unfold affineValuePosition
      ring
    _ = _ := by
      rw [affineValueProfile_index, localMemoryCore_scores_rowSum]
      unfold affineValuePosition
      ring

/-- Exactly two coefficient rows generate all output entries on demand.
Source: the new constant-storage value chart after arXiv:1602.02068v2, Eq. (1). -/
def affineValueOutput (N : ℕ) {D : ℕ} (W : Matrix (Fin 2) (Fin D) ℝ) :
    Matrix (Fin (N + 2)) (Fin D) ℝ :=
  fun i d => W 0 d + affineValuePosition N i * W 1 d

/-- Generated outputs are linear in the complete small coefficient table.
Source: the invariant compact-value chart before sparsemax Eq. (1). -/
theorem affineValueOutput_linear (N : ℕ) {D : ℕ} (W V : Matrix (Fin 2) (Fin D) ℝ) (a b : ℝ) :
    affineValueOutput N (a • W + b • V) = a • affineValueOutput N W + b • affineValueOutput N V := by
  ext i d
  change a * W 0 d + b * V 0 d + affineValuePosition N i * (a * W 1 d + b * V 1 d) = _
  change _ = a * (W 0 d + affineValuePosition N i * W 1 d) +
    b * (V 0 d + affineValuePosition N i * V 1 d)
  ring

/-- Endpoint responses identify every learned coefficient, for arbitrary output dimension.
Source: actual observation injectivity of the compact chart after sparsemax §2.5. -/
theorem affineValueOutput_injective (N D : ℕ) :
    Function.Injective (affineValueOutput N : Matrix (Fin 2) (Fin D) ℝ → _) := by
  intro W V h
  have h0 (d : Fin D) := congrArg (fun Z => Z 0 d) h
  have h1 (d : Fin D) := congrArg (fun Z => Z (Fin.last (N + 1)) d) h
  simp only [affineValueOutput, affineValuePosition_zero, affineValuePosition_last,
    neg_one_mul, one_mul] at h0 h1
  ext k d
  fin_cases k
  · change W 0 d = V 0 d
    linarith [h0 d, h1 d]
  · change W 1 d = V 1 d
    linarith [h0 d, h1 d]

/-- Ordinary observed distance controls all compact coefficients with uniform factor two.
Source: the two distinct endpoint observations after arXiv:1602.02068v2, §2.5. -/
theorem affineValueOutput_distance (N : ℕ) {D : ℕ} (W V : Matrix (Fin 2) (Fin D) ℝ) :
    2 * matrixOutputError V W ≤ matrixOutputError (affineValueOutput N V) (affineValueOutput N W) := by
  classical
  have hn : (0 : Fin (N + 2)) ≠ Fin.last (N + 1) := by
    intro h
    have hv := congrArg Fin.val h
    simp only [Fin.val_zero, Fin.val_last] at hv
    omega
  have hs := Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ {0, Fin.last (N + 1)})
    (f := fun i => ∑ d, (affineValueOutput N W i d - affineValueOutput N V i d) ^ 2)
    (fun i hi hnot => Finset.sum_nonneg (fun d hd => sq_nonneg _))
  rw [Finset.sum_pair hn] at hs
  simp only [affineValueOutput, affineValuePosition_zero, affineValuePosition_last,
    neg_one_mul, one_mul] at hs
  have he : (∑ d, (W 0 d + -W 1 d - (V 0 d + -V 1 d)) ^ 2) +
      (∑ d, (W 0 d + W 1 d - (V 0 d + V 1 d)) ^ 2) = 2 * matrixOutputError V W := by
    unfold matrixOutputError
    rw [Fin.sum_univ_two, mul_add, Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib,
      ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro d hd
    ring
  rw [he] at hs
  exact hs

/-- Any dictionary size uses exactly two learned value coefficients per output channel.
Source: the explicit compact parameter type following sparsemax Eq. (1). -/
theorem affineValueCoefficient_count (D : ℕ) : Fintype.card (Fin 2 × Fin D) = 2 * D := by
  rw [Fintype.card_prod, Fintype.card_fin, Fintype.card_fin]

/-- Four-slot outputs already vary beyond the two learned coefficient rows. -/
example : affineValueOutput 2 (Matrix.of (fun k : Fin 2 => fun _ : Fin 1 => (k.val : ℝ)))
    (1 : Fin 4) 0 = -1 / 3 := by
  norm_num [affineValueOutput, affineValuePosition, Matrix.of_apply]

/-- A thousand generated positions retain the same uniform feature bound. -/
example : ∀ i : Fin 1000, -1 ≤ affineValuePosition 998 i ∧ affineValuePosition 998 i ≤ 1 :=
  affineValuePosition_bounds 998

end Transformer.GPTMini.Sparsemax

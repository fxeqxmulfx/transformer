/-
# AdaFisher: min-max normalization

arXiv:2405.16397v3, §3.2, Proposition 3.2 and Appendix A.2, Part 2.
The source omits the constant-diagonal case. We extend that case by zero;
on a nonconstant vector this is precisely the printed affine formula.
-/

import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Tactic

open scoped BigOperators

noncomputable section

namespace Transformer.AdaFisher

variable {n : ℕ} [NeZero n]

/-- Minimum of a nonempty diagonal, Proposition 3.2, Appendix A.2. -/
def diagonalMin (x : Fin n → ℝ) : ℝ := Finset.univ.inf' Finset.univ_nonempty x

/-- Maximum of a nonempty diagonal, Proposition 3.2, Appendix A.2. -/
def diagonalMax (x : Fin n → ℝ) : ℝ := Finset.univ.sup' Finset.univ_nonempty x

/-- Min-max scaling in Proposition 3.2, Appendix A.2. The missing domain
case `max = min` is explicitly mapped to zero, rather than divided by zero. -/
def minMaxDiagonal (x : Fin n → ℝ) (i : Fin n) : ℝ :=
  if diagonalMin x < diagonalMax x then
    (x i - diagonalMin x) / (diagonalMax x - diagonalMin x) else 0

/-- Every diagonal entry is at least the minimum, Appendix A.2, Part 2. -/
theorem diagonalMin_le (x : Fin n → ℝ) (i : Fin n) : diagonalMin x ≤ x i :=
  Finset.inf'_le x (Finset.mem_univ i)

/-- Every diagonal entry is at most the maximum, Appendix A.2, Part 2. -/
theorem le_diagonalMax (x : Fin n → ℝ) (i : Fin n) : x i ≤ diagonalMax x :=
  Finset.le_sup' x (Finset.mem_univ i)

/-- Normalized diagonal entries lie in `[0,1]`, Proposition 3.2.
This includes the explicitly documented constant-vector extension. -/
theorem minMaxDiagonal_bounds (x : Fin n → ℝ) (i : Fin n) :
    0 ≤ minMaxDiagonal x i ∧ minMaxDiagonal x i ≤ 1 := by
  unfold minMaxDiagonal
  split_ifs with h
  · have hd : 0 < diagonalMax x - diagonalMin x := sub_pos.mpr h
    constructor
    · exact div_nonneg (sub_nonneg.mpr (diagonalMin_le x i)) hd.le
    · apply (div_le_one hd).mpr
      linarith [le_diagonalMax x i]
  · norm_num

/-- With a nonzero range the implementation equals the paper's formula,
Proposition 3.2, Appendix A.2. The source needs this domain hypothesis. -/
theorem minMaxDiagonal_eq (x : Fin n → ℝ)
    (h : diagonalMin x < diagonalMax x) (i : Fin n) :
    minMaxDiagonal x i = (x i - diagonalMin x) / (diagonalMax x - diagonalMin x) := by
  simp [minMaxDiagonal, h]

example : diagonalMin (fun i : Fin 2 => (i : ℝ)) <
    diagonalMax (fun i : Fin 2 => (i : ℝ)) := by
  have h0 := diagonalMin_le (fun i : Fin 2 => (i : ℝ)) 0
  have h1 := le_diagonalMax (fun i : Fin 2 => (i : ℝ)) 1
  norm_num at h0 h1
  linarith

/-- The literal formula's denominator vanishes for constant KFs,
Appendix A.2, Part 2. In particular this occurs for the identity fallback
of Appendix A.3 and for every one-dimensional factor. -/
theorem constant_diagonal_range (r : ℝ) :
    diagonalMin (fun _ : Fin n => r) = r ∧
    diagonalMax (fun _ : Fin n => r) = r := by
  simp [diagonalMin, diagonalMax]

/-- Constant KFs produce zero normalized factors under our documented
extension of Proposition 3.2, Appendix A.2. -/
theorem minMaxDiagonal_constant (r : ℝ) (i : Fin n) :
    minMaxDiagonal (fun _ : Fin n => r) i = 0 := by
  obtain ⟨hmin, hmax⟩ := constant_diagonal_range (n := n) r
  simp [minMaxDiagonal, hmin, hmax]

/-- Min-max normalization preserves entry ordering on a nonconstant
diagonal, Proposition 3.2, Appendix A.2, “relative magnitudes”. -/
theorem minMaxDiagonal_le_iff (x : Fin n → ℝ)
    (h : diagonalMin x < diagonalMax x) (i j : Fin n) :
    minMaxDiagonal x i ≤ minMaxDiagonal x j ↔ x i ≤ x j := by
  rw [minMaxDiagonal_eq x h i, minMaxDiagonal_eq x h j,
    div_le_div_iff_of_pos_right (sub_pos.mpr h)]
  exact sub_le_sub_iff_right _

example : diagonalMin (fun i : Fin 2 => (i : ℝ)) <
    diagonalMax (fun i : Fin 2 => (i : ℝ)) := by
  have h0 := diagonalMin_le (fun i : Fin 2 => (i : ℝ)) 0
  have h1 := le_diagonalMax (fun i : Fin 2 => (i : ℝ)) 1
  norm_num at h0 h1
  linarith

/-- Distances are scaled by the reciprocal range, not preserved as
absolute distances, Proposition 3.2, Appendix A.2, Part 2. -/
theorem minMaxDiagonal_sub (x : Fin n → ℝ)
    (h : diagonalMin x < diagonalMax x) (i j : Fin n) :
    minMaxDiagonal x i - minMaxDiagonal x j =
      (x i - x j) / (diagonalMax x - diagonalMin x) := by
  rw [minMaxDiagonal_eq x h i, minMaxDiagonal_eq x h j]
  ring

example : diagonalMin (fun i : Fin 2 => (i : ℝ)) <
    diagonalMax (fun i : Fin 2 => (i : ℝ)) := by
  have h0 := diagonalMin_le (fun i : Fin 2 => (i : ℝ)) 0
  have h1 := le_diagonalMax (fun i : Fin 2 => (i : ℝ)) 1
  norm_num at h0 h1
  linarith

end Transformer.AdaFisher

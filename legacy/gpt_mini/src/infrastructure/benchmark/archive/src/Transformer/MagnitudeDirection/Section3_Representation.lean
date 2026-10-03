/-
# Representational scope of the gains

arXiv:2606.25971v2, §3.1, “do not provide any additional representational
capacity”, and Appendix A. Positive gains on a positive fixed sphere
represent every nonzero matrix, but never the zero matrix.
-/

import Transformer.MagnitudeDirection.Section3_Sphere

noncomputable section

namespace Transformer.MagnitudeDirection

variable {m n : ℕ}

/-- Every nonzero weight has a fixed-radius direction and positive scalar
gain (a special case of the row/column model). Zero is excluded because
strictly positive gains on a positive sphere cannot represent it.
Source: arXiv:2606.25971v2, §3.1, “Updating the magnitude”. -/
theorem fixed_sphere_representation (W : Matrix (Fin m) (Fin n) ℝ) (c : ℝ)
    (hW : W ≠ 0) (hc : 0 < c) :
    ∃ (D : Matrix (Fin m) (Fin n) ℝ) (gamma : ℝ),
      frobeniusNorm D = c ∧ 0 < gamma ∧ fuse (fun _ => gamma) (fun _ => 1) D = W := by
  have hn := (frobeniusNorm_pos_iff W).mpr hW
  refine ⟨matrixProject c W, frobeniusNorm W / c,
    matrixProject_norm c W hc.le hW, div_pos hn hc, ?_⟩
  ext i j
  dsimp [fuse, matrixProject]
  field_simp

/-- Nonzero weights and positive sphere radii exist, arXiv:2606.25971v2, §3.1. -/
example : (1 : Matrix (Fin 1) (Fin 1) ℝ) ≠ 0 ∧ (0 : ℝ) < 1 := by norm_num

/-- With nonzero gains, a fused matrix vanishes exactly when its direction
vanishes. Source: arXiv:2606.25971v2, §3 and Appendix A. -/
theorem fuse_zero_iff (row : Fin m → ℝ) (col : Fin n → ℝ)
    (D : Matrix (Fin m) (Fin n) ℝ) (hr : ∀ i, row i ≠ 0) (hc : ∀ j, col j ≠ 0) :
    fuse row col D = 0 ↔ D = 0 := by
  constructor
  · intro h
    have := unfuse_fuse row col D hr hc
    rw [h] at this
    calc
      D = unfuse row col 0 := this.symm
      _ = 0 := by ext i j; simp [unfuse]
  · intro h
    ext i j
    simp [fuse, h]

/-- The nonzero-gain hypotheses hold, arXiv:2606.25971v2, §3. -/
example : (∀ i : Fin 1, (fun _ => (1 : ℝ)) i ≠ 0) ∧
    (∀ j : Fin 1, (fun _ => (1 : ℝ)) j ≠ 0) := by norm_num

/-- Strictly positive gains on a positive sphere exclude the zero weight.
This qualifies full representational equivalence with an unconstrained matrix.
Source: arXiv:2606.25971v2, §3.1 and §4.1.3. -/
theorem positive_sphere_fuse_ne_zero (row : Fin m → ℝ) (col : Fin n → ℝ)
    (D : Matrix (Fin m) (Fin n) ℝ) (c : ℝ)
    (hr : ∀ i, 0 < row i) (hc : ∀ j, 0 < col j)
    (hD : frobeniusNorm D = c) (hpos : 0 < c) : fuse row col D ≠ 0 := by
  intro h
  have hz := (fuse_zero_iff row col D (fun i => (hr i).ne') (fun j => (hc j).ne')).mp h
  have hn := (frobeniusNorm_pos_iff D).mp (hD ▸ hpos)
  exact hn hz

/-- All positive-sphere hypotheses hold, arXiv:2606.25971v2, §3.1. -/
example : (∀ i : Fin 1, (fun _ => (1 : ℝ)) i > 0) ∧
    (∀ j : Fin 1, (fun _ => (1 : ℝ)) j > 0) ∧
    frobeniusNorm (1 : Matrix (Fin 1) (Fin 1) ℝ) = 1 ∧ (0 : ℝ) < 1 := by
  have hn := frobeniusNorm_sq (1 : Matrix (Fin 1) (Fin 1) ℝ)
  norm_num only [Fin.sum_univ_one, Matrix.one_apply, ite_true, one_pow] at hn
  have hnonneg : 0 ≤ frobeniusNorm (1 : Matrix (Fin 1) (Fin 1) ℝ) := norm_nonneg _
  refine ⟨by norm_num, by norm_num, ?_, by norm_num⟩
  nlinarith

/-- Row and column gains are not individually identifiable: reciprocal
rescaling preserves every fused entry. Source: arXiv:2606.25971v2, §3,
the displayed factorization. -/
theorem reciprocal_gain_rescaling (row : Fin m → ℝ) (col : Fin n → ℝ)
    (D : Matrix (Fin m) (Fin n) ℝ) (a : ℝ) (ha : a ≠ 0) :
    fuse (fun i => a * row i) (fun j => col j / a) D = fuse row col D := by
  ext i j
  dsimp [fuse]
  field_simp

/-- Reciprocal rescaling admits a nonzero factor, arXiv:2606.25971v2, §3. -/
example : (2 : ℝ) ≠ 0 := by norm_num

/-- Fine-grained gains alter direction as well as scale: a column gain of
`(1,2)` on `(1,1)` cannot be reproduced by any overall scalar.
Source: arXiv:2606.25971v2, §3, “Fine-grained scales”. -/
theorem fine_grained_gain_changes_direction :
    ¬ ∃ a : ℝ, fuse (fun _ : Fin 1 => 1) ![1, 2] (fun _ _ => 1) =
      a • (fun _ _ => 1 : Matrix (Fin 1) (Fin 2) ℝ) := by
  rintro ⟨a, h⟩
  have h0 := congrArg (fun W : Matrix (Fin 1) (Fin 2) ℝ => W 0 0) h
  have h1 := congrArg (fun W : Matrix (Fin 1) (Fin 2) ℝ => W 0 1) h
  norm_num [fuse] at h0 h1
  linarith

/-- With fixed anisotropic gains, rotating a direction on the same sphere
can change the fused magnitude: the two unit basis directions give norms
one and two under the same column gains `(1,2)`.
Thus §3.1's “magnitude of W is determined by the gains” is exact for a
scalar gain, but for row/column gains the direction also matters.
Source: arXiv:2606.25971v2, §3.1, “Updating the magnitude”. -/
theorem fixed_gains_can_change_magnitude :
    let D : Matrix (Fin 1) (Fin 2) ℝ := !![1, 0]
    let E : Matrix (Fin 1) (Fin 2) ℝ := !![0, 1]
    frobeniusNorm D = 1 ∧ frobeniusNorm E = 1 ∧
      frobeniusNorm (fuse (fun _ => 1) ![1, 2] D) = 1 ∧
      frobeniusNorm (fuse (fun _ => 1) ![1, 2] E) = 2 := by
  norm_num [frobeniusNorm_eq_sqrt, fuse, Fin.sum_univ_one, Fin.sum_univ_two]

end Transformer.MagnitudeDirection

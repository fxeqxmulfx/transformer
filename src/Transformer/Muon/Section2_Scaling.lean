/-
# Muon — shape-dependent and direct RMS scaling

arXiv:2502.16982, §2.2 and §3.1. Shape scaling cancels the dimension factor
for an exact full-rank polar update. Direct normalization cancels the
actual RMS, provided that RMS is positive; the zero-update exception is
made explicit rather than hidden in division.
-/

import Transformer.Muon.AppendixA_RMS

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.Muon

variable {a b : ℕ}

/-- RMS is nonnegative, arXiv:2502.16982, Appendix A. -/
theorem matrixRMS_nonneg (X : Matrix (Fin a) (Fin b) ℝ) : 0 ≤ matrixRMS X :=
  Real.sqrt_nonneg _

/-- Shape scaling makes the exact full-rank update RMS one,
arXiv:2502.16982, §2.2, “Consistent update RMS”. -/
theorem shape_scaled_rms (U : Matrix (Fin a) (Fin (min a b)) ℝ)
    (V : Matrix (Fin b) (Fin (min a b)) ℝ) (ha : 0 < a) (hb : 0 < b)
    (hU : OrthonormalColumns U) (hV : OrthonormalColumns V) :
    matrixRMS (shapeScale a b • polarUpdate U V) = 1 := by
  have hd : 0 < max (a : ℝ) b := lt_of_lt_of_le (by exact_mod_cast ha) (le_max_left _ _)
  rw [matrixRMS_smul, shapeScale, abs_of_nonneg (Real.sqrt_nonneg _),
    fullRank_update_rms U V ha hb hU hV]
  change Real.sqrt (max (a : ℝ) b) * Real.sqrt (1 / max (a : ℝ) b) = 1
  rw [← Real.sqrt_mul hd.le]
  simp [hd.ne']

/-- The shape-scaling hypotheses are satisfiable, arXiv:2502.16982, §2.2. -/
example : 0 < (1 : ℕ) ∧ 0 < (1 : ℕ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  norm_num [OrthonormalColumns]

/-- The adjusted update has theoretical RMS `0.2`, arXiv:2502.16982, §2.2,
“Matching update RMS of AdamW”, for exact full-rank orthogonalization. -/
theorem adjustedUpdate_rms (U : Matrix (Fin a) (Fin (min a b)) ℝ)
    (V : Matrix (Fin b) (Fin (min a b)) ℝ) (ha : 0 < a) (hb : 0 < b)
    (hU : OrthonormalColumns U) (hV : OrthonormalColumns V) :
    matrixRMS (adjustedUpdate (polarUpdate U V)) = 1 / 5 := by
  rw [adjustedUpdate, ← smul_smul, matrixRMS_smul, shape_scaled_rms U V ha hb hU hV]
  norm_num

/-- The adjusted-update hypotheses are satisfiable, arXiv:2502.16982, §2.2. -/
example : 0 < (1 : ℕ) ∧ 0 < (1 : ℕ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  norm_num [OrthonormalColumns]

/-- “Update Norm” has exactly RMS `0.2` when the input RMS is positive.
The source omits this necessary condition: a zero update stays zero.
Source: arXiv:2502.16982, §3.1, “Update Norm”. -/
theorem normalizedUpdate_rms (O : Matrix (Fin a) (Fin b) ℝ) (hO : 0 < matrixRMS O) :
    matrixRMS (normalizedUpdate O) = 1 / 5 := by
  rw [normalizedUpdate, matrixRMS_smul, abs_of_pos (by positivity)]
  field_simp

/-- A positive-RMS input exists, arXiv:2502.16982, §3.1. -/
example : 0 < matrixRMS (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  norm_num [matrixRMS, squaredFrobenius]

/-- The zero matrix refutes unconditional exact RMS `0.2` for “Update Norm”.
Source: arXiv:2502.16982, §3.1, “Update Norm”. -/
theorem normalizedUpdate_zero_rms :
    matrixRMS (normalizedUpdate (0 : Matrix (Fin 1) (Fin 1) ℝ)) ≠ 1 / 5 := by
  norm_num [normalizedUpdate_zero, matrixRMS, squaredFrobenius]

/-- The original Muon shape factor differs from the proposed factor by the
global multiplier `1/√b`, when all matrices share their second dimension.
Source: arXiv:2502.16982, §2.2, footnote on `√max(1,A/B)`. -/
theorem original_shape_scale (a b : ℕ) (hb : 0 < b) :
    Real.sqrt (max 1 ((a : ℝ) / b)) * Real.sqrt b = shapeScale a b := by
  have hb' : 0 < (b : ℝ) := by exact_mod_cast hb
  rw [← Real.sqrt_mul (by positivity : 0 ≤ max 1 ((a : ℝ) / b))]
  unfold shapeScale
  congr 1
  rcases le_total (a : ℝ) b with h | h
  · rw [max_eq_left ((div_le_one hb').mpr h), one_mul, max_eq_right h]
  · rw [max_eq_right ((one_le_div hb').mpr h), max_eq_left h]
    field_simp

/-- A common positive second dimension exists, arXiv:2502.16982, §2.2. -/
example : 0 < (1 : ℕ) := by norm_num

/-- The `[H,4H]` MLP adjustment is twice the baseline factor,
arXiv:2502.16982, §3.1, analysis of Table 1. -/
theorem mlp_shape_ratio (H : ℝ) (hH : 0 < H) :
    Real.sqrt (max H (4 * H)) / Real.sqrt H = 2 := by
  rw [max_eq_right (by linarith), Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 4)]
  norm_num
  field_simp

/-- Positive hidden dimensions exist, arXiv:2502.16982, §3.1. -/
example : (0 : ℝ) < 1 := by norm_num

/-- The `[H,H]` query adjustment equals the baseline factor,
arXiv:2502.16982, §3.1, analysis of Table 1. -/
theorem query_shape_ratio (H : ℝ) (hH : 0 < H) :
    Real.sqrt (max H H) / Real.sqrt H = 1 := by
  simp [max_self, (Real.sqrt_pos.mpr hH).ne']

/-- Positive hidden dimensions exist, arXiv:2502.16982, §3.1. -/
example : (0 : ℝ) < 1 := by norm_num

end Transformer.Muon

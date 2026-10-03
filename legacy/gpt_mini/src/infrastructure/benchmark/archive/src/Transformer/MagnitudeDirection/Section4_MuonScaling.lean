/-
# Muon update scale and the sphere radius

arXiv:2606.25971v2, §4.1.1 and Appendix D. RMS matching uses an
exact full-rank polar factor. Finite Newton--Schulz iterations provide
only approximate orthogonalization, not an exact norm identity.
-/

import Transformer.MagnitudeDirection.Section3_Factorization
import Transformer.Muon.AppendixA_RMS

open scoped BigOperators

noncomputable section

namespace Transformer.MagnitudeDirection

variable {m n : ℕ}

/-- The printed “shape up” factor, arXiv:2606.25971v2, §4.1.1 and Appendix D. -/
def muonShapeFactor (m n : ℕ) : ℝ :=
  Real.sqrt (max ((m : ℝ) / n) ((n : ℝ) / m))

/-- The two orientations equal `sqrt(max/min)` for positive dimensions,
arXiv:2606.25971v2, §4.1.1 and Appendix D. -/
theorem muonShapeFactor_eq (hm : 0 < m) (hn : 0 < n) :
    muonShapeFactor m n = Real.sqrt (max (m : ℝ) n / min (m : ℝ) n) := by
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  unfold muonShapeFactor
  rcases le_total (m : ℝ) n with h | h
  · rw [max_eq_right ((div_le_div_iff₀ hn' hm').mpr (by nlinarith)),
      max_eq_right h, min_eq_left h]
  · rw [max_eq_left ((div_le_div_iff₀ hm' hn').mpr (by nlinarith)),
      max_eq_left h, min_eq_right h]

/-- Positive matrix shapes exist, arXiv:2606.25971v2, §4.1.1. -/
example : 0 < (1 : ℕ) ∧ 0 < (2 : ℕ) := by norm_num

/-- Frobenius norm of an exact rank-`r` polar factor, arXiv:2606.25971v2,
§4.1.1, “assuming proper orthogonalization”. The SVD frames explicitly
carry orthonormal columns; this does not assume full-rank weights. -/
theorem polar_frobeniusNorm {r : ℕ} (U : Matrix (Fin m) (Fin r) ℝ)
    (V : Matrix (Fin n) (Fin r) ℝ) (hU : Muon.OrthonormalColumns U)
    (hV : Muon.OrthonormalColumns V) :
    frobeniusNorm (Muon.polarUpdate U V) = Real.sqrt r := by
  have hs : frobeniusNorm (Muon.polarUpdate U V) ^ 2 = r := by
    rw [frobeniusNorm_sq]
    exact Muon.squaredFrobenius_polarUpdate U V hU hV
  rw [← hs]
  exact (Real.sqrt_sq (show 0 ≤ frobeniusNorm (Muon.polarUpdate U V) from norm_nonneg _)).symm

/-- Exact polar frames exist, arXiv:2606.25971v2, §4.1.1. -/
example : Muon.OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    Muon.OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  norm_num [Muon.OrthonormalColumns]

/-- The printed shape factor makes a full-rank exact polar update match
the nominal sphere radius `sqrt(max(m,n))`.
Source: arXiv:2606.25971v2, §4.1.1 and Appendix D, Muon scale factor. -/
theorem shape_scaled_polar_norm (U : Matrix (Fin m) (Fin (min m n)) ℝ)
    (V : Matrix (Fin n) (Fin (min m n)) ℝ) (hm : 0 < m) (hn : 0 < n)
    (hU : Muon.OrthonormalColumns U) (hV : Muon.OrthonormalColumns V) :
    frobeniusNorm (muonShapeFactor m n • Muon.polarUpdate U V) =
      Real.sqrt (max (m : ℝ) n) := by
  have hmin : (0 : ℝ) < min (m : ℝ) n := by
    exact lt_min (by exact_mod_cast hm) (by exact_mod_cast hn)
  rw [frobeniusNorm_smul, muonShapeFactor_eq hm hn,
    abs_of_nonneg (Real.sqrt_nonneg _), polar_frobeniusNorm U V hU hV]
  have hcast : ((min m n : ℕ) : ℝ) = min (m : ℝ) n := by simp
  rw [hcast, ← Real.sqrt_mul (by positivity), div_mul_cancel₀ _ hmin.ne']

/-- Full-rank positive shapes admit valid polar frames,
arXiv:2606.25971v2, §4.1.1. -/
example : 0 < (1 : ℕ) ∧ 0 < (1 : ℕ) ∧
    Muon.OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    Muon.OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  norm_num [Muon.OrthonormalColumns]

/-- General second-moment radius under variance `1/d` initialization,
arXiv:2606.25971v2, Appendix D. It is the square root of expected squared
Frobenius norm, not the realized norm of an arbitrary random sample. -/
def initializationRadius (m n d : ℕ) : ℝ := Real.sqrt ((m : ℝ) * n / d)

/-- The abbreviated `sqrt(max)` target is valid precisely in the intended
`min(m,n)=d` shape regime (with positive dimensions).
Source: arXiv:2606.25971v2, §4.1.1, qualified explicitly in Appendix D. -/
theorem initializationRadius_eq_target (d : ℕ) (hm : 0 < m) (hn : 0 < n)
    (hd : min m n = d) : initializationRadius m n d = Real.sqrt (max (m : ℝ) n) := by
  unfold initializationRadius
  rw [← hd]
  have hm' : (m : ℝ) ≠ 0 := by positivity
  have hn' : (n : ℝ) ≠ 0 := by positivity
  rcases le_total m n with h | h
  · rw [min_eq_left h, max_eq_right (by exact_mod_cast h)]
    congr 1
    field_simp
  · rw [min_eq_right h, max_eq_left (by exact_mod_cast h)]
    congr 1
    field_simp

/-- The intended hidden-size regime is inhabited, arXiv:2606.25971v2,
Appendix D. -/
example : 0 < (1 : ℕ) ∧ 0 < (2 : ℕ) ∧ min (1 : ℕ) 2 = 1 := by norm_num

/-- The shape qualification in Appendix D is necessary as well as sufficient:
for positive dimensions and hidden size the general second-moment radius
equals `sqrt(max(m,n))` exactly when `min(m,n)=d`.
Source: arXiv:2606.25971v2, Appendix D, “exact only when”. -/
theorem initializationRadius_eq_target_iff (d : ℕ) (hm : 0 < m) (hn : 0 < n)
    (hd : 0 < d) :
    initializationRadius m n d = Real.sqrt (max (m : ℝ) n) ↔ min m n = d := by
  constructor
  · intro h
    have hm' : (m : ℝ) ≠ 0 := by positivity
    have hn' : (n : ℝ) ≠ 0 := by positivity
    have hd' : (d : ℝ) ≠ 0 := by positivity
    have he := (Real.sqrt_inj (by positivity) (by positivity)).mp h
    rw [div_eq_iff hd'] at he
    rcases le_total m n with hmn | hnm
    · rw [max_eq_right (by exact_mod_cast hmn)] at he
      rw [min_eq_left hmn]
      have hc : (m : ℝ) = d := mul_right_cancel₀ hn' (by nlinarith [he])
      exact_mod_cast hc
    · rw [max_eq_left (by exact_mod_cast hnm)] at he
      rw [min_eq_right hnm]
      have hc : (n : ℝ) = d := mul_left_cancel₀ hm' he
      exact_mod_cast hc
  · exact initializationRadius_eq_target d hm hn

/-- Positive shape and hidden-size assumptions hold,
arXiv:2606.25971v2, Appendix D. -/
example : 0 < (1 : ℕ) ∧ 0 < (2 : ℕ) ∧ 0 < (1 : ℕ) := by norm_num

/-- A key/value-shaped matrix with half as many outputs has nominal radius
one, whereas the unqualified `sqrt(max)` recipe gives `sqrt(2)>1`.
Source: arXiv:2606.25971v2, Appendix D, GQA exception. -/
theorem gqa_radius_counterexample : initializationRadius 1 2 2 = 1 ∧
    1 < Real.sqrt (max (1 : ℝ) 2) := by
  constructor
  · norm_num [initializationRadius]
  · rw [max_eq_right (by norm_num : (1 : ℝ) ≤ 2)]
    nlinarith [Real.sq_sqrt (show (0 : ℝ) ≤ 2 by norm_num), Real.sqrt_nonneg 2]

end Transformer.MagnitudeDirection

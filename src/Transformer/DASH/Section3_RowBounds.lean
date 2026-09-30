/-
# DASH — inexpensive spectral certificates for guarded scaling

arXiv:2602.02016v2, §3.4. The manuscript treats twice a Rayleigh
estimate as a guaranteed upper bound. The correction checks it against
an independently computed upper bound: the smaller of the Frobenius
norm and the maximum absolute row sum. Neither requires an EVD.
-/

import Transformer.DASH.Section3_PowerCoordinates
import Transformer.DASH.Section3_Scaling

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- Maximum absolute row sum, with value zero for an empty matrix.
The norm is the finite vector supremum norm of the nonnegative row sums.
Source: arXiv:2602.02016v2, §3.4, corrected PI scaling certificate. -/
def absoluteRowBound (A : Matrix (Fin n) (Fin n) ℝ) : ℝ :=
  ‖fun i : Fin n => ∑ j : Fin n, |A i j|‖

/-- The row-sum bound controls matrix-vector multiplication in the
vector supremum norm. Source: arXiv:2602.02016v2, §3.4, the correction
that independently certifies the PI-based scaling factor. -/
theorem absoluteRowBound_mulVec (A : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) :
    ‖A *ᵥ x‖ ≤ absoluteRowBound A * ‖x‖ := by
  apply (pi_norm_le_iff_of_nonneg (mul_nonneg (norm_nonneg _) (norm_nonneg _))).mpr
  intro i
  change |∑ j, A i j * x j| ≤ _
  have hrow : (∑ j, |A i j|) ≤ absoluteRowBound A := by
    have hsum : 0 ≤ ∑ j : Fin n, |A i j| :=
      Finset.sum_nonneg fun j _ => abs_nonneg (A i j)
    simpa only [Real.norm_eq_abs, abs_of_nonneg hsum, absoluteRowBound]
      using norm_le_pi_norm (fun k : Fin n => ∑ j : Fin n, |A k j|) i
  calc
    _ ≤ ∑ j, |A i j * x j| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j, |A i j| * ‖x‖ := by
      apply Finset.sum_le_sum
      intro j hj
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left
        (by simpa only [Real.norm_eq_abs] using norm_le_pi_norm x j) (abs_nonneg _)
    _ = (∑ j, |A i j|) * ‖x‖ := (Finset.sum_mul _ _ _).symm
    _ ≤ absoluteRowBound A * ‖x‖ := mul_le_mul_of_nonneg_right hrow (norm_nonneg _)

/-- Every absolute eigenvalue is bounded by the maximum absolute row
sum. This certificate is valid even when the PI pool misses the largest
eigenspace. Source: arXiv:2602.02016v2, §3.4–3.5, repaired scaling guarantee. -/
theorem eigenvalue_le_absoluteRowBound (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (hQ : Orthogonal Q) (i : Fin n) :
    |s i| ≤ absoluteRowBound (spectralMatrix Q s) := by
  let z : Fin n → ℝ := Pi.single i 1
  have hz : z ≠ 0 := by
    intro h
    have := congrFun h i
    simp [z] at this
  have hv := orthogonal_mulVec_ne_zero Q z hQ hz
  have heigen : spectralMatrix Q s *ᵥ (Q *ᵥ z) = s i • (Q *ᵥ z) := by
    rw [spectralMatrix_mulVec Q s z hQ, ← Matrix.mulVec_smul]
    congr 1
    funext j
    by_cases h : j = i <;> simp [z, h]
  have h := absoluteRowBound_mulVec (spectralMatrix Q s) (Q *ᵥ z)
  rw [heigen, norm_smul, Real.norm_eq_abs] at h
  exact le_of_mul_le_mul_right h (norm_pos_iff.mpr hv)

/-- Orthogonal spectral data for the row certificate exist,
arXiv:2602.02016v2, §3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- Use whichever inexpensive bound is tighter. This is an upper bound,
whereas the PI Rayleigh estimate is generally a lower bound.
Source: arXiv:2602.02016v2, §3.4, corrected matrix-scaling procedure. -/
def certifiedSpectralBound (A : Matrix (Fin n) (Fin n) ℝ) : ℝ :=
  min (frobeniusNorm A) (absoluteRowBound A)

/-- The computed certificate is nonnegative, including at zero input.
Source: arXiv:2602.02016v2, §3.4, guarded normalization. -/
theorem certifiedSpectralBound_nonneg (A : Matrix (Fin n) (Fin n) ℝ) :
    0 ≤ certifiedSpectralBound A :=
  le_min (Real.sqrt_nonneg _) (norm_nonneg _)

/-- The actual certificate bounds every absolute eigenvalue; callers
do not supply a hypothesis that a PI estimate is accurate.
Source: arXiv:2602.02016v2, §3.4, repairing the unconditional factor-two claim. -/
theorem eigenvalue_le_certifiedSpectralBound (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (hQ : Orthogonal Q) (i : Fin n) :
    |s i| ≤ certifiedSpectralBound (spectralMatrix Q s) :=
  le_min (eigenvalue_le_frobenius Q s hQ i) (eigenvalue_le_absoluteRowBound Q s hQ i)

/-- The combined certificate admits an orthogonal frame,
arXiv:2602.02016v2, §3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

end Transformer.DASH

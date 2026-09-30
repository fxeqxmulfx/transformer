/-
# DASH — spectral bounds and inverse-root rescaling

arXiv:2602.02016v2, §3.4. Frobenius scaling bounds the entire
spectrum. The scaling factor must also be removed from the output.
-/

import Transformer.DASH.Section3_Spectral

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- Frobenius norm bounds the absolute value of each eigenvalue.
Source: arXiv:2602.02016v2, §3.4, “upper bound on λ_max”. -/
theorem eigenvalue_le_frobenius (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (hQ : Orthogonal Q) (i : Fin n) :
    |s i| ≤ frobeniusNorm (spectralMatrix Q s) := by
  rw [frobeniusNorm, spectralMatrix_energy Q s hQ]
  apply (Real.le_sqrt (abs_nonneg _) (Finset.sum_nonneg fun j _ => sq_nonneg (s j))).mpr
  simpa only [sq_abs] using
    Finset.single_le_sum (fun j _ => sq_nonneg (s j)) (Finset.mem_univ i)

/-- Orthogonal spectral data exist, arXiv:2602.02016v2, §3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- The worst Frobenius/maximum-eigenvalue gap is √n for a nonnegative
spectrum bounded by μ. The source's factors 10–100 are observations,
not bounds independent of matrix dimension.
Source: arXiv:2602.02016v2, §3.4. -/
theorem frobenius_le_dimension_bound (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (μ : ℝ) (hQ : Orthogonal Q)
    (hs : ∀ i, 0 ≤ s i) (hupper : ∀ i, s i ≤ μ) (hμ : 0 ≤ μ) :
    frobeniusNorm (spectralMatrix Q s) ≤ Real.sqrt (n : ℝ) * μ := by
  rw [frobeniusNorm, spectralMatrix_energy Q s hQ]
  have hsum : (∑ i, s i ^ 2) ≤ (n : ℝ) * μ ^ 2 := by
    calc
      (∑ i, s i ^ 2) ≤ ∑ _ : Fin n, μ ^ 2 :=
        Finset.sum_le_sum fun i _ => pow_le_pow_left₀ (hs i) (hupper i) 2
      _ = (n : ℝ) * μ ^ 2 := by simp
  calc
    Real.sqrt (∑ i, s i ^ 2) ≤ Real.sqrt ((n : ℝ) * μ ^ 2) := Real.sqrt_le_sqrt hsum
    _ = Real.sqrt (n : ℝ) * μ := by rw [Real.sqrt_mul (by positivity), Real.sqrt_sq hμ]

/-- Dimension-bound hypotheses are satisfiable, arXiv:2602.02016v2, §3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (∀ i : Fin 1, (0 : ℝ) ≤ (fun _ : Fin 1 => (1 : ℝ)) i) ∧
    (∀ i : Fin 1, (fun _ : Fin 1 => (1 : ℝ)) i ≤ 1) ∧ (0 : ℝ) ≤ 1 := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

/-- Equal unit eigenvalues attain the dimension bound.
Source: arXiv:2602.02016v2, §3.4, the gap between Frobenius and spectral scaling. -/
theorem unit_spectrum_frobenius (Q : Matrix (Fin n) (Fin n) ℝ) (hQ : Orthogonal Q) :
    frobeniusNorm (spectralMatrix Q (fun _ => 1)) = Real.sqrt (n : ℝ) := by
  simp [frobeniusNorm, spectralMatrix_energy Q _ hQ]

/-- The sharp-bound hypotheses are satisfiable, arXiv:2602.02016v2, §3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- Any positive upper bound provides a spectrum in `[0,1]` after division.
Source: arXiv:2602.02016v2, §3.4 and Appendix A, the matrix normalization. -/
theorem normalized_spectrum_bounds (s : Fin n → ℝ) (c : ℝ) (hc : 0 < c)
    (hs : ∀ i, 0 ≤ s i) (hupper : ∀ i, s i ≤ c) (i : Fin n) :
    0 ≤ s i / c ∧ s i / c ≤ 1 :=
  ⟨div_nonneg (hs i) hc.le, (div_le_one hc).mpr (hupper i)⟩

/-- Normalization hypotheses are satisfiable, arXiv:2602.02016v2, §3.4. -/
example : (0 : ℝ) < 1 ∧ (∀ i : Fin 1, (0 : ℝ) ≤ (fun _ : Fin 1 => (1 : ℝ)) i) ∧
    (∀ i : Fin 1, (fun _ : Fin 1 => (1 : ℝ)) i ≤ 1) := by norm_num

/-- Undoing the input normalization requires multiplying the computed power
by `c^p`. Without it the result is the power of `A/c`, not of `A`.
Source: arXiv:2602.02016v2, §3.4, and Appendix A's matrix algorithm. -/
theorem spectralPower_rescale (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ)
    (c p : ℝ) (hc : 0 < c) (hs : ∀ i, 0 ≤ s i) :
    c ^ p • spectralPower Q (fun i => s i / c) p = spectralPower Q s p := by
  rw [spectralPower, spectralMatrix_smul, spectralPower]
  congr 1
  funext i
  rw [Real.div_rpow (hs i) hc.le]
  exact mul_div_cancel₀ _ (Real.rpow_pos_of_pos hc p).ne'

/-- Rescaling hypotheses are satisfiable, arXiv:2602.02016v2, §3.4. -/
example : (0 : ℝ) < 2 ∧ ∀ i : Fin 1, (0 : ℝ) ≤ (fun _ : Fin 1 => (1 : ℝ)) i := by
  norm_num

end Transformer.DASH

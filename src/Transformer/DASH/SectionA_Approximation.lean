/-
# DASH — transferring scalar approximation accuracy to matrices

arXiv:2602.02016v2, Appendix A. A verified scalar approximation bound
controls the matrix polynomial in Frobenius norm. The normalized target
and the output rescaling are explicit.
-/

import Transformer.DASH.SectionA_OptimizedClenshaw

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- Uniform eigenvalue error gives a dimension-dependent Frobenius error.
Source: arXiv:2602.02016v2, Appendix A, extending the scalar polynomial
approximation to the matrix eigendecomposition. -/
theorem spectral_approximation_error (Q : Matrix (Fin n) (Fin n) ℝ)
    (s t : Fin n → ℝ) (δ : ℝ) (hQ : Orthogonal Q) (hδ : 0 ≤ δ)
    (herror : ∀ i, |s i - t i| ≤ δ) :
    frobeniusNorm (spectralMatrix Q s - spectralMatrix Q t) ≤ Real.sqrt (n : ℝ) * δ := by
  rw [spectralMatrix_sub, frobeniusNorm, spectralMatrix_energy Q _ hQ]
  have hsum : (∑ i, (s i - t i) ^ 2) ≤ (n : ℝ) * δ ^ 2 := by
    calc
      (∑ i, (s i - t i) ^ 2) = ∑ i, |s i - t i| ^ 2 := by simp only [sq_abs]
      _ ≤ ∑ _ : Fin n, δ ^ 2 :=
        Finset.sum_le_sum fun i _ => pow_le_pow_left₀ (abs_nonneg _) (herror i) 2
      _ = (n : ℝ) * δ ^ 2 := by simp
  calc
    Real.sqrt (∑ i, (s i - t i) ^ 2) ≤ Real.sqrt ((n : ℝ) * δ ^ 2) :=
      Real.sqrt_le_sqrt hsum
    _ = Real.sqrt (n : ℝ) * δ := by rw [Real.sqrt_mul (by positivity), Real.sqrt_sq hδ]

/-- Error-bound hypotheses are satisfiable, arXiv:2602.02016v2, Appendix A. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧ (0 : ℝ) ≤ 0 ∧
    (∀ i : Fin 1, |(fun _ => (1 : ℝ)) i - (fun _ => (1 : ℝ)) i| ≤ 0) := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

/-- A scalar approximation on `[ε,1+ε]` bounds the actual optimized output.
The source labels its target `A^(-1/p)`; the algorithm instead targets
`(A/c+εI)^(-1/p)`. No accuracy of an arbitrary fitted coefficient list is
assumed implicitly: its scalar accuracy is the explicit premise.
Source: arXiv:2602.02016v2, Appendix A, both matrix Clenshaw algorithms. -/
theorem optimizedClenshaw_approximation (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (cs : List ℝ) (c ε exponent δ : ℝ)
    (hQ : Orthogonal Q) (hc : 0 < c) (hs : ∀ i, 0 ≤ s i)
    (hupper : ∀ i, s i ≤ c) (hδ : 0 ≤ δ)
    (happrox : ∀ x : ℝ, ε ≤ x → x ≤ 1 + ε →
      |clenshaw (2 * (x - ε) - 1) cs - x ^ exponent| ≤ δ) :
    frobeniusNorm
      ((optimizedClenshaw (chebMatrixArgument (spectralMatrix Q s) c) cs).1 -
        spectralPower Q (fun i => s i / c + ε) exponent) ≤ Real.sqrt (n : ℝ) * δ := by
  rw [chebMatrixArgument_spectrum Q s c hQ, optimizedClenshaw_spectrum Q _ cs hQ]
  apply spectral_approximation_error Q _ _ δ hQ hδ
  intro i
  obtain ⟨hi, hi'⟩ := normalized_spectrum_bounds s c hc hs hupper i
  have h := happrox (s i / c + ε) (by linarith) (by linarith)
  simpa only [add_sub_cancel_right] using h

/-- The transfer assumptions have an exact degree-zero instance.
Source: arXiv:2602.02016v2, Appendix A. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧ (0 : ℝ) < 1 ∧
    (∀ i : Fin 1, (0 : ℝ) ≤ (fun _ => (1 : ℝ)) i) ∧
    (∀ i : Fin 1, (fun _ => (1 : ℝ)) i ≤ 1) ∧ (0 : ℝ) ≤ 0 ∧
    (∀ x : ℝ, (0 : ℝ) ≤ x → x ≤ 1 → |clenshaw (2 * x - 1) [1] - x ^ (0 : ℝ)| ≤ 0) := by
  norm_num [Orthogonal, Muon.OrthonormalColumns, clenshaw, clenshawState]

/-- Undoing normalization recovers the power of `A+cεI`, with the
regularization scaled as well. To target `A+εI`, fit with `ε/c` instead.
Source: arXiv:2602.02016v2, Appendix A, the omitted output scaling. -/
theorem regularized_spectralPower_rescale (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (c ε exponent : ℝ) (hc : 0 < c)
    (hs : ∀ i, 0 ≤ s i) (hε : 0 ≤ ε) :
    c ^ exponent • spectralPower Q (fun i => s i / c + ε) exponent =
      spectralPower Q (fun i => s i + c * ε) exponent := by
  have hcoord : (fun i => s i / c + ε) = fun i => (s i + c * ε) / c := by
    funext i
    field_simp
  rw [hcoord]
  exact spectralPower_rescale Q _ c exponent hc
    (fun i => add_nonneg (hs i) (mul_nonneg hc.le hε))

/-- Regularized-rescaling assumptions are satisfiable,
arXiv:2602.02016v2, Appendix A. -/
example : (0 : ℝ) < 2 ∧ (∀ i : Fin 1, (0 : ℝ) ≤ (fun _ => (1 : ℝ)) i) ∧
    (0 : ℝ) ≤ 1 := by norm_num

end Transformer.DASH

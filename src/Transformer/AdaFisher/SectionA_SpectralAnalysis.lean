/-
# AdaFisher: Fourier and SNR formulas used in the structural experiments

arXiv:2405.16397v3, Appendix A.1, `eq:fft` and `eq:SNR`.
The definitions give exact formulas; the measured diagonal concentration
and noise robustness require the source's trained matrices and are empirical.
-/

import Transformer.AdaFisher.SectionA_Gershgorin
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.SpecialFunctions.Log.Base

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.AdaFisher

variable {m n : ℕ}

/-- Two-dimensional discrete Fourier transform, Appendix A.1, `eq:fft`.
This is the mathematical DFT; “FFT” describes algorithms evaluating it. -/
def matrixDFT (A : Matrix (Fin m) (Fin n) ℂ) : Matrix (Fin m) (Fin n) ℂ :=
  fun k l => ∑ p, ∑ q, A p q * Complex.exp
    (-2 * (Real.pi : ℂ) * Complex.I * ((p : ℂ) * (k : ℂ) / m + (q : ℂ) * (l : ℂ) / n))

/-- Diagonal signal energy, Appendix A.1, `eq:SNR`. -/
def diagonalEnergy (A : Matrix (Fin n) (Fin n) ℂ) : ℝ := ∑ i, ‖A i i‖ ^ 2

/-- Strictly upper-triangular noise energy, Appendix A.1, `eq:SNR`. -/
def upperEnergy (A : Matrix (Fin n) (Fin n) ℂ) : ℝ :=
  ∑ i, ∑ j ∈ Finset.univ.filter (fun j => i < j), ‖A i j‖ ^ 2

/-- The reported SNR in decibels, Appendix A.1, `eq:SNR`.
The physically meaningful finite value requires positive signal and noise;
Lean's total log/division merely extends the formula outside that domain. -/
def signalNoiseRatio (A perturbed : Matrix (Fin n) (Fin n) ℂ) : ℝ :=
  10 * Real.logb 10 (diagonalEnergy A / upperEnergy perturbed)

/-- The DFT at zero frequency is the sum of matrix entries,
Appendix A.1, `eq:fft`. -/
theorem matrixDFT_zeroFrequency [NeZero m] [NeZero n]
    (A : Matrix (Fin m) (Fin n) ℂ) : matrixDFT A 0 0 = ∑ p, ∑ q, A p q := by
  simp [matrixDFT]

/-- Fourier transformation is additive, Appendix A.1, the noisy
matrix comparison immediately after `eq:fft`. -/
theorem matrixDFT_add (A E : Matrix (Fin m) (Fin n) ℂ) :
    matrixDFT (A + E) = matrixDFT A + matrixDFT E := by
  ext k l
  simp [matrixDFT, add_mul, Finset.sum_add_distrib]

/-- Off-diagonal perturbations preserve diagonal signal energy,
Appendix A.1, `eq:SNR`. They need not preserve eigenvalues. -/
theorem diagonalEnergy_offDiagonal_noise (A E : Matrix (Fin n) (Fin n) ℂ)
    (hE : ∀ i, E i i = 0) : diagonalEnergy (A + E) = diagonalEnergy A := by
  simp [diagonalEnergy, hE]

example : ∀ i : Fin 2, (0 : Matrix (Fin 2) (Fin 2) ℂ) i i = 0 := by simp

/-- Both energies in the paper's SNR are nonnegative, Appendix A.1. -/
theorem spectral_energy_nonneg (A : Matrix (Fin n) (Fin n) ℂ) :
    0 ≤ diagonalEnergy A ∧ 0 ≤ upperEnergy A := by
  constructor
  · exact Finset.sum_nonneg fun i _ => sq_nonneg ‖A i i‖
  · exact Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ => sq_nonneg ‖A i j‖

end Transformer.AdaFisher

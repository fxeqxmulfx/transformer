/-
# DASH — symmetric spectral calculus

arXiv:2602.02016v2, §3.1, and Appendix B. A square orthogonal frame
and real eigenvalues specify the eigendecomposition. Negative powers
used for Shampoo require positive eigenvalues; symmetry alone does
not give a real inverse root on a zero or negative eigenvalue.
-/

import Transformer.DASH.Section2_Models
import Mathlib.Analysis.SpecialFunctions.Pow.Real

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- Orthogonality of the full eigenvector frame,
arXiv:2602.02016v2, §3.1, `A = Q Λ Qᵀ`. -/
def Orthogonal (Q : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  Muon.OrthonormalColumns Q ∧ Q * Q.transpose = 1

/-- The symmetric matrix with the specified eigendecomposition,
arXiv:2602.02016v2, §3.1. -/
def spectralMatrix (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ) :
    Matrix (Fin n) (Fin n) ℝ := Muon.singularMatrix Q s Q

/-- Real spectral power in the supplied orthogonal frame,
arXiv:2602.02016v2, §3.1. Negative powers are interpreted as inverse
roots only on positive spectra. -/
def spectralPower (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ) (p : ℝ) :
    Matrix (Fin n) (Fin n) ℝ := spectralMatrix Q (fun i => s i ^ p)

/-- A constant unit spectrum gives the identity,
arXiv:2602.02016v2, §3.1. -/
theorem spectralMatrix_one (Q : Matrix (Fin n) (Fin n) ℝ) (hQ : Orthogonal Q) :
    spectralMatrix Q (fun _ => 1) = 1 := by
  simpa only [spectralMatrix, Muon.singularMatrix, Matrix.diagonal_one,
    Matrix.mul_one] using hQ.2

/-- Orthogonal frames exist, arXiv:2602.02016v2, §3.1. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- Spectral multiplication multiplies the eigenvalues,
arXiv:2602.02016v2, §3.1. -/
theorem spectralMatrix_mul (Q : Matrix (Fin n) (Fin n) ℝ) (s t : Fin n → ℝ)
    (hQ : Orthogonal Q) :
    spectralMatrix Q s * spectralMatrix Q t = spectralMatrix Q (fun i => s i * t i) :=
  Muon.singularMatrix_mul Q s Q t Q hQ.1

/-- Multiplication assumptions are satisfiable, arXiv:2602.02016v2, §3.1. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- Spectral addition adds eigenvalues, arXiv:2602.02016v2, §3.1. -/
theorem spectralMatrix_add (Q : Matrix (Fin n) (Fin n) ℝ) (s t : Fin n → ℝ) :
    spectralMatrix Q s + spectralMatrix Q t = spectralMatrix Q (fun i => s i + t i) :=
  Muon.singularMatrix_add Q s t Q

/-- Spectral scaling scales the eigenvalues, arXiv:2602.02016v2, §3.1 and §3.4. -/
theorem spectralMatrix_smul (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ) (c : ℝ) :
    c • spectralMatrix Q s = spectralMatrix Q (fun i => c * s i) :=
  Muon.singularMatrix_smul c Q s Q

/-- Spectral subtraction subtracts the eigenvalues,
arXiv:2602.02016v2, §3.2–3.3, the correction matrices. -/
theorem spectralMatrix_sub (Q : Matrix (Fin n) (Fin n) ℝ) (s t : Fin n → ℝ) :
    spectralMatrix Q s - spectralMatrix Q t = spectralMatrix Q (fun i => s i - t i) := by
  simp only [spectralMatrix, Muon.singularMatrix, ← Matrix.sub_mul,
    ← Matrix.mul_sub, Matrix.diagonal_sub]

/-- The zero spectral matrix is zero, arXiv:2602.02016v2, §3.1. -/
theorem spectralMatrix_zero (Q : Matrix (Fin n) (Fin n) ℝ) :
    spectralMatrix Q (fun _ => 0) = 0 := by simp [spectralMatrix, Muon.singularMatrix]

/-- Adding `ε I` shifts every eigenvalue by `ε`,
arXiv:2602.02016v2, §2.2 and Appendix B, “Regularization”. -/
theorem spectralMatrix_shift (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ)
    (ε : ℝ) (hQ : Orthogonal Q) :
    spectralMatrix Q s + ε • (1 : Matrix (Fin n) (Fin n) ℝ) =
      spectralMatrix Q (fun i => s i + ε) := by
  rw [← spectralMatrix_one Q hQ, spectralMatrix_smul, spectralMatrix_add]
  simp only [mul_one]

/-- Shift assumptions are satisfiable, arXiv:2602.02016v2, Appendix B. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- Integer spectral powers agree with matrix powers,
arXiv:2602.02016v2, §3.1–3.2. -/
theorem spectralMatrix_pow (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ)
    (hQ : Orthogonal Q) (k : ℕ) :
    spectralMatrix Q s ^ k = spectralMatrix Q (fun i => s i ^ k) := by
  induction k with
  | zero => simpa only [pow_zero] using (spectralMatrix_one Q hQ).symm
  | succ k ih => rw [pow_succ, ih, spectralMatrix_mul Q _ s hQ]; simp only [pow_succ]

/-- Matrix-power assumptions are satisfiable, arXiv:2602.02016v2, §3.1. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- Powers of a positive spectrum add their exponents,
arXiv:2602.02016v2, §3.1 and §3.3. -/
theorem spectralPower_mul (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ)
    (p q : ℝ) (hQ : Orthogonal Q) (hs : ∀ i, 0 < s i) :
    spectralPower Q s p * spectralPower Q s q = spectralPower Q s (p + q) := by
  rw [spectralPower, spectralPower, spectralMatrix_mul Q _ _ hQ, spectralPower]
  congr 1
  funext i
  exact (Real.rpow_add (hs i) p q).symm

/-- Positive spectra and orthogonal frames coexist, arXiv:2602.02016v2, §3.1. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    ∀ i : Fin 1, (0 : ℝ) < (fun _ : Fin 1 => (2 : ℝ)) i := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

/-- Chaining spectral powers multiplies their exponents. In particular,
the inverse square root of a square root is the inverse fourth root.
Source: arXiv:2602.02016v2, §3.3, after `equation:NDB-Y-Z`. -/
theorem spectralPower_nested (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ)
    (p q : ℝ) (hs : ∀ i, 0 ≤ s i) :
    spectralPower Q (fun i => s i ^ p) q = spectralPower Q s (p * q) := by
  unfold spectralPower
  congr 1
  funext i
  exact (Real.rpow_mul (hs i) p q).symm

/-- Nonnegative spectra exist, arXiv:2602.02016v2, §3.3. -/
example : ∀ i : Fin 1, (0 : ℝ) ≤ (fun _ : Fin 1 => (2 : ℝ)) i := by norm_num

/-- The Frobenius energy of a symmetric matrix is the sum of squared
eigenvalues, arXiv:2602.02016v2, §3.4. -/
theorem spectralMatrix_energy (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ)
    (hQ : Orthogonal Q) : Muon.squaredFrobenius (spectralMatrix Q s) = ∑ i, s i ^ 2 :=
  Muon.squaredFrobenius_singularMatrix Q s Q hQ.1 hQ.1

/-- Frobenius spectral assumptions are satisfiable, arXiv:2602.02016v2, §3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

end Transformer.DASH

/-
# Muon — Newton–Schulz in singular coordinates

arXiv:2502.16982, §2.1, `eq:iteration`. Orthonormal singular frames are
preserved, and the singular coefficients follow the printed polynomial.
This is an exact real-arithmetic statement, without a bf16 error model.
-/

import Transformer.Muon.Section2_Spectral

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.Muon

variable {a b c r : ℕ}

/-- Transposing a singular representation swaps its two frames,
arXiv:2502.16982, §2.1. -/
theorem singularMatrix_transpose (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ)
    (V : Matrix (Fin b) (Fin r) ℝ) :
    (singularMatrix U s V).transpose = singularMatrix V s U := by
  have hd : (Matrix.diagonal s).transpose = Matrix.diagonal s := by
    ext i j
    by_cases hij : i = j
    · subst j; simp
    · simp [hij, Ne.symm hij]
  simp only [singularMatrix, Matrix.transpose_mul, Matrix.transpose_transpose, hd,
    Matrix.mul_assoc]

/-- Composition in matching orthonormal singular coordinates multiplies the
diagonal coefficients, arXiv:2502.16982, §2.1, `eq:iteration`. -/
theorem singularMatrix_mul (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ)
    (V : Matrix (Fin b) (Fin r) ℝ) (t : Fin r → ℝ) (T : Matrix (Fin c) (Fin r) ℝ)
    (hV : OrthonormalColumns V) :
    singularMatrix U s V * singularMatrix V t T =
      singularMatrix U (fun k => s k * t k) T := by
  simp only [singularMatrix, Matrix.mul_assoc]
  rw [← Matrix.mul_assoc V.transpose V, hV, Matrix.one_mul]
  rw [← Matrix.mul_assoc (Matrix.diagonal s) (Matrix.diagonal t), Matrix.diagonal_mul_diagonal]

/-- A matching orthonormal frame exists, arXiv:2502.16982, §2.1. -/
example : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [OrthonormalColumns]

/-- Scaling the represented matrix scales its singular coefficients,
arXiv:2502.16982, §2.1. Coefficients are allowed to be signed. -/
theorem singularMatrix_smul (z : ℝ) (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ)
    (V : Matrix (Fin b) (Fin r) ℝ) :
    z • singularMatrix U s V = singularMatrix U (fun k => z * s k) V := by
  change z • (U * Matrix.diagonal s * V.transpose) =
    U * Matrix.diagonal (z • s) * V.transpose
  simp only [Matrix.diagonal_smul, Matrix.mul_smul, Matrix.smul_mul]

/-- Adding matrices in the same singular coordinates adds their coefficients,
arXiv:2502.16982, §2.1, `eq:iteration`. -/
theorem singularMatrix_add (U : Matrix (Fin a) (Fin r) ℝ) (s t : Fin r → ℝ)
    (V : Matrix (Fin b) (Fin r) ℝ) :
    singularMatrix U s V + singularMatrix U t V =
      singularMatrix U (fun k => s k + t k) V := by
  simp only [singularMatrix, ← Matrix.add_mul, ← Matrix.mul_add, Matrix.diagonal_add]

/-- Newton–Schulz acts on each singular coefficient by `αx + βx³ + γx⁵`,
arXiv:2502.16982, §2.1, `eq:iteration`. -/
theorem schulzStep_spectrum (α β γ : ℝ) (U : Matrix (Fin a) (Fin r) ℝ)
    (s : Fin r → ℝ) (V : Matrix (Fin b) (Fin r) ℝ)
    (hU : OrthonormalColumns U) (hV : OrthonormalColumns V) :
    schulzStep α β γ (singularMatrix U s V) =
      singularMatrix U (fun k => schulzPolynomial α β γ (s k)) V := by
  let X := singularMatrix U s V
  have hgram : X * X.transpose = singularMatrix U (fun k => s k ^ 2) U :=
    singularMatrix_gram U s V hV
  have hcube : (X * X.transpose) * X = singularMatrix U (fun k => s k ^ 3) V := by
    rw [hgram, singularMatrix_mul U _ U s V hU]
    congr 1
  have hfifth : (X * X.transpose) ^ 2 * X = singularMatrix U (fun k => s k ^ 5) V := by
    rw [hgram, pow_two, singularMatrix_mul U _ U _ U hU,
      singularMatrix_mul U _ U s V hU]
    congr 1
    funext k
    ring
  change α • X + β • ((X * X.transpose) * X) + γ • ((X * X.transpose) ^ 2 * X) = _
  rw [hcube, hfifth, singularMatrix_smul, singularMatrix_smul, singularMatrix_smul,
    singularMatrix_add, singularMatrix_add]
  rfl

/-- Both frames can be orthonormal, arXiv:2502.16982, §2.1. -/
example : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [OrthonormalColumns]

/-- Scalar Newton–Schulz iteration from an initial singular coefficient,
arXiv:2502.16982, §2.1, `eq:iteration`. -/
def scalarSchulz (α β γ x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | k + 1 => schulzPolynomial α β γ (scalarSchulz α β γ x₀ k)

/-- Finite matrix iteration follows the scalar polynomial from the normalized
initial singular coefficients, arXiv:2502.16982, §2.1, `eq:iteration`. -/
theorem schulzIterate_spectrum (α β γ : ℝ) (U : Matrix (Fin a) (Fin r) ℝ)
    (s : Fin r → ℝ) (V : Matrix (Fin b) (Fin r) ℝ)
    (hU : OrthonormalColumns U) (hV : OrthonormalColumns V) (N : ℕ) :
    schulzIterate α β γ (singularMatrix U s V) N =
      singularMatrix U (fun k => scalarSchulz α β γ
        ((Real.sqrt (squaredFrobenius (singularMatrix U s V)))⁻¹ * s k) N) V := by
  induction N with
  | zero => exact singularMatrix_smul _ U s V
  | succ N ih =>
    rw [schulzIterate, ih, schulzStep_spectrum α β γ U _ V hU hV]
    rfl

/-- Orthonormal frames for the iteration exist, arXiv:2502.16982, §2.1. -/
example : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [OrthonormalColumns]

end Transformer.Muon

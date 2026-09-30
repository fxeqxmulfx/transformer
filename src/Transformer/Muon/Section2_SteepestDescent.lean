/-
# Muon — steepest descent under a spectral norm constraint

arXiv:2502.16982, §2.1 and §4. The exact polar factor maximizes the
Frobenius pairing over Euclidean contractions. Its negative minimizes
the linearized objective over the same ball, proving the stated
steepest-descent interpretation without assuming the duality result.
-/

import Transformer.Muon.Section2_Euclidean
import Transformer.Muon.Section2_NewtonSchulz

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.Muon

variable {a b r : ℕ}

/-- The linearized objective in singular coordinates, arXiv:2502.16982, §2.1. -/
theorem singularMatrix_pair (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ)
    (V : Matrix (Fin b) (Fin r) ℝ) (D : Matrix (Fin a) (Fin b) ℝ) :
    frobeniusPair (singularMatrix U s V) D =
      ∑ k, s k * (U.transpose * D * V) k k := by
  rw [frobeniusPair_eq_trace, singularMatrix_transpose]
  simp only [singularMatrix, Matrix.mul_assoc]
  rw [Matrix.trace_mul_comm]
  simp [Matrix.trace, Matrix.diag, Matrix.diagonal_mul, Matrix.mul_assoc]

/-- Each diagonal coefficient of a contraction in orthonormal singular
coordinates is at most one, arXiv:2502.16982, §2.1. -/
theorem contraction_singular_coeff_le (U : Matrix (Fin a) (Fin r) ℝ)
    (V : Matrix (Fin b) (Fin r) ℝ) (hU : OrthonormalColumns U)
    (hV : OrthonormalColumns V) (D : Matrix (Fin a) (Fin b) ℝ)
    (hD : EuclideanContraction D) (k : Fin r) :
    (U.transpose * D * V) k k ≤ 1 := by
  let E : Matrix (Fin r) (Fin 1) ℝ := Matrix.single k 0 1
  have hbound : squaredFrobenius ((U.transpose * D * V) * E) ≤ 1 := by
    calc
      _ = squaredFrobenius (U.transpose * (D * (V * E))) := by
        simp only [Matrix.mul_assoc]
      _ ≤ squaredFrobenius (D * (V * E)) := frame_transpose_contraction U hU _
      _ ≤ squaredFrobenius (V * E) := hD _
      _ = 1 := by rw [squaredFrobenius_frame_mul V hV, unitColumn_energy]
  have hentry := entry_sq_le_squaredFrobenius ((U.transpose * D * V) * E) k 0
  change ((U.transpose * D * V) * Matrix.single k (0 : Fin 1) (1 : ℝ)) k 0 ^ 2 ≤ _ at hentry
  rw [Matrix.mul_single_apply_same, mul_one] at hentry
  have hsq := hentry.trans hbound
  nlinarith [sq_nonneg ((U.transpose * D * V) k k - 1)]

/-- The singular-coefficient hypotheses are satisfiable: identity frames and
identity contraction, arXiv:2502.16982, §2.1. -/
example : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    EuclideanContraction (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  refine ⟨?_, ?_, ?_⟩
  · simp [OrthonormalColumns]
  · simp [OrthonormalColumns]
  · intro X; simp

/-- The exact polar factor attains the nuclear-norm value `∑σ`,
arXiv:2502.16982, §2.1, “Steepest Descent Under Norm Constraints”. -/
theorem polarUpdate_pair (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ)
    (V : Matrix (Fin b) (Fin r) ℝ) (hU : OrthonormalColumns U)
    (hV : OrthonormalColumns V) :
    frobeniusPair (singularMatrix U s V) (polarUpdate U V) = ∑ k, s k := by
  rw [singularMatrix_pair]
  have h : U.transpose * polarUpdate U V * V = (1 : Matrix (Fin r) (Fin r) ℝ) := by
    simp only [polarUpdate, Matrix.mul_assoc]
    rw [← Matrix.mul_assoc U.transpose U, hU, Matrix.one_mul, hV]
  simp [h]

/-- The pairing hypotheses are satisfiable, arXiv:2502.16982, §2.1. -/
example : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [OrthonormalColumns]

/-- The exact polar update maximizes the first-order decrease coefficient
over the spectral unit ball, arXiv:2502.16982, §2.1 and §4.
Nonnegative coefficients are precisely the singular-value condition. -/
theorem polarUpdate_maximizes_pair (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ)
    (V : Matrix (Fin b) (Fin r) ℝ) (hU : OrthonormalColumns U)
    (hV : OrthonormalColumns V) (hs : ∀ k, 0 ≤ s k) :
    EuclideanContraction (polarUpdate U V) ∧
      ∀ D : Matrix (Fin a) (Fin b) ℝ, EuclideanContraction D →
        frobeniusPair (singularMatrix U s V) D ≤
          frobeniusPair (singularMatrix U s V) (polarUpdate U V) := by
  refine ⟨polarUpdate_contraction U V hU hV, fun D hD => ?_⟩
  rw [singularMatrix_pair, polarUpdate_pair U s V hU hV]
  apply Finset.sum_le_sum
  intro k hk
  simpa using mul_le_mul_of_nonneg_left (contraction_singular_coeff_le U V hU hV D hD k) (hs k)

/-- Positive singular values and orthonormal frames coexist,
arXiv:2502.16982, §2.1. -/
example : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    ∀ k : Fin 1, (0 : ℝ) ≤ (fun _ : Fin 1 => (2 : ℝ)) k := by
  norm_num [OrthonormalColumns]

/-- Negation preserves the spectral unit ball, arXiv:2502.16982, §2.1. -/
theorem contraction_neg (D : Matrix (Fin a) (Fin b) ℝ) (hD : EuclideanContraction D) :
    EuclideanContraction (-D) := by
  intro X
  have henergy : squaredFrobenius (-D * X) = squaredFrobenius (D * X) := by
    simp [squaredFrobenius, Matrix.neg_mul]
  rw [henergy]
  exact hD X

/-- A contractive direction exists, arXiv:2502.16982, §2.1. -/
example : EuclideanContraction (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  intro X; simp

/-- Negating an update negates its first-order objective,
arXiv:2502.16982, §2.1. -/
theorem frobeniusPair_neg_right (M D : Matrix (Fin a) (Fin b) ℝ) :
    frobeniusPair M (-D) = -frobeniusPair M D := by
  simp [frobeniusPair]

/-- Exact Muon is steepest descent under the Euclidean induced-operator-norm
constraint: `-U Vᵀ` minimizes the linearized objective among all feasible
directions. Source: arXiv:2502.16982, §2.1 and §4. -/
theorem polarUpdate_steepestDescent (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ)
    (V : Matrix (Fin b) (Fin r) ℝ) (hU : OrthonormalColumns U)
    (hV : OrthonormalColumns V) (hs : ∀ k, 0 ≤ s k) :
    EuclideanContraction (-polarUpdate U V) ∧
      ∀ D : Matrix (Fin a) (Fin b) ℝ, EuclideanContraction D →
        frobeniusPair (singularMatrix U s V) (-polarUpdate U V) ≤
          frobeniusPair (singularMatrix U s V) D := by
  have hp := polarUpdate_maximizes_pair U s V hU hV hs
  refine ⟨contraction_neg _ hp.1, fun D hD => ?_⟩
  have h := hp.2 (-D) (contraction_neg D hD)
  rw [frobeniusPair_neg_right] at h ⊢
  linarith

/-- The steepest-descent hypotheses are satisfiable, arXiv:2502.16982, §2.1. -/
example : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    ∀ k : Fin 1, (0 : ℝ) ≤ (fun _ : Fin 1 => (2 : ℝ)) k := by
  norm_num [OrthonormalColumns]

end Transformer.Muon

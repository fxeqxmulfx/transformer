/-
# The homogeneous and heterogeneous quadratic Hessians

arXiv:2506.12543v1, Section 3.3 and Appendix D.
Orthogonal rotations are arbitrary inputs: the spectral claims do not
depend on how the experiment samples them. The coordinate index is
(within-block position, block number), matching Matrix.blockDiagonal.
-/

import Transformer.BatchSize.Section4_ErrorFunction

open Matrix
open scoped BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- Nine coordinates in three blocks of three, Appendix D. -/
abbrev QuadIndex := Fin 3 × Fin 3

/-- The heterogeneous grouping [[1,2,3],[99,100,101],[4998,4999,5000]];
Appendix D. The second coordinate is the block number. -/
def heterogeneousEigenvalue (k : QuadIndex) : ℕ :=
  (![![1, 2, 3], ![99, 100, 101], ![4998, 4999, 5000]] : Fin 3 → Fin 3 → ℕ) k.2 k.1

/-- The homogeneous grouping [[1,99,4998],[2,100,4999],[3,101,5000]];
Appendix D. It transposes the heterogeneous eigenvalue arrangement. -/
def homogeneousEigenvalue (k : QuadIndex) : ℕ := heterogeneousEigenvalue (k.2, k.1)

/-- All nine prescribed eigenvalues are positive, Appendix D. -/
theorem heterogeneousEigenvalue_pos (k : QuadIndex) : 0 < heterogeneousEigenvalue k := by
  obtain ⟨i, b⟩ := k
  fin_cases i <;> fin_cases b <;> decide

/-- Both groupings contain exactly the same eigenvalues, Appendix D. -/
theorem eigenvalue_groupings_same_range :
    Set.range homogeneousEigenvalue = Set.range heterogeneousEigenvalue := by
  ext n
  constructor
  · rintro ⟨⟨i, b⟩, rfl⟩
    exact ⟨(b, i), rfl⟩
  · rintro ⟨⟨i, b⟩, rfl⟩
    exact ⟨(b, i), rfl⟩

/-- The global block-diagonal rotation, Appendix D. -/
def blockRotation (R : Fin 3 → Matrix (Fin 3) (Fin 3) ℝ) : Matrix QuadIndex QuadIndex ℝ :=
  Matrix.blockDiagonal R

/-- Independent orthogonal rotations, the mathematical input required
by Appendix D's construction. This predicate states an assumption,
not the paper's claimed Haar distribution of the sampling procedure. -/
def BlockOrthogonal (R : Fin 3 → Matrix (Fin 3) (Fin 3) ℝ) : Prop :=
  ∀ b, (R b).transpose * R b = 1

/-- Orthogonality is preserved by assembling the blocks, Appendix D. -/
theorem blockRotation_orthogonal (R : Fin 3 → Matrix (Fin 3) (Fin 3) ℝ)
    (hR : BlockOrthogonal R) : (blockRotation R).transpose * blockRotation R = 1 := by
  rw [blockRotation, Matrix.blockDiagonal_transpose, ← Matrix.blockDiagonal_mul]
  have heq : (fun b => (R b).transpose * R b) = (1 : Fin 3 → Matrix (Fin 3) (Fin 3) ℝ) :=
    funext hR
  rw [heq, Matrix.blockDiagonal_one]

/-- Nonvacuity of the block rotations, Appendix D. -/
example : BlockOrthogonal (fun _ => (1 : Matrix (Fin 3) (Fin 3) ℝ)) := by
  intro b
  simp

/-- Actual rotated Hessian with the prescribed block eigenvalues;
Appendix D. -/
def quadraticHessian (eigenvalue : QuadIndex → ℕ)
    (R : Fin 3 → Matrix (Fin 3) (Fin 3) ℝ) : Matrix QuadIndex QuadIndex ℝ :=
  blockRotation R * Matrix.diagonal (fun k => (eigenvalue k : ℝ)) * (blockRotation R).transpose

/-- Orthogonal block rotations preserve the full characteristic
polynomial, and hence the eigenvalue multiset; Appendix D. -/
theorem quadraticHessian_charpoly (eigenvalue : QuadIndex → ℕ)
    (R : Fin 3 → Matrix (Fin 3) (Fin 3) ℝ) (hR : BlockOrthogonal R) :
    (quadraticHessian eigenvalue R).charpoly =
      (Matrix.diagonal (fun k => (eigenvalue k : ℝ))).charpoly := by
  rw [quadraticHessian, Matrix.mul_assoc, Matrix.charpoly_mul_comm,
    Matrix.mul_assoc, blockRotation_orthogonal R hR, Matrix.mul_one]

/-- Nonvacuity of spectral preservation, Appendix D. -/
example : BlockOrthogonal (fun _ => (1 : Matrix (Fin 3) (Fin 3) ℝ)) := by
  intro b
  simp

/-- The two eigenvalue arrangements have identical characteristic
polynomials before rotation, Appendix D. -/
theorem eigenvalue_groupings_same_charpoly :
    (Matrix.diagonal (fun k => (homogeneousEigenvalue k : ℝ))).charpoly =
      (Matrix.diagonal (fun k => (heterogeneousEigenvalue k : ℝ))).charpoly := by
  let e : QuadIndex ≃ QuadIndex := Equiv.prodComm _ _
  have heq : Matrix.diagonal (fun k => (homogeneousEigenvalue k : ℝ)) =
      Matrix.reindex e e (Matrix.diagonal (fun k => (heterogeneousEigenvalue k : ℝ))) := by
    ext i j
    simp [Matrix.reindex, Matrix.diagonal, homogeneousEigenvalue, e, Prod.swap,
      Prod.ext_iff, and_comm]
  rw [heq, Matrix.charpoly_reindex]

/-- The homogeneous and heterogeneous rotated problems have identical
global spectra, even with different rotations; Section 3.3 and Appendix D. -/
theorem homogeneous_heterogeneous_same_charpoly
    (R S : Fin 3 → Matrix (Fin 3) (Fin 3) ℝ)
    (hR : BlockOrthogonal R) (hS : BlockOrthogonal S) :
    (quadraticHessian homogeneousEigenvalue R).charpoly =
      (quadraticHessian heterogeneousEigenvalue S).charpoly := by
  rw [quadraticHessian_charpoly _ R hR, quadraticHessian_charpoly _ S hS]
  exact eigenvalue_groupings_same_charpoly

/-- Nonvacuity of the equal-spectrum comparison, Appendix D. -/
example : BlockOrthogonal (fun _ => (1 : Matrix (Fin 3) (Fin 3) ℝ)) ∧
    BlockOrthogonal (fun _ => (1 : Matrix (Fin 3) (Fin 3) ℝ)) := by
  constructor <;> intro b <;> simp

end Transformer.BatchSize

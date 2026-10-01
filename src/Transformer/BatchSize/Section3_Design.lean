/-
# The quadratic design matrix and loss expansion

arXiv:2506.12543v1, Section 3.2, equation (1), and Appendix D.
The design is constructed from the actual rotated eigenvalues and is
proved to satisfy X^T X=H. The quadratic Taylor expansion is exact.
-/

import Transformer.BatchSize.Section3_Quadratic

open Matrix
open scoped BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- The spectral square root used as the design X in Appendix D. -/
def spectralDesign (eigenvalue : QuadIndex → ℕ)
    (R : Fin 3 → Matrix (Fin 3) (Fin 3) ℝ) : Matrix QuadIndex QuadIndex ℝ :=
  blockRotation R * Matrix.diagonal (fun k => Real.sqrt (eigenvalue k)) *
    (blockRotation R).transpose

/-- The design is symmetric, Appendix D's X=H^(1/2). -/
theorem spectralDesign_symmetric (eigenvalue : QuadIndex → ℕ)
    (R : Fin 3 → Matrix (Fin 3) (Fin 3) ℝ) :
    (spectralDesign eigenvalue R).transpose = spectralDesign eigenvalue R := by
  simp only [spectralDesign, Matrix.transpose_mul, Matrix.transpose_transpose,
    Matrix.diagonal_transpose, Matrix.mul_assoc]

/-- The design is positive semidefinite, selecting the positive spectral
square root rather than an arbitrary symmetric factor; Appendix D. -/
theorem spectralDesign_posSemidef (eigenvalue : QuadIndex → ℕ)
    (R : Fin 3 → Matrix (Fin 3) (Fin 3) ℝ) :
    (spectralDesign eigenvalue R).PosSemidef := by
  have hD : (Matrix.diagonal (fun k => Real.sqrt (eigenvalue k))).PosSemidef :=
    Matrix.posSemidef_diagonal_iff.mpr (fun _ => Real.sqrt_nonneg _)
  simpa [spectralDesign, Matrix.conjTranspose_eq_transpose_of_trivial] using
    hD.mul_mul_conjTranspose_same (blockRotation R)

/-- The constructed design has exactly the stated Gram matrix H=X^T X,
Appendix D. -/
theorem spectralDesign_gram (eigenvalue : QuadIndex → ℕ)
    (R : Fin 3 → Matrix (Fin 3) (Fin 3) ℝ) (hR : BlockOrthogonal R) :
    (spectralDesign eigenvalue R).transpose * spectralDesign eigenvalue R =
      quadraticHessian eigenvalue R := by
  let Q := blockRotation R
  let D := Matrix.diagonal (fun k => Real.sqrt (eigenvalue k))
  have hQ : Q.transpose * Q = 1 := blockRotation_orthogonal R hR
  rw [spectralDesign_symmetric]
  change (Q * D * Q.transpose) * (Q * D * Q.transpose) =
    Q * Matrix.diagonal (fun k => (eigenvalue k : ℝ)) * Q.transpose
  calc
    (Q * D * Q.transpose) * (Q * D * Q.transpose) =
        Q * (D * (Q.transpose * Q) * D) * Q.transpose := by simp only [Matrix.mul_assoc]
    _ = Q * (D * D) * Q.transpose := by rw [hQ, Matrix.mul_one]
    _ = Q * Matrix.diagonal (fun k => (eigenvalue k : ℝ)) * Q.transpose := by
      dsimp [D]
      rw [Matrix.diagonal_mul_diagonal]
      have heq : (fun k => Real.sqrt (eigenvalue k) * Real.sqrt (eigenvalue k)) =
          (fun k => (eigenvalue k : ℝ)) := by
        funext k
        exact Real.mul_self_sqrt (Nat.cast_nonneg _)
      rw [heq]

/-- Nonvacuity of the design factorization, Appendix D. -/
example : BlockOrthogonal (fun _ => (1 : Matrix (Fin 3) (Fin 3) ℝ)) := by
  intro b
  simp

/-- The exact quadratic loss L(w)=w^T H w/2 in Appendix D. -/
def quadraticLoss (H : Matrix QuadIndex QuadIndex ℝ) (w : QuadIndex → ℝ) : ℝ :=
  (w ⬝ᵥ H *ᵥ w) / 2

/-- The toy loss is the least-squares objective for its actual design,
Appendix D. -/
theorem quadraticLoss_eq_design_norm (eigenvalue : QuadIndex → ℕ)
    (R : Fin 3 → Matrix (Fin 3) (Fin 3) ℝ) (hR : BlockOrthogonal R)
    (w : QuadIndex → ℝ) :
    quadraticLoss (quadraticHessian eigenvalue R) w =
      (∑ k, ((spectralDesign eigenvalue R) *ᵥ w) k ^ 2) / 2 := by
  rw [quadraticLoss, ← spectralDesign_gram eigenvalue R hR, ← Matrix.mulVec_mulVec,
    Matrix.dotProduct_mulVec, Matrix.vecMul_transpose]
  simp only [dotProduct, sq]

/-- Nonvacuity of the least-squares loss representation, Appendix D. -/
example : BlockOrthogonal (fun _ => (1 : Matrix (Fin 3) (Fin 3) ℝ)) := by
  intro b
  simp

/-- The objective is nonnegative, Appendix D's positive quadratic example. -/
theorem quadraticLoss_nonneg (eigenvalue : QuadIndex → ℕ)
    (R : Fin 3 → Matrix (Fin 3) (Fin 3) ℝ) (hR : BlockOrthogonal R)
    (w : QuadIndex → ℝ) : 0 ≤ quadraticLoss (quadraticHessian eigenvalue R) w := by
  rw [quadraticLoss_eq_design_norm eigenvalue R hR w]
  exact div_nonneg (Finset.sum_nonneg (fun k _ => sq_nonneg _)) (by norm_num)

/-- Nonvacuity of a nonnegative quadratic loss, Appendix D. -/
example : BlockOrthogonal (fun _ => (1 : Matrix (Fin 3) (Fin 3) ℝ)) := by
  intro b
  simp

/-- The rotated Hessian is symmetric, Section 3.3 and Appendix D. -/
theorem quadraticHessian_symmetric (eigenvalue : QuadIndex → ℕ)
    (R : Fin 3 → Matrix (Fin 3) (Fin 3) ℝ) :
    (quadraticHessian eigenvalue R).transpose = quadraticHessian eigenvalue R := by
  simp only [quadraticHessian, Matrix.transpose_mul, Matrix.transpose_transpose,
    Matrix.diagonal_transpose, Matrix.mul_assoc]

/-- Exact second-order expansion of the quadratic objective; Section 3.2,
equation (1), specialized to the Appendix D model. There is no remainder. -/
theorem quadraticLoss_update (H : Matrix QuadIndex QuadIndex ℝ)
    (hH : H.transpose = H) (w u : QuadIndex → ℝ) :
    quadraticLoss H (w + u) = quadraticLoss H w + (w ⬝ᵥ H *ᵥ u) + quadraticLoss H u := by
  have hvec : w ᵥ* H = H *ᵥ w := by
    calc
      w ᵥ* H = w ᵥ* H.transpose := by rw [hH]
      _ = H *ᵥ w := Matrix.vecMul_transpose _ _
  have hcross : w ⬝ᵥ H *ᵥ u = u ⬝ᵥ H *ᵥ w := by
    rw [Matrix.dotProduct_mulVec, hvec, dotProduct_comm]
  simp only [quadraticLoss, Matrix.mulVec_add, add_dotProduct, dotProduct_add]
  rw [hcross]
  ring

/-- Nonvacuity of the exact symmetric Taylor expansion, Section 3.2. -/
example : (1 : Matrix QuadIndex QuadIndex ℝ).transpose = 1 := by simp

end Transformer.BatchSize

/-
# Fused row-and-column gains

arXiv:2606.25971v2, §3, the displayed factorization, and Appendix A,
Algorithm 2. A matrix is a genuine finite array; its geometric norm is
the Euclidean norm of its entries, rather than the sup norm on functions.
-/

import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Data.Matrix.Mul
import Mathlib.Tactic

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.MagnitudeDirection

variable {m n : ℕ}

/-- Flattening a matrix into Euclidean space, arXiv:2606.25971v2, §2–3,
where the norm is explicitly the Frobenius norm. -/
def flatten (W : Matrix (Fin m) (Fin n) ℝ) : EuclideanSpace ℝ (Fin m × Fin n) :=
  WithLp.toLp 2 (fun p => W p.1 p.2)

/-- Frobenius norm of a weight matrix, arXiv:2606.25971v2, §2–3. -/
def frobeniusNorm (W : Matrix (Fin m) (Fin n) ℝ) : ℝ := ‖flatten W‖

/-- The fused weight `diag(row) D diag(col)`, arXiv:2606.25971v2, §3,
the displayed factorization. -/
def fuse (row : Fin m → ℝ) (col : Fin n → ℝ)
    (D : Matrix (Fin m) (Fin n) ℝ) : Matrix (Fin m) (Fin n) ℝ :=
  fun i j => row i * D i j * col j

/-- Recovering the direction, arXiv:2606.25971v2, Appendix A, Algorithm 2.
Its inverse property requires nonzero gains. -/
def unfuse (row : Fin m → ℝ) (col : Fin n → ℝ)
    (W : Matrix (Fin m) (Fin n) ℝ) : Matrix (Fin m) (Fin n) ℝ :=
  fun i j => W i j / (row i * col j)

/-- The entry formula agrees with the paper's diagonal-matrix product,
arXiv:2606.25971v2, §3. -/
theorem fuse_eq_diagonal (row : Fin m → ℝ) (col : Fin n → ℝ)
    (D : Matrix (Fin m) (Fin n) ℝ) :
    fuse row col D = Matrix.diagonal row * D * Matrix.diagonal col := by
  ext i j
  simp [fuse, Matrix.diagonal_mul, Matrix.mul_diagonal]

/-- Unit gains preserve the initial model, arXiv:2606.25971v2, §4.1.3. -/
theorem fuse_one (D : Matrix (Fin m) (Fin n) ℝ) :
    fuse (fun _ => 1) (fun _ => 1) D = D := by
  ext i j
  simp [fuse]

/-- Unit gains recover the original direction, arXiv:2606.25971v2,
§3.1, Algorithm 1, at unit initialization. -/
@[simp] theorem unfuse_one (W : Matrix (Fin m) (Fin n) ℝ) :
    unfuse (fun _ => 1) (fun _ => 1) W = W := by
  ext i j
  simp [unfuse]

/-- Recovery of the zero weight returns zero, arXiv:2606.25971v2,
Algorithms 1–2. This does not belong to a positive sphere. -/
@[simp] theorem unfuse_zero (row : Fin m → ℝ) (col : Fin n → ℝ) :
    unfuse row col 0 = 0 := by
  ext i j
  simp [unfuse]

/-- A zero direct row gain kills the stored weight, arXiv:2606.25971v2,
§4.1.3, direct gain parametrization. -/
@[simp] theorem fuse_zero_row (col : Fin n → ℝ) (D : Matrix (Fin m) (Fin n) ℝ) :
    fuse (fun _ => 0) col D = 0 := by
  ext i j
  simp [fuse]

/-- Recovering the stored direction is exact for nonzero gains,
arXiv:2606.25971v2, Appendix A, Algorithm 2, line 2. -/
theorem unfuse_fuse (row : Fin m → ℝ) (col : Fin n → ℝ)
    (D : Matrix (Fin m) (Fin n) ℝ)
    (hr : ∀ i, row i ≠ 0) (hc : ∀ j, col j ≠ 0) :
    unfuse row col (fuse row col D) = D := by
  ext i j
  dsimp [unfuse, fuse]
  field_simp [hr i, hc j]

/-- Nonzero row and column gains exist, arXiv:2606.25971v2, Appendix A. -/
example : (∀ i : Fin 1, (fun _ => (1 : ℝ)) i ≠ 0) ∧
    (∀ j : Fin 1, (fun _ => (1 : ℝ)) j ≠ 0) := by norm_num

/-- Re-fusing also recovers an arbitrary stored weight, arXiv:2606.25971v2,
Appendix A, Algorithm 2. -/
theorem fuse_unfuse (row : Fin m → ℝ) (col : Fin n → ℝ)
    (W : Matrix (Fin m) (Fin n) ℝ)
    (hr : ∀ i, row i ≠ 0) (hc : ∀ j, col j ≠ 0) :
    fuse row col (unfuse row col W) = W := by
  ext i j
  dsimp [unfuse, fuse]
  field_simp [hr i, hc j]

/-- The inverse's hypotheses are satisfiable, arXiv:2606.25971v2, Appendix A. -/
example : (∀ i : Fin 1, (fun _ => (1 : ℝ)) i ≠ 0) ∧
    (∀ j : Fin 1, (fun _ => (1 : ℝ)) j ≠ 0) := by norm_num

/-- Frobenius norm is the sum-of-squares matrix norm, arXiv:2606.25971v2, §2. -/
theorem frobeniusNorm_sq (W : Matrix (Fin m) (Fin n) ℝ) :
    frobeniusNorm W ^ 2 = ∑ i, ∑ j, W i j ^ 2 := by
  rw [frobeniusNorm, EuclideanSpace.real_norm_sq_eq]
  simp [flatten, Fintype.sum_prod_type]

/-- The literal Frobenius square-root formula,
arXiv:2606.25971v2, §2, norm convention. -/
theorem frobeniusNorm_eq_sqrt (W : Matrix (Fin m) (Fin n) ℝ) :
    frobeniusNorm W = Real.sqrt (∑ i, ∑ j, W i j ^ 2) := by
  rw [← frobeniusNorm_sq]
  exact (Real.sqrt_sq (show 0 ≤ frobeniusNorm W from norm_nonneg _)).symm

/-- Flattening preserves scalar multiplication, arXiv:2606.25971v2, §3. -/
theorem flatten_smul (a : ℝ) (W : Matrix (Fin m) (Fin n) ℝ) :
    flatten (a • W) = a • flatten W := by
  ext p
  rfl

/-- Nonzero matrices have nonzero Frobenius norm, arXiv:2606.25971v2, §3. -/
theorem frobeniusNorm_pos_iff (W : Matrix (Fin m) (Fin n) ℝ) :
    0 < frobeniusNorm W ↔ W ≠ 0 := by
  rw [frobeniusNorm, norm_pos_iff]
  constructor
  · intro h hW
    apply h
    ext p
    simp [flatten, hW]
  · intro h hW
    apply h
    ext i j
    exact congrArg (fun v : EuclideanSpace ℝ (Fin m × Fin n) => v (i, j)) hW

/-- Frobenius norm scales by the absolute gain, arXiv:2606.25971v2, §3. -/
theorem frobeniusNorm_smul (a : ℝ) (W : Matrix (Fin m) (Fin n) ℝ) :
    frobeniusNorm (a • W) = |a| * frobeniusNorm W := by
  rw [frobeniusNorm, flatten_smul, norm_smul, Real.norm_eq_abs]
  rfl

/-- The scalar specialization fixes the other side's gain to one.
Appendix A says that tying *both* gains to a scalar recovers `W = gamma D`;
that instead gives `gamma² D`. This correct specialization matches §3,
Algorithm 1. Source: arXiv:2606.25971v2, Appendix A, introductory paragraph. -/
theorem scalar_gain_specialization (gamma : ℝ) (D : Matrix (Fin m) (Fin n) ℝ) :
    fuse (fun _ => gamma) (fun _ => 1) D = gamma • D ∧
      fuse (fun _ => gamma) (fun _ => gamma) D = (gamma ^ 2) • D := by
  constructor <;> ext i j <;> simp [fuse, pow_two, mul_comm, mul_left_comm]

/-- The literal shared-scalar prescription in Appendix A fails at `gamma=2`.
Source: arXiv:2606.25971v2, Appendix A, introductory paragraph. -/
theorem shared_scalar_counterexample :
    fuse (fun _ : Fin 1 => (2 : ℝ)) (fun _ : Fin 1 => (2 : ℝ))
      (fun _ _ => 1) ≠ (2 : ℝ) • (fun _ _ => 1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  intro h
  have := congrArg (fun W : Matrix (Fin 1) (Fin 1) ℝ => W 0 0) h
  norm_num [fuse] at this

end Transformer.MagnitudeDirection

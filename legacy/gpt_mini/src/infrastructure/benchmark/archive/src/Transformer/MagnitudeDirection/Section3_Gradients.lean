/-
# Gradient splitting for the fused weight

arXiv:2606.25971v2, §3.1, Algorithm 1, and Appendix A, Algorithm 2.
These are adjoint identities for the actual differential of the factorization.
The following module uses them to prove a loss chain rule.
-/

import Transformer.MagnitudeDirection.Section3_Factorization
import Mathlib.Analysis.Calculus.Deriv.Prod
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.FDeriv.Linear
import Mathlib.Topology.Algebra.Module.FiniteDimension
import Mathlib.Analysis.Matrix.Normed

open scoped BigOperators Matrix.Norms.Elementwise

noncomputable section

namespace Transformer.MagnitudeDirection

variable {m n : ℕ}

/-- Frobenius pairing with a loss gradient, arXiv:2606.25971v2, §3.1. -/
def frobeniusPairing (G H : Matrix (Fin m) (Fin n) ℝ) : ℝ :=
  ∑ i, ∑ j, G i j * H i j

/-- The continuous linear differential encoded by a Frobenius gradient,
arXiv:2606.25971v2, §3.1, `G = partial L / partial W`.
Finite-dimensional sup and Frobenius norms give the same differentiability. -/
def gradientFunctional (G : Matrix (Fin m) (Fin n) ℝ) :
    Matrix (Fin m) (Fin n) ℝ →L[ℝ] ℝ :=
  LinearMap.toContinuousLinearMap
    { toFun := frobeniusPairing G
      map_add' := by intro A B; simp [frobeniusPairing, mul_add, Finset.sum_add_distrib]
      map_smul' := by
        intro a H
        simp only [frobeniusPairing, Matrix.smul_apply, smul_eq_mul, RingHom.id_apply,
          Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro i hi
        apply Finset.sum_congr rfl
        intro j hj
        ring }

/-- The continuous differential has the intended Frobenius pairing,
arXiv:2606.25971v2, §3.1. -/
theorem gradientFunctional_apply (G H : Matrix (Fin m) (Fin n) ℝ) :
    gradientFunctional G H = frobeniusPairing G H := by
  rfl

/-- Direction gradient, arXiv:2606.25971v2, Appendix A, Algorithm 2, line 6. -/
def directionGradient (row : Fin m → ℝ) (col : Fin n → ℝ)
    (G : Matrix (Fin m) (Fin n) ℝ) : Matrix (Fin m) (Fin n) ℝ := fuse row col G

/-- Row gain gradient, arXiv:2606.25971v2, Appendix A, Algorithm 2, line 3. -/
def rowGradient (col : Fin n → ℝ) (D G : Matrix (Fin m) (Fin n) ℝ) : Fin m → ℝ :=
  fun i => ∑ j, D i j * G i j * col j

/-- Column gain gradient, arXiv:2606.25971v2, Appendix A, Algorithm 2, line 4. -/
def colGradient (row : Fin m → ℝ) (D G : Matrix (Fin m) (Fin n) ℝ) : Fin n → ℝ :=
  fun j => ∑ i, row i * D i j * G i j

/-- The actual first variation of the factorization, arXiv:2606.25971v2,
§3 and Appendix A, Algorithm 2. -/
def fusionVariation (row dr : Fin m → ℝ) (col dc : Fin n → ℝ)
    (D H : Matrix (Fin m) (Fin n) ℝ) : Matrix (Fin m) (Fin n) ℝ :=
  fuse row col H + fuse dr col D + fuse row dc D

/-- Adjoint identity giving the direction gradient,
arXiv:2606.25971v2, Appendix A, Algorithm 2, line 6. -/
theorem directionGradient_pairing (row : Fin m → ℝ) (col : Fin n → ℝ)
    (G H : Matrix (Fin m) (Fin n) ℝ) :
    frobeniusPairing G (fuse row col H) = frobeniusPairing (directionGradient row col G) H := by
  apply Finset.sum_congr rfl
  intro i hi
  apply Finset.sum_congr rfl
  intro j hj
  dsimp [frobeniusPairing, directionGradient, fuse]
  ring

/-- Adjoint identity giving each row gain gradient,
arXiv:2606.25971v2, Appendix A, Algorithm 2, line 3. -/
theorem rowGradient_pairing (dr : Fin m → ℝ) (col : Fin n → ℝ)
    (D G : Matrix (Fin m) (Fin n) ℝ) :
    frobeniusPairing G (fuse dr col D) = ∑ i, rowGradient col D G i * dr i := by
  simp only [frobeniusPairing, rowGradient, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro i hi
  apply Finset.sum_congr rfl
  intro j hj
  dsimp [fuse]
  ring

/-- Adjoint identity giving each column gain gradient,
arXiv:2606.25971v2, Appendix A, Algorithm 2, line 4. -/
theorem colGradient_pairing (row : Fin m → ℝ) (dc : Fin n → ℝ)
    (D G : Matrix (Fin m) (Fin n) ℝ) :
    frobeniusPairing G (fuse row dc D) = ∑ j, colGradient row D G j * dc j := by
  simp only [frobeniusPairing, colGradient, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j hj
  apply Finset.sum_congr rfl
  intro i hi
  dsimp [fuse]
  ring

/-- Splitting the gradient preserves the complete directional derivative,
arXiv:2606.25971v2, Appendix A, Algorithm 2, lines 3–6. -/
theorem fusionVariation_pairing (row dr : Fin m → ℝ) (col dc : Fin n → ℝ)
    (D H G : Matrix (Fin m) (Fin n) ℝ) :
    frobeniusPairing G (fusionVariation row dr col dc D H) =
      frobeniusPairing (directionGradient row col G) H +
        (∑ i, rowGradient col D G i * dr i) + (∑ j, colGradient row D G j * dc j) := by
  have hadd (A B : Matrix (Fin m) (Fin n) ℝ) :
      frobeniusPairing G (A + B) = frobeniusPairing G A + frobeniusPairing G B := by
    simp [frobeniusPairing, mul_add, Finset.sum_add_distrib]
  rw [fusionVariation, hadd, hadd, directionGradient_pairing, rowGradient_pairing,
    colGradient_pairing]

/-- The product rule for the actual weight curve, arXiv:2606.25971v2,
Appendix A, Algorithm 2, lines 3–6. -/
theorem fuse_curve_hasDerivAt (row dr : Fin m → ℝ) (col dc : Fin n → ℝ)
    (D H : Matrix (Fin m) (Fin n) ℝ) :
    HasDerivAt (fun t : ℝ => fuse (row + t • dr) (col + t • dc) (D + t • H))
      (fusionVariation row dr col dc D H) 0 := by
  apply hasDerivAt_pi.mpr
  intro i
  apply hasDerivAt_pi.mpr
  intro j
  have hd (a b : ℝ) : HasDerivAt (fun t : ℝ => a + t * b) b 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const b).const_add a
  convert ((hd (row i) (dr i)).mul (hd (D i j) (H i j))).mul (hd (col j) (dc j)) using 1
  · rfl
  · dsimp [fusionVariation, fuse]
    ring

end Transformer.MagnitudeDirection

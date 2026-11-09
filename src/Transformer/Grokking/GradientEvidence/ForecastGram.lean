module

public import Transformer.Grokking.GradientEvidence.ForecastSecant
public import Mathlib.Analysis.CStarAlgebra.Matrix

/-!
# Gram-based amplification bounds

Source: `review/flat_spectrum.md`, *Forecast tightening*, FT2. Bound the
operator norm by the square root of a Gram Frobenius norm. This uses all
Gram entries, including correlations, without assuming an estimated leading
singular vector is exact. Native outward rounding is audited separately.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence
open Finset
open scoped Matrix.Norms.L2Operator

variable {m n : Type*} [Fintype m] [Fintype n]

/-- FT2: squared Frobenius norm with explicit entries, independent of which
norm instance is currently used for matrices. -/
def forecastMatrixEnergy (A : Matrix m n ℝ) : ℝ := ∑ i, ∑ j, A i j ^ 2

/-- FT2: finite rowwise Cauchy--Schwarz bounds the actual matrix action. -/
theorem forecast_matrix_action_sq (A : Matrix m n ℝ) (x : n → ℝ) :
    (∑ i, (∑ j, A i j * x j) ^ 2) ≤ forecastMatrixEnergy A * ∑ j, x j ^ 2 := by
  calc
    _ ≤ ∑ i, (∑ j, A i j ^ 2) * ∑ j, x j ^ 2 := by
      apply sum_le_sum
      intro i hi
      exact sum_mul_sq_le_sq_mul_sq univ (A i) x
    _ = _ := by rw [← sum_mul]; rfl

example := forecast_matrix_action_sq (fun i j : Fin 2 => if i = j then (1 : ℝ) else 0)
  (fun i => if i = 0 then 2 else -1)

variable [DecidableEq n]

/-- FT2: the L2 operator norm is bounded by the explicit Frobenius norm. -/
theorem forecast_matrix_op_le (A : Matrix m n ℝ) :
    ‖A‖ ≤ Real.sqrt (forecastMatrixEnergy A) := by
  have he : 0 ≤ forecastMatrixEnergy A := by unfold forecastMatrixEnergy; positivity
  rw [Matrix.l2_opNorm_def]
  apply ContinuousLinearMap.opNorm_le_bound _ (Real.sqrt_nonneg _)
  intro x
  change ‖(Matrix.toEuclideanLin A) x‖ ≤ Real.sqrt (forecastMatrixEnergy A) * ‖x‖
  have ha := forecast_matrix_action_sq A x
  have hx := EuclideanSpace.real_norm_sq_eq x
  have hy := EuclideanSpace.real_norm_sq_eq ((Matrix.toEuclideanLin A) x)
  change ‖(Matrix.toEuclideanLin A) x‖ ^ 2 = ∑ i, (∑ j, A i j * x j) ^ 2 at hy
  rw [← hy, ← hx] at ha
  calc
    _ = Real.sqrt (‖(Matrix.toEuclideanLin A) x‖ ^ 2) := (Real.sqrt_sq (norm_nonneg _)).symm
    _ ≤ Real.sqrt (forecastMatrixEnergy A * ‖x‖ ^ 2) := Real.sqrt_le_sqrt ha
    _ = _ := by rw [Real.sqrt_mul he, Real.sqrt_sq (norm_nonneg x)]

example := forecast_matrix_op_le (fun i j : Fin 2 => if i = j then (1 : ℝ) else 0)

/-- FT2: the column Gram yields a fourth-root entrywise bound on the
original operator norm. Rectangular matrices are allowed. -/
theorem forecast_column_gram_op_le (A : Matrix m n ℝ) :
    ‖A‖ ≤ Real.sqrt (Real.sqrt (forecastMatrixEnergy (A.conjTranspose * A))) := by
  have h := forecast_matrix_op_le (A.conjTranspose * A)
  rw [Matrix.l2_opNorm_conjTranspose_mul_self] at h
  exact (Real.le_sqrt (norm_nonneg _) (Real.sqrt_nonneg _)).mpr (by nlinarith)

example := forecast_column_gram_op_le (fun i : Fin 3 => fun j : Fin 2 => (i : ℝ) + j)

variable [DecidableEq m]

/-- FT2: the row Gram has the same validity, allowing the implementation
to use the smaller dimension without changing the parameter matrix. -/
theorem forecast_row_gram_op_le (A : Matrix m n ℝ) :
    ‖A‖ ≤ Real.sqrt (Real.sqrt (forecastMatrixEnergy (A * A.conjTranspose))) := by
  have h := forecast_column_gram_op_le A.conjTranspose
  simpa only [Matrix.conjTranspose_conjTranspose, Matrix.l2_opNorm_conjTranspose] using h

example := forecast_row_gram_op_le (fun i : Fin 2 => fun j : Fin 3 => (i : ℝ) - j)

omit [DecidableEq m] in
/-- FT2: an outward-rounded Gram-energy upper bound remains a valid
amplification bound. The hypothesis exposes the native numerical obligation. -/
theorem forecast_gram_upper (A : Matrix m n ℝ) (U : ℝ)
    (hU : forecastMatrixEnergy (A.conjTranspose * A) ≤ U) :
    ‖A‖ ≤ Real.sqrt (Real.sqrt U) := by
  exact (forecast_column_gram_op_le A).trans
    (Real.sqrt_le_sqrt (Real.sqrt_le_sqrt hU))

example := forecast_gram_upper (1 : Matrix (Fin 1) (Fin 1) ℝ) 1
  (by norm_num [forecastMatrixEnergy])

/-- FT2: any certified linear amplification transports the existing
forecast and its actual mixed term. The error is not hidden in a definition. -/
theorem linear_forecast_of_bound {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F]
    (A D : E →L[ℝ] F) (x v y : E) (r B : ℝ)
    (hy : ‖y - (x + v)‖ ≤ r) (hB : ‖A + D‖ ≤ B) :
    ‖(A + D) y - (A x + (A v + D x))‖ ≤ B * r + ‖D v‖ := by
  have hr := (norm_nonneg (y - (x + v))).trans hy
  exact (linear_forecast A D x v y r hy).trans
    (add_le_add (mul_le_mul_of_nonneg_right hB hr) le_rfl)

example := linear_forecast_of_bound (ContinuousLinearMap.id ℝ ℝ)
  (0 : ℝ →L[ℝ] ℝ) 2 1 4 1 1 (by norm_num) (by simp)

end Transformer.Grokking.GradientEvidence

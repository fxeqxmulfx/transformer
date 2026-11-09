module

public import Transformer.Grokking.GradientEvidence.ForecastGram
public import Transformer.Grokking.GradientEvidence.FeatureMetricCE

/-!
# Probability-weighted feature operators

Source: `review/flat_spectrum.md`, *Query-weighted feature covariance*, QG1.
The operator retains the probability ceiling of every minibatch row and the
fixed feature metric. Its squared L2 norm divided by the actual batch size
will replace the worst-probability times feature-trace coefficient.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence.CEGram

open Finset
open scoped Matrix.Norms.L2Operator
variable {I K : Type*} [Fintype I] [Fintype K] [DecidableEq I]

/-- QG1: columns are actual features scaled by their individual probability
ceilings and the class-shared feature metric. -/
def weightedFeatures (q : I → ℝ) (r : K → ℝ) (T : I → K → ℝ) : Matrix K I ℝ :=
  fun k i => Real.sqrt (r k) * T i k * Real.sqrt (q i)

/-- QG1: the spectral coefficient uses the actual nonuniform feature table. -/
def coefficient (q : I → ℝ) (r : K → ℝ) (T : I → K → ℝ) : ℝ :=
  ‖weightedFeatures q r T‖ ^ 2 / Fintype.card I

/-- QG1: a cheaper trace coefficient still retains the rowwise probabilities. -/
def traceCoefficient (q : I → ℝ) (r : K → ℝ) (T : I → K → ℝ) : ℝ :=
  (∑ i, q i * ∑ k, r k * (T i k)^2) / Fintype.card I

/-- QG1: the exact operator coefficient is nonnegative, including empty batches. -/
theorem coefficient_nonneg (q : I → ℝ) (r : K → ℝ) (T : I → K → ℝ) :
    0 ≤ coefficient q r T := div_nonneg (sq_nonneg _) (Nat.cast_nonneg _)

/-- QG1: squared L2 amplification controls the finite matrix action. -/
theorem matrix_action_sq (A : Matrix K I ℝ) (a : I → ℝ) :
    (∑ k, (∑ i, A k i * a i)^2) ≤ ‖A‖^2 * ∑ i, (a i)^2 := by
  let x : EuclideanSpace ℝ I := (EuclideanSpace.equiv I ℝ).symm a
  have h := Matrix.l2_opNorm_mulVec A x
  have hs := (sq_le_sq₀ (norm_nonneg _) (mul_nonneg (norm_nonneg A) (norm_nonneg x))).2 h
  rw [mul_pow, EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq] at hs
  exact hs

/-- QG1: weighted Cauchy--Schwarz in operator form. The probability weights
cancel from the true gradient mean, but remain in its amplification bound. -/
theorem weighted_mean_energy_le (q : I → ℝ) (r : K → ℝ) (T : I → K → ℝ)
    (a : I → ℝ) (hq : ∀ i, 0 < q i) (hr : ∀ k, 0 ≤ r k) :
    (∑ k, r k * ((∑ i, a i * T i k) / Fintype.card I)^2) ≤
      coefficient q r T * ((∑ i, (a i)^2 / q i) / Fintype.card I) := by
  have h := matrix_action_sq (weightedFeatures q r T) (fun i => a i / Real.sqrt (q i))
  have ha (k : K) :
      (∑ i, weightedFeatures q r T k i * (a i / Real.sqrt (q i))) =
        Real.sqrt (r k) * ∑ i, a i * T i k := by
    rw [mul_sum]
    apply sum_congr rfl
    intro i hi
    unfold weightedFeatures
    field_simp [ne_of_gt (Real.sqrt_pos.mpr (hq i))]
  have hk (k : K) : (Real.sqrt (r k) * ∑ i, a i * T i k)^2 =
      r k * (∑ i, a i * T i k)^2 := by rw [mul_pow, Real.sq_sqrt (hr k)]
  have hi (i : I) : (a i / Real.sqrt (q i))^2 = (a i)^2 / q i := by
    rw [div_pow, Real.sq_sqrt (hq i).le]
  simp_rw [ha, hk, hi] at h
  calc
    _ = (∑ k, r k * (∑ i, a i * T i k)^2) / (Fintype.card I : ℝ)^2 := by
      symm
      rw [sum_div]
      apply sum_congr rfl
      intro k hk
      rw [div_pow, mul_div_assoc]
    _ ≤ (‖weightedFeatures q r T‖^2 * ∑ i, (a i)^2 / q i) /
        (Fintype.card I : ℝ)^2 := div_le_div_of_nonneg_right h (sq_nonneg _)
    _ = _ := by unfold coefficient; simp only [div_mul_div_comm, pow_two]

omit [DecidableEq I] in
/-- QG1: the weighted matrix Frobenius energy equals the explicit weighted
feature trace, so no probability weight is replaced by a maximum here. -/
theorem weighted_features_energy (q : I → ℝ) (r : K → ℝ) (T : I → K → ℝ)
    (hq : ∀ i, 0 ≤ q i) (hr : ∀ k, 0 ≤ r k) :
    forecastMatrixEnergy (weightedFeatures q r T) = ∑ i, q i * ∑ k, r k * (T i k)^2 := by
  unfold forecastMatrixEnergy weightedFeatures
  simp_rw [mul_pow, Real.sq_sqrt (hq _), Real.sq_sqrt (hr _)]
  rw [sum_comm]
  simp only [mul_sum]
  apply sum_congr rfl
  intro i hi
  apply sum_congr rfl
  intro k hk
  ring

/-- QG1: the operator coefficient is at most the rowwise trace coefficient. -/
theorem coefficient_le_trace (q : I → ℝ) (r : K → ℝ) (T : I → K → ℝ)
    (hq : ∀ i, 0 ≤ q i) (hr : ∀ k, 0 ≤ r k) :
    coefficient q r T ≤ traceCoefficient q r T := by
  have he : 0 ≤ forecastMatrixEnergy (weightedFeatures q r T) := by
    unfold forecastMatrixEnergy
    positivity
  have h := forecast_matrix_op_le (weightedFeatures q r T)
  have hs := (sq_le_sq₀ (norm_nonneg _) (Real.sqrt_nonneg _)).2 h
  rw [Real.sq_sqrt he, weighted_features_energy q r T hq hr] at hs
  exact div_le_div_of_nonneg_right hs (Nat.cast_nonneg _)

omit [DecidableEq I] in
/-- QG1: replacing every row's probability bound by a common maximum recovers
the previous trace coefficient. The sharper trace never makes it larger. -/
theorem trace_le_uniform (q : I → ℝ) (r : K → ℝ) (T : I → K → ℝ) (Q : ℝ)
    (hr : ∀ k, 0 ≤ r k) (hq : ∀ i, q i ≤ Q) :
    traceCoefficient q r T ≤ Q * (metricFeatureEnergy r T / Fintype.card I) := by
  have h : (∑ i, q i * ∑ k, r k * (T i k)^2) ≤
      ∑ i, Q * ∑ k, r k * (T i k)^2 := by
    exact sum_le_sum fun i _ => mul_le_mul_of_nonneg_right (hq i)
      (sum_nonneg fun k _ => mul_nonneg (hr k) (sq_nonneg (T i k)))
  have hd := div_le_div_of_nonneg_right h (Nat.cast_nonneg (Fintype.card I) :
    (0 : ℝ) ≤ Fintype.card I)
  simpa only [traceCoefficient, metricFeatureEnergy, ← mul_sum, mul_div_assoc] using hd

example := matrix_action_sq (fun i j : Fin 2 => if i = j then (1 : ℝ) else 0)
  (fun i => if i = 0 then 2 else -1)
example := weighted_mean_energy_le (I := Bool) (K := Bool)
  (fun i => if i then 2 else 1) (fun k => if k then 3 else 1)
  (fun i k => if i == k then 2 else -1) (fun i => if i then 3 else -1)
  (by intro i; cases i <;> norm_num) (by intro k; cases k <;> norm_num)
example := weighted_features_energy (I := Bool) (K := Bool)
  (fun i => if i then 2 else 1) (fun k => if k then 3 else 1)
  (fun i k => if i == k then 2 else -1)
  (by intro i; cases i <;> norm_num) (by intro k; cases k <;> norm_num)
example := coefficient_le_trace (I := Bool) (K := Bool)
  (fun i => if i then 2 else 1) (fun k => if k then 3 else 1)
  (fun i k => if i == k then 2 else -1)
  (by intro i; cases i <;> norm_num) (by intro k; cases k <;> norm_num)
example := trace_le_uniform (I := Bool) (K := Bool)
  (fun i => if i then 2 else 1) (fun k => if k then 3 else 1)
  (fun i k => if i == k then 2 else -1) 2
  (by intro k; cases k <;> norm_num) (by intro i; cases i <;> norm_num)

-- Two orthogonal nonzero features have half the trace coefficient.
example : coefficient (fun _ : Fin 2 => (1 : ℝ)) (fun _ : Fin 2 => 1)
    (fun i k => if i = k then 1 else 0) = 1/2 := by
  have h : weightedFeatures (fun _ : Fin 2 => (1 : ℝ)) (fun _ : Fin 2 => 1)
      (fun i k => if i = k then 1 else 0) = (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
    ext i k
    simp [weightedFeatures, Matrix.one_apply, eq_comm]
  rw [coefficient, h]
  norm_num
example : traceCoefficient (fun _ : Fin 2 => (1 : ℝ)) (fun _ : Fin 2 => 1)
    (fun i k => if i = k then 1 else 0) = 1 := by
  norm_num [traceCoefficient, Fin.sum_univ_two]

end Transformer.Grokking.GradientEvidence.CEGram

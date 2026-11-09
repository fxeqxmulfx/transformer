module

public import Transformer.Grokking.GradientEvidence.CEGram.Gram

/-!
# Transport a weighted feature operator from a checkpoint

Source: `review/flat_spectrum.md`, *Query-weighted feature covariance*, QG3.
All input features may move. Every row retains its own probability ceiling
and feature-motion radius, while the feature metric remains fixed. Numerical
execution errors must already be included in the supplied feature radii.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence.CEGram

open Finset
open scoped Matrix.Norms.L2Operator
variable {I K : Type*} [Fintype I] [Fintype K] [DecidableEq I]

omit [DecidableEq I] in
/-- QG3: complete feature motion bounds the exact weighted matrix difference
by a sum retaining each example's probability ceiling and radius. -/
theorem difference_energy_le (q : I → ℝ) (r : K → ℝ) (T T' : I → K → ℝ)
    (R : ℝ) (S : I → ℝ) (hq : ∀ i, 0 ≤ q i) (hr : ∀ k, 0 ≤ r k)
    (hR : 0 ≤ R) (hrR : ∀ k, r k ≤ R)
    (hS : ∀ i, ∑ k, (T' i k-T i k)^2 ≤ (S i)^2) :
    forecastMatrixEnergy (weightedFeatures q r T'-weightedFeatures q r T) ≤
      R * ∑ i, q i * (S i)^2 := by
  have heq : weightedFeatures q r T'-weightedFeatures q r T =
      weightedFeatures q r (fun i k => T' i k-T i k) := by
    ext k i
    simp only [weightedFeatures, Matrix.sub_apply]
    ring
  rw [heq, weighted_features_energy q r _ hq hr, mul_sum]
  apply sum_le_sum
  intro i hi
  have ht : (∑ k, r k * (T' i k-T i k)^2) ≤ R * (S i)^2 := by
    calc
      _ ≤ ∑ k, R * (T' i k-T i k)^2 :=
        sum_le_sum fun k _ => mul_le_mul_of_nonneg_right (hrR k) (sq_nonneg _)
      _ = R * ∑ k, (T' i k-T i k)^2 := (mul_sum _ _ _).symm
      _ ≤ R * (S i)^2 := mul_le_mul_of_nonneg_left (hS i) hR
  have h := mul_le_mul_of_nonneg_left ht (hq i)
  nlinarith

/-- QG3: the same explicit sum bounds operator-norm motion by Frobenius motion. -/
theorem difference_norm_le (q : I → ℝ) (r : K → ℝ) (T T' : I → K → ℝ)
    (R : ℝ) (S : I → ℝ) (hq : ∀ i, 0 ≤ q i) (hr : ∀ k, 0 ≤ r k)
    (hR : 0 ≤ R) (hrR : ∀ k, r k ≤ R)
    (hS : ∀ i, ∑ k, (T' i k-T i k)^2 ≤ (S i)^2) :
    ‖weightedFeatures q r T'-weightedFeatures q r T‖ ≤ Real.sqrt (R * ∑ i, q i * (S i)^2) :=
  (forecast_matrix_op_le _).trans
    (Real.sqrt_le_sqrt (difference_energy_le q r T T' R S hq hr hR hrR hS))

/-- QG3: an old operator cap and finite feature-motion radii give a future
actual-CE coefficient, without observing the future feature table. -/
theorem coefficient_transport (q : I → ℝ) (r : K → ℝ) (T T' : I → K → ℝ)
    (R B : ℝ) (S : I → ℝ) (hq : ∀ i, 0 ≤ q i) (hr : ∀ k, 0 ≤ r k)
    (hR : 0 ≤ R) (hrR : ∀ k, r k ≤ R)
    (hB : ‖weightedFeatures q r T‖ ≤ B)
    (hS : ∀ i, ∑ k, (T' i k-T i k)^2 ≤ (S i)^2) :
    coefficient q r T' ≤ (B + Real.sqrt (R * ∑ i, q i * (S i)^2))^2 / Fintype.card I := by
  have hd := difference_norm_le q r T T' R S hq hr hR hrR hS
  have h : ‖weightedFeatures q r T'‖ ≤ B + Real.sqrt (R * ∑ i, q i * (S i)^2) := by
    calc
      _ = ‖weightedFeatures q r T + (weightedFeatures q r T'-weightedFeatures q r T)‖ := by
        congr 1
        abel
      _ ≤ ‖weightedFeatures q r T‖ + ‖weightedFeatures q r T'-weightedFeatures q r T‖ :=
        norm_add_le _ _
      _ ≤ _ := add_le_add hB hd
  have hb : 0 ≤ B := (norm_nonneg _).trans hB
  have hs := (sq_le_sq₀ (norm_nonneg _) (add_nonneg hb (Real.sqrt_nonneg _))).2 h
  exact div_le_div_of_nonneg_right hs (Nat.cast_nonneg _)

variable [DecidableEq K]

/-- QG3: the native polynomial Gram upper bound supplies the old amplification
in the finite-motion coefficient. The numerical upper-bound obligation is explicit. -/
theorem coefficient_transport_gram (q : I → ℝ) (r : K → ℝ) (T T' : I → K → ℝ)
    (R U : ℝ) (S : I → ℝ) (hq : ∀ i, 0 ≤ q i) (hr : ∀ k, 0 ≤ r k)
    (hR : 0 ≤ R) (hrR : ∀ k, r k ≤ R) (hU : gramEnergy q r T ≤ U)
    (hS : ∀ i, ∑ k, (T' i k-T i k)^2 ≤ (S i)^2) :
    coefficient q r T' ≤
      (Real.sqrt (Real.sqrt U) + Real.sqrt (R * ∑ i, q i * (S i)^2))^2 / Fintype.card I :=
  coefficient_transport q r T T' R _ S hq hr hR hrR (op_le_gram_upper q r T U hq hr hU) hS

omit [DecidableEq K] in
/-- QG3: a pointwise future energy cap retains the mean of the individual
probability ceilings, rather than replacing them by their largest value. -/
theorem coefficient_trace_cap (q : I → ℝ) (r : K → ℝ) (T : I → K → ℝ)
    (R M : ℝ) (hq : ∀ i, 0 ≤ q i) (hr : ∀ k, 0 ≤ r k)
    (hR : 0 ≤ R) (hrR : ∀ k, r k ≤ R) (hT : ∀ i, ∑ k, (T i k)^2 ≤ M) :
    coefficient q r T ≤ R*M*((∑ i, q i)/Fintype.card I) := by
  apply (coefficient_le_trace q r T hq hr).trans
  have ht (i : I) : (∑ k, r k*(T i k)^2) ≤ R*M := by
    calc
      _ ≤ ∑ k, R*(T i k)^2 :=
        sum_le_sum fun k _ => mul_le_mul_of_nonneg_right (hrR k) (sq_nonneg _)
      _ = R*(∑ k, (T i k)^2) := (mul_sum _ _ _).symm
      _ ≤ R*M := mul_le_mul_of_nonneg_left (hT i) hR
  have h : (∑ i, q i * ∑ k, r k*(T i k)^2) ≤ ∑ i, q i*(R*M) :=
    sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (ht i) (hq i)
  calc
    _ ≤ (∑ i, q i*(R*M))/Fintype.card I :=
      div_le_div_of_nonneg_right h (Nat.cast_nonneg _)
    _ = _ := by rw [← sum_mul, mul_comm _ (R*M), mul_div_assoc]

/-- QG3: intersect transported Gram and pointwise trace bounds. Neither
branch assumes future feature alignment or favorable encoder motion. -/
theorem coefficient_transport_min (q : I → ℝ) (r : K → ℝ) (T T' : I → K → ℝ)
    (R M U : ℝ) (S : I → ℝ) (hq : ∀ i, 0 ≤ q i) (hr : ∀ k, 0 ≤ r k)
    (hR : 0 ≤ R) (hrR : ∀ k, r k ≤ R) (hU : gramEnergy q r T ≤ U)
    (hS : ∀ i, ∑ k, (T' i k-T i k)^2 ≤ (S i)^2)
    (hT : ∀ i, ∑ k, (T' i k)^2 ≤ M) :
    coefficient q r T' ≤ min
      ((Real.sqrt (Real.sqrt U) + Real.sqrt (R * ∑ i, q i * (S i)^2))^2 / Fintype.card I)
      (R*M*((∑ i, q i)/Fintype.card I)) :=
  le_min (coefficient_transport_gram q r T T' R U S hq hr hR hrR hU hS)
    (coefficient_trace_cap q r T' R M hq hr hR hrR hT)

example := difference_energy_le (I := Fin 1) (K := Fin 1)
  (fun _ => 1/2) (fun _ => 3) (fun _ _ => 2) (fun _ _ => 3) 3 (fun _ => 1)
  (by intro i; norm_num) (by intro k; norm_num) (by norm_num)
  (by intro k; norm_num) (by intro i; norm_num)
example := difference_norm_le (I := Fin 1) (K := Fin 1)
  (fun _ => 1/2) (fun _ => 3) (fun _ _ => 2) (fun _ _ => 3) 3 (fun _ => 1)
  (by intro i; norm_num) (by intro k; norm_num) (by norm_num)
  (by intro k; norm_num) (by intro i; norm_num)
example := coefficient_transport (I := Fin 1) (K := Fin 1)
  (fun _ => 1) (fun _ => 1) (fun _ _ => 2) (fun _ _ => 3) 1 2 (fun _ => 1)
  (by intro i; norm_num) (by intro k; norm_num) (by norm_num)
  (by intro k; norm_num)
  (by
    have h := forecast_matrix_op_le (weightedFeatures (fun _ : Fin 1 => (1 : ℝ))
      (fun _ : Fin 1 => 1) (fun _ _ => 2))
    norm_num [forecastMatrixEnergy, weightedFeatures] at h
    exact h)
  (by intro i; norm_num)
example := coefficient_transport_gram (I := Fin 1) (K := Fin 1)
  (fun _ => 1) (fun _ => 1) (fun _ _ => 2) (fun _ _ => 3) 1 16 (fun _ => 1)
  (by intro i; norm_num) (by intro k; norm_num) (by norm_num)
  (by intro k; norm_num) (by norm_num [gramEnergy, covariance]) (by intro i; norm_num)
example := coefficient_trace_cap (I := Fin 1) (K := Fin 1)
  (fun _ => 1/2) (fun _ => 3) (fun _ _ => 2) 3 4
  (by intro i; norm_num) (by intro k; norm_num) (by norm_num)
  (by intro k; norm_num) (by intro i; norm_num)
example := coefficient_transport_min (I := Fin 1) (K := Fin 1)
  (fun _ => 1) (fun _ => 1) (fun _ _ => 2) (fun _ _ => 3) 1 9 16 (fun _ => 1)
  (by intro i; norm_num) (by intro k; norm_num) (by norm_num)
  (by intro k; norm_num) (by norm_num [gramEnergy, covariance])
  (by intro i; norm_num) (by intro i; norm_num)

end Transformer.Grokking.GradientEvidence.CEGram

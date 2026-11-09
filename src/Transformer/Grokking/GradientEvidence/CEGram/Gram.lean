module

public import Transformer.Grokking.GradientEvidence.CEGram.Basic

/-!
# Computable weighted feature Gram bounds

Source: `review/flat_spectrum.md`, *Query-weighted feature covariance*, QG3.
The explicit covariance avoids square roots inside its sum. Feature weights
enter its Gram energy as r_k*r_l, so every correlation remains observable.
FT2 supplies a certified operator cap; no estimated singular vector is used.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence.CEGram

open Finset
open scoped Matrix.Norms.L2Operator
variable {I K : Type*} [Fintype I] [Fintype K] [DecidableEq I] [DecidableEq K]

/-- QG3: the unnormalized probability-weighted feature covariance. -/
def covariance (q : I → ℝ) (T : I → K → ℝ) : Matrix K K ℝ :=
  fun k l => ∑ i, q i * T i k * T i l

/-- QG3: squared Frobenius energy of the full weighted feature Gram. -/
def gramEnergy (q : I → ℝ) (r : K → ℝ) (T : I → K → ℝ) : ℝ :=
  ∑ k, ∑ l, r k * r l * (covariance q T k l)^2

omit [Fintype K] [DecidableEq I] [DecidableEq K] in
/-- QG3: each actual Gram entry is an explicitly weighted covariance entry. -/
theorem row_gram_entry (q : I → ℝ) (r : K → ℝ) (T : I → K → ℝ)
    (hq : ∀ i, 0 ≤ q i) (k l : K) :
    (weightedFeatures q r T * (weightedFeatures q r T).conjTranspose) k l =
      Real.sqrt (r k) * Real.sqrt (r l) * covariance q T k l := by
  simp only [Matrix.mul_apply, Matrix.conjTranspose_apply, star_trivial, weightedFeatures]
  unfold covariance
  rw [mul_sum]
  apply sum_congr rfl
  intro i hi
  calc
    _ = (Real.sqrt (r k) * Real.sqrt (r l) * (T i k * T i l)) *
        (Real.sqrt (q i))^2 := by ring
    _ = _ := by rw [Real.sq_sqrt (hq i)]; ring

omit [DecidableEq I] [DecidableEq K] in
/-- QG3: the explicit polynomial covariance formula is exactly the weighted
Gram energy, including every off-diagonal entry. -/
theorem row_gram_energy (q : I → ℝ) (r : K → ℝ) (T : I → K → ℝ)
    (hq : ∀ i, 0 ≤ q i) (hr : ∀ k, 0 ≤ r k) :
    forecastMatrixEnergy (weightedFeatures q r T * (weightedFeatures q r T).conjTranspose) =
      gramEnergy q r T := by
  unfold forecastMatrixEnergy gramEnergy
  simp_rw [row_gram_entry q r T hq, mul_pow, Real.sq_sqrt (hr _)]

/-- QG3: all weighted correlations supply a finite fourth-root operator cap. -/
theorem op_le_gram (q : I → ℝ) (r : K → ℝ) (T : I → K → ℝ)
    (hq : ∀ i, 0 ≤ q i) (hr : ∀ k, 0 ≤ r k) :
    ‖weightedFeatures q r T‖ ≤ Real.sqrt (Real.sqrt (gramEnergy q r T)) := by
  have h := forecast_row_gram_op_le (weightedFeatures q r T)
  rw [row_gram_energy q r T hq hr] at h
  exact h

/-- QG3: a verified upper bound on the polynomial Gram energy gives the
operator cap used by the prospective constructor. -/
theorem op_le_gram_upper (q : I → ℝ) (r : K → ℝ) (T : I → K → ℝ) (U : ℝ)
    (hq : ∀ i, 0 ≤ q i) (hr : ∀ k, 0 ≤ r k) (hU : gramEnergy q r T ≤ U) :
    ‖weightedFeatures q r T‖ ≤ Real.sqrt (Real.sqrt U) :=
  (op_le_gram q r T hq hr).trans (Real.sqrt_le_sqrt (Real.sqrt_le_sqrt hU))

/-- QG3: the corresponding actual-CE coefficient uses the square root of
Gram energy divided by the real batch size. -/
theorem coefficient_le_gram (q : I → ℝ) (r : K → ℝ) (T : I → K → ℝ)
    (hq : ∀ i, 0 ≤ q i) (hr : ∀ k, 0 ≤ r k) :
    coefficient q r T ≤ Real.sqrt (gramEnergy q r T) / Fintype.card I := by
  have h := (sq_le_sq₀ (norm_nonneg _) (Real.sqrt_nonneg _)).2 (op_le_gram q r T hq hr)
  rw [Real.sq_sqrt (Real.sqrt_nonneg _)] at h
  exact div_le_div_of_nonneg_right h (Nat.cast_nonneg _)

example := row_gram_entry (I := Fin 1) (K := Fin 1)
  (fun _ => 1/2) (fun _ => 3) (fun _ _ => 2) (by intro i; norm_num) 0 0
example := row_gram_energy (I := Fin 1) (K := Fin 1)
  (fun _ => 1/2) (fun _ => 3) (fun _ _ => 2)
  (by intro i; norm_num) (by intro k; norm_num)
example := op_le_gram (I := Fin 1) (K := Fin 1)
  (fun _ => 1/2) (fun _ => 3) (fun _ _ => 2)
  (by intro i; norm_num) (by intro k; norm_num)
example := op_le_gram_upper (I := Fin 1) (K := Fin 1)
  (fun _ => 1/2) (fun _ => 3) (fun _ _ => 2) 36
  (by intro i; norm_num) (by intro k; norm_num)
  (by norm_num [gramEnergy, covariance])
example := coefficient_le_gram (I := Fin 1) (K := Fin 1)
  (fun _ => 1/2) (fun _ => 3) (fun _ _ => 2)
  (by intro i; norm_num) (by intro k; norm_num)

-- The computable Gram cap also improves the trace for a flat covariance.
example : gramEnergy (fun _ : Fin 2 => (1 : ℝ)) (fun _ : Fin 2 => 1)
    (fun i k => if i = k then 1 else 0) = 2 := by
  norm_num [gramEnergy, covariance, Fin.sum_univ_two]

end Transformer.Grokking.GradientEvidence.CEGram

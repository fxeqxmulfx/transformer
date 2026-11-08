module

public import Transformer.Grokking.GradientEvidence.ReadoutGradient

/-!
# Training loss bounds the projected readout gradient

Source: `review/flat_spectrum.md`, *Residual learning dynamics*, RD2.
For any fixed linear map Q, the actual readout gradient obeys
`||QG||_F^2 <= 2 * training_CE * mean_training ||Qh||^2`.
The feature factor includes every training row, not only the answer orbit
used to select a subspace. This bounds gradient forcing, not residual size.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.GradientEvidence

open Finset Geometry FlatSpectrum.FeatureCodes Landscape.Neural

variable {I C K : Type*} [Fintype I] [Fintype C] [Fintype K]

/-- A finite linear map, including a training-basis residual map, RD2. -/
def matrixApply (Q : K → K → ℝ) (x : K → ℝ) (k : K) : ℝ := linearScore (Q k) x

/-- Squared Frobenius norm of the explicit row array, RD2. -/
def matrixEnergy (M : C → K → ℝ) : ℝ := ∑ c, ∑ k, M c k ^ 2

omit [Fintype I] in
/-- The coordinate energy is nonnegative, RD2. -/
theorem matrixEnergy_nonneg (M : C → K → ℝ) : 0 ≤ matrixEnergy M :=
  sum_nonneg fun c _ => sum_nonneg fun k _ => sq_nonneg (M c k)

omit [Fintype I] [Fintype C] in
/-- Linearity keeps decay and gradient forcing separate, RD2–RD3. -/
theorem matrixApply_sub_smul (Q : K → K → ℝ) (x g : K → ℝ) (a b : ℝ) (k : K) :
    matrixApply Q (fun l => a * x l - b * g l) k =
      a * matrixApply Q x k - b * matrixApply Q g k := by
  simp only [matrixApply, linearScore, mul_sub, sum_sub_distrib, mul_sum]
  congr 1 <;> apply sum_congr rfl <;> intro l hl <;> ring

/-- Cauchy–Schwarz for a product of two finite row arrays, RD2. -/
theorem product_energy_le (A : I → C → ℝ) (R : I → K → ℝ) :
    matrixEnergy (fun c k => ∑ i, A i c * R i k) ≤ matrixEnergy A * matrixEnergy R := by
  calc
    _ ≤ ∑ c, ∑ k, (∑ i, A i c ^ 2) * ∑ i, R i k ^ 2 := by
      apply sum_le_sum
      intro c hc
      exact sum_le_sum fun k hk => sum_mul_sq_le_sq_mul_sq univ (fun i => A i c) (fun i => R i k)
    _ = (∑ c, ∑ i, A i c ^ 2) * ∑ k, ∑ i, R i k ^ 2 := by
      symm
      rw [sum_mul]
      apply sum_congr rfl
      intro c hc
      exact mul_sum _ _ _
    _ = _ := by
      rw [sum_comm, sum_comm (f := fun k i => R i k ^ 2)]
      rfl

/-- Averaging divides both training factors by the same row count, RD2. -/
theorem mean_product_energy_le (A : I → C → ℝ) (R : I → K → ℝ) :
    matrixEnergy (fun c k => (∑ i, A i c * R i k) / Fintype.card I) ≤
      (matrixEnergy A / Fintype.card I) * (matrixEnergy R / Fintype.card I) := by
  have h := div_le_div_of_nonneg_right (product_energy_le A R)
    (sq_nonneg (Fintype.card I : ℝ))
  simpa only [matrixEnergy, div_pow, ← sum_div, div_mul_div_comm, ← pow_two] using h

variable [DecidableEq C] [DecidableEq K]

/-- The projected derivative is the probability-error mean of projected
features. This follows from the actual CE derivative, RD2. -/
theorem projected_readoutGradient (T : I → K → ℝ) (y : I → C)
    (E : C → K → ℝ) (Q : K → K → ℝ) (c : C) (k : K) :
    matrixApply Q (readoutGradient T y E c) k =
      (∑ i, (probability (fun v => linearScore (E v) (T i)) c -
        if c = y i then 1 else 0) * matrixApply Q (T i) k) / Fintype.card I := by
  unfold matrixApply
  conv_lhs =>
    arg 2
    ext l
    rw [readoutGradient_eq]
  rw [linearScore_mean]
  congr 1
  apply sum_congr rfl
  intro i hi
  simp only [linearScore, mul_sum]
  apply sum_congr rfl
  intro l hl
  ring

/-- **Small CE bounds readout-gradient leakage into any fixed directions**,
RD2. This does not bound encoder gradients, Adam-normalized updates, current
readout residuals or future feature/subspace drift. -/
theorem projected_readoutGradient_energy_le (T : I → K → ℝ) (y : I → C)
    (E : C → K → ℝ) (Q : K → K → ℝ) :
    matrixEnergy (fun c => matrixApply Q (readoutGradient T y E c)) ≤
      2 * readoutLoss T y E *
        (matrixEnergy (fun i => matrixApply Q (T i)) / Fintype.card I) := by
  let A := fun i c => probability (fun v => linearScore (E v) (T i)) c -
    if c = y i then 1 else 0
  let R := fun i => matrixApply Q (T i)
  have ha : matrixEnergy A / Fintype.card I ≤ 2 * readoutLoss T y E := by
    have he := sum_le_sum (fun i (_ : i ∈ (univ : Finset I)) =>
      probability_error_le_twice_loss (fun v => linearScore (E v) (T i)) (y i))
    have hd := div_le_div_of_nonneg_right he (Nat.cast_nonneg (Fintype.card I) :
      (0 : ℝ) ≤ Fintype.card I)
    simpa only [A, matrixEnergy, readoutLoss, meanCrossEntropy, card_univ,
      ← mul_sum, mul_div_assoc] using hd
  calc
    _ = matrixEnergy (fun c k => (∑ i, A i c * R i k) / Fintype.card I) := by
      simp only [matrixEnergy, projected_readoutGradient, A, R]
    _ ≤ (matrixEnergy A / Fintype.card I) * (matrixEnergy R / Fintype.card I) :=
      mean_product_energy_le A R
    _ ≤ (2 * readoutLoss T y E) * (matrixEnergy R / Fintype.card I) :=
      mul_le_mul_of_nonneg_right ha (div_nonneg (matrixEnergy_nonneg R) (Nat.cast_nonneg _))

example := projected_readoutGradient_energy_le
  (fun (i : Bool) (_ : Fin 1) => if i then (2 : ℝ) else 1) id
  (fun (c : Bool) (_ : Fin 1) => if c then (1 : ℝ) else -1)
  (fun (_ : Fin 1) (_ : Fin 1) => 1)

end Transformer.Grokking.GradientEvidence

module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.AnswerContrast

/-!
# Contrast control from training-state transport

Source: `review/flat_spectrum.md`, *Training-state transport*, TT1–TT3.
A query state is reconstructed from centered training states. Its residual
is retained explicitly, including directions outside the training span.
Cauchy–Schwarz on the joint training/residual coordinates bounds each linear
contrast using training contrast variation and the query's transport energy.
No query target error or query contrast appears on the bound's right side.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open Finset

variable {I K : Type*} [Fintype I] [Fintype K]

/-- A real linear score, including a difference of two tied readout rows, TT1. -/
def linearScore (w x : K → ℝ) : ℝ := ∑ k, w k * x k

/-- The part of the query displacement not reconstructed by training states,
TT1. This is computed from states; it is not an assumed prediction error. -/
def transportResidual (T : I → K → ℝ) (center x : K → ℝ) (a : I → ℝ) (k : K) : ℝ :=
  x k - center k - ∑ i, a i * (T i k - center k)

/-- Squared reconstruction energy at positive scale `s`, TT2. -/
def transportEnergy (a : I → ℝ) (r : K → ℝ) (s : ℝ) : ℝ :=
  (∑ i, a i ^ 2) + (∑ k, r k ^ 2) / s ^ 2

/-- Training contrast variation plus the residual-direction penalty, TT2.
The center need not be a mean for the theorem; the diagnostic uses the mean. -/
def trainingContrastEnergy (T : I → K → ℝ) (center w : K → ℝ) (s : ℝ) : ℝ :=
  (∑ i, (linearScore w (T i) - linearScore w center) ^ 2) + s ^ 2 * ∑ k, w k ^ 2

omit [Fintype I] in
/-- The Euclidean training-center baseline, TT2. This bound uses the query
displacement from the center instead of the norm of the complete query state. -/
theorem center_contrast_error_le (center x w : K → ℝ) :
    |linearScore w x - linearScore w center| ≤
      Real.sqrt ((∑ k, (x k - center k) ^ 2) * ∑ k, w k ^ 2) := by
  apply Real.abs_le_sqrt
  have h := sum_mul_sq_le_sq_mul_sq univ w (fun k => x k - center k)
  simpa only [linearScore, mul_sub, sum_sub_distrib, mul_comm] using h

/-- **Exact transport decomposition of a linear contrast**, TT1. -/
theorem linearScore_transport (T : I → K → ℝ) (center x w : K → ℝ) (a : I → ℝ) :
    linearScore w x - linearScore w center =
      (∑ i, a i * (linearScore w (T i) - linearScore w center)) +
        linearScore w (transportResidual T center x a) := by
  have hc : ∑ k, w k * (∑ i, a i * (T i k - center k)) =
      ∑ i, a i * (linearScore w (T i) - linearScore w center) := by
    simp only [mul_sum]
    rw [sum_comm]
    apply sum_congr rfl
    intro i hi
    calc
      _ = a i * ∑ k, w k * (T i k - center k) := by
        rw [mul_sum]
        apply sum_congr rfl
        intro k hk
        ring
      _ = _ := by simp only [linearScore, mul_sub, sum_sub_distrib]
  simp only [linearScore, transportResidual, mul_sub, sum_sub_distrib] at *
  rw [hc]
  ring

/-- **The joint training/residual Cauchy–Schwarz bound**, TT2. Scaling the
residual in one factor and the weights inversely in the other preserves all
terms. Dropping the residual would invalidate the bound outside the span. -/
theorem weighted_transport_sq_le (a d : I → ℝ) (r w : K → ℝ)
    {s : ℝ} (hs : s ≠ 0) :
    ((∑ i, a i * d i) + ∑ k, r k * w k) ^ 2 ≤
      transportEnergy a r s * ((∑ i, d i ^ 2) + s ^ 2 * ∑ k, w k ^ 2) := by
  have h := sum_mul_sq_le_sq_mul_sq univ
    (Sum.elim a (fun k => r k / s)) (Sum.elim d (fun k => s * w k))
  simp only [Fintype.sum_sum_type, Sum.elim_inl, Sum.elim_inr] at h
  have hm (k : K) : r k / s * (s * w k) = r k * w k := by
    rw [← mul_assoc, div_mul_cancel₀ _ hs]
  simp_rw [hm, div_pow, mul_pow] at h
  simpa only [transportEnergy, sum_div, ← mul_sum] using h

/-- **Query contrast error from training evidence and state transport**, TT3.
The right side uses training contrasts, readout weights and the query state,
without requiring the query's target label or target error. -/
theorem training_transport_error_le (T : I → K → ℝ) (center x w : K → ℝ)
    (a : I → ℝ) {s : ℝ} (hs : s ≠ 0) :
    |linearScore w x - linearScore w center| ≤
      Real.sqrt (transportEnergy a (transportResidual T center x a) s *
        trainingContrastEnergy T center w s) := by
  apply Real.abs_le_sqrt
  rw [linearScore_transport T center x w a]
  simpa only [trainingContrastEnergy, linearScore, mul_comm] using
    weighted_transport_sq_le a (fun i => linearScore w (T i) - linearScore w center)
      (transportResidual T center x a) w hs

example := linearScore_transport (I := Bool) (K := Bool)
  (fun i k => if k then (if i then 1 else -1) else 1)
  (fun k => if k then 0 else 1) (fun k => if k then 1 / 4 else 9 / 8)
  (fun k => if k then 0 else 2) (fun i => if i then 1 / 8 else -1 / 8)

example := center_contrast_error_le (K := Bool) (fun k => if k then 0 else 1)
  (fun k => if k then 1 / 4 else 9 / 8) (fun k => if k then 0 else 2)

example := weighted_transport_sq_le (I := Bool) (K := Bool)
  (fun i => if i then 1 / 8 else -1 / 8) (fun _ => 0)
  (fun k => if k then 0 else 1 / 8) (fun k => if k then 0 else 2)
  (s := 1 / 2) (by norm_num)

example := training_transport_error_le (I := Bool) (K := Bool)
  (fun i k => if k then (if i then 1 else -1) else 1)
  (fun k => if k then 0 else 1) (fun k => if k then 1 / 4 else 9 / 8)
  (fun k => if k then 0 else 2) (fun i => if i then 1 / 8 else -1 / 8)
  (s := 1 / 2) (by norm_num)

/-- Training-span reconstruction alone cannot control an unseen direction,
TT2. The training variations are zero although the query contrast changes. -/
theorem residual_penalty_necessary :
    let T : Fin 1 → Fin 1 → ℝ := fun _ _ => 0
    let center : Fin 1 → ℝ := fun _ => 0
    let x : Fin 1 → ℝ := fun _ => 1
    let w : Fin 1 → ℝ := fun _ => 1
    (∑ i, (linearScore w (T i) - linearScore w center) ^ 2) = 0 ∧
      |linearScore w x - linearScore w center| = 1 := by
  norm_num [linearScore]

/-- A positive margin budget with a genuinely nonzero reconstruction residual,
TT5. The query displacement includes a direction absent from the training
variations, and the residual penalty still leaves a strictly positive margin. -/
theorem training_transport_nonzero_witness :
    let T : Bool → Bool → ℝ := fun i k => if k then (if i then 1 else -1) else 1
    let center : Bool → ℝ := fun k => if k then 0 else 1
    let x : Bool → ℝ := fun k => if k then 1 / 4 else 9 / 8
    let w : Bool → ℝ := fun k => if k then 0 else 2
    let a : Bool → ℝ := fun i => if i then 1 / 8 else -1 / 8
    transportResidual T center x a false = 1 / 8 ∧
      Real.sqrt (transportEnergy a (transportResidual T center x a) (1 / 2) *
        trainingContrastEnergy T center w (1 / 2)) < linearScore w center := by
  norm_num [transportResidual, transportEnergy, trainingContrastEnergy, linearScore]
  rw [← Real.sqrt_div (by norm_num : (0 : ℝ) ≤ 3)]
  exact (Real.sqrt_lt (by norm_num : (0 : ℝ) ≤ 3 / 32)
    (by norm_num : (0 : ℝ) ≤ 2)).mpr (by norm_num)

example := training_transport_nonzero_witness

end Transformer.Grokking.FlatSpectrum.FeatureCodes

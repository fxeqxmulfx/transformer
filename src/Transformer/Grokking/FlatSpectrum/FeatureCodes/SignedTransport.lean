module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.ProbabilityTransfer

/-!
# Signed training reconstruction and directional residuals

Source: `review/flat_spectrum.md`, *Signed directional transport*, ST1–ST3.
Retain the signed training contribution and bound only the unreconstructed
state. A second decomposition removes readout directions represented in a
training basis. Its orthogonality defect remains explicit, so no numerical
orthogonality or successful optimizer discovery is assumed.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open Finset

variable {I J K : Type*} [Fintype I] [Fintype J] [Fintype K]

/-- Signed reconstruction of a score from training scores, ST1. -/
def transportPrediction (T : I → K → ℝ) (center w : K → ℝ) (a : I → ℝ) : ℝ :=
  linearScore w center + ∑ i, a i * (linearScore w (T i) - linearScore w center)

/-- Part of the readout not represented by the selected training basis, ST2. -/
def readoutResidual (B : J → K → ℝ) (w : K → ℝ) (b : J → ℝ) (k : K) : ℝ :=
  w k - ∑ j, b j * B j k

/-- Residual bound retaining the measured geometric orthogonality defect,
ST2. Query scores against the original readout do not appear on this side. -/
def directionalResidualBound (B : J → K → ℝ) (r w : K → ℝ) (b : J → ℝ) : ℝ :=
  Real.sqrt ((∑ k, r k ^ 2) * ∑ k, readoutResidual B w b k ^ 2) +
    ∑ j, |b j| * |linearScore (B j) r|

omit [Fintype J] in
/-- **The training contribution keeps its sign**, ST1. -/
theorem linearScore_signed_residual (T : I → K → ℝ) (center x w : K → ℝ)
    (a : I → ℝ) :
    linearScore w x = transportPrediction T center w a +
      linearScore w (transportResidual T center x a) := by
  have h := linearScore_transport T center x w a
  unfold transportPrediction
  linarith

omit [Fintype J] in
/-- Signed reconstruction with the full readout norm as a baseline, ST1.
Only the residual is bounded; the reconstructed contribution is not replaced
by an absolute-value or covariance penalty. -/
theorem signed_transport_norm_le (T : I → K → ℝ) (center x w : K → ℝ) (a : I → ℝ) :
    |linearScore w x - transportPrediction T center w a| ≤
      Real.sqrt ((∑ k, transportResidual T center x a k ^ 2) * ∑ k, w k ^ 2) := by
  have h := linearScore_signed_residual T center x w a
  have he : linearScore w x - transportPrediction T center w a =
      linearScore w (transportResidual T center x a) := by linarith
  rw [he]
  simpa [linearScore] using center_contrast_error_le (fun _ => 0)
    (transportResidual T center x a) w

omit [Fintype I] in
/-- Exact readout decomposition for any basis and coefficients, ST2. -/
theorem linearScore_readout_residual (B : J → K → ℝ) (r w : K → ℝ) (b : J → ℝ) :
    linearScore w r = linearScore (readoutResidual B w b) r +
      ∑ j, b j * linearScore (B j) r := by
  have hs : (∑ k, (∑ j, b j * B j k) * r k) =
      ∑ j, b j * linearScore (B j) r := by
    simp_rw [sum_mul]
    rw [sum_comm]
    apply sum_congr rfl
    intro j hj
    simp only [linearScore, mul_sum, mul_assoc]
  simp only [linearScore, readoutResidual, sub_mul, sum_sub_distrib] at *
  rw [hs]
  ring

omit [Fintype I] in
/-- **Two residual norms plus the explicit orthogonality defect**, ST2.
This inequality does not require exact orthonormality of a computed basis. -/
theorem directional_residual_abs_le (B : J → K → ℝ) (r w : K → ℝ) (b : J → ℝ) :
    |linearScore w r| ≤ directionalResidualBound B r w b := by
  have hc : |linearScore (readoutResidual B w b) r| ≤
      Real.sqrt ((∑ k, r k ^ 2) * ∑ k, readoutResidual B w b k ^ 2) := by
    apply Real.abs_le_sqrt
    simpa only [linearScore, mul_comm] using
      sum_mul_sq_le_sq_mul_sq univ r (readoutResidual B w b)
  have hs : |∑ j, b j * linearScore (B j) r| ≤ ∑ j, |b j| * |linearScore (B j) r| := by
    simpa only [abs_mul] using abs_sum_le_sum_abs (fun j => b j * linearScore (B j) r) univ
  rw [linearScore_readout_residual B r w b]
  exact (abs_add_le _ _).trans (add_le_add hc hs)

/-- The signed training prediction has a directional error bound, ST3. -/
theorem signed_transport_abs_le (T : I → K → ℝ) (center x w : K → ℝ) (a : I → ℝ)
    (B : J → K → ℝ) (b : J → ℝ) :
    |linearScore w x - transportPrediction T center w a| ≤
      directionalResidualBound B (transportResidual T center x a) w b := by
  have h := linearScore_signed_residual T center x w a
  have he : linearScore w x - transportPrediction T center w a =
      linearScore w (transportResidual T center x a) := by linarith
  rw [he]
  exact directional_residual_abs_le B _ w b

/-- The one-sided bound used for competitor log odds, ST3. -/
theorem signed_transport_upper (T : I → K → ℝ) (center x w : K → ℝ) (a : I → ℝ)
    (B : J → K → ℝ) (b : J → ℝ) :
    linearScore w x ≤ transportPrediction T center w a +
      directionalResidualBound B (transportResidual T center x a) w b := by
  have h := (abs_le.mp (signed_transport_abs_le T center x w a B b)).2
  linarith

omit [Fintype J] in
/-- Exact reconstruction merely re-expresses the original score, ST4.
It does not show that a low-rank representation or the task was learned. -/
theorem exact_transport_of_zero_residual (T : I → K → ℝ) (center x w : K → ℝ)
    (a : I → ℝ) (h : ∀ k, transportResidual T center x a k = 0) :
    linearScore w x = transportPrediction T center w a := by
  rw [linearScore_signed_residual T center x w a]
  simp only [linearScore, h, mul_zero, sum_const_zero, add_zero]

omit [Fintype I] in
/-- Exact residual orthogonality removes the defect, ST2. The diagnostic
keeps the defect instead of assuming this hypothesis in floating point. -/
theorem orthogonal_residual_abs_le (B : J → K → ℝ) (r w : K → ℝ) (b : J → ℝ)
    (h : ∀ j, linearScore (B j) r = 0) :
    |linearScore w r| ≤
      Real.sqrt ((∑ k, r k ^ 2) * ∑ k, readoutResidual B w b k ^ 2) := by
  have hd := directional_residual_abs_le B r w b
  simpa only [directionalResidualBound, h, abs_zero, mul_zero, sum_const_zero, add_zero] using hd

/-- Dropping the defect for a nonorthogonal residual gives a false zero
bound even with a unit basis, ST2. -/
theorem orthogonality_defect_necessary :
    let B : Fin 1 → Fin 1 → ℝ := fun _ _ => 1
    let r : Fin 1 → ℝ := fun _ => 1
    let w : Fin 1 → ℝ := fun _ => 1
    let b : Fin 1 → ℝ := fun _ => 1
    (∑ k, readoutResidual B w b k ^ 2) = 0 ∧ |linearScore w r| = 1 := by
  norm_num [readoutResidual, linearScore]

example := exact_transport_of_zero_residual (fun (_ : Fin 1) (_ : Fin 1) => (2 : ℝ))
  (fun _ => 1) (fun _ => 2) (fun _ => 3) (fun _ => 1)
  (by intro k; norm_num [transportResidual])
example := orthogonal_residual_abs_le (fun (_ : Fin 1) (k : Bool) => if k then (0 : ℝ) else 1)
  (fun k => if k then 1 / 4 else 0) (fun k => if k then 1 else -2) (fun _ => -2)
  (by intro j; norm_num [linearScore])
example := orthogonality_defect_necessary

end Transformer.Grokking.FlatSpectrum.FeatureCodes

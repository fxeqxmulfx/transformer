module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.SignedProbability

/-!
# Signed training transport across a forecast step

Source: `review/flat_spectrum.md`, *Training-to-forecast bridge*, TF1–TF2.
Keep the old training reconstruction fixed, retain the signed readout and
feature changes including their mixed term, and bound only the forecast
remainder. The radius can be supplied by FT1/FT2; an independently valid
FT3 endpoint gives a second bridge. No optimizer discovery is assumed.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open Finset
variable {I J K : Type*} [Fintype I] [Fintype J] [Fintype K]

/-- TF1: the updated readout retains its product with the feature tangent. -/
def forecastScoreChange (w next x tangent : K → ℝ) : ℝ :=
  linearScore (fun k => next k - w k) x + linearScore next tangent

omit [Fintype I] [Fintype J] in
/-- TF1: exact finite expansion, with no discarded mixed term. -/
theorem linearScore_forecast_identity (w next x tangent future : K → ℝ) :
    linearScore next future = linearScore w x + forecastScoreChange w next x tangent +
      linearScore next (fun k => future k - (x k + tangent k)) := by
  simp only [linearScore, forecastScoreChange, sub_mul, mul_sub, mul_add,
    sum_sub_distrib, sum_add_distrib]
  ring

omit [Fintype I] [Fintype J] in
/-- TF1: a finite squared Euclidean forecast radius controls any readout. -/
theorem linearScore_forecast_error_le (next x tangent future : K → ℝ) {r : ℝ}
    (hr : 0 ≤ r) (hball : (∑ k, (future k - (x k + tangent k)) ^ 2) ≤ r ^ 2) :
    |linearScore next (fun k => future k - (x k + tangent k))| ≤
      Real.sqrt (∑ k, next k ^ 2) * r := by
  have hc := center_contrast_error_le (fun _ : K => 0)
    (fun k => future k - (x k + tangent k)) next
  have hz : linearScore next (fun _ : K => 0) = 0 := by simp [linearScore]
  rw [hz, sub_zero] at hc
  simp only [sub_zero] at hc
  have hn : 0 ≤ ∑ k, next k ^ 2 := sum_nonneg (fun k _ => sq_nonneg (next k))
  have hs := Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_right hball hn)
  rw [Real.sqrt_mul (sq_nonneg r), Real.sqrt_sq hr] at hs
  simpa only [mul_comm] using hc.trans hs

/-- TF1: old signed geometry plus computable change and forecast remainder. -/
def transportForecastUpper (T : I → K → ℝ) (center x w next tangent : K → ℝ)
    (a : I → ℝ) (B : J → K → ℝ) (b : J → ℝ) (r : ℝ) : ℝ :=
  transportPrediction T center w a +
    directionalResidualBound B (transportResidual T center x a) w b +
    forecastScoreChange w next x tangent + Real.sqrt (∑ k, next k ^ 2) * r

/-- TF1: the training-to-forecast bound controls the actual future score. -/
theorem signed_transport_forecast_upper (T : I → K → ℝ)
    (center x w next tangent future : K → ℝ) (a : I → ℝ)
    (B : J → K → ℝ) (b : J → ℝ) {r : ℝ} (hr : 0 ≤ r)
    (hball : (∑ k, (future k - (x k + tangent k)) ^ 2) ≤ r ^ 2) :
    linearScore next future ≤ transportForecastUpper T center x w next tangent a B b r := by
  have hold := signed_transport_upper T center x w a B b
  have herror := (abs_le.mp (linearScore_forecast_error_le next x tangent future hr hball)).2
  rw [linearScore_forecast_identity w next x tangent future]
  unfold transportForecastUpper
  linarith

/-- TF2: transport an independent new-score endpoint while retaining old
training geometry. The old score is used only to express the signed drift. -/
def transportEndpointUpper (T : I → K → ℝ) (center x w : K → ℝ)
    (a : I → ℝ) (B : J → K → ℝ) (b : J → ℝ) (upper : ℝ) : ℝ :=
  transportPrediction T center w a +
    directionalResidualBound B (transportResidual T center x a) w b +
    (upper - linearScore w x)

/-- TF1/TF3: a separately bounded executed-feature discrepancy adds to the
forecast radius; a scalar readout discrepancy is then added only once.
For a class contrast the scalar allowance contains both logit errors. -/
theorem signed_transport_forecast_execution_upper (T : I → K → ℝ)
    (center x w next tangent future measured : K → ℝ) (a : I → ℝ)
    (B : J → K → ℝ) (b : J → ℝ) {r δ η executed : ℝ}
    (hr : 0 ≤ r) (hδ : 0 ≤ δ)
    (hball : (∑ k, (future k - (x k + tangent k)) ^ 2) ≤ r ^ 2)
    (hfeature : (∑ k, (measured k - future k) ^ 2) ≤ δ ^ 2)
    (hreadout : |executed - linearScore next measured| ≤ η) :
    executed ≤ transportForecastUpper T center x w next tangent a B b (r + δ) + η := by
  have hold := signed_transport_forecast_upper T center x w next tangent future a B b hr hball
  have hf := linearScore_forecast_error_le next future (fun _ => 0) measured hδ
    (by simpa only [add_zero] using hfeature)
  simp only [add_zero] at hf
  have he := (abs_le.mp hf).2
  have hn := (abs_le.mp hreadout).2
  have hid : linearScore next (fun k => measured k - future k) =
      linearScore next measured - linearScore next future := by
    simp only [linearScore, mul_sub, sum_sub_distrib]
  rw [hid] at he
  unfold transportForecastUpper at hold ⊢
  rw [mul_add]
  linarith

/-- TF2: any valid FT3 endpoint transports through the old training bound. -/
theorem signed_transport_endpoint_upper (T : I → K → ℝ) (center x w : K → ℝ)
    (a : I → ℝ) (B : J → K → ℝ) (b : J → ℝ) {future upper : ℝ}
    (hfuture : future ≤ upper) :
    future ≤ transportEndpointUpper T center x w a B b upper := by
  have hold := signed_transport_upper T center x w a B b
  unfold transportEndpointUpper
  linarith

/-- TF2: this composition cannot beat the independent endpoint's tightness.
Its added slack records the remaining training-to-query explanation. -/
theorem endpoint_le_transportEndpointUpper (T : I → K → ℝ) (center x w : K → ℝ)
    (a : I → ℝ) (B : J → K → ℝ) (b : J → ℝ) (upper : ℝ) :
    upper ≤ transportEndpointUpper T center x w a B b upper := by
  have hold := signed_transport_upper T center x w a B b
  unfold transportEndpointUpper
  linarith

/-- TF1: nonzero training residual, feature movement, readout movement and
forecast remainder can jointly leave a strict negative competitor score. -/
theorem transport_forecast_nonzero_witness :
    let T : Bool → Bool → ℝ := fun i k => if k then 0 else if i then 3 / 2 else 1 / 2
    let center : Bool → ℝ := fun k => if k then 0 else 1
    let x : Bool → ℝ := fun k => if k then 1 / 4 else 3
    let w : Bool → ℝ := fun k => if k then 1 else -2
    let next : Bool → ℝ := fun k => if k then 0 else -3
    let tangent : Bool → ℝ := fun k => if k then 0 else 1 / 4
    let future : Bool → ℝ := fun k => if k then 1 / 4 else 27 / 8
    let a : Bool → ℝ := fun i => if i then 2 else -2
    let B : Fin 1 → Bool → ℝ := fun _ k => if k then 0 else 1
    let b : Fin 1 → ℝ := fun _ => -2
    transportResidual T center x a true = 1 / 4 ∧
      (∑ k, (future k - (x k + tangent k)) ^ 2) = (1 / 8 : ℝ) ^ 2 ∧
      transportForecastUpper T center x w next tangent a B b (1 / 8) = -75 / 8 ∧
      linearScore next future = -81 / 8 := by
  norm_num [transportResidual, transportForecastUpper, transportPrediction,
    directionalResidualBound, readoutResidual, forecastScoreChange, linearScore]

example := linearScore_forecast_identity (fun _ : Fin 1 => (2 : ℝ))
  (fun _ => 3) (fun _ => 2) (fun _ => -1) (fun _ => 3 / 2)
example := linearScore_forecast_error_le (fun _ : Fin 1 => (3 : ℝ))
  (fun _ => 2) (fun _ => -1) (fun _ => 3 / 2) (r := 1 / 2)
  (by norm_num) (by norm_num)
example := signed_transport_forecast_upper (fun (_ : Fin 1) (_ : Fin 1) => (1 : ℝ))
  (fun _ => 1) (fun _ => 2) (fun _ => -2) (fun _ => -3) (fun _ => 1)
  (fun _ => 7 / 2) (fun _ => 0) (fun (_ : Fin 1) _ => 1) (fun _ => -2)
  (r := 1 / 2) (by norm_num) (by norm_num)
example := signed_transport_forecast_execution_upper
  (fun (_ : Fin 1) (_ : Fin 1) => (1 : ℝ))
  (fun _ => 1) (fun _ => 2) (fun _ => -2) (fun _ => -3) (fun _ => 1)
  (fun _ => 7 / 2) (fun _ => 15 / 4) (fun _ => 0)
  (fun (_ : Fin 1) _ => 1) (fun _ => -2)
  (r := 1 / 2) (δ := 1 / 4) (η := 1 / 8) (executed := -89 / 8)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num [linearScore])
example := signed_transport_endpoint_upper (fun (_ : Fin 1) (_ : Fin 1) => (1 : ℝ))
  (fun _ => 1) (fun _ => 2) (fun _ => -2) (fun _ => 0)
  (fun (_ : Fin 1) _ => 1) (fun _ => -2) (future := -7) (upper := -6) (by norm_num)
example := endpoint_le_transportEndpointUpper (fun (_ : Fin 1) (_ : Fin 1) => (1 : ℝ))
  (fun _ => 1) (fun _ => 2) (fun _ => -2) (fun _ => 0)
  (fun (_ : Fin 1) _ => 1) (fun _ => -2) (-6)
example := transport_forecast_nonzero_witness

end Transformer.Grokking.FlatSpectrum.FeatureCodes

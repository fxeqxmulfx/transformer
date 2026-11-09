module

public import Transformer.Rotate90.ComplexProduct.SignedTransport
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.TransportForecastOutcomes

/-!
# Training-to-forecast bridge for the actual complex-product network

Source: `review/flat_spectrum.md`, *Training-to-forecast bridge*, TF4.
Both old and new features are the source's actual four-normalization states.
The rounded proposed parameters may be an actual AdamW update, but the bound
does not assume how they were found. FT propagation supplies the feature-ball
premise and FT3 supplies the alternative endpoint. Backend errors stay explicit.
-/

@[expose] public section
noncomputable section
namespace Transformer.Rotate90.ComplexProduct

open Landscape.Neural Grokking.FlatSpectrum.FeatureCodes
open scoped BigOperators
variable {I J V K H : Type*} [Fintype I] [Fintype J] [Fintype K] [Fintype H]

/-- TF4: a forecast of the actual normalized future state transports an old
training certificate through the changed tied readout, including execution. -/
theorem executed_gap_le_transport_forecast (eps : ℝ) (old next : Weights V K H)
    (T : I → Channels K → ℝ) (center tangent : Channels K → ℝ)
    (left right target competitor : V) (a : I → ℝ)
    (B : J → Channels K → ℝ) (b : V → J → ℝ) (executed : V → ℝ) {r η : ℝ}
    (hr : 0 ≤ r)
    (hball : (∑ k, (readoutState eps next left right k -
      (readoutState eps old left right k + tangent k)) ^ 2) ≤ r ^ 2)
    (herr : ∀ v, |executed v - realLogit eps next left right v| ≤ η) :
    executed competitor - executed target ≤ transportForecastGapUpper
      (fun v => realify (old.embedding v)) (fun v => realify (next.embedding v))
      T center (readoutState eps old left right) tangent a B b r η target competitor := by
  apply executed_gap_le_transportForecast _ _ T center _ tangent
    (readoutState eps next left right) a B b executed target competitor hr hball
  intro v
  simpa only [readoutState_logit] using herr v

/-- TF4: a directional normalized forecast endpoint composes with training
transport without requiring a future feature evaluation on the right side. -/
theorem executed_gap_le_transport_endpoint (eps : ℝ) (old : Weights V K H)
    (T : I → Channels K → ℝ) (center : Channels K → ℝ)
    (left right target competitor : V) (a : I → ℝ)
    (B : J → Channels K → ℝ) (b : J → ℝ) (executed : V → ℝ) {upper : ℝ}
    (hfuture : executed competitor - executed target ≤ upper) :
    executed competitor - executed target ≤ transportEndpointUpper T center
      (readoutState eps old left right)
      (fun k => realify (old.embedding competitor) k - realify (old.embedding target) k)
      a B b upper := by
  exact signed_transport_endpoint_upper T center (readoutState eps old left right)
    (fun k => realify (old.embedding competitor) k - realify (old.embedding target) k)
    a B b hfuture

/-- TF4: source-normalized forecast error, executed-feature error and arithmetic
readout error remain distinct. This yields the native allowance
`norm(updated contrast) * feature_error + 2 * logit_error` exactly. -/
theorem executed_gap_le_transport_forecast_with_feature_error
    (eps : ℝ) (old next : Weights V K H)
    (T : I → Channels K → ℝ) (center tangent measured : Channels K → ℝ)
    (left right target competitor : V) (a : I → ℝ)
    (B : J → Channels K → ℝ) (b : V → J → ℝ) (executed : V → ℝ) {r δ η : ℝ}
    (hr : 0 ≤ r) (hδ : 0 ≤ δ)
    (hball : (∑ k, (readoutState eps next left right k -
      (readoutState eps old left right k + tangent k)) ^ 2) ≤ r ^ 2)
    (hfeature : (∑ k, (measured k - readoutState eps next left right k) ^ 2) ≤ δ ^ 2)
    (herr : ∀ v, |executed v - linearScore (realify (next.embedding v)) measured| ≤ η) :
    executed competitor - executed target ≤ transportForecastGapUpper
      (fun v => realify (old.embedding v)) (fun v => realify (next.embedding v))
      T center (readoutState eps old left right) tangent a B b (r + δ) η target competitor := by
  apply signed_transport_forecast_execution_upper T center (readoutState eps old left right)
    (fun k => realify (old.embedding competitor) k - realify (old.embedding target) k)
    (fun k => realify (next.embedding competitor) k - realify (next.embedding target) k)
    tangent (readoutState eps next left right) measured a B (b competitor) hr hδ hball hfeature
  have hc := abs_le.mp (herr competitor)
  have ht := abs_le.mp (herr target)
  rw [abs_le]
  simp only [linearScore, sub_mul, Finset.sum_sub_distrib] at hc ht ⊢
  constructor <;> linarith

-- The active source has nonzero complex branch parameters. These source
-- adapter witnesses use an exact zero-step ball; TF1 has the separate explicit
-- witness with nonzero readout motion, tangent, residual and forecast error.
example := executed_gap_le_transport_forecast (I := Fin 1) (J := Fin 1)
  1 activeWeights activeWeights (fun _ => readoutState 1 activeWeights 0 0)
  (readoutState 1 activeWeights 0 0) (fun _ => 0) 0 1 0 1 (fun _ => 0)
  (fun _ => readoutState 1 activeWeights 0 0) (fun _ _ => 0)
  (realLogit 1 activeWeights 0 1) (r := 0) (η := 0) (by norm_num)
  (by simp) (by intro v; simp)
example := executed_gap_le_transport_endpoint (I := Fin 1) (J := Fin 1)
  1 activeWeights (fun _ => readoutState 1 activeWeights 0 0)
  (readoutState 1 activeWeights 0 0) 0 1 0 1 (fun _ => 0)
  (fun _ => readoutState 1 activeWeights 0 0) (fun _ => 0)
  (realLogit 1 activeWeights 0 1) le_rfl
example := executed_gap_le_transport_forecast_with_feature_error
  (I := Fin 1) (J := Fin 1) 1 activeWeights activeWeights
  (fun _ => readoutState 1 activeWeights 0 0) (readoutState 1 activeWeights 0 0)
  (fun _ => 0) (readoutState 1 activeWeights 0 1) 0 1 0 1 (fun _ => 0)
  (fun _ => readoutState 1 activeWeights 0 0) (fun _ _ => 0)
  (realLogit 1 activeWeights 0 1) (r := 0) (δ := 0) (η := 0)
  (by norm_num) (by norm_num) (by simp) (by simp)
  (by intro v; simp only [readoutState_logit, sub_self, abs_zero, le_refl])

end Transformer.Rotate90.ComplexProduct

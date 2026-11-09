module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.TransportForecast

/-!
# Executed margins, probabilities and loss across training transport

Source: `review/flat_spectrum.md`, *Training-to-forecast bridge*, TF3.
Every competitor retains its own old signed geometry and updated readout.
Execution contributes both logit errors. A probability or loss guarantee
uses all competitors, including a future change of the worst competitor.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open Finset Geometry Landscape.Neural
variable {I J K C : Type*} [Fintype I] [Fintype J] [Fintype K]

/-- TF3: complete forecast bound on one executed competitor-minus-target gap. -/
def transportForecastGapUpper (E next : C → K → ℝ) (T : I → K → ℝ)
    (center x tangent : K → ℝ) (a : I → ℝ) (B : J → K → ℝ)
    (b : C → J → ℝ) (r η : ℝ) (t c : C) : ℝ :=
  transportForecastUpper T center x (fun k => E c k - E t k)
    (fun k => next c k - next t k) tangent a B (b c) r + 2 * η

/-- TF3: old training transport reaches the actual executed future gap. -/
theorem executed_gap_le_transportForecast (E next : C → K → ℝ) (T : I → K → ℝ)
    (center x tangent future : K → ℝ) (a : I → ℝ) (B : J → K → ℝ)
    (b : C → J → ℝ) (executed : C → ℝ) {r η : ℝ} (t c : C)
    (hr : 0 ≤ r) (hball : (∑ k, (future k - (x k + tangent k)) ^ 2) ≤ r ^ 2)
    (herr : ∀ v, |executed v - linearScore (next v) future| ≤ η) :
    executed c - executed t ≤ transportForecastGapUpper E next T center x tangent a B b r η t c := by
  have h := signed_transport_forecast_upper T center x (fun k => E c k - E t k)
    (fun k => next c k - next t k) tangent future a B (b c) hr hball
  have hc := (abs_le.mp (herr c)).2
  have ht := (abs_le.mp (herr t)).1
  simp only [linearScore, sub_mul, sum_sub_distrib] at h
  simp only [linearScore] at hc ht
  unfold transportForecastGapUpper
  linarith

/-- TF3: checking every competitor proves a margin for the future execution. -/
theorem margin_of_transportForecast (E next : C → K → ℝ) (T : I → K → ℝ)
    (center x tangent future : K → ℝ) (a : I → ℝ) (B : J → K → ℝ)
    (b : C → J → ℝ) (executed : C → ℝ) {r η margin : ℝ} (t : C)
    (hr : 0 ≤ r) (hball : (∑ k, (future k - (x k + tangent k)) ^ 2) ≤ r ^ 2)
    (herr : ∀ v, |executed v - linearScore (next v) future| ≤ η)
    (hmargin : ∀ c, c ≠ t →
      transportForecastGapUpper E next T center x tangent a B b r η t c ≤ -margin) :
    ∀ c, c ≠ t → margin ≤ executed t - executed c := by
  intro c hc
  have h := executed_gap_le_transportForecast E next T center x tangent future a B b
    executed t c hr hball herr
  linarith [hmargin c hc]

/-- TF3: a positive certified margin gives a strict correct winner. -/
theorem strict_winner_of_transportForecast (E next : C → K → ℝ) (T : I → K → ℝ)
    (center x tangent future : K → ℝ) (a : I → ℝ) (B : J → K → ℝ)
    (b : C → J → ℝ) (executed : C → ℝ) {r η : ℝ} (t : C)
    (hr : 0 ≤ r) (hball : (∑ k, (future k - (x k + tangent k)) ^ 2) ≤ r ^ 2)
    (herr : ∀ v, |executed v - linearScore (next v) future| ≤ η)
    (hnegative : ∀ c, c ≠ t →
      transportForecastGapUpper E next T center x tangent a B b r η t c < 0) :
    ∀ c, c ≠ t → executed c < executed t := by
  intro c hc
  have h := executed_gap_le_transportForecast E next T center x tangent future a B b
    executed t c hr hball herr
  linarith [hnegative c hc]

variable [Fintype C] [DecidableEq C]

/-- TF3: the same pre-evaluation bounds supply a future target probability floor. -/
theorem probability_ge_transportForecast (E next : C → K → ℝ) (T : I → K → ℝ)
    (center x tangent future : K → ℝ) (a : I → ℝ) (B : J → K → ℝ)
    (b : C → J → ℝ) (executed : C → ℝ) {r η : ℝ} (t : C)
    (hr : 0 ≤ r) (hball : (∑ k, (future k - (x k + tangent k)) ^ 2) ≤ r ^ 2)
    (herr : ∀ v, |executed v - linearScore (next v) future| ≤ η) :
    1 / (1 + oddsEnvelope (transportForecastGapUpper E next T center x tangent a B b r η t) t)
      ≤ probability executed t := by
  apply probability_ge_of_contrast_bounds
  intro c hc
  exact executed_gap_le_transportForecast E next T center x tangent future a B b executed t c
    hr hball herr

/-- TF3: all-query CE ceilings retain uncertified queries as well. -/
theorem crossEntropy_le_transportForecast (E next : C → K → ℝ) (T : I → K → ℝ)
    (center x tangent future : K → ℝ) (a : I → ℝ) (B : J → K → ℝ)
    (b : C → J → ℝ) (executed : C → ℝ) {r η : ℝ} (t : C)
    (hr : 0 ≤ r) (hball : (∑ k, (future k - (x k + tangent k)) ^ 2) ≤ r ^ 2)
    (herr : ∀ v, |executed v - linearScore (next v) future| ≤ η) :
    crossEntropy executed t ≤ Real.log (1 + oddsEnvelope
      (transportForecastGapUpper E next T center x tangent a B b r η t) t) := by
  apply crossEntropy_le_of_contrast_bounds
  intro c hc
  exact executed_gap_le_transportForecast E next T center x tangent future a B b executed t c
    hr hball herr

-- Nonzero readout movement, tangent and forecast error, with two competing classes.
private def oldCode (c : Bool) (_ : Fin 1) : ℝ := if c then 2 else 0
private def nextCode (c : Bool) (_ : Fin 1) : ℝ := if c then 3 else 0
private def futureScores (c : Bool) : ℝ := if c then 21 / 2 else 0

example := executed_gap_le_transportForecast oldCode nextCode
  (fun (_ : Fin 1) (_ : Fin 1) => (1 : ℝ)) (fun _ => 1) (fun _ => 2) (fun _ => 1)
  (fun _ => 7 / 2) (fun _ => 0) (fun (_ : Fin 1) _ => 1) (fun c _ => if c then 0 else -2)
  futureScores (r := 1 / 2) (η := 0) true false (by norm_num) (by norm_num)
  (by intro c; cases c <;> norm_num [futureScores, nextCode, linearScore])
example := margin_of_transportForecast oldCode nextCode
  (fun (_ : Fin 1) (_ : Fin 1) => (1 : ℝ)) (fun _ => 1) (fun _ => 2) (fun _ => 1)
  (fun _ => 7 / 2) (fun _ => 0) (fun (_ : Fin 1) _ => 1) (fun c _ => if c then 0 else -2)
  futureScores (r := 1 / 2) (η := 0) (margin := 3) true (by norm_num) (by norm_num)
  (by intro c; cases c <;> norm_num [futureScores, nextCode, linearScore])
  (by intro c hc; cases c <;> norm_num [transportForecastGapUpper, transportForecastUpper,
      transportPrediction, directionalResidualBound, transportResidual, readoutResidual,
      forecastScoreChange, linearScore, oldCode, nextCode] at *)
example := strict_winner_of_transportForecast oldCode nextCode
  (fun (_ : Fin 1) (_ : Fin 1) => (1 : ℝ)) (fun _ => 1) (fun _ => 2) (fun _ => 1)
  (fun _ => 7 / 2) (fun _ => 0) (fun (_ : Fin 1) _ => 1) (fun c _ => if c then 0 else -2)
  futureScores (r := 1 / 2) (η := 0) true (by norm_num) (by norm_num)
  (by intro c; cases c <;> norm_num [futureScores, nextCode, linearScore])
  (by intro c hc; cases c <;> norm_num [transportForecastGapUpper, transportForecastUpper,
      transportPrediction, directionalResidualBound, transportResidual, readoutResidual,
      forecastScoreChange, linearScore, oldCode, nextCode] at *)
example := probability_ge_transportForecast oldCode nextCode
  (fun (_ : Fin 1) (_ : Fin 1) => (1 : ℝ)) (fun _ => 1) (fun _ => 2) (fun _ => 1)
  (fun _ => 7 / 2) (fun _ => 0) (fun (_ : Fin 1) _ => 1) (fun c _ => if c then 0 else -2)
  futureScores (r := 1 / 2) (η := 0) true (by norm_num) (by norm_num)
  (by intro c; cases c <;> norm_num [futureScores, nextCode, linearScore])
example := crossEntropy_le_transportForecast oldCode nextCode
  (fun (_ : Fin 1) (_ : Fin 1) => (1 : ℝ)) (fun _ => 1) (fun _ => 2) (fun _ => 1)
  (fun _ => 7 / 2) (fun _ => 0) (fun (_ : Fin 1) _ => 1) (fun c _ => if c then 0 else -2)
  futureScores (r := 1 / 2) (η := 0) true (by norm_num) (by norm_num)
  (by intro c; cases c <;> norm_num [futureScores, nextCode, linearScore])

end Transformer.Grokking.FlatSpectrum.FeatureCodes

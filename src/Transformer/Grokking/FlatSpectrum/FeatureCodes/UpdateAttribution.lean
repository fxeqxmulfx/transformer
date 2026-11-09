module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.TransportForecast

/-!
# Signed attribution of a finite proposed update

Source: `review/flat_spectrum.md`, *Observable spectral geometry*, OG5.
Decompose the actual readout increment and old feature tangent into signed
components. The full forecast change retains the mixed readout/feature term,
and the future score also retains its nonlinear remainder. These identities
attribute a known update; they do not assert that a component causes eventual
discovery or that positive readout credit implies a positive complete change.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open Finset
variable {I K : Type*} [Fintype I] [Fintype K]

/-- Each component contributes both direct readout motion and feature motion
through the old readout, OG5. -/
def componentScoreChange (w x dw v : K → ℝ) : ℝ :=
  linearScore dw x + linearScore w v

/-- Distributing a readout increment over its finite components, OG5. -/
theorem linearScore_sum_readout (dw : I → K → ℝ) (x : K → ℝ) :
    linearScore (fun k => ∑ i, dw i k) x = ∑ i, linearScore (dw i) x := by
  simp only [linearScore, sum_mul]
  rw [sum_comm]

/-- Distributing a feature tangent over its finite components, OG5. -/
theorem linearScore_sum_feature (w : K → ℝ) (v : I → K → ℝ) :
    linearScore w (fun k => ∑ i, v i k) = ∑ i, linearScore w (v i) := by
  simp only [linearScore, mul_sum]
  rw [sum_comm]

/-- Exact signed attribution of the proposed change, including every mixed
component through the product of total increments, OG5. -/
theorem forecast_component_identity (w next x tangent : K → ℝ)
    (dw v : I → K → ℝ)
    (hreadout : ∀ k, next k - w k = ∑ i, dw i k)
    (hfeature : ∀ k, tangent k = ∑ i, v i k) :
    forecastScoreChange w next x tangent =
      (∑ i, componentScoreChange w x (dw i) (v i)) +
        linearScore (fun k => next k - w k) tangent := by
  have hr : linearScore (fun k => next k - w k) x = ∑ i, linearScore (dw i) x := by
    simp_rw [hreadout]
    exact linearScore_sum_readout dw x
  have hf : linearScore w tangent = ∑ i, linearScore w (v i) := by
    have ht : tangent = fun k => ∑ i, v i k := funext hfeature
    rw [ht, linearScore_sum_feature]
  have hm : linearScore next tangent = linearScore w tangent +
      linearScore (fun k => next k - w k) tangent := by
    simp only [linearScore, sub_mul, sum_sub_distrib]
    ring
  unfold forecastScoreChange
  rw [hm, hr, hf]
  simp only [componentScoreChange, sum_add_distrib]
  ring

/-- The finite actual score keeps a separate nonlinear feature remainder,
OG5. Directional component attribution alone is not the full future score. -/
theorem finite_update_component_identity (w next x tangent future : K → ℝ)
    (dw v : I → K → ℝ)
    (hreadout : ∀ k, next k - w k = ∑ i, dw i k)
    (hfeature : ∀ k, tangent k = ∑ i, v i k) :
    linearScore next future = linearScore w x +
      ((∑ i, componentScoreChange w x (dw i) (v i)) +
        linearScore (fun k => next k - w k) tangent) +
      linearScore next (fun k => future k - (x k + tangent k)) := by
  rw [linearScore_forecast_identity w next x tangent future,
    forecast_component_identity w next x tangent dw v hreadout hfeature]

/-- OG5: positive credit for one component is compatible with a negative
complete update. A readout contribution cannot stand in for the actual step. -/
theorem positive_component_negative_update_witness :
    let w : Fin 1 → ℝ := fun _ => 2
    let next : Fin 1 → ℝ := fun _ => 1
    let x : Fin 1 → ℝ := fun _ => 3
    let dw : Bool → Fin 1 → ℝ := fun i _ => if i then 1 else -2
    let v : Bool → Fin 1 → ℝ := fun i _ => if i then 1 / 2 else -1 / 4
    (∀ k, next k - w k = ∑ i, dw i k) ∧
      0 < componentScoreChange w x (dw true) (v true) ∧
      forecastScoreChange w next x (fun k => ∑ i, v i k) < 0 := by
  norm_num [componentScoreChange, forecastScoreChange, linearScore]

example := forecast_component_identity (I := Bool) (K := Fin 1)
  (fun _ => 2) (fun _ => 3) (fun _ => 3) (fun _ => 3 / 4)
  (fun i _ => if i then 2 else -1) (fun i _ => if i then 1 / 2 else 1 / 4)
  (by intro k; norm_num) (by intro k; norm_num)

example := finite_update_component_identity (I := Bool) (K := Fin 1)
  (fun _ => 2) (fun _ => 3) (fun _ => 3) (fun _ => 3 / 4) (fun _ => 4)
  (fun i _ => if i then 2 else -1) (fun i _ => if i then 1 / 2 else 1 / 4)
  (by intro k; norm_num) (by intro k; norm_num)

end Transformer.Grokking.FlatSpectrum.FeatureCodes

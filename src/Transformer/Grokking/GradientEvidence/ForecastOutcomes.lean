module

public import Transformer.Grokking.GradientEvidence.ForecastRMS
public import Mathlib.Analysis.InnerProductSpace.Basic

/-!
# Margins and loss from a feature forecast ball

Source: `review/flat_spectrum.md`, *Feature forecasts*, FF3. The proposed
rounded readout is known. A feature ball and a separately declared execution
allowance yield all-class gap intervals, margins and actual CE bounds.
These statements do not certify the CUDA arithmetic allowance itself.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

open Geometry Landscape.Neural Finset
variable {H C : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]

omit [InnerProductSpace ℝ H] in
/-- FF3: add an independently checked execution allowance to a real feature ball. -/
theorem forecast_execution_ball (center exact executed : H) (r δ : ℝ)
    (hm : ‖exact - center‖ ≤ r) (he : ‖executed - exact‖ ≤ δ) :
    ‖executed - center‖ ≤ r + δ := by
  have h := (norm_sub_le_norm_sub_add_norm_sub executed exact center).trans (add_le_add he hm)
  simpa only [add_comm] using h

example := forecast_execution_ball (1 : ℝ) 2 3 1 1 (by norm_num) (by norm_num)

/-- Predicted competitor gap with the proposed readout, FF3. -/
def forecastGap (W : C → H) (center : H) (t c : C) : ℝ :=
  inner ℝ (W c - W t) center

/-- Feature radius and per-class execution allowance, FF3. -/
def forecastGapRadius (W : C → H) (r δ : ℝ) (t c : C) : ℝ :=
  ‖W c - W t‖ * r + 2 * δ

/-- Upper competing-class gap without a future feature evaluation, FF3. -/
def forecastGapUpper (W : C → H) (center : H) (r δ : ℝ) (t c : C) : ℝ :=
  forecastGap W center t c + forecastGapRadius W r δ t c

/-- Lower competing-class gap from the same forecast ball, FF3. -/
def forecastGapLower (W : C → H) (center : H) (r δ : ℝ) (t c : C) : ℝ :=
  forecastGap W center t c - forecastGapRadius W r δ t c

/-- FF3: the readout contrast, not a common class norm, controls gap error.
The feature hypothesis is supplied by the layerwise forecast certificate. -/
theorem forecast_contrast_error (W : C → H) (center x : H) (e : C → ℝ)
    (r δ : ℝ) (t c : C) (hx : ‖x - center‖ ≤ r) (he : ∀ a, |e a| ≤ δ) :
    |((inner ℝ (W c) x + e c) - (inner ℝ (W t) x + e t)) -
      forecastGap W center t c| ≤ forecastGapRadius W r δ t c := by
  have hi : ((inner ℝ (W c) x + e c) - (inner ℝ (W t) x + e t)) -
      forecastGap W center t c = inner ℝ (W c - W t) (x - center) + (e c - e t) := by
    simp only [forecastGap, inner_sub_left, inner_sub_right]
    ring
  have hscore := (abs_real_inner_le_norm (W c - W t) (x - center)).trans
    (mul_le_mul_of_nonneg_left hx (norm_nonneg _))
  have hnum : |e c - e t| ≤ 2 * δ := by
    have h := (abs_sub_le (e c) 0 (e t))
    simp only [sub_zero, zero_sub, abs_neg] at h
    linarith [he c, he t]
  rw [hi]
  exact (abs_add_le _ _).trans (add_le_add hscore hnum)

example := forecast_contrast_error (fun c : Bool => if c then (2 : ℝ) else 0)
  2 3 (fun _ => 0) 1 0 true false (by norm_num) (by intro c; norm_num)

/-- FF3: both endpoints enclose every new gap, including a new worst competitor. -/
theorem forecast_gap_interval (W : C → H) (center x : H) (e : C → ℝ)
    (r δ : ℝ) (t c : C) (hx : ‖x - center‖ ≤ r) (he : ∀ a, |e a| ≤ δ) :
    forecastGapLower W center r δ t c ≤
      (inner ℝ (W c) x + e c) - (inner ℝ (W t) x + e t) ∧
    (inner ℝ (W c) x + e c) - (inner ℝ (W t) x + e t) ≤
      forecastGapUpper W center r δ t c := by
  obtain ⟨hl,hu⟩ := abs_le.mp (forecast_contrast_error W center x e r δ t c hx he)
  unfold forecastGapLower forecastGapUpper
  constructor <;> linarith

example := forecast_gap_interval (fun c : Bool => if c then (2 : ℝ) else 0)
  2 3 (fun _ => 0) 1 0 true false (by norm_num) (by intro c; norm_num)

/-- FF3: negative forecast ceilings ensure a prescribed margin against all
competitors. The conclusion contains actual features, not predicted ones. -/
theorem forecast_margin (W : C → H) (center x : H) (e : C → ℝ)
    (r δ margin : ℝ) (t : C) (hx : ‖x - center‖ ≤ r) (he : ∀ a, |e a| ≤ δ)
    (h : ∀ c, c ≠ t → forecastGapUpper W center r δ t c ≤ -margin) :
    ∀ c, c ≠ t → margin ≤ (inner ℝ (W t) x + e t) - (inner ℝ (W c) x + e c) := by
  intro c hc
  have hu := (forecast_gap_interval W center x e r δ t c hx he).2
  have hb := h c hc
  linarith

example := forecast_margin (fun c : Bool => if c then (2 : ℝ) else 0)
  2 3 (fun _ => 0) 1 0 2 true (by norm_num) (by intro c; norm_num)
  (by intro c hc; cases c <;> norm_num [forecastGapUpper, forecastGap, forecastGapRadius] at *)

variable [Fintype C] [DecidableEq C]

/-- FF3: sum every wrong-class exponential to control actual query CE. -/
theorem forecast_loss_upper (W : C → H) (center x : H) (e : C → ℝ)
    (r δ : ℝ) (t : C) (hx : ‖x - center‖ ≤ r) (he : ∀ a, |e a| ≤ δ) :
    crossEntropy (fun c => inner ℝ (W c) x + e c) t ≤
      Real.log (1 + oddsEnvelope (forecastGapUpper W center r δ t) t) := by
  apply crossEntropy_le_of_contrast_bounds
  intro c hc
  exact (forecast_gap_interval W center x e r δ t c hx he).2

example := forecast_loss_upper (fun c : Bool => if c then (2 : ℝ) else 0)
  2 3 (fun _ => 0) 1 0 true (by norm_num) (by intro c; norm_num)

/-- FF3: the lower loss endpoint can also certify worsening loss. -/
theorem forecast_loss_lower (W : C → H) (center x : H) (e : C → ℝ)
    (r δ : ℝ) (t : C) (hx : ‖x - center‖ ≤ r) (he : ∀ a, |e a| ≤ δ) :
    Real.log (1 + oddsEnvelope (forecastGapLower W center r δ t) t) ≤
      crossEntropy (fun c => inner ℝ (W c) x + e c) t := by
  rw [crossEntropy_eq_log_odds]
  apply Real.log_le_log (by have := oddsEnvelope_nonneg (forecastGapLower W center r δ t) t; linarith)
  apply add_le_add le_rfl
  apply sum_le_sum
  intro c hc
  exact Real.exp_le_exp.mpr (forecast_gap_interval W center x e r δ t c hx he).1

example := forecast_loss_lower (fun c : Bool => if c then (2 : ℝ) else 0)
  2 3 (fun _ => 0) 1 0 true (by norm_num) (by intro c; norm_num)

/-- FF3: independently certified forecasts average over a fixed query set;
the bound does not select queries by their future margins or losses. -/
theorem forecast_mean_loss_upper {I : Type*} (rows : Finset I) (W : C → H)
    (center x : I → H) (e : I → C → ℝ) (r δ : I → ℝ) (t : I → C)
    (hx : ∀ i ∈ rows, ‖x i - center i‖ ≤ r i) (he : ∀ i ∈ rows, ∀ a, |e i a| ≤ δ i) :
    meanCrossEntropy rows (fun i c => inner ℝ (W c) (x i) + e i c) t ≤
      (∑ i ∈ rows, Real.log (1 + oddsEnvelope
        (forecastGapUpper W (center i) (r i) (δ i) (t i)) (t i))) / rows.card := by
  apply meanCrossEntropy_le_of_contrast_bounds
  intro i hi c hc
  exact (forecast_gap_interval W (center i) (x i) (e i) (r i) (δ i) (t i) c (hx i hi) (he i hi)).2

example := forecast_mean_loss_upper ({false, true} : Finset Bool)
  (fun c : Bool => if c then (2 : ℝ) else 0) (fun _ => 2) (fun _ => 3)
  (fun _ _ => 0) (fun _ => 1) (fun _ => 0) (fun _ => true)
  (by intro i hi; norm_num) (by intro i hi c; norm_num)

end Transformer.Grokking.GradientEvidence

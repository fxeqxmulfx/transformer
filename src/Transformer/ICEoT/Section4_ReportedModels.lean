/-
# IC-EoT: reported model dimensions and predictive metrics

arXiv:2603.22095v2, §3.4, Table 1, and §4.3, Tables 2–4.
All decimals are exact rationals transcribed from the published tables.
Theorems check consequences of those reported values; they do not assert
that unprovided training trajectories or fitted weights were reproduced.
-/

import Transformer.ICEoT.Section4_EncoderMPC

namespace Transformer.ICEoT

/-- The two simulated testbeds of §4.2. -/
inductive Testbed | apartment | office
  deriving DecidableEq

/-- The two-element testbed enumeration; §4.2. -/
instance : Fintype Testbed where
  elems := {.apartment, .office}
  complete := by intro t; cases t <;> simp

/-- The four model architectures compared in §4.3.2, Table 3. -/
inductive Architecture | iceot | iclstm | eot | lstm
  deriving DecidableEq

/-- The four compared models; §4.3.2, Table 3. -/
instance : Fintype Architecture where
  elems := {.iceot, .iclstm, .eot, .lstm}
  complete := by intro a; cases a <;> simp

/-- The two models in the multi-seed study; §4.3.4, Table 5. -/
inductive ConvexArchitecture | iceot | iclstm
  deriving DecidableEq

/-- The two multi-seed models; §4.3.4, Table 5. -/
instance : Fintype ConvexArchitecture where
  elems := {.iceot, .iclstm}
  complete := by intro a; cases a <;> simp

/-- The columns of Table 4, §4.3.3, as reported rational data. -/
structure PredictionMetrics where
  mse : ℚ
  r2 : ℚ
  worstZoneR2 : ℚ
  electricityR2 : ℚ
  epochSeconds : ℚ
  deriving DecidableEq

/-- Every row of §4.3.3, Table 4, including both conventional baselines. -/
def reportedPrediction : Testbed → Architecture → PredictionMetrics
  | .apartment, .iceot => ⟨0.0427, 0.9357, 0.9778, 0.5918, 0.5317⟩
  | .apartment, .iclstm => ⟨0.0379, 0.9390, 0.9722, 0.6702, 0.7201⟩
  | .apartment, .eot => ⟨0.0196, 0.9710, 0.9881, 0.8919, 0.5989⟩
  | .apartment, .lstm => ⟨0.0299, 0.9614, 0.9828, 0.8038, 1.1671⟩
  | .office, .iceot => ⟨0.0332, 0.9670, 0.9139, 0.8217, 0.5243⟩
  | .office, .iclstm => ⟨0.0272, 0.9729, 0.9293, 0.8499, 0.7472⟩
  | .office, .eot => ⟨0.0127, 0.9875, 0.9798, 0.8693, 0.5787⟩
  | .office, .lstm => ⟨0.0134, 0.9867, 0.9796, 0.8719, 1.1480⟩

/-- The conventional models have lower reported MSE and higher reported
average R² than their paired convex models; §4.3.3, Table 4. -/
theorem reported_baseline_accuracy (t : Testbed) :
    (reportedPrediction t .eot).mse < (reportedPrediction t .iceot).mse ∧
    (reportedPrediction t .lstm).mse < (reportedPrediction t .iclstm).mse ∧
    (reportedPrediction t .iceot).r2 < (reportedPrediction t .eot).r2 ∧
    (reportedPrediction t .iclstm).r2 < (reportedPrediction t .lstm).r2 := by
  cases t <;> norm_num [reportedPrediction]

/-- The reported apartment R² gap is exactly 0.0033; §4.3.3, Table 4. -/
theorem reported_apartment_r2_gap :
    (reportedPrediction .apartment .iclstm).r2 -
      (reportedPrediction .apartment .iceot).r2 = 0.0033 := by norm_num [reportedPrediction]

/-- Arithmetic correction to §4.3.3's prose: it says the office average
R² gap is 0.0060, but Table 4 prints 0.9729 and 0.9670, whose difference
is 0.0059. This theorem concerns the printed numbers, without imputing
an error to unavailable unrounded experimental values. -/
theorem reported_office_r2_gap_corrected :
    (reportedPrediction .office .iclstm).r2 -
      (reportedPrediction .office .iceot).r2 = 0.0059 := by norm_num [reportedPrediction]

/-- Neither convex model dominates every reported predictive metric;
§4.3.3, Table 4. -/
theorem reported_mixed_accuracy :
    (reportedPrediction .apartment .iclstm).electricityR2 >
      (reportedPrediction .apartment .iceot).electricityR2 ∧
    (reportedPrediction .office .iclstm).electricityR2 >
      (reportedPrediction .office .iceot).electricityR2 ∧
    (reportedPrediction .apartment .iceot).worstZoneR2 >
      (reportedPrediction .apartment .iclstm).worstZoneR2 := by norm_num [reportedPrediction]

/-- IC-EoT has the lowest reported epoch time in each testbed;
§4.3.3, Table 4. -/
theorem reported_epoch_time_minimum (t : Testbed) (a : Architecture) :
    (reportedPrediction t .iceot).epochSeconds ≤ (reportedPrediction t a).epochSeconds := by
  cases t <;> cases a <;> norm_num [reportedPrediction]

/-- Percentage reduction relative to IC-LSTM, using Table 4, §4.3.3. -/
def reportedEpochReduction (t : Testbed) : ℚ :=
  100 * (1 - (reportedPrediction t .iceot).epochSeconds /
    (reportedPrediction t .iclstm).epochSeconds)

/-- The reported 26.2% and 29.8% reductions are correct when rounded to
one decimal percent; §4.3.3 and §5. -/
theorem reported_epoch_reductions :
    26.15 ≤ reportedEpochReduction .apartment ∧ reportedEpochReduction .apartment < 26.25 ∧
    29.75 ≤ reportedEpochReduction .office ∧ reportedEpochReduction .office < 29.85 := by
  norm_num [reportedEpochReduction, reportedPrediction]

/-- Selective ICNN input dimensions derive from the actual concatenated
index; §4.3.1, Eq. (48), Table 2. -/
theorem selective_input_dimension (y u : ℕ) :
    Fintype.card (Fin y ⊕ (Bool × Fin u)) = y + 2 * u := by simp

/-- All standard, selectively extended and output dimensions in Table 2,
§4.3.1. Both physically exogenous and endogenous outputs are counted. -/
theorem reported_testbed_dimensions :
    11 + 4 = (15 : ℕ) ∧ Fintype.card (Fin 11 ⊕ (Bool × Fin 4)) = 19 ∧
    16 + 14 = (30 : ℕ) ∧ Fintype.card (Fin 16 ⊕ (Bool × Fin 14)) = 44 := by decide

/-- MSE and R² for all six rows of §3.4, Table 1.
`false` selects IC-EoT, `true` selects IC-LSTM. -/
def reportedToyMetrics (k : Fin 3) (lstm : Bool) : ℚ × ℚ :=
  if lstm then ![(0.3975, 0.2245), (0.4915, 0.8352), (0.1552, 0.8077)] k
  else ![(0.3924, 0.2345), (0.5810, 0.8051), (0.1490, 0.8154)] k

/-- The two architectures' reported toy-fit comparisons in both metrics;
§3.4, Table 1. IC-EoT wins on f₁ and f₃, IC-LSTM on f₂. -/
theorem reported_toy_comparisons :
    (reportedToyMetrics 0 false).1 < (reportedToyMetrics 0 true).1 ∧
    (reportedToyMetrics 0 true).2 < (reportedToyMetrics 0 false).2 ∧
    (reportedToyMetrics 1 true).1 < (reportedToyMetrics 1 false).1 ∧
    (reportedToyMetrics 1 false).2 < (reportedToyMetrics 1 true).2 ∧
    (reportedToyMetrics 2 false).1 < (reportedToyMetrics 2 true).1 ∧
    (reportedToyMetrics 2 true).2 < (reportedToyMetrics 2 false).2 := by norm_num [reportedToyMetrics]

end Transformer.ICEoT

/-
# IC-EoT: the complete reported multi-seed stability table

arXiv:2603.22095v2, §4.3.4, Table 5. These are exact counts of the
published S/R/A/I classifications, not claims about unprovided loss traces.
No theorem extrapolates numerical stability outside the tested runs.
-/

import Transformer.ICEoT.Section4_ReportedModels

namespace Transformer.ICEoT

/-- Stable, Recovered, Completed abnormal, Interrupted counts;
§4.3.4, Table 5. -/
structure TrainingCounts where
  stable : ℕ
  recovered : ℕ
  abnormal : ℕ
  interrupted : ℕ
  deriving DecidableEq

/-- The historical windows `{5,10,15,20,25,30}`, §4.3.4. -/
def trainingWindow (k : Fin 6) : ℕ := 5 * (k.val + 1)

/-- The five seeds `{42,43,44,45,46}`, §4.3.4. -/
def trainingSeed (k : Fin 5) : ℕ := 42 + k.val

/-- All 24 S/R/A/I cells in Table 5, §4.3.4. -/
def reportedTraining : Testbed → ConvexArchitecture → Fin 6 → TrainingCounts
  | _, .iceot, _ => ⟨5, 0, 0, 0⟩
  | .apartment, .iclstm, k =>
    ![⟨5, 0, 0, 0⟩, ⟨5, 0, 0, 0⟩, ⟨5, 0, 0, 0⟩,
      ⟨4, 0, 0, 1⟩, ⟨3, 0, 1, 1⟩, ⟨4, 0, 0, 1⟩] k
  | .office, .iclstm, k =>
    ![⟨5, 0, 0, 0⟩, ⟨5, 0, 0, 0⟩, ⟨4, 0, 1, 0⟩,
      ⟨3, 1, 0, 1⟩, ⟨4, 1, 0, 0⟩, ⟨2, 1, 1, 1⟩] k

/-- Every classified run is included in the denominator; §4.3.4. -/
def TrainingCounts.total (c : TrainingCounts) : ℕ :=
  c.stable + c.recovered + c.abnormal + c.interrupted

/-- The numerically successful proportion `(S+R)/Total`, §4.3.4. -/
def TrainingCounts.successRate (c : TrainingCounts) : ℚ :=
  (c.stable + c.recovered : ℕ) / (c.total : ℚ)

/-- Table 5 includes exactly five runs for every configuration; §4.3.4. -/
theorem reported_training_cell_total (t : Testbed) (m : ConvexArchitecture) (k : Fin 6) :
    (reportedTraining t m k).total = 5 := by cases t <;> cases m <;> fin_cases k <;> decide

/-- Thirty runs per model and testbed, and sixty across both testbeds;
§4.3.4, combinatorial design. -/
theorem training_design_size :
    Fintype.card (Fin 6 × Fin 5) = 30 ∧
      Fintype.card (Testbed × Fin 6 × Fin 5) = 60 := by decide

/-- The aggregate reported outcome counts, §4.3.4, Table 5. -/
def totalReportedTraining (t : Testbed) (m : ConvexArchitecture) : TrainingCounts :=
  ⟨∑ k, (reportedTraining t m k).stable, ∑ k, (reportedTraining t m k).recovered,
    ∑ k, (reportedTraining t m k).abnormal, ∑ k, (reportedTraining t m k).interrupted⟩

/-- Every reported IC-EoT run is classified Stable: 30 in each testbed
and 60 in total, with zero recovered/abnormal/interrupted counts;
§4.3.4, Table 5, and §5. -/
theorem reported_iceot_sixty_stable :
    totalReportedTraining .apartment .iceot = ⟨30, 0, 0, 0⟩ ∧
    totalReportedTraining .office .iceot = ⟨30, 0, 0, 0⟩ ∧
    (totalReportedTraining .apartment .iceot).stable +
      (totalReportedTraining .office .iceot).stable = 60 := by decide

/-- All reported IC-LSTM aggregate outcomes, §4.3.4, Table 5. -/
theorem reported_iclstm_training_totals :
    totalReportedTraining .apartment .iclstm = ⟨26, 0, 1, 3⟩ ∧
    totalReportedTraining .office .iclstm = ⟨23, 3, 2, 2⟩ := by decide

/-- All reported IC-EoT successful rates are 100%, §4.3.4, Table 5. -/
theorem reported_iceot_training_rates (t : Testbed) (k : Fin 6) :
    (reportedTraining t .iceot k).successRate = 1 := by
  cases t <;> fin_cases k <;> norm_num [reportedTraining, TrainingCounts.successRate, TrainingCounts.total]

/-- Every reported IC-LSTM successful rate, including recovered office
runs in the numerator; §4.3.4, Table 5. -/
theorem reported_iclstm_training_rates (k : Fin 6) :
    (reportedTraining .apartment .iclstm k).successRate = ![1, 1, 1, 4/5, 3/5, 4/5] k ∧
    (reportedTraining .office .iclstm k).successRate = ![1, 1, 4/5, 4/5, 1, 3/5] k := by
  fin_cases k <;> norm_num [reportedTraining, TrainingCounts.successRate, TrainingCounts.total]

/-- The shortest windows have no reported anomalies; the longest window
does in both testbeds, exactly as stated in §4.3.4. -/
theorem reported_iclstm_short_long_comparison (t : Testbed) :
    (reportedTraining t .iclstm 0).stable = 5 ∧
    (reportedTraining t .iclstm 1).stable = 5 ∧
    (reportedTraining t .iclstm 5).stable < 5 := by cases t <;> decide

/-- Failures occur only at windows ≥20 in apartments and ≥15 in offices;
§4.3.4. "Anomaly" here includes recovered non-finite events, so it is
equivalent to fewer than five Stable classifications. -/
theorem reported_anomaly_thresholds (k : Fin 6) :
    ((reportedTraining .apartment .iclstm k).stable < 5 → 20 ≤ trainingWindow k) ∧
    ((reportedTraining .office .iclstm k).stable < 5 → 15 ≤ trainingWindow k) := by
  fin_cases k <;> decide

example : (reportedTraining .apartment .iclstm (5 : Fin 6)).stable < 5 ∧
    (reportedTraining .office .iclstm (5 : Fin 6)).stable < 5 := by decide

/-- Reported success is not monotone with history length: apartment success
rises from 60% to 80%, office success from 80% to 100%; §4.3.4. -/
theorem reported_training_not_monotone :
    (reportedTraining .apartment .iclstm 4).successRate <
      (reportedTraining .apartment .iclstm 5).successRate ∧
    (reportedTraining .office .iclstm 3).successRate <
      (reportedTraining .office .iclstm 4).successRate := by
  norm_num [reportedTraining, TrainingCounts.successRate, TrainingCounts.total]

end Transformer.ICEoT

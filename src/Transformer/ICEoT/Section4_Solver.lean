/-
# IC-EoT: complete reported solver-termination counts

arXiv:2603.22095v2, §4.4.2, Table 6. Every MPC call is retained,
including the iteration-limit calls; acceptable termination is valid.
These are arithmetic checks on the reported data, not a solver certificate.
-/

import Transformer.ICEoT.Section4_Training

namespace Transformer.ICEoT

/-- Desired-tolerance, acceptable-tolerance, and maximum-iteration counts;
§4.4.2, Table 6, ordered S/A/M. -/
structure TerminationCounts where
  desired : ℕ
  acceptable : ℕ
  iterationLimit : ℕ
  deriving DecidableEq

/-- The eight prediction horizons, §4.4, Eq. (51). -/
def solverHorizon (k : Fin 8) : ℕ := 4 * (k.val + 1)

/-- All 64 S/A/M cells of §4.4.2, Table 6. -/
def reportedTermination : Testbed → Architecture → Fin 8 → TerminationCounts
  | .apartment, .iceot, k =>
    ![⟨85, 0, 1⟩, ⟨86, 0, 0⟩, ⟨86, 0, 0⟩, ⟨86, 0, 0⟩,
      ⟨86, 0, 0⟩, ⟨86, 0, 0⟩, ⟨86, 0, 0⟩, ⟨86, 0, 0⟩] k
  | .apartment, .eot, k =>
    ![⟨61, 0, 25⟩, ⟨74, 0, 12⟩, ⟨74, 0, 12⟩, ⟨84, 0, 2⟩,
      ⟨65, 0, 21⟩, ⟨74, 0, 12⟩, ⟨80, 0, 6⟩, ⟨76, 0, 10⟩] k
  | .office, .iceot, k =>
    ![⟨86, 0, 0⟩, ⟨86, 0, 0⟩, ⟨85, 1, 0⟩, ⟨86, 0, 0⟩,
      ⟨86, 0, 0⟩, ⟨86, 0, 0⟩, ⟨86, 0, 0⟩, ⟨86, 0, 0⟩] k
  | .office, .eot, k =>
    ![⟨81, 5, 0⟩, ⟨79, 7, 0⟩, ⟨62, 24, 0⟩, ⟨51, 35, 0⟩,
      ⟨46, 40, 0⟩, ⟨47, 39, 0⟩, ⟨53, 33, 0⟩, ⟨59, 27, 0⟩] k
  | .office, .lstm, k =>
    ![⟨86, 0, 0⟩, ⟨81, 5, 0⟩, ⟨84, 2, 0⟩, ⟨86, 0, 0⟩,
      ⟨86, 0, 0⟩, ⟨85, 1, 0⟩, ⟨86, 0, 0⟩, ⟨86, 0, 0⟩] k
  | _, .iclstm, _ => ⟨86, 0, 0⟩
  | .apartment, .lstm, _ => ⟨86, 0, 0⟩

/-- The denominator includes every termination status; §4.4.2. -/
def TerminationCounts.total (c : TerminationCounts) : ℕ :=
  c.desired + c.acceptable + c.iterationLimit

/-- Valid means either desired or acceptable convergence; §4.4.2. -/
def TerminationCounts.validPercent (c : TerminationCounts) : ℚ :=
  100 * (c.desired + c.acceptable : ℕ) / (c.total : ℚ)

/-- Aggregating all eight tested horizons, §4.4.2, Table 6. -/
def totalReportedTermination (t : Testbed) (m : Architecture) : TerminationCounts :=
  ⟨∑ k, (reportedTermination t m k).desired, ∑ k, (reportedTermination t m k).acceptable,
    ∑ k, (reportedTermination t m k).iterationLimit⟩

/-- Every reported model/horizon experiment has exactly 86 solves;
§4.4.2, Table 6. -/
theorem reported_solver_cell_total (t : Testbed) (m : Architecture) (k : Fin 8) :
    (reportedTermination t m k).total = 86 := by
  cases t <;> cases m <;> fin_cases k <;> decide

/-- Each model has 688 calls per testbed and 1,376 across both;
§4.4.2, Table 6. -/
theorem reported_solver_totals (t : Testbed) (m : Architecture) :
    (totalReportedTermination t m).total = 688 ∧
    (totalReportedTermination .apartment m).total +
      (totalReportedTermination .office m).total = 1376 := by cases t <;> cases m <;> decide

/-- All stated IC-EoT/IC-LSTM termination totals, §4.4.2 and §5.
The two exceptional IC-EoT calls are kept rather than discarded. -/
theorem reported_convex_termination_counts :
    totalReportedTermination .apartment .iceot = ⟨687, 0, 1⟩ ∧
    totalReportedTermination .office .iceot = ⟨687, 1, 0⟩ ∧
    totalReportedTermination .apartment .iclstm = ⟨688, 0, 0⟩ ∧
    totalReportedTermination .office .iclstm = ⟨688, 0, 0⟩ ∧
    (totalReportedTermination .apartment .iceot).desired +
      (totalReportedTermination .office .iceot).desired = 1374 := by decide

/-- The reported conventional-model termination totals, §4.4.2, Table 6. -/
theorem reported_conventional_termination_counts :
    totalReportedTermination .apartment .eot = ⟨588, 0, 100⟩ ∧
    totalReportedTermination .office .eot = ⟨478, 210, 0⟩ ∧
    totalReportedTermination .apartment .lstm = ⟨688, 0, 0⟩ ∧
    totalReportedTermination .office .lstm = ⟨680, 8, 0⟩ := by decide

/-- The 99.9% and 85.5% apartment rates round correctly to one decimal;
§4.4.2, Table 6. They are not asserted as exact percentages. -/
theorem reported_apartment_valid_rates :
    99.85 ≤ (totalReportedTermination .apartment .iceot).validPercent ∧
    (totalReportedTermination .apartment .iceot).validPercent < 99.95 ∧
    85.45 ≤ (totalReportedTermination .apartment .eot).validPercent ∧
    (totalReportedTermination .apartment .eot).validPercent < 85.55 := by
  norm_num [TerminationCounts.validPercent, TerminationCounts.total, totalReportedTermination,
    reportedTermination, Fin.sum_univ_succ]

/-- The remaining reported valid-termination rates are exactly 100%;
§4.4.2, Table 6. -/
theorem reported_full_valid_rates (m : Architecture) :
    (totalReportedTermination .office m).validPercent = 100 ∧
    (totalReportedTermination .apartment .iclstm).validPercent = 100 ∧
    (totalReportedTermination .apartment .lstm).validPercent = 100 := by
  cases m <;> norm_num [TerminationCounts.validPercent, TerminationCounts.total, totalReportedTermination,
    reportedTermination, Fin.sum_univ_succ]

/-- Every apartment EoT horizon has iteration-limit events;
§4.4.2, Table 6. -/
theorem reported_eot_every_horizon_limit (k : Fin 8) :
    0 < (reportedTermination .apartment .eot k).iterationLimit := by fin_cases k <;> decide

/-- With the stated 15-minute control interval, the tested horizons
cover one through eight hours; §4.4, Eq. (51). -/
theorem solver_horizon_hours (k : Fin 8) : solverHorizon k * 15 = (k.val + 1) * 60 := by
  simp [solverHorizon]
  omega

end Transformer.ICEoT

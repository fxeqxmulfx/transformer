/-
# IC-EoT: reported solve times and closed-loop performance

arXiv:2603.22095v2, §4.4.2–4.4.3, Figure 8 and Table 7.
Theorems check exact reported values and one-decimal rounding intervals,
without extrapolating to other solvers, hardware, horizons or true plants.
-/

import Transformer.ICEoT.Section4_Solver

namespace Transformer.ICEoT

/-- Reported mean solve times at `N_p=32`, §4.4.2. -/
def reportedLongestSolve : Testbed → Architecture → ℚ
  | .apartment, .iceot => 8.883
  | .apartment, .iclstm => 51.500
  | .apartment, .eot => 184.394
  | .apartment, .lstm => 35.157
  | .office, .iceot => 20.503
  | .office, .iclstm => 108.497
  | .office, .eot => 82.713
  | .office, .lstm => 50.188

/-- All six speed-up factors stated for the longest horizon, with explicit
rounding intervals: apartments 5.8/4.0/20.8, offices 5.3/2.4/4.0.
Source: §4.4.2; the IC-LSTM comparison is also in the abstract and §5. -/
theorem reported_long_horizon_speedups :
    5.75 ≤ reportedLongestSolve .apartment .iclstm / reportedLongestSolve .apartment .iceot ∧
    reportedLongestSolve .apartment .iclstm / reportedLongestSolve .apartment .iceot < 5.85 ∧
    3.95 ≤ reportedLongestSolve .apartment .lstm / reportedLongestSolve .apartment .iceot ∧
    reportedLongestSolve .apartment .lstm / reportedLongestSolve .apartment .iceot < 4.05 ∧
    20.75 ≤ reportedLongestSolve .apartment .eot / reportedLongestSolve .apartment .iceot ∧
    reportedLongestSolve .apartment .eot / reportedLongestSolve .apartment .iceot < 20.85 ∧
    5.25 ≤ reportedLongestSolve .office .iclstm / reportedLongestSolve .office .iceot ∧
    reportedLongestSolve .office .iclstm / reportedLongestSolve .office .iceot < 5.35 ∧
    2.35 ≤ reportedLongestSolve .office .lstm / reportedLongestSolve .office .iceot ∧
    reportedLongestSolve .office .lstm / reportedLongestSolve .office .iceot < 2.45 ∧
    3.95 ≤ reportedLongestSolve .office .eot / reportedLongestSolve .office .iceot ∧
    reportedLongestSolve .office .eot / reportedLongestSolve .office .iceot < 4.05 := by
  norm_num [reportedLongestSolve]

/-- The stated IC-EoT endpoint growth is present in both testbeds;
§4.4.2. This does not assert monotonicity at unreported intermediate points. -/
theorem reported_solve_time_growth :
    (0.990 : ℚ) < reportedLongestSolve .apartment .iceot ∧
    (0.920 : ℚ) < reportedLongestSolve .office .iceot := by norm_num [reportedLongestSolve]

/-- The five reported closed-loop statistics in Table 7, §4.4.3. -/
structure ClosedLoopMetrics where
  minimumBill : ℚ
  minimumBillHorizon : ℕ
  meanBill : ℚ
  meanDegreeHours : ℚ
  maxDegreeHours : ℚ
  deriving DecidableEq

/-- Every row of the full-day accounting summary, §4.4.3, Table 7. -/
def reportedClosedLoop : Testbed → Architecture → ClosedLoopMetrics
  | .apartment, .iceot => ⟨11.44, 24, 12.34, 4.0826, 4.0826⟩
  | .apartment, .iclstm => ⟨11.53, 12, 12.48, 4.0826, 4.0826⟩
  | .apartment, .eot => ⟨11.51, 8, 12.39, 6.0366, 7.3886⟩
  | .apartment, .lstm => ⟨11.93, 20, 12.72, 6.2531, 8.7266⟩
  | .office, .iceot => ⟨15.40, 12, 19.53, 0.7859, 0.7954⟩
  | .office, .iclstm => ⟨16.38, 12, 20.33, 0.7850, 0.7867⟩
  | .office, .eot => ⟨18.20, 4, 23.67, 0.8522, 1.2973⟩
  | .office, .lstm => ⟨16.59, 8, 22.63, 0.7978, 0.8864⟩

/-- IC-EoT has the lowest reported mean and minimum apartment bills;
§4.4.3, Table 7. -/
theorem reported_apartment_bill_minimum (m : Architecture) :
    (reportedClosedLoop .apartment .iceot).meanBill ≤ (reportedClosedLoop .apartment m).meanBill ∧
    (reportedClosedLoop .apartment .iceot).minimumBill ≤ (reportedClosedLoop .apartment m).minimumBill := by
  cases m <;> norm_num [reportedClosedLoop]

/-- The convex controllers have equal reported apartment mean/maximum
degree-hours, and lower means than the conventional baselines;
§4.4.3, Table 7. No full active-cooling capability is inferred. -/
theorem reported_apartment_comfort :
    (reportedClosedLoop .apartment .iceot).meanDegreeHours =
      (reportedClosedLoop .apartment .iclstm).meanDegreeHours ∧
    (reportedClosedLoop .apartment .iceot).maxDegreeHours =
      (reportedClosedLoop .apartment .iclstm).maxDegreeHours ∧
    (reportedClosedLoop .apartment .iceot).meanDegreeHours <
      (reportedClosedLoop .apartment .eot).meanDegreeHours ∧
    (reportedClosedLoop .apartment .iceot).meanDegreeHours <
      (reportedClosedLoop .apartment .lstm).meanDegreeHours := by norm_num [reportedClosedLoop]

/-- The reported office IC-EoT/IC-LSTM bill gap is €0.80 and the mean
comfort gap is only 0.0009 degree-hours; §4.4.3, Table 7. -/
theorem reported_office_bill_comfort_gaps :
    (reportedClosedLoop .office .iclstm).meanBill - (reportedClosedLoop .office .iceot).meanBill = 0.80 ∧
    (reportedClosedLoop .office .iceot).meanDegreeHours -
      (reportedClosedLoop .office .iclstm).meanDegreeHours = 0.0009 ∧
    (reportedClosedLoop .office .iceot).meanBill < (reportedClosedLoop .office .eot).meanBill ∧
    (reportedClosedLoop .office .iceot).meanBill < (reportedClosedLoop .office .lstm).meanBill := by
  norm_num [reportedClosedLoop]

/-- The reported conventional models have greater office comfort variation
(maximum minus mean) than either convex model; §4.4.3, Table 7. -/
theorem reported_office_comfort_variation :
    (reportedClosedLoop .office .iceot).maxDegreeHours - (reportedClosedLoop .office .iceot).meanDegreeHours <
      (reportedClosedLoop .office .eot).maxDegreeHours - (reportedClosedLoop .office .eot).meanDegreeHours ∧
    (reportedClosedLoop .office .iceot).maxDegreeHours - (reportedClosedLoop .office .iceot).meanDegreeHours <
      (reportedClosedLoop .office .lstm).maxDegreeHours - (reportedClosedLoop .office .lstm).meanDegreeHours ∧
    (reportedClosedLoop .office .iclstm).maxDegreeHours - (reportedClosedLoop .office .iclstm).meanDegreeHours <
      (reportedClosedLoop .office .lstm).maxDegreeHours - (reportedClosedLoop .office .lstm).meanDegreeHours := by
  norm_num [reportedClosedLoop]

end Transformer.ICEoT

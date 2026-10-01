/-
# Arithmetic of reported downstream evaluations

arXiv:2606.25971v2, Appendix C, Table 3 (`tab:moe-evals`). These theorems
check arithmetic and rankings of the literal printed scores; they do not
assert that the training/evaluation process produced those scores.
-/

import Mathlib.Tactic

open scoped BigOperators

namespace Transformer.MagnitudeDirection

/-- Optimizers in the printed evaluation table, arXiv:2606.25971v2, Appendix C. -/
inductive ReportedOptimizer where
  | adamW | muon | muonMD
  deriving DecidableEq

/-- Reported accuracies in percent, ordered MMLU, HellaSwag, ARC-c, ARC-e,
WinoGrande. Source: arXiv:2606.25971v2, Appendix C, `tab:moe-evals`. -/
def reportedScore (opt : ReportedOptimizer) : Fin 5 → ℚ :=
  match opt with
  | .adamW => ![25.6, 53.0, 32.3, 61.5, 53.8]
  | .muon => ![25.6, 55.9, 35.0, 63.7, 58.0]
  | .muonMD => ![23.3, 57.0, 36.6, 67.1, 59.6]

/-- Exact mean of the five printed task scores, before rounding to one
decimal place. Source: arXiv:2606.25971v2, Appendix C, `tab:moe-evals`. -/
def reportedAverage (opt : ReportedOptimizer) : ℚ := (∑ t, reportedScore opt t) / 5

/-- Exact averages are 45.24, 47.64 and 48.72; the source rounds them to
45.2, 47.6 and 48.7. Source: arXiv:2606.25971v2, Appendix C, `tab:moe-evals`. -/
theorem reported_exact_averages : reportedAverage .adamW = 45.24 ∧
    reportedAverage .muon = 47.64 ∧ reportedAverage .muonMD = 48.72 := by
  norm_num [reportedAverage, reportedScore, Fin.sum_univ_succ]

/-- The printed scores support the reported average ranking,
arXiv:2606.25971v2, Appendix C. -/
theorem reported_average_ranking : reportedAverage .adamW < reportedAverage .muon ∧
    reportedAverage .muon < reportedAverage .muonMD := by
  rcases reported_exact_averages with ⟨ha, hm, hd⟩
  rw [ha, hm, hd]
  norm_num

/-- Tasks on which the printed MD score strictly beats both baselines,
arXiv:2606.25971v2, Appendix C. -/
def reportedMDWinningTasks : Finset (Fin 5) := Finset.univ.filter fun t =>
  reportedScore .adamW t < reportedScore .muonMD t ∧
    reportedScore .muon t < reportedScore .muonMD t

/-- The four claimed leading benchmarks are exactly the four non-MMLU tasks,
arXiv:2606.25971v2, §4.2 and Appendix C. -/
theorem reported_winning_tasks : reportedMDWinningTasks = {1, 2, 3, 4} := by
  ext t
  fin_cases t <;> norm_num [reportedMDWinningTasks, reportedScore]

/-- MD leads on four of five tasks in the printed table,
arXiv:2606.25971v2, §4.2 and Appendix C. -/
theorem reported_four_of_five : reportedMDWinningTasks.card = 4 := by
  rw [reported_winning_tasks]
  decide

end Transformer.MagnitudeDirection

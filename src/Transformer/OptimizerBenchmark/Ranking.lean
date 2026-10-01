/-
# What the observed optimizer winner actually establishes

The source is our RTX 3050 GPTMini/Tiny Shakespeare experiment, REPORT.md,
Protocol and results, with the complete final run log encoded in Basic.
The arithmetic mean uses exact binary64 encodings before report rounding.
Seed numbers are preserved, so paired comparisons use the correct runs.
Both attention modes include every measured optimizer variant.
The winner's small numerical margin and its seedwise exceptions are proved
directly from the table, without an assumption asserting the desired ranking.
These statements audit logged outputs. They do not certify GPU execution,
convergence, statistical significance, or a causal explanation of the win.

Algorithm references: arXiv:2502.16982, Section 2.2, and
arXiv:2602.02016v2, Sections 2-4.
-/

import Transformer.OptimizerBenchmark.Basic
import Mathlib.Tactic

namespace Transformer.OptimizerBenchmark

/-- AdamW has strictly the smallest encoded three-seed mean among all 18
measured methods, separately for each attention mode. Source: REPORT.md,
Protocol and results; related algorithms: arXiv:2502.16982, Section 2.2,
and arXiv:2602.02016v2, Sections 2-4. -/
theorem adamw_mean_winner (attention : Attention) (method : Method)
    (h : method ≠ .adamw) : meanLoss attention .adamw < meanLoss attention method := by
  cases attention <;> cases method <;>
    first | exact (h rfl).elim | norm_num [meanLoss, seedLoss]

/-- The winner theorem compares actual competitors. Source: REPORT.md,
results; arXiv:2502.16982, Section 2.2, algorithm context. -/
example : Method.adam ≠ Method.adamw := by decide

/-- The softmax margin over Adam is between 0.0042 and 0.0043 test CE.
Source: REPORT.md, softmax results, exact final records; optimizer context:
arXiv:2502.16982, Section 2.2. -/
theorem softmax_margin :
    (42 / 10000 : ℚ) < meanLoss .softmax .adam - meanLoss .softmax .adamw ∧
    meanLoss .softmax .adam - meanLoss .softmax .adamw < 43 / 10000 := by
  norm_num [meanLoss, seedLoss]

/-- The Sparsemax margin over Adam is between 0.0014 and 0.0015 test CE.
Source: REPORT.md, sparsemax results, exact final records; optimizer context:
arXiv:2502.16982, Section 2.2. -/
theorem sparsemax_margin :
    (14 / 10000 : ℚ) < meanLoss .sparsemax .adam - meanLoss .sparsemax .adamw ∧
    meanLoss .sparsemax .adam - meanLoss .sparsemax .adamw < 15 / 10000 := by
  norm_num [meanLoss, seedLoss]

/-- On softmax, Adam beats AdamW exactly at seed 1. Source: runs.jsonl,
phase=final, paired by seed; REPORT.md, Protocol. Optimizer context:
arXiv:2502.16982, Section 2.2. -/
theorem softmax_adam_better_iff (seed : Fin 3) :
    seedLoss .softmax .adam seed < seedLoss .softmax .adamw seed ↔ seed = 1 := by
  fin_cases seed <;> norm_num [seedLoss]

/-- On Sparsemax, Adam beats AdamW exactly at seeds 0 and 2, despite losing
the mean. Source: runs.jsonl, phase=final, paired by seed; REPORT.md, Protocol.
Optimizer context: arXiv:2502.16982, Section 2.2. -/
theorem sparsemax_adam_better_iff (seed : Fin 3) :
    seedLoss .sparsemax .adam seed < seedLoss .sparsemax .adamw seed ↔ seed ≠ 1 := by
  fin_cases seed <;> norm_num [seedLoss]

/-- The empirical mean winner does not dominate Adam on every paired seed
in either attention mode. Source: REPORT.md, Protocol and exact final
records; arXiv:2502.16982, Section 2.2, optimizer context. -/
theorem adamw_not_seedwise_dominant (attention : Attention) :
    ¬ ∀ seed, seedLoss attention .adamw seed ≤ seedLoss attention .adam seed := by
  cases attention with
  | softmax =>
      intro h
      have hs := h 1
      norm_num [seedLoss] at hs
  | sparsemax =>
      intro h
      have hs := h 0
      norm_num [seedLoss] at hs

/-- Both safeguarded variants logged exactly the SGD loss on every final
seed. Equality of parameters requires the recurrence theorem and rejection
evidence, not equality of loss alone. Source: REPORT.md, Artifact
verification, and runs.jsonl; training correction to arXiv:2502.16982,
Section 2.2, and arXiv:2602.02016v2, Sections 2-4. -/
theorem guarded_losses_equal_sgd (attention : Attention) (seed : Fin 3) :
    seedLoss attention .muon_guarded seed = seedLoss attention .sgd seed ∧
    seedLoss attention .dash_ndb_guarded seed = seedLoss attention .sgd seed := by
  cases attention <;> fin_cases seed <;> norm_num [seedLoss]

end Transformer.OptimizerBenchmark

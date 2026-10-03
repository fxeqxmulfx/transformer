import Transformer.GPTMini.Sparsemax.Basic

/-!
# A wrong saturated route with zero corrective score derivative

Counterexample to the inference that a convex sparsemax row program
guarantees useful gradients for every outer training loss. Extension of
arXiv:2211.11052v1, §3.1; this is not a claim refuted from the paper.
The example is a row-level squared loss, not the entire GPTMini loss or
a causal explanation of the measured final held-out accuracy.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- Scores that separate one of two allowed positions by two units.
Source context: §3.1's simplex, causal content-score extension. -/
def separatedScores (winner : Fin 2) : Fin 2 → ℝ :=
  fun j => if j = winner then 2 else 0

/-- Squared outer error relative to a desired row, using the actual
causal sparsemax projection. Source context: §3.1's quadratic routing
extension; this is an illustrative outer objective, not benchmark CE. -/
def rowSquaredError (target : Fin 2) (scores : Fin 2 → ℝ) : ℝ :=
  ∑ j, (sparseWeights scores 1 j - basis target j) ^ 2

/-- Separated two-slot scores have a strict unit gap. Source context:
the content-score extension of arXiv:2211.11052v1, §3.1. -/
theorem separatedScores_gap (winner : Fin 2) :
    ∀ j : Fin 2, j ≠ winner →
      separatedScores winner j + 1 < separatedScores winner winner := by
  intro j hne
  simp [separatedScores, hne]

/-- The competitor can differ from the winner. Source context:
§3.1's simplex, two-slot causal extension. -/
example : ∃ j winner : Fin 2, j ≠ winner := ⟨1, 0, by decide⟩

/-- The illustrative outer loss is positive at a point where its full
score derivative is zero, even though a score vector with zero loss
exists. Thus convex row inference alone does not prevent bad stationary
score regions. Source context: §3.1's simplex, content-score extension. -/
theorem wrong_route_positive_stationary_point :
    rowSquaredError 1 (separatedScores 0) = 2 ∧
    HasFDerivAt (𝕜 := ℝ) (rowSquaredError 1) 0 (separatedScores 0) ∧
    rowSquaredError 1 (separatedScores 1) = 0 := by
  have hzero := sparseWeights_eq_basis_of_gap (separatedScores 0) 1 0
    (by decide) (fun j _ hn => le_of_lt (separatedScores_gap 0 j hn))
  have hone := sparseWeights_eq_basis_of_gap (separatedScores 1) 1 1
    (by decide) (fun j _ hn => le_of_lt (separatedScores_gap 1 j hn))
  refine ⟨?_, ?_, ?_⟩
  · norm_num [rowSquaredError, hzero, basis, Fin.sum_univ_two]
  · exact rowLoss_hasFDerivAt_zero
      (fun row : Fin 2 → ℝ => ∑ j, (row j - basis 1 j) ^ 2)
      (separatedScores 0) 1 0 (by decide) (fun j _ hn => separatedScores_gap 0 j hn)
  · simp [rowSquaredError, hone]

end Transformer.GPTMini.Sparsemax

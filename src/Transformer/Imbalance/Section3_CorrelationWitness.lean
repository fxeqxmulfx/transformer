/-
# A nondegenerate assignment witness for arbitrarily many classes

arXiv:2402.19449v2, Proposition 2, equation (2). One observed class
has frequency one, input one, probability one half, and second moment one.
The other output classes remain available to the softmax. This witnesses
the consistency of all corrected correlation hypotheses.
-/

import Transformer.Imbalance.Section3_AssignmentBounds
import Transformer.Imbalance.Section3_SimpleModel

open scoped BigOperators

noncomputable section

namespace Transformer.Imbalance

/-- Nonzero parameter matrix with the correct-class logit log(c-1);
Proposition 2's hypothesis witness. -/
def correlationWitnessWeights (N : ℕ) : Parameters (N + 2) 1 :=
  fun k _ => if k = 0 then Real.log (N + 1 : ℝ) else 0

/-- The witness assigns its sample to the correct class with probability
one half for every class count; Assumption 1. -/
theorem correlationWitness_probability (N : ℕ) :
    probability (correlationWitnessWeights N) (fun _ : Fin 1 => 1) 0 = 1 / 2 := by
  have hp : (0 : ℝ) < N + 1 := by positivity
  have hs : scores (correlationWitnessWeights N) (fun _ : Fin 1 => 1) =
      twoLevel (Real.log (N + 1 : ℝ) + 0) 0 (0 : Fin (N + 2)) := by
    funext k
    simp [scores, correlationWitnessWeights, twoLevel]
  unfold probability
  rw [hs, softmax_twoLevel, ite_eq_left rfl, Real.exp_log hp]
  push_cast
  rw [show (N : ℝ) + 1 + ((N : ℝ) + 2 - 1) = 2 * ((N : ℝ) + 1) by ring]
  field_simp

/-- Nonvacuity of the corrected asymptotic assignment assumptions:
nonzero class moments, p=1/2, bounded data, and zero off-class remainder.
This is a concrete softmax model, not an assumed list of compatible facts. -/
theorem correlationWitness_conditions (N : ℕ) :
    (classSamples (fun _ : Fin 1 => (0 : Fin (N + 2))) 0).Nonempty ∧
    frequency (fun _ : Fin 1 => (0 : Fin (N + 2))) 0 = 1 ∧
    (∀ r : Fin 1, classMean (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ))
      (fun _ : Fin 1 => (0 : Fin (N + 2))) 0 r = 1) ∧
    (∑ r : Fin 1, classSecondMoment (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ))
      (fun _ : Fin 1 => (0 : Fin (N + 2))) 0 r r) = 1 ∧
    CorrectAssignment (correlationWitnessWeights N) (fun _ : Fin 1 => fun _ : Fin 1 => 1)
      (fun _ : Fin 1 => (0 : Fin (N + 2))) 0 (1 / 2) 0 := by
  refine ⟨⟨0, by simp [classSamples]⟩, ?_, ?_, ?_, ?_⟩
  · simp [frequency, classSamples]
  · intro r
    simp [classMean, classSamples]
  · simp [classSecondMoment, classSamples]
  · exact ⟨fun _ _ => correlationWitness_probability N, fun _ h => False.elim (h rfl)⟩

end Transformer.Imbalance

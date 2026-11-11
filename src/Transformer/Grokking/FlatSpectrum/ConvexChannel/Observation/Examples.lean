module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Observation.Span

/-!
# Three observed Boolean dictionaries determine a fourth

Source: `review/flat_spectrum.md`, *Observable features determine unseen
predictions*, O4. The observed dictionaries are 00, 01 and 10; 11 is absent.
Its categorical features are reconstructed with coefficients -1, 1, 1.
These are witnesses for all conditional results of O1-O3, including an
unseen prediction shared by every feasible global answer-CE optimizer.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.Observation

open scoped BigOperators

/-- Three distinct training inputs, excluding the all-true dictionary (O4). -/
def trainingData (s : Fin 3) (k : Bool) : Bool :=
  if s = 0 then false else if s = 1 then k else !k

/-- The signed input reconstruction coefficients, independent of labels (O4). -/
def coefficients (s : Fin 3) : ℝ := if s = 0 then -1 else 1

/-- No observed dictionary is the dictionary used for the conclusion (O4). -/
theorem test_input_unseen (s : Fin 3) : trainingData s ≠ (fun _ => true) := by
  intro h
  fin_cases s
  · have := congrFun h false
    simp [trainingData] at this
  · have := congrFun h false
    simp [trainingData] at this
  · have := congrFun h true
    simp [trainingData] at this

/-- The fourth input lies in the signed span of the observed inputs (O4). -/
theorem example_reconstructs :
    Reconstructs (fun _ : Fin 3 => true) trainingData true (fun _ => true) coefficients := by
  refine ⟨fun s h => (h rfl).elim, ?_⟩
  funext kv
  rcases kv with ⟨k, v⟩
  cases k <;> cases v <;>
    norm_num [features, Fin.sum_univ_three, trainingData, coefficients]

/-- This input certificate can amplify uniform observed error by three (O4). -/
theorem example_amplification : ∑ s : Fin 3, |coefficients s| = 3 := by
  norm_num [Fin.sum_univ_three, coefficients]

/-- A feasible shared copying channel realizes the three observed labels.
The generator is a parameter witness, not an additional model input (O4). -/
theorem example_fits : Fits (fun _ : Fin 3 => true) trainingData
    (fun s => trainingData s true) (generator (fun _ : Bool => id)) := by
  intro s
  funext c
  simp [logits_generator, ConvexRecall.targetLogits]

/-- Every global answer-CE optimizer predicts the missing fourth label.
No fourth example or test-answer premise occurs among the hypotheses (O4). -/
theorem every_optimizer_predicts_unseen (W : Channel Bool Bool Bool Bool)
    (hW : Feasible W)
    (hmin : IsMinOn
      (answerLoss (fun _ : Fin 3 => true) trainingData (fun s => trainingData s true))
      {B | Feasible B} W) :
    logits W true (fun _ => true) true = 1 := by
  have he := unseen_logits_of_min _ _ _
    (feasible_generator (fun _ : Bool => (id : Bool → Bool))) example_fits hW hmin
    true (fun _ => true) coefficients example_reconstructs
  rw [he]
  simp [logits_generator]

example := fits_loss_le _ _ _ example_fits
  (feasible_generator (fun _ : Bool => (id : Bool → Bool)))

example := isMinOn_of_fits _ _ _ example_fits

example := observed_mass_one _ _ _ example_fits
  (feasible_generator (fun _ : Bool => (id : Bool → Bool))) le_rfl (0 : Fin 3)

example : (fun c : Bool => if c = true then (1 : ℝ) else 0) =
    ConvexRecall.targetLogits true := by
  apply simplex_eq_target true
  · intro c
    split_ifs <;> norm_num
  · simp
  · simp

example := observed_logits_eq _ _ _ example_fits
  (feasible_generator (fun _ : Bool => (id : Bool → Bool))) le_rfl (0 : Fin 3)

example := logits_reconstruct _ _ _ _ _ example_reconstructs
  (generator (fun _ : Bool => (id : Bool → Bool))) true

example := logits_eq_of_reconstruct _ _ _ _ _ example_reconstructs
  (A := generator (fun _ : Bool => (id : Bool → Bool))) (fun _ => rfl)

example := logit_error_of_reconstruct _ _ _ _ _ example_reconstructs
  (A := generator (fun _ : Bool => (id : Bool → Bool))) (W := generator (fun _ => id))
  true 0 (fun _ => by simp)

example := unseen_logits_of_min _ _ _
  (feasible_generator (fun _ : Bool => (id : Bool → Bool))) example_fits
  (feasible_generator (fun _ : Bool => (id : Bool → Bool)))
  (isMinOn_of_fits _ _ _ example_fits) _ _ _ example_reconstructs

example := every_optimizer_predicts_unseen _
  (feasible_generator (fun _ : Bool => (id : Bool → Bool)))
  (isMinOn_of_fits _ _ _ example_fits)

example := unseen_logits_eq_targets _ _ _
  (feasible_generator (fun _ : Bool => (id : Bool → Bool))) example_fits
  (feasible_generator (fun _ : Bool => (id : Bool → Bool)))
  (isMinOn_of_fits _ _ _ example_fits) _ _ _ example_reconstructs true

end Transformer.Grokking.FlatSpectrum.ConvexChannel.Observation

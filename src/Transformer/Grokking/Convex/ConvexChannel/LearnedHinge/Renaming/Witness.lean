module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Renaming.Objects

/-!
# A nonconstant fitted paired network and a genuinely unseen next object

Source: `review/convex/moments.md`, CM78. Two observed prefixes have different
object labels and ordinary CE below log 2. Shared learned complex passes then
return the third object at a distinct unseen prefix. These are inhabited
feasible points and a conditional identification theorem, not an optimizer
trajectory or a stress-test pass. All three output classes remain present.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Renaming.Witness

open PatternReduction

/-- The actual paired forward has a positive margin at both nonconstant
observed targets, CM78. The two base passes share the same learned point. -/
theorem paired_fit_exists :
    ∃ p : ReducedSequenceParameters Object 1,
      p ∈ complexDomain rawInput Set.univ diagonal ∧
      (∀ c, pairedLogits flip p seen c = if c = some false then 2 else 0) ∧
      (∀ c, pairedLogits flip p control c = if c = none then 2 else 0) := by
  obtain ⟨p, hp, hpred⟩ := base_exists
  refine ⟨p, hp, ?_, ?_⟩
  · intro c
    rw [pairedLogits_average, renamed_seen, hpred, hpred]
    cases c with
    | none => norm_num [seen, unseen, flip, Function.Involutive.toPerm]
    | some b => cases b <;> norm_num [seen, unseen, flip, Function.Involutive.toPerm]
  · intro c
    rw [pairedLogits_average, renamed_control, hpred, hpred]
    cases c with
    | none => norm_num [control, flip, Function.Involutive.toPerm]
    | some b => cases b <;> norm_num [control, flip, Function.Involutive.toPerm]

/-- Three-class ordinary CE, with both competing objects retained, is
below log 2 at margin two, CM78. No binary reduction of the vocabulary occurs. -/
theorem margin_two_fit (target : Object) :
    Landscape.Neural.crossEntropy (fun c : Object => if c = target then 2 else 0) target <
      Real.log 2 := by
  have hexp : (2 : ℝ) < Real.exp 2 := by
    have he := Real.add_one_le_exp (2 : ℝ)
    linarith
  have hsmall : (2 : ℝ) * Real.exp (-2) < 1 := by
    rw [Real.exp_neg, mul_inv_lt_iff₀ (Real.exp_pos _)]
    simpa using hexp
  have hl : Real.log (1 + 2 * Real.exp (-2)) < Real.log 2 :=
    Real.log_lt_log (by positivity) (by linarith)
  rw [ConvexRecall.crossEntropy_relative]
  cases target with
  | none => simpa [Fintype.sum_option, Fintype.sum_bool] using hl
  | some b => cases b <;>
      simpa [Fintype.sum_option, Fintype.sum_bool, two_mul, add_comm, add_left_comm, add_assoc] using hl

/-- An actual feasible model fits both different observed answers, CM78.
The third prefix is not included as a training record or supplied target. -/
theorem fitted_exists :
    ∃ p : ReducedSequenceParameters Object 1,
      p ∈ complexDomain rawInput Set.univ diagonal ∧
      pairedLoss flip seen (some false) p < Real.log 2 ∧
      pairedLoss flip control none p < Real.log 2 := by
  obtain ⟨p, hp, hseen, hcontrol⟩ := paired_fit_exists
  refine ⟨p, hp, ?_, ?_⟩
  · unfold pairedLoss
    rw [funext hseen]
    exact margin_two_fit _
  · unfold pairedLoss
    rw [funext hcontrol]
    exact margin_two_fit _

/-- A fixed witness point, selected only to exercise the mathematical
theorems. Its existence does not specify or stand in for training, CM78. -/
noncomputable def model : ReducedSequenceParameters Object 1 := Classical.choose fitted_exists

/-- The inhabited point satisfies the original complete cone and both
strict observed fits, CM78. -/
theorem model_spec : model ∈ complexDomain rawInput Set.univ diagonal ∧
    pairedLoss flip seen (some false) model < Real.log 2 ∧
    pairedLoss flip control none model < Real.log 2 := Classical.choose_spec fitted_exists

/-- The mathematical predictor returns the two original observed objects
and the different unseen renamed object, CM78. The witness `none` only proves
vocabulary nonemptiness and is constant across all three calls. -/
theorem predicted_objects : predict flip model seen none = some false ∧
    predict flip model control none = none ∧ predict flip model unseen none = some true := by
  refine ⟨predict_of_loss flip model seen none (some false) model_spec.2.1,
    predict_of_loss flip model control none none model_spec.2.2, ?_⟩
  simpa only [renamed_seen, show flip (some false) = some true from rfl] using
    predict_renamed flip flip_involutive model seen none (some false) model_spec.2.1

example := low_loss_unique flip model seen (some false) model_spec.2.1
example := renamed_unique flip flip_involutive model seen (some false) model_spec.2.1
example := task_transfer flip flip_involutive answer answer_symmetry model seen model_spec.2.1
example := predict_of_unique flip model seen none (some false)
  (low_loss_unique flip model seen (some false) model_spec.2.1)
example := predict_of_loss flip model seen none (some false) model_spec.2.1
example := predict_renamed flip flip_involutive model seen none (some false) model_spec.2.1

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Renaming.Witness

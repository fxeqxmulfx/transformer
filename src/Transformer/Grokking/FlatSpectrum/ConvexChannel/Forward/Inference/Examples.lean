module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference.SelfLabel
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference.Certified

/-!
# Real feedback corrects a default that self-labeling can poison

Source: `review/flat_spectrum.md`, *Inference-time learning*, CI4.
The same actual candidate updater is used in every example. The initial
history observes only value false with label false. Its default prediction
at true is wrong when the target is the identity. Genuine feedback corrects
it; first recording a self-generated false label instead causes failure.
The conservative guard abstains and preserves the ability to learn.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference

/-- Main training observes only value false at the enumerated key false (CI4). -/
theorem bool_main_snapshot :
    finish [false] (trainedState [(id, false)]) false =
      some (false, fun _ : Bool => false) := by
  simp [trainedState, finish, trainKey, scan, step]
  funext v
  cases v <;> rfl

/-- An actual newly revealed true label corrects the previously wrong default (CI4). -/
theorem bool_feedback_recovers :
    finish [false] (advance (trainedState [(id, false)]) [(fun _ => true, true)]) false =
      some (false, id) := by
  rw [feedback_eq_learn]
  simp [learn, trainKey, scan, step]
  funext v
  cases v <;> rfl

/-- An unverified default recorded as a label makes the same genuine feedback
eliminate the only enumerated candidate; the original history was realizable (CI4). -/
theorem bool_self_label_blocks_feedback :
    finish [false]
      (advance (inferBatch [false] (trainedState [(id, false)]) false [fun _ => true])
        [(fun _ => true, true)]) false = none := by
  rw [inferBatch, bool_main_snapshot]
  simp [labelInputs, advance, trainedState, finish, trainKey, scan, step]

/-- The guard detects the unknown entry instead of accepting its default (CI4). -/
theorem bool_certificate_abstains :
    certify [false] (trainedState [(id, false)]) (fun _ : Bool => true) = none := by
  simp [certify, trainedState, trainKey, scan, step]

/-- Guarding the unlabeled inference input preserves the ability to recover
the correct decoder when the actual answer is later observed (CI4). -/
theorem bool_guard_preserves_feedback :
    finish [false]
      (advance (guardedObserve [false] (trainedState [(id, false)]) (fun _ => true))
        [(fun _ => true, true)]) false = some (false, id) := by
  rw [guardedObserve, bool_certificate_abstains]
  exact bool_feedback_recovers

/-- A label already determined by observations can pass the guard (CI4). -/
theorem bool_certificate_accepts :
    certify [false] (trainedState [(id, false)]) id = some false := by
  simp [certify, agrees, trainedState, trainKey, scan, step]

/-- Individually observed entries are insufficient when surviving keys disagree.
Both candidates fit the two historical labels, but differ on the all-true input (CI4). -/
theorem bool_certificate_rejects_disagreement :
    certify [false, true] (trainedState [(id, false), (Bool.not, true)])
      (fun _ => true) = none := by
  simp [certify, agrees, trainedState, trainKey, scan, step]

/-- Any answer at the unseen value can be wrong for a decoder fitting the same
actual labeled history. Extra unlabeled inputs alone do not distinguish these
single-key target rules without further restrictions or evidence (CI4). -/
theorem no_unlabeled_answer_guarantee (answer : Bool) :
    ∃ f : Bool → Bool,
      (∀ s ∈ ([(id, false)] : List (Sample Bool Bool Bool)), f (s.1 false) = s.2) ∧
      answer ≠ f true := by
  cases answer with
  | false => exact ⟨id, by simp, by decide⟩
  | true => exact ⟨fun _ => false, by simp, by decide⟩

example := finish_own [false] (trainedState [(id, false)]) false
  bool_main_snapshot [fun _ => true]

example := learn_self_append [false] [(id, false)] false
  (result := (false, fun _ : Bool => false))
  ((finish_trained [false] [(id, false)] false).symm.trans bool_main_snapshot)
  [fun _ => true]

end Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference

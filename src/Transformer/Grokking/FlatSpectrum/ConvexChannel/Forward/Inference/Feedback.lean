module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference.Basic

/-!
# Continuing with observed answers after main training

Source: `review/flat_spectrum.md`, *Inference-time learning*, CI1.
These are actual labels supplied by feedback or an observable target in
the incoming stream. Their truth is not inferred from the model's prediction.
Continuing the retained state equals scanning the combined history exactly.
Single-key realizability guarantees a successful result fitting both phases.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference

variable {K V C : Type*} [DecidableEq V] [DecidableEq C]

/-- Main training and later labeled feedback retain the same complete state
as one scan of the concatenated history, including elimination markers (CI1). -/
theorem advance_trained (history feedback : List (Sample K V C)) :
    advance (trainedState history) feedback = trainedState (history ++ feedback) := by
  funext k
  unfold advance trainedState trainKey
  exact (scan_append k history feedback _).symm

/-- The deployed learner's next snapshot equals the original combined-data fit (CI1). -/
theorem feedback_eq_learn (keys : List K) (history feedback : List (Sample K V C))
    (default : C) :
    finish keys (advance (trainedState history) feedback) default =
      learn keys (history ++ feedback) default := by
  rw [advance_trained]
  exact finish_trained keys (history ++ feedback) default

/-- Correct later feedback under a common realizable rule yields a successful
model fitting both the training history and the new observations (CI1).
The target key and decoder occur only in the specification, not the update. -/
theorem feedback_fits (keys : List K) (history feedback : List (Sample K V C))
    (default : C) (target : K) (f : V → C) (hk : target ∈ keys)
    (hhistory : ∀ s ∈ history, f (s.1 target) = s.2)
    (hfeedback : ∀ s ∈ feedback, f (s.1 target) = s.2) :
    ∃ result, finish keys (advance (trainedState history) feedback) default = some result ∧
      ∀ s ∈ history ++ feedback, result.2 (s.1 result.1) = s.2 := by
  have hfit : ∀ s ∈ history ++ feedback, f (s.1 target) = s.2 := by
    intro s hs
    rcases List.mem_append.mp hs with hp | hn
    · exact hhistory s hp
    · exact hfeedback s hn
  obtain ⟨k, g, hg⟩ := learn_complete keys (history ++ feedback) default target f hk hfit
  refine ⟨(k, g), ?_, (learn_sound keys (history ++ feedback) default hg).2⟩
  rw [feedback_eq_learn]
  exact hg

example := feedback_fits [false] [(id, false)] [(fun _ => true, true)] false false
  (id : Bool → Bool) (by simp) (by simp) (by simp)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference

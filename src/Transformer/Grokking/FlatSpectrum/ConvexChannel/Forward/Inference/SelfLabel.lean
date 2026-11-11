module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference.Feedback

/-!
# Unverified self-labeling cannot change the selected model

Source: `review/flat_spectrum.md`, *Inference-time learning*, CI2.
Earlier candidates remain eliminated. The selected candidate records only
its own predictions, so it survives with precisely the same decoder.
This proves exact model equality after any finite number of inference batches,
even though the records of this and other candidates may have changed.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference

variable {K V C : Type*} [DecidableEq V] [DecidableEq C]

/-- Self-labeled observations preserve the current key and complete decoder (CI2). -/
theorem finish_own (keys : List K) (state : Candidates K V C) (default : C)
    {result : K × (V → C)} (h : finish keys state default = some result)
    (inputs : List (K → V)) :
    finish keys (advance state (labelInputs result inputs)) default = some result := by
  induction keys with
  | nil => simp [finish] at h
  | cons k rest ih =>
    cases hs : state k with
    | none =>
      have hr : finish rest state default = some result := by
        simpa only [finish, hs] using h
      have hn : advance state (labelInputs result inputs) k = none := by
        simp only [advance, hs, scan_none]
      simp only [finish, hn]
      exact ih hr
    | some record =>
      have he : (k, decode record default) = result := by
        simpa only [finish, hs, Option.some.injEq] using h
      subst result
      obtain ⟨next, hn, hd⟩ := scan_own k inputs record default
      have ha : advance state (labelInputs (k, decode record default) inputs) k = some next := by
        simpa only [advance, hs] using hn
      simp only [finish, ha, hd]

/-- One actual unverified inference update leaves the snapshot unchanged,
including the no-surviving-candidate result (CI2). -/
theorem finish_inferBatch (keys : List K) (state : Candidates K V C) (default : C)
    (inputs : List (K → V)) :
    finish keys (inferBatch keys state default inputs) default = finish keys state default := by
  cases h : finish keys state default with
  | none => simp only [inferBatch, h]
  | some result =>
    simpa only [inferBatch, h] using finish_own keys state default h inputs

/-- Refreshing the snapshot between any finite sequence of self-labeled
inference batches still preserves exactly the original prediction function (CI2). -/
theorem finish_inferBatches (keys : List K) (batches : List (List (K → V)))
    (state : Candidates K V C) (default : C) :
    finish keys (inferBatches keys batches state default) default = finish keys state default := by
  induction batches generalizing state with
  | nil => rfl
  | cons batch rest ih =>
    rw [inferBatches, ih, finish_inferBatch]

/-- Appending a successful learner's hard predictions to its actual training
history cannot change the result of refitting this same categorical learner (CI2). -/
theorem learn_self_append (keys : List K) (history : List (Sample K V C)) (default : C)
    {result : K × (V → C)} (h : learn keys history default = some result)
    (inputs : List (K → V)) :
    learn keys (history ++ labelInputs result inputs) default = some result := by
  rw [← feedback_eq_learn]
  exact finish_own keys (trainedState history) default
    ((finish_trained keys history default).trans h) inputs

end Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference

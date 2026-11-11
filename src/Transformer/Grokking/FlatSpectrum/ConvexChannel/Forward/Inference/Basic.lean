module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Minibatch

/-!
# Inference observations and the unchanged decoded table

Source: `review/flat_spectrum.md`, *Inference-time learning*, CI1-CI2.
The candidate state is retained after training. Unverified self-labeling
really calls the existing update on predictions from the current snapshot;
it is not defined to preserve the model. The preservation lemmas below
prove that recording a table's own decoded answers cannot change its decoder.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference

variable {K V C : Type*} [DecidableEq V] [DecidableEq C]

/-- Retain all candidates after main training, including unselected ones (CI1). -/
def trainedState (samples : List (Sample K V C)) : Candidates K V C :=
  fun k => trainKey k samples

/-- Attach the selected model's own answers to an unlabeled input batch (CI2). -/
def labelInputs (result : K × (V → C)) (inputs : List (K → V)) : List (Sample K V C) :=
  inputs.map (fun D => (D, result.2 (D result.1)))

/-- Unverified self-labeling updates every candidate using the current snapshot.
If no candidate survives, no label can be generated and the state is retained (CI2). -/
def inferBatch (keys : List K) (state : Candidates K V C) (default : C)
    (inputs : List (K → V)) : Candidates K V C :=
  match finish keys state default with
  | none => state
  | some result => advance state (labelInputs result inputs)

/-- Refresh the snapshot between successive unverified self-labeling batches (CI2). -/
def inferBatches (keys : List K) (batches : List (List (K → V)))
    (state : Candidates K V C) (default : C) : Candidates K V C :=
  match batches with
  | [] => state
  | batch :: rest => inferBatches keys rest (inferBatch keys state default batch) default

/-- Observations never revive an eliminated candidate (CI2). -/
theorem scan_none (k : K) (samples : List (Sample K V C)) :
    scan k samples none = none := by
  induction samples with
  | nil => rfl
  | cons s rest ih => simpa only [scan, step] using ih

/-- Inserting the decoder's own answer may fill a cell, but leaves every
decoded value unchanged and cannot eliminate the candidate (CI2). -/
theorem step_decode (record : Record V C) (default : C) (v : V) :
    ∃ next, step (some record) v (decode record default v) = some next ∧
      decode next default = decode record default := by
  cases hv : record v with
  | none =>
    refine ⟨Function.update record v (some default), ?_, ?_⟩
    · simp [step, decode, hv]
    · funext w
      by_cases hw : w = v
      · subst w
        simp [decode, hv]
      · simp only [decode, Function.update_of_ne hw]
  | some c =>
    refine ⟨record, ?_, rfl⟩
    simp [step, decode, hv]

/-- An arbitrary batch labeled by this decoder preserves its entire function.
The post-scan record can differ where unseen values acquired default labels (CI2). -/
theorem scan_own (k : K) (inputs : List (K → V)) (record : Record V C) (default : C) :
    ∃ next, scan k (labelInputs (k, decode record default) inputs) (some record) = some next ∧
      decode next default = decode record default := by
  induction inputs generalizing record with
  | nil => exact ⟨record, rfl, rfl⟩
  | cons D rest ih =>
    obtain ⟨mid, hm, hd⟩ := step_decode record default (D k)
    obtain ⟨next, hn, he⟩ := ih mid
    refine ⟨next, ?_, he.trans hd⟩
    change scan k (labelInputs (k, decode record default) rest)
      (step (some record) (D k) (decode record default (D k))) = some next
    rw [hm]
    simpa only [hd] using hn

end Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference

module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Training

/-!
# Persistent minibatches compute exactly the same channel

Source: `review/flat_spectrum.md`, *Forward-only channel learning*, CF6.
Every candidate retains its partial record and elimination state. Batches
advance these records; selecting a prediction does not change them. Flattening
the batches gives exactly the original learner, including its failure result,
tie order and default for unseen values. The theorem concerns the same ordered
observations and imposes no realizability or equal-batch-size assumption.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward

variable {Q K V C : Type*} [DecidableEq V] [DecidableEq C]

/-- Keep every candidate, including its permanent elimination marker (CF6). -/
abbrev Candidates (K V C : Type*) := K → Option (Record V C)

/-- No candidate has observed any labels initially (CF6). -/
def initialCandidates : Candidates K V C := fun _ => some (fun _ => none)

/-- Consume one batch while retaining all candidates for future batches (CF6). -/
def advance (state : Candidates K V C) (batch : List (Sample K V C)) :
    Candidates K V C := fun k => scan k batch (state k)

/-- Process a list of minibatches, without selecting or resetting candidates (CF6). -/
def runBatches (batches : List (List (Sample K V C))) (state : Candidates K V C) :
    Candidates K V C :=
  match batches with
  | [] => state
  | batch :: rest => runBatches rest (advance state batch)

/-- A read-only snapshot: first surviving key, with defaults only in the decoder (CF6). -/
def finish (keys : List K) (state : Candidates K V C) (default : C) :
    Option (K × (V → C)) :=
  match keys with
  | [] => none
  | k :: rest =>
    match state k with
    | none => finish rest state default
    | some record => some (k, decode record default)

/-- The persistent learner starts empty and selects after consuming every batch (CF6). -/
def learnBatches (keys : List K) (batches : List (List (Sample K V C))) (default : C) :
    Option (K × (V → C)) :=
  finish keys (runBatches batches initialCandidates) default

/-- Continuing a candidate's scan is exactly scanning the concatenation (CF6). -/
theorem scan_append (k : K) (xs ys : List (Sample K V C))
    (state : Option (Record V C)) :
    scan k (xs ++ ys) state = scan k ys (scan k xs state) := by
  induction xs generalizing state with
  | nil => rfl
  | cons s rest ih =>
    simp only [List.cons_append, scan]
    exact ih (step state (s.1 k) s.2)

/-- Every candidate's actual state agrees with one uninterrupted scan (CF6). -/
theorem runBatches_apply (batches : List (List (Sample K V C)))
    (state : Candidates K V C) (k : K) :
    runBatches batches state k = scan k batches.flatten (state k) := by
  induction batches generalizing state with
  | nil => rfl
  | cons batch rest ih =>
    rw [runBatches, ih, List.flatten_cons, scan_append]
    rfl

/-- Selecting from completely scanned records agrees with the original search (CF6). -/
theorem finish_trained (keys : List K) (samples : List (Sample K V C)) (default : C) :
    finish keys (fun k => trainKey k samples) default = learn keys samples default := by
  induction keys with
  | nil => rfl
  | cons k rest ih =>
    simp only [finish, learn]
    cases trainKey k samples with
    | none => exact ih
    | some record => rfl

/-- Batch boundaries preserve the exact computed result, including failure (CF6). -/
theorem learnBatches_eq_learn (keys : List K)
    (batches : List (List (Sample K V C))) (default : C) :
    learnBatches keys batches default = learn keys batches.flatten default := by
  have hstate : runBatches batches (initialCandidates : Candidates K V C) =
      fun k => trainKey k batches.flatten := by
    funext k
    exact runBatches_apply batches initialCandidates k
  rw [learnBatches, hstate, finish_trained]

/-- Assemble the snapshot rows into the same explicit-fallback channel as CF3 (CF6). -/
def learnedBatches [DecidableEq K] (keys : List K)
    (batches : Q → List (List (Sample K V C))) (default : C)
    (fallback : K × (V → C)) : Channel Q K V C :=
  channelOf (fun q => (learnBatches keys (batches q) default).getD fallback)

/-- Exact tensor equality transfers the existing CE and identification results (CF6). -/
theorem learnedBatches_eq_channel [DecidableEq K] (keys : List K)
    (batches : Q → List (List (Sample K V C))) (default : C)
    (fallback : K × (V → C)) :
    learnedBatches keys batches default fallback =
      learnedChannel keys (fun q => (batches q).flatten) default fallback := by
  unfold learnedBatches learnedChannel learnedRow
  congr 1
  funext q
  rw [learnBatches_eq_learn]

/-- A prefix can select the wrong key: snapshots must not discard alternatives (CF6). -/
theorem prefix_selects_first :
    learnBatches [false, true] [[((fun _ => false), true)]] false =
      some (false, Bool.not) := by
  rw [learnBatches_eq_learn]
  simp only [List.flatten_cons, List.flatten_nil, List.append_nil]
  simp [learn, trainKey, scan, step]
  funext v
  cases v <;> rfl

/-- A later conflict changes the selected key, retaining its earlier label (CF6). -/
theorem later_batch_changes_key :
    learnBatches [false, true] [[((fun _ => false), true)], [], [(id, false)]] false =
      some (true, Bool.not) := by
  rw [learnBatches_eq_learn]
  simp [learn, trainKey, scan, step]
  funext v
  cases v <;> rfl

end Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward

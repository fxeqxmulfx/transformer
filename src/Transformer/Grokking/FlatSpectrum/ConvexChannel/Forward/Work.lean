module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Learning

/-!
# A bound on executed candidate observations

Source: `review/flat_spectrum.md`, *Forward-only channel learning*, CF5.
The instrumented implementation counts one unit for every call to `step`.
It returns the same answer as the uninstrumented learner and uses at most
`keys.length * samples.length` candidate observations. This is an abstract
operation count, not a bound on the runtime of Lean's functional tables or
a benchmark of an array implementation.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward

variable {K V C : Type*} [DecidableEq V] [DecidableEq C]

/-- The actual scan, with one counted call to `step` per consumed sample (CF5). -/
def countedScan (k : K) (samples : List (Sample K V C)) (state : Option (Record V C)) :
    Option (Record V C) × ℕ :=
  match samples with
  | [] => (state, 0)
  | s :: rest =>
    let result := countedScan k rest (step state (s.1 k) s.2)
    (result.1, result.2 + 1)

/-- Instrumentation preserves the returned state and counts exactly one pass (CF5). -/
theorem countedScan_eq (k : K) (samples : List (Sample K V C))
    (state : Option (Record V C)) :
    countedScan k samples state = (scan k samples state, samples.length) := by
  induction samples generalizing state with
  | nil => rfl
  | cons s rest ih => simp [countedScan, scan, ih]

/-- The key search with the executed scan counts accumulated (CF5). -/
def countedLearn (keys : List K) (samples : List (Sample K V C)) (default : C) :
    Option (K × (V → C)) × ℕ :=
  match keys with
  | [] => (none, 0)
  | k :: rest =>
    let result := countedScan k samples (some (fun _ => none))
    match result.1 with
    | some record => (some (k, decode record default), result.2)
    | none =>
      let next := countedLearn rest samples default
      (next.1, result.2 + next.2)

/-- The instrumented learner returns exactly the same key and decoder (CF5). -/
theorem countedLearn_result (keys : List K) (samples : List (Sample K V C)) (default : C) :
    (countedLearn keys samples default).1 = learn keys samples default := by
  induction keys with
  | nil => rfl
  | cons k rest ih =>
    cases h : trainKey k samples with
    | none =>
      have hs : scan k samples (some (fun _ => none)) = none := h
      simp [countedLearn, countedScan_eq, hs, learn, h, ih]
    | some record =>
      have hs : scan k samples (some (fun _ => none)) = some record := h
      simp [countedLearn, countedScan_eq, hs, learn, h]

/-- At most one data pass is executed per supplied key (CF5). -/
theorem countedLearn_bound (keys : List K) (samples : List (Sample K V C)) (default : C) :
    (countedLearn keys samples default).2 ≤ keys.length * samples.length := by
  induction keys with
  | nil => simp [countedLearn]
  | cons k rest ih =>
    cases h : trainKey k samples with
    | none =>
      have hs : scan k samples (some (fun _ => none)) = none := h
      simp only [countedLearn, countedScan_eq, hs, List.length_cons, Nat.succ_mul]
      omega
    | some record =>
      have hs : scan k samples (some (fun _ => none)) = some record := h
      simp only [countedLearn, countedScan_eq, hs, List.length_cons, Nat.succ_mul]
      omega

end Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward

module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference.Basic

/-!
# A conservative guard for inference labels

Source: `review/flat_spectrum.md`, *Inference-time learning*, CI3.
An answer is accepted only if every surviving enumerated candidate has an
observed entry for its input value and all these entries agree. No default
or confidence score is evidence. Accepted answers are correct for every
decoder consistent with a surviving candidate; the guarded update leaves
all enumerated records unchanged. New information must come from actual
feedback or an independently justified observable target.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference

variable {K V C : Type*} [DecidableEq C]

/-- All surviving enumerated candidates have already observed this answer (CI3). -/
def agrees (keys : List K) (state : Candidates K V C) (D : K → V) (c : C) : Bool :=
  keys.all fun k =>
    match state k with
    | none => true
    | some record => decide (record (D k) = some c)

/-- Return a unanimously recorded answer, or abstain. An unknown value at
even one surviving candidate is insufficient evidence; defaults are unused (CI3). -/
def certify (keys : List K) (state : Candidates K V C) (D : K → V) : Option C :=
  match keys with
  | [] => none
  | k :: rest =>
    match state k with
    | none => certify rest state D
    | some record =>
      match record (D k) with
      | none => none
      | some c => if agrees rest state D c = true then some c else none

/-- Only a certified inference label may enter the candidate update (CI3). -/
def guardedObserve [DecidableEq V] (keys : List K) (state : Candidates K V C) (D : K → V) :
    Candidates K V C :=
  match certify keys state D with
  | none => state
  | some c => advance state [(D, c)]

/-- Agreement witnesses an actual observed entry at every surviving key (CI3). -/
theorem agrees_observed (keys : List K) (state : Candidates K V C) (D : K → V) (c : C)
    (h : agrees keys state D c = true) {k : K} (hk : k ∈ keys)
    {record : Record V C} (hr : state k = some record) : record (D k) = some c := by
  have he := List.all_eq_true.mp h k hk
  simpa only [hr, decide_eq_true_eq] using he

/-- The executable certificate checks every surviving enumerated key (CI3). -/
theorem certify_agrees (keys : List K) (state : Candidates K V C) (D : K → V) {c : C}
    (h : certify keys state D = some c) : agrees keys state D c = true := by
  induction keys with
  | nil => simp [certify] at h
  | cons k rest ih =>
    cases hs : state k with
    | none =>
      have hr : certify rest state D = some c := by simpa only [certify, hs] using h
      simpa only [agrees, List.all_cons, hs, Bool.true_and] using ih hr
    | some record =>
      cases hv : record (D k) with
      | none => simp [certify, hs, hv] at h
      | some a =>
        by_cases ha : agrees rest state D a = true
        · have he : a = c := by simpa [certify, hs, hv, ha] using h
          subst c
          simpa only [agrees, List.all_cons, hs, hv, decide_true, Bool.true_and] using ha
        · simp [certify, hs, hv, ha] at h

/-- A certified answer is correct for any surviving target key and decoder
consistent with its observations. The target is not an input to the guard (CI3). -/
theorem certify_sound (keys : List K) (state : Candidates K V C) (D : K → V) {c : C}
    (h : certify keys state D = some c) (target : K) (f : V → C)
    (hk : target ∈ keys) (hfit : Allows (state target) f) : f (D target) = c := by
  cases hs : state target with
  | none => simp [hs, Allows] at hfit
  | some record =>
    have hc := agrees_observed keys state D c (certify_agrees keys state D h) hk hs
    have he : Extends record f := by simpa only [hs, Allows] using hfit
    exact he (D target) c hc

/-- Guarded inference cannot contaminate or eliminate an enumerated candidate (CI3). -/
theorem guarded_unchanged_on [DecidableEq V] (keys : List K) (state : Candidates K V C) (D : K → V)
    (k : K) (hk : k ∈ keys) : guardedObserve keys state D k = state k := by
  cases hc : certify keys state D with
  | none => simp only [guardedObserve, hc]
  | some c =>
    cases hs : state k with
    | none => simp [guardedObserve, hc, advance, hs, scan, step]
    | some record =>
      have hv := agrees_observed keys state D c (certify_agrees keys state D hc) hk hs
      simp [guardedObserve, hc, advance, hs, scan, step, hv]

/-- With every candidate enumerated, the guarded observation preserves the
whole state. This prevents damage but supplies no new training information (CI3). -/
theorem guarded_unchanged [DecidableEq V] (keys : List K) (state : Candidates K V C) (D : K → V)
    (hkeys : ∀ k, k ∈ keys) : guardedObserve keys state D = state := by
  funext k
  exact guarded_unchanged_on keys state D k (hkeys k)

example := agrees_observed [false] (fun _ : Bool => some (fun v : Bool => some v))
  id false (by rfl) (k := false) (by simp) (by rfl)

example := certify_agrees [false] (fun _ : Bool => some (fun v : Bool => some v))
  id (c := false) (by rfl)

example := certify_sound [false] (fun _ : Bool => some (fun v : Bool => some v))
  id (c := false) (by rfl) false id (by simp)
  (by intro v c hc; simpa using hc)

example := guarded_unchanged_on [false] (trainedState [(id, false)]) id false (by simp)

example := guarded_unchanged [false, true]
  (fun _ : Bool => some (fun v : Bool => some v)) (fun _ => false)
  (by intro k; cases k <;> simp)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference

module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.SharedPrograms.Learner

/-!
# Executable dictionaries for shared latent input programs

Source: `review/convex/moments.md`, CM53. The program bank supplies input states,
not raw labels. Decoder entries come only from observed state/label pairs.
Conflicting labels eliminate a program; an unseen state remains unidentified.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.LatentPrograms

variable {J X S C : Type*} [DecidableEq S]

/-- The first observed raw label at a latent state, CM53. No default label
or unknown target is supplied to this executable partial decoder. -/
def lookup (state : X → S) : List (X × C) → S → Option C
  | [], _ => none
  | t :: rest, s => if state t.1 = s then some t.2 else lookup state rest s

/-- A successful lookup names an actual observed state/label pair, CM53. -/
theorem lookup_some_mem (state : X → S) (samples : List (X × C)) (s : S) (y : C) :
    lookup state samples s = some y →
      ∃ t ∈ samples, state t.1 = s ∧ t.2 = y := by
  induction samples with
  | nil => simp [lookup]
  | cons t rest ih =>
    intro h
    by_cases ht : state t.1 = s
    · have hy : t.2 = y := by simpa [lookup, ht] using h
      exact ⟨t, by simp, ht, hy⟩
    · have hr : lookup state rest s = some y := by simpa [lookup, ht] using h
      obtain ⟨u, hu, hs, hy⟩ := ih hr
      exact ⟨u, by simp [hu], hs, hy⟩

/-- Observable coverage and agreement of the recorded fiber determine the
lookup, CM53. Coverage is over latent states, not equality of whole inputs. -/
theorem lookup_eq_of_covered (state : X → S) (samples : List (X × C)) (s : S) (y : C)
    (hseen : ∃ t ∈ samples, state t.1 = s)
    (hagree : ∀ t ∈ samples, state t.1 = s → t.2 = y) :
    lookup state samples s = some y := by
  induction samples with
  | nil => simp at hseen
  | cons t rest ih =>
    by_cases ht : state t.1 = s
    · simp [lookup, ht, hagree t (by simp) ht]
    · have hrest : ∃ u ∈ rest, state u.1 = s := by
        obtain ⟨u, hu, hs⟩ := hseen
        rcases List.mem_cons.mp hu with he | hr
        · subst u
          exact False.elim (ht hs)
        · exact ⟨u, hr, hs⟩
      simpa [lookup, ht] using ih hrest (fun u hu => hagree u (by simp [hu]))

/-- A dictionary refusal is exactly absence of the query state from the
observations, CM53. It is not absence of the raw class from the vocabulary. -/
theorem lookup_none_iff (state : X → S) (samples : List (X × C)) (s : S) :
    lookup state samples s = none ↔ ∀ t ∈ samples, state t.1 ≠ s := by
  induction samples with
  | nil => simp [lookup]
  | cons t rest ih =>
    by_cases ht : state t.1 = s
    · simp [lookup, ht]
    · simp [lookup, ht, ih]

/-- The observable constraint that one latent state cannot carry two
different recorded raw labels, CM53. The decoder need not be injective. -/
def fiberConsistent (state : X → S) (samples : List (X × C)) : Prop :=
  ∀ t ∈ samples, ∀ u ∈ samples, state t.1 = state u.1 → t.2 = u.2

/-- Checking every original row against the executable dictionary is
equivalent to absence of conflicting state labels, CM53. -/
theorem dictionary_consistent_iff (state : X → S) (samples : List (X × C)) :
    (∀ t ∈ samples, lookup state samples (state t.1) = some t.2) ↔
      fiberConsistent state samples := by
  constructor
  · intro h t ht u hu he
    have a := h t ht
    have b := h u hu
    rw [he] at a
    exact Option.some.inj (a.symm.trans b)
  · intro h t ht
    apply lookup_eq_of_covered state samples (state t.1) t.2
    · exact ⟨t, ht, rfl⟩
    · intro u hu he
      exact h u hu t ht he

variable [DecidableEq C]

/-- All consistent programs survive, in the original candidate order,
CM53. The filter sees only observed inputs and labels. -/
def retained (program : J → X → S) (order : List J) (samples : List (X × C)) : List J :=
  order.filter fun j => samples.all fun t =>
    decide (lookup (program j) samples (program j t.1) = some t.2)

/-- The executable filter retains exactly the candidates whose observed
latent fibers have a well-defined raw decoder, CM53. -/
theorem mem_retained (program : J → X → S) (order : List J) (samples : List (X × C))
    (j : J) : j ∈ retained program order samples ↔
      j ∈ order ∧ fiberConsistent (program j) samples := by
  simp only [retained, List.mem_filter, List.all_eq_true, decide_eq_true_eq]
  rw [dictionary_consistent_iff]

/-- A retained program's learned dictionary fits each original raw label,
CM53. This does not impose a label at an unobserved latent state. -/
theorem retained_observed (program : J → X → S) (order : List J)
    (samples : List (X × C)) (j : J) (hj : j ∈ retained program order samples)
    (t : X × C) (ht : t ∈ samples) :
    lookup (program j) samples (program j t.1) = some t.2 := by
  have hc := ((mem_retained program order samples j).mp hj).2
  exact (dictionary_consistent_iff (program j) samples).mpr hc t ht

/-- A declared total completion supplies a tensor initializer, CM53. The
default is explicit and carries no target-identification certificate. -/
def completion (state : X → S) (samples : List (X × C)) (default : C) (s : S) : C :=
  (lookup state samples s).getD default

omit [DecidableEq C] in
/-- Every completion agrees with an accepted partial dictionary entry,
CM53, independently of the chosen default for other states. -/
theorem completion_of_lookup (state : X → S) (samples : List (X × C)) (default : C)
    (s : S) (y : C) (h : lookup state samples s = some y) :
    completion state samples default s = y := by
  unfold completion
  rw [h]
  rfl

example := lookup_some_mem (id : Bool → Bool) [(true, false)] true false (by decide)

example := lookup_eq_of_covered (id : Bool → Bool) [(true, false)] true false
  (by exact ⟨(true, false), by simp, rfl⟩) (by simp)

example := dictionary_consistent_iff (id : Bool → Bool) [(true, false), (false, false)]

example := lookup_none_iff (id : Bool → Bool) [(true, false)] false

example := lookup_none_iff (id : Bool → Bool) [(true, false)] true

example := mem_retained (fun j x : Bool => x == j) [false, true] [(false, false)] true

example := retained_observed (fun j x : Bool => x == j) [false, true]
  [(false, false)] true (by decide) (false, false) (by simp)

example := completion_of_lookup (id : Bool → Bool) [(true, false)] true
  true false (by decide)

end Transformer.Grokking.FlatSpectrum.ConvexDepth.LatentPrograms

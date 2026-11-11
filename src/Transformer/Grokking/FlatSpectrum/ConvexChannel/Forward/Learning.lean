module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Basic

/-!
# Selection, explicit failure and identification

Source: `review/flat_spectrum.md`, *Forward-only channel learning*, CF2-CF4.
One consistent candidate suffices for successful learning. To identify the
true key, each wrong candidate needs just one conflicting pair. Coverage
of the true key's values then identifies its complete decoder. The algorithm
never receives the target key or rule; they occur only in its specification.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward

variable {K V C : Type*} [DecidableEq V] [DecidableEq C]

/-- A returned key belongs to the supplied list and its decoder fits all answers (CF2). -/
theorem learn_sound (keys : List K) (samples : List (Sample K V C)) (default : C)
    {k : K} {f : V → C} (h : learn keys samples default = some (k, f)) :
    k ∈ keys ∧ ∀ s ∈ samples, f (s.1 k) = s.2 := by
  induction keys with
  | nil => simp [learn] at h
  | cons key rest ih =>
    cases ht : trainKey key samples with
    | none =>
      obtain ⟨hk, hf⟩ := ih (by simpa [learn, ht] using h)
      exact ⟨List.mem_cons.mpr (Or.inr hk), hf⟩
    | some record =>
      simp only [learn, ht, Option.some.injEq, Prod.mk.injEq] at h
      rcases h with ⟨rfl, rfl⟩
      exact ⟨by simp, trainKey_sound key samples default ht⟩

/-- Realizability by any enumerated key guarantees a concrete successful return (CF2). -/
theorem learn_complete (keys : List K) (samples : List (Sample K V C)) (default : C)
    (target : K) (f : V → C) (hk : target ∈ keys)
    (hfit : ∀ s ∈ samples, f (s.1 target) = s.2) :
    ∃ k g, learn keys samples default = some (k, g) := by
  induction keys with
  | nil => simp at hk
  | cons key rest ih =>
    cases ht : trainKey key samples with
    | some record => exact ⟨key, decode record default, by simp [learn, ht]⟩
    | none =>
      rcases List.mem_cons.mp hk with he | hm
      · subst target
        obtain ⟨record, hr⟩ := trainKey_complete key samples f hfit
        simp [ht] at hr
      · obtain ⟨k, g, hg⟩ := ih hm
        exact ⟨k, g, by simpa [learn, ht] using hg⟩

/-- Failure means no supplied key has a decoder fitting all observed labels.
It does not exclude a lower-confidence optimum elsewhere in the convex family (CF2). -/
theorem learn_failure_iff (keys : List K) (samples : List (Sample K V C)) (default : C) :
    learn keys samples default = none ↔
      ∀ k ∈ keys, ¬ ∃ f : V → C, ∀ s ∈ samples, f (s.1 k) = s.2 := by
  constructor
  · intro hn k hk ⟨f, hf⟩
    obtain ⟨a, g, hg⟩ := learn_complete keys samples default k f hk hf
    simp [hn] at hg
  · intro h
    cases hl : learn keys samples default with
    | none => rfl
    | some result =>
      obtain ⟨hk, hf⟩ := learn_sound keys samples default hl
      exact False.elim (h result.1 hk ⟨result.2, hf⟩)

/-- One observed collision per wrong candidate identifies the key; observed
coverage of the true values identifies the decoder, including on new dictionaries (CF4). -/
theorem learn_identifies (keys : List K) (samples : List (Sample K V C)) (default : C)
    (target : K) (f : V → C)
    (hlabel : ∀ s ∈ samples, f (s.1 target) = s.2)
    (hcover : ∀ v, ∃ s ∈ samples, s.1 target = v)
    (hconflict : ∀ k ∈ keys, k ≠ target → ∃ s ∈ samples, ∃ t ∈ samples,
      s.1 k = t.1 k ∧ s.2 ≠ t.2)
    {k : K} {g : V → C} (h : learn keys samples default = some (k, g)) :
    k = target ∧ g = f := by
  obtain ⟨hk, hg⟩ := learn_sound keys samples default h
  have he : k = target := by
    by_contra hne
    obtain ⟨s, hs, t, ht, hv, hc⟩ := hconflict k hk hne
    exact hc ((hg s hs).symm.trans ((congrArg g hv).trans (hg t ht)))
  refine ⟨he, ?_⟩
  subst k
  funext v
  obtain ⟨s, hs, hv⟩ := hcover v
  rw [← hv]
  exact (hg s hs).trans (hlabel s hs).symm

/-- The learner returns the entire target rule after a finite data pass,
without assuming that any optimization algorithm has reached an optimum (CF4). -/
theorem learn_eq_identified (keys : List K) (samples : List (Sample K V C)) (default : C)
    (target : K) (f : V → C) (hk : target ∈ keys)
    (hlabel : ∀ s ∈ samples, f (s.1 target) = s.2)
    (hcover : ∀ v, ∃ s ∈ samples, s.1 target = v)
    (hconflict : ∀ k ∈ keys, k ≠ target → ∃ s ∈ samples, ∃ t ∈ samples,
      s.1 k = t.1 k ∧ s.2 ≠ t.2) :
    learn keys samples default = some (target, f) := by
  obtain ⟨k, g, hg⟩ := learn_complete keys samples default target f hk hlabel
  obtain ⟨rfl, rfl⟩ := learn_identifies keys samples default target f hlabel hcover hconflict hg
  exact hg

/-- Two dictionaries identify Boolean negation at the first key, although
the other two dictionaries are absent. The wrong key is tried first (CF4). -/
theorem bool_negation_learned :
    learn [true, false] [(id, true), (fun _ => true, false)] false =
      some (false, Bool.not) := by
  apply learn_eq_identified
  · simp
  · intro s hs
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hs
    rcases hs with rfl | rfl <;> rfl
  · intro v
    cases v
    · exact ⟨(id, true), by simp, rfl⟩
    · exact ⟨(fun _ => true, false), by simp, rfl⟩
  · intro k _ hk
    cases k
    · exact False.elim (hk rfl)
    · exact ⟨(id, true), by simp, (fun _ => true, false), by simp, rfl, by decide⟩

example := learn_sound [true, false] [(id, true), (fun _ => true, false)] false
  bool_negation_learned

example := learn_complete [false] [(id, true)] false false Bool.not (by simp) (by simp)

example := learn_identifies [false] [(id, true), (fun _ => true, false)] false false Bool.not
  (k := false) (g := Bool.not)
  (by intro s hs; simp only [List.mem_cons, List.not_mem_nil, or_false] at hs
      rcases hs with rfl | rfl <;> rfl)
  (by intro v; cases v
      · exact ⟨(id, true), by simp, rfl⟩
      · exact ⟨(fun _ => true, false), by simp, rfl⟩)
  (by intro k hk hne; simp only [List.mem_singleton] at hk; exact False.elim (hne hk))
  (by simpa [learn, trainKey, scan, step, Function.update_apply] using bool_negation_learned)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward

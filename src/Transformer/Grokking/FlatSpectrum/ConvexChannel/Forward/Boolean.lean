module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Training
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Work
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Capacity

/-!
# Nonempty witnesses and held-out generated answers

Source: `review/flat_spectrum.md`, *Forward-only channel learning*, CF4-CF5.
For each Boolean query, two of the four dictionaries identify negation.
The other key is constant in the observations, so its conflicting answers
eliminate it. The all-false dictionary is held out and the computed learner
generates the absent value `true`. Inconsistent data instead return failure.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward

/-- Two input dictionaries and ordinary labels, grouped by query (CF4). -/
def boolRows (q : Bool) : List (Sample Bool Bool Bool) :=
  [(fun k => if k = q then false else true, true), (fun _ => true, false)]

/-- The finite Boolean training rows have the claimed generated answers (CF4). -/
theorem boolRows_labels (q : Bool) (s : Sample Bool Bool Bool) (hs : s ∈ boolRows q) :
    Bool.not (s.1 q) = s.2 := by
  simp only [boolRows, List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with rfl | rfl <;> simp

/-- Both possible values of the queried key are observed (CF4). -/
theorem boolRows_cover (q v : Bool) : ∃ s ∈ boolRows q, s.1 q = v := by
  cases v
  · exact ⟨(fun k => if k = q then false else true, true), by simp [boolRows], by simp⟩
  · exact ⟨(fun _ => true, false), by simp [boolRows], rfl⟩

/-- The wrong key has one repeated value with contradictory labels (CF4). -/
theorem boolRows_conflict (q k : Bool) (hk : k ≠ q) :
    ∃ s ∈ boolRows q, ∃ t ∈ boolRows q, s.1 k = t.1 k ∧ s.2 ≠ t.2 := by
  exact ⟨(fun j => if j = q then false else true, true), by simp [boolRows],
    (fun _ => true, false), by simp [boolRows], by simp [hk], by simp⟩

/-- Both query rows are learned exactly from the actual finite scan (CF4). -/
theorem bool_channel_learned :
    learnedChannel [true, false] boolRows false (true, id) =
      generator (fun _ : Bool => Bool.not) := by
  apply learned_eq_generator
  · intro q
    cases q <;> simp
  · exact boolRows_labels
  · exact boolRows_cover
  · intro q k _ hk
    exact boolRows_conflict q k hk

/-- On an unobserved all-false dictionary, the learned channel generates
`true`; it cannot obtain this answer by copying a supplied value (CF4). -/
theorem bool_unseen_generated :
    (∀ c, ((fun _ => false), c) ∉ boolRows false) ∧
      logits (learnedChannel [true, false] boolRows false (true, id))
        false (fun _ => false) true = 1 := by
  constructor
  · intro c
    cases c <;> decide
  · rw [bool_channel_learned]
    simp [logits_generator]

/-- The actual learned outputs over all four dictionaries have a flat
rank-two signed kernel, with squared singular value four (CF4). -/
theorem bool_learned_flat :
    IsFlat (outputKernel boolSign
      (learnedChannel [true, false] boolRows false (true, id))
      (fun x : Bool × Bool => fun k => if k then x.2 else x.1)) 4 := by
  rw [bool_channel_learned]
  right
  ext q r
  cases q <;> cases r <;>
    norm_num [Matrix.mul_apply, Matrix.conjTranspose_apply, outputKernel,
      logits_generator, boolSign, Fintype.sum_prod_type, Fintype.sum_bool, Matrix.one_apply]

/-- Contradictory labels on the same input eliminate every candidate;
the learner returns explicit failure rather than a claimed optimizer (CF2). -/
theorem bool_inconsistent_failure :
    learn [false, true] [((fun _ => false), false), ((fun _ => false), true)] false = none := by
  simp [learn, trainKey, scan, step]

example := boolRows_labels false (fun k => if k = false then false else true, true)
  (by simp [boolRows])

example := boolRows_conflict false true (by decide)

example := learn_eq_identified [true, false] (boolRows false) false false Bool.not
  (by simp) (boolRows_labels false) (boolRows_cover false)
  (fun k _ hk => boolRows_conflict false k hk)

example := learned_eq_generator [true, false] boolRows false (true, id)
  (fun _ : Bool => Bool.not) (by intro q; cases q <;> simp)
  boolRows_labels boolRows_cover (fun q k _ hk => boolRows_conflict q k hk)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward

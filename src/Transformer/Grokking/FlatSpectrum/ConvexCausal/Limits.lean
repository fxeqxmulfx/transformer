module

public import Transformer.Grokking.FlatSpectrum.ConvexCausal.Generation
public import Transformer.Grokking.FlatSpectrum.ConvexCausal.Features
public import Transformer.Grokking.FlatSpectrum.ConvexCausal.Calibration

/-!
# Universality and conditional recovery are different guarantees

Source: `review/convex_causal.md`, CCa8, and the random-table/random-language
controls in `diff-gpt/stress_tests_for_gpt.md`. A universal rule bank cannot
have separating training signatures while leaving an arbitrary input unseen.
The concrete learner refuses ambiguity and inconsistency instead of treating
an optimum or a feasible flat spectrum as evidence for missing labels.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexCausal

variable {X S : Type*}

/-- An unseen input admits two indistinguishable binary completions (CCa8). -/
theorem unseen_twins (input : S → X) (x : X) (hx : ∀ s, input s ≠ x) :
    ∃ f g : X → Bool, (∀ s, f (input s) = g (input s)) ∧ f x ≠ g x := by
  classical
  refine ⟨fun _ => false, fun z => if z = x then true else false, ?_, ?_⟩
  · intro s
    simp [hx s]
  · simp

/-- Full expressivity alone cannot discharge the learning condition on a proper split (CCa8). -/
theorem universal_not_separates (input : S → X) (x : X)
    (hx : ∀ s, input s ≠ x) : ¬ Separates (fun f : X → Bool => f) input := by
  obtain ⟨f, g, hagree, hdiff⟩ := unseen_twins input x hx
  intro hsep
  have hne : f ≠ g := fun he => hdiff (congrFun he x)
  obtain ⟨s, hs⟩ := hsep f g hne
  exact hs (hagree s)

/-- The ambiguity uses genuinely distinct causal predictions, not only redundant parameters (CCa8). -/
def ambiguousRules (h : Bool) (x : Bool) : Bool := if h then true else x

/-- The learner refuses two rules that agree on training but disagree elsewhere (CCa8 witness). -/
theorem refuses_ambiguity :
    learn ambiguousRules (fun _ : Unit => true) (fun _ => true) = .ambiguous := by
  have hs : survivors ambiguousRules (fun _ : Unit => true) (fun _ => true) = Finset.univ := by
    ext h
    cases h <;> simp [survivors, ambiguousRules]
  simp [learn, hs]

/-- Conflicting observed labels on the same input cause explicit failure (CCa8 witness). -/
theorem refuses_inconsistency :
    learn parityRule (fun _ : Bool => (false, false)) id = .inconsistent := by
  have hs : survivors parityRule (fun _ : Bool => (false, false)) id = ∅ := by
    ext h
    cases h <;> simp [survivors, parityRule, Bool.forall_bool]
  simp [learn, hs]

/-- The positive witness identifies an actually unseen input and both output classes (CCa8 witness). -/
theorem parity_unseen :
    (true, false) ∉ Set.range (fun _ : Unit => (false, false)) ∧
      trained parityRule (fun _ : Unit => (false, false)) (fun _ => false)
        (true, false) true = 1 ∧
      trained parityRule (fun _ : Unit => (false, false)) (fun _ => false)
        (true, true) false = 1 := by
  refine ⟨by simp, ?_, ?_⟩ <;>
    rw [trained_target parityRule (fun _ : Unit => (false, false)) (fun _ => false)
      false (by intro s; rfl) parity_separates] <;> norm_num [point, parityRule]

example := unseen_twins (fun _ : Unit => false) true (by decide)
example := universal_not_separates (fun _ : Unit => false) true (by decide)

end Transformer.Grokking.FlatSpectrum.ConvexCausal

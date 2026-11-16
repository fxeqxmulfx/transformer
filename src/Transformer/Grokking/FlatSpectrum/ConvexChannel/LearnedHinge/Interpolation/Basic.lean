module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Unseen.Neuron

/-!
# Full-word interpolation inside the actual complete learned complex family

Source: `review/flat_spectrum.md`, CM84. CM77's learned singleton neurons
can be combined with arbitrary signed class readouts. CM76's exact coverage
places every such physical model in the same complete reduced convex domain.
This proves the size of the existing function class; the witness is not a
fixed-feature learner or a proposed independent whole-prefix table model.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Interpolation

open scoped BigOperators
open PatternReduction Unseen

variable {O : Type*} [Fintype O] [DecidableEq O] {T : ℕ}

/-- The original immutable full-length prefix, CM84. -/
def fullPrefix (word : Fin T → O) : Prefix O T := ⟨List.ofFn word, by simp⟩

/-- Actual complex neurons supply the singleton response on every full word,
CM84. Both components of their learned representations are used. -/
theorem neuron_entry (word needle : Fin T → O) :
    complexNeuron (complexWeights needle) (rawInput (fullPrefix word)) =
      if word = needle then (⟨1, 3⟩ : ℂ) else 0 := by
  rw [neuron_singleton]
  simp only [fullPrefix, List.ofFn_inj]

/-- Arbitrary signed readouts on actual learned neurons interpolate every
full-word class table, CM84. This is an identity of full complex outputs. -/
theorem physical_table (table : (Fin T → O) → O → ℝ) (word : Fin T → O) (c : O) :
    factoredForward complexWeights table (rawInput (fullPrefix word)) c =
      (table word c : ℂ) * (⟨1, 3⟩ : ℂ) := by
  unfold factoredForward
  simp_rw [mul_assoc, ← complexNeuron_eq, neuron_entry, mul_ite, mul_zero]
  exact Fintype.sum_ite_eq word _

/-- Every real full-word logit table belongs to the same complete learned
family, CM84. Feasibility covers all bounded prefixes, not only full words. -/
theorem complex_interpolation (table : (Fin T → O) → O → ℝ) :
    ∃ p : ReducedSequenceParameters O T,
      p ∈ complexDomain rawInput Set.univ diagonal ∧
      ∀ word : Fin T → O, ∀ c : O,
        complexForward p (rawInput (fullPrefix word)) c =
          (table word c : ℂ) * (⟨1, 3⟩ : ℂ) := by
  obtain ⟨p, hp, hforward⟩ := reducedSequence_covers complexWeights table
  refine ⟨p, hp, ?_⟩
  intro word c
  rw [hforward]
  exact physical_table table word c

/-- The real logits of the actual complete complex computation interpolate
every full-word table, CM84. Targets do not enter the forward definition. -/
theorem logit_interpolation (table : (Fin T → O) → O → ℝ) :
    ∃ p : ReducedSequenceParameters O T,
      p ∈ complexDomain rawInput Set.univ diagonal ∧
      ∀ word : Fin T → O, sequenceLogits p (fullPrefix word) = table word := by
  obtain ⟨p, hp, hforward⟩ := complex_interpolation table
  refine ⟨p, hp, ?_⟩
  intro word
  funext c
  change (complexForward p (rawInput (fullPrefix word)) c).re = table word c
  rw [hforward]
  simp

/-- Every binary contrast between distinct original objects can be prescribed
independently on every full word, CM84. Common logit shifts are irrelevant here. -/
theorem margin_interpolation (base selected : O) (hne : base ≠ selected)
    (values : (Fin T → O) → ℝ) :
    ∃ p : ReducedSequenceParameters O T,
      p ∈ complexDomain rawInput Set.univ diagonal ∧
      ∀ word : Fin T → O,
        sequenceLogits p (fullPrefix word) selected -
          sequenceLogits p (fullPrefix word) base = values word := by
  obtain ⟨p, hp, hlogits⟩ := logit_interpolation
    (fun word c => if c = selected then values word else 0)
  refine ⟨p, hp, ?_⟩
  intro word
  rw [hlogits]
  simp [hne]

example := margin_interpolation (T := 2) false true (by decide)
  (fun word => if word 0 then 2 else -1)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Interpolation

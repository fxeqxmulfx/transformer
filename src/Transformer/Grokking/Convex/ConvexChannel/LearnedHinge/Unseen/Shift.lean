module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Unseen.Addition

/-!
# Arbitrary unseen-logit shifts inside the actual complete learned model

Source: `review/convex/moments.md`, CM77. A feasible complex singleton-prefix
direction can be added to any feasible model. The ordinary raw forward stays
unchanged at every other prefix. At the selected full prefix, one selected
class logit changes by any prescribed real amount; other logits stay fixed.
The target class here selects a parameter coordinate, not a supplied answer.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Unseen

open PatternReduction

variable {O : Type*} [Fintype O] [DecidableEq O] {T : ℕ}

/-- Exact full complex singleton response inside the reduced complete domain,
CM77. The coefficient may be positive, negative, or zero. -/
theorem spike_exists (needle : Fin T → O) (target : O) (r : ℝ) :
    ∃ q : ReducedSequenceParameters O T,
      q ∈ complexDomain rawInput Set.univ diagonal ∧
      ∀ tokens : Prefix O T, ∀ c : O, complexForward q (rawInput tokens) c =
        if tokens.val = List.ofFn needle ∧ c = target then (r : ℂ) * ⟨1, 3⟩ else 0 := by
  let w : Unit → ObjectWeights O T := fun _ => complexWeights needle
  let a : Unit → O → ℝ := fun _ c => if c = target then r else 0
  obtain ⟨q, hq, hpred⟩ := reducedSequence_covers w a
  refine ⟨q, hq, ?_⟩
  intro tokens c
  rw [hpred]
  simp only [factoredForward, Fintype.sum_unique, w, a]
  rw [mul_assoc, ← complexNeuron_eq, neuron_singleton]
  by_cases hx : tokens.val = List.ofFn needle <;>
    by_cases hc : c = target <;> simp [hx, hc]

/-- Feasible whole raw-sequence models add at every admitted prefix, CM77.
The recovered factors can change; their actual entire complex result adds. -/
theorem sequence_add {p q : ReducedSequenceParameters O T}
    (hp : p ∈ complexDomain rawInput Set.univ diagonal)
    (hq : q ∈ complexDomain rawInput Set.univ diagonal) (tokens : Prefix O T) :
    complexForward (p + q) (rawInput tokens) =
      complexForward p (rawInput tokens) + complexForward q (rawInput tokens) := by
  let input : Prefix O T → (Coordinate O T → ℝ) →ₗ[ℝ] ℝ := realInput
  have he : (fun x => complexInput (input x)) = (rawInput (O := O) (T := T)) := by
    funext x
    exact (rawInput_complexInput x).symm
  have hp' : p ∈ complexDomain (fun x => complexInput (input x)) Set.univ diagonal := by
    rw [he]
    exact hp
  have hq' : q ∈ complexDomain (fun x => complexInput (input x)) Set.univ diagonal := by
    rw [he]
    exact hq
  rw [rawInput_complexInput]
  exact complexForward_add input Set.univ diagonal hp' hq' (Set.mem_univ tokens)

/-- An arbitrary already fitted feasible point admits an exact complex
perturbation localized to one full prefix, CM77. No fit assumption is weakened. -/
theorem shifted_forward_exists (p : ReducedSequenceParameters O T)
    (hp : p ∈ complexDomain rawInput Set.univ diagonal) (needle : Fin T → O) (target : O) (r : ℝ) :
    ∃ q : ReducedSequenceParameters O T,
      q ∈ complexDomain rawInput Set.univ diagonal ∧
      ∀ tokens : Prefix O T, ∀ c : O, complexForward q (rawInput tokens) c =
        complexForward p (rawInput tokens) c +
          if tokens.val = List.ofFn needle ∧ c = target then (r : ℂ) * ⟨1, 3⟩ else 0 := by
  obtain ⟨v, hv, hpred⟩ := spike_exists needle target r
  let input : Prefix O T → ObjectWeights O T →ₗ[ℝ] ℂ := rawInput
  have hm : p + v ∈ complexDomain input Set.univ diagonal :=
    complexDomain_add (E := ObjectWeights O T) (X := Prefix O T)
      (J := ScalarPattern (Prefix O T)) (C := O) input Set.univ diagonal (p := p) (q := v) hp hv
  refine ⟨p + v, hm, ?_⟩
  intro tokens c
  rw [sequence_add hp hv tokens]
  change complexForward p (rawInput tokens) c + complexForward v (rawInput tokens) c = _
  rw [hpred]

/-- Every selected unseen next-object logit can change by any real amount
while all other raw-prefix logits remain exactly equal, CM77. -/
theorem shifted_logits_exists (p : ReducedSequenceParameters O T)
    (hp : p ∈ complexDomain rawInput Set.univ diagonal) (needle : Fin T → O) (target : O) (r : ℝ) :
    ∃ q : ReducedSequenceParameters O T,
      q ∈ complexDomain rawInput Set.univ diagonal ∧
      ∀ tokens : Prefix O T, ∀ c : O, sequenceLogits q tokens c = sequenceLogits p tokens c +
        if tokens.val = List.ofFn needle ∧ c = target then r else 0 := by
  obtain ⟨q, hq, hpred⟩ := shifted_forward_exists p hp needle target r
  refine ⟨q, hq, ?_⟩
  intro tokens c
  change (complexForward q (rawInput tokens) c).re = (complexForward p (rawInput tokens) c).re + _
  rw [hpred, Complex.add_re]
  by_cases h : tokens.val = List.ofFn needle ∧ c = target <;> simp [h]

example := sequence_add (O := Bool) (T := 2) (p := 0) (q := 0)
  (zero_mem _ _ _) (zero_mem _ _ _) ⟨[true, false], by decide⟩
example := shifted_forward_exists (O := Bool) (T := 2) 0 (zero_mem _ _ _)
  (fun _ => true) true (-2)
example := shifted_logits_exists (O := Bool) (T := 2) 0 (zero_mem _ _ _)
  (fun _ => true) true (-2)

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Unseen

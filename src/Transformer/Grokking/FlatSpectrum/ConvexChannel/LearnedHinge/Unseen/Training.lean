module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Unseen.Shift

/-!
# Observed ordinary CE cannot identify an omitted full-prefix answer

Source: `review/convex/moments.md`, CM77. The perturbation applies to any
feasible starting model, including a fitted point. Every original observed
logit and every indexed CE term is unchanged. For binary objects, two such
points can prefer opposite next objects at the omitted prefix. This is an
identification failure, not a statement that an optimizer fails to converge.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Unseen

open scoped BigOperators
open PatternReduction

variable {O : Type*} [Fintype O] [DecidableEq O] {T N : ℕ}

/-- Any unseen class-logit shift preserves all original observed predictions
and the complete indexed ordinary CE, CM77. Multiplicities and tails remain. -/
theorem training_shift (p : ReducedSequenceParameters O T)
    (hp : p ∈ complexDomain rawInput Set.univ diagonal)
    (data : Fin N → Prefix O T) (labels : Fin N → O) (needle : Fin T → O)
    (hmiss : ∀ i, (data i).val ≠ List.ofFn needle) (target : O) (r : ℝ) :
    ∃ q : ReducedSequenceParameters O T,
      q ∈ complexDomain rawInput Set.univ diagonal ∧
      (∀ i, sequenceLogits q (data i) = sequenceLogits p (data i)) ∧
      (∑ i, sequenceLoss (data i) (labels i) q) = (∑ i, sequenceLoss (data i) (labels i) p) ∧
      ∀ c, sequenceLogits q ⟨List.ofFn needle, by simp⟩ c =
        sequenceLogits p ⟨List.ofFn needle, by simp⟩ c + if c = target then r else 0 := by
  obtain ⟨q, hq, hpred⟩ := shifted_logits_exists p hp needle target r
  have hfit (i : Fin N) : sequenceLogits q (data i) = sequenceLogits p (data i) := by
    funext c
    rw [hpred]
    simp [hmiss i]
  refine ⟨q, hq, hfit, ?_, ?_⟩
  · apply Finset.sum_congr rfl
    intro i hi
    unfold sequenceLoss
    rw [hfit]
  · intro c
    rw [hpred]
    simp

/-- For any feasible binary-object fit, the same observed logits and CE
permit either strict unseen next-object preference, CM77. -/
theorem binary_ambiguity (p : ReducedSequenceParameters Bool T)
    (hp : p ∈ complexDomain rawInput Set.univ diagonal)
    (data : Fin N → Prefix Bool T) (labels : Fin N → Bool) (needle : Fin T → Bool)
    (hmiss : ∀ i, (data i).val ≠ List.ofFn needle) :
    ∃ qPlus qMinus : ReducedSequenceParameters Bool T,
      qPlus ∈ complexDomain rawInput Set.univ diagonal ∧
      qMinus ∈ complexDomain rawInput Set.univ diagonal ∧
      (∀ i, sequenceLogits qPlus (data i) = sequenceLogits p (data i) ∧
        sequenceLogits qMinus (data i) = sequenceLogits p (data i)) ∧
      ((∑ i, sequenceLoss (data i) (labels i) qPlus) = (∑ i, sequenceLoss (data i) (labels i) p) ∧
        (∑ i, sequenceLoss (data i) (labels i) qMinus) = (∑ i, sequenceLoss (data i) (labels i) p)) ∧
      (sequenceLogits qPlus ⟨List.ofFn needle, by simp⟩ false <
        sequenceLogits qPlus ⟨List.ofFn needle, by simp⟩ true) ∧
      (sequenceLogits qMinus ⟨List.ofFn needle, by simp⟩ true <
        sequenceLogits qMinus ⟨List.ofFn needle, by simp⟩ false) := by
  let x : Prefix Bool T := ⟨List.ofFn needle, by simp⟩
  let δ := sequenceLogits p x false - sequenceLogits p x true
  obtain ⟨qPlus, hPlus, hfPlus, hlPlus, hvPlus⟩ := training_shift p hp data labels needle hmiss true (δ + 1)
  obtain ⟨qMinus, hMinus, hfMinus, hlMinus, hvMinus⟩ := training_shift p hp data labels needle hmiss true (δ - 1)
  refine ⟨qPlus, qMinus, hPlus, hMinus, fun i => ⟨hfPlus i, hfMinus i⟩, ⟨hlPlus, hlMinus⟩, ?_, ?_⟩
  · have ht := hvPlus true
    have hf := hvPlus false
    simp only [Bool.false_eq_true, ↓reduceIte, add_zero] at ht hf
    dsimp [δ, x] at ht
    linarith
  · have ht := hvMinus true
    have hf := hvMinus false
    simp only [Bool.false_eq_true, ↓reduceIte, add_zero] at ht hf
    dsimp [δ, x] at ht
    linarith

example := training_shift (O := Bool) (T := 2) 0 (zero_mem _ _ _)
  (fun _ : Fin 3 => ⟨[false, true], by decide⟩) (fun _ => true) (fun _ => true)
  (by intro i; decide) true 2
example := binary_ambiguity (T := 2) 0 (zero_mem _ _ _)
  (fun _ : Fin 3 => ⟨[false, true], by decide⟩) (fun _ => true) (fun _ => true)
  (by intro i; decide)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Unseen

module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Unseen.Training

/-!
# Nonconstant observed answers can be fitted without identifying a missing one

Source: `review/convex/moments.md`, CM77. Two raw prefixes have different
observed next objects. An inhabited feasible model fits both with an arbitrary
positive margin. Its observed CE is below log 2, yet feasible changes preserve
that exact fit and give opposite strict answers at the omitted prefix.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Unseen.Witness

open PatternReduction

/-- Two observed raw prefixes, with different intended output objects. -/
def observed (label : Bool) : Prefix Bool 2 :=
  if label then ⟨[false, true], by decide⟩ else ⟨[false, false], by decide⟩

/-- Original indexed labels retain their two distinct object identities. -/
def labels (i : Fin 2) : Bool := if i = 0 then false else true

/-- Original indexed data for the nonconstant fit witness. -/
def data (i : Fin 2) : Prefix Bool 2 := observed (labels i)

/-- A different full prefix whose next object was not observed. -/
def missing : Prefix Bool 2 := ⟨[true, true], by decide⟩

/-- The two labels really differ, CM77; the example is not constant-label fitting. -/
theorem labels_differ : labels 0 ≠ labels 1 := by decide

/-- Every observed input differs from the omitted full prefix, CM77. -/
theorem data_missing (i : Fin 2) : (data i).val ≠ List.ofFn (fun _ : Fin 2 => true) := by
  fin_cases i <;> decide

/-- The original actual zero model has zero logits, CM77. -/
theorem sequenceLogits_zero {O : Type*} [Fintype O] [DecidableEq O] {T : ℕ}
    (tokens : Prefix O T) : sequenceLogits (0 : ReducedSequenceParameters O T) tokens = 0 := by
  funext c
  simp [sequenceLogits, complexLogits, complexForward, complexNeuron_eq, splitHinge, hinge]

/-- Both distinct observed next objects admit an arbitrary exact positive
logit margin in the same complete learned model, CM77. -/
theorem observed_logits_exists (r : ℝ) :
    ∃ p : ReducedSequenceParameters Bool 2,
      p ∈ complexDomain rawInput Set.univ diagonal ∧
      ∀ b c : Bool, sequenceLogits p (observed b) c = if c = b then r else 0 := by
  obtain ⟨p, hp, hpred⟩ := shifted_logits_exists (O := Bool) (T := 2) 0 (zero_mem _ _ _)
    (fun _ => false) false r
  obtain ⟨q, hq, hqpred⟩ := shifted_logits_exists p hp ![false, true] true r
  refine ⟨q, hq, ?_⟩
  intro b c
  rw [hqpred, hpred, sequenceLogits_zero]
  cases b <;> cases c <;> simp [observed, List.ofFn_succ]

/-- Ordinary observed CE is strictly below log 2 for both different labels,
CM77. Such a bound excludes an incorrect winner or tie. -/
theorem observed_fit_exists (r : ℝ) (hr : 0 < r) :
    ∃ p : ReducedSequenceParameters Bool 2,
      p ∈ complexDomain rawInput Set.univ diagonal ∧
      ∀ i : Fin 2, sequenceLoss (data i) (labels i) p < Real.log 2 := by
  obtain ⟨p, hp, hpred⟩ := observed_logits_exists r
  refine ⟨p, hp, ?_⟩
  intro i
  have he : Real.exp (-r) < 1 := by
    simpa using Real.exp_lt_exp.mpr (by linarith : -r < 0)
  have hl : Real.log (1 + Real.exp (-r)) < Real.log 2 :=
    Real.log_lt_log (by positivity) (by linarith)
  unfold sequenceLoss data
  rw [ConvexRecall.crossEntropy_relative]
  cases h : labels i <;> simpa [Fintype.sum_bool, hpred, add_comm] using hl

/-- The ambiguity theorem is inhabited by a genuinely nonconstant fitted
dataset, CM77. Both alternative unseen answers retain that strict observed fit. -/
theorem fitted_ambiguity :
    ∃ p q : ReducedSequenceParameters Bool 2,
      p ∈ complexDomain rawInput Set.univ diagonal ∧
      q ∈ complexDomain rawInput Set.univ diagonal ∧
      (∀ i : Fin 2, sequenceLoss (data i) (labels i) p < Real.log 2 ∧
        sequenceLoss (data i) (labels i) q < Real.log 2) ∧
      sequenceLogits p missing false < sequenceLogits p missing true ∧
      sequenceLogits q missing true < sequenceLogits q missing false := by
  obtain ⟨b, hb, hfit⟩ := observed_fit_exists 1 (by norm_num)
  obtain ⟨p, q, hp, hq, hpred, hloss, hpos, hneg⟩ :=
    binary_ambiguity b hb data labels (fun _ => true) data_missing
  refine ⟨p, q, hp, hq, ?_, ?_, ?_⟩
  · intro i
    have hpfit : sequenceLoss (data i) (labels i) p = sequenceLoss (data i) (labels i) b := by
      unfold sequenceLoss
      rw [(hpred i).1]
    have hqfit : sequenceLoss (data i) (labels i) q = sequenceLoss (data i) (labels i) b := by
      unfold sequenceLoss
      rw [(hpred i).2]
    constructor
    · rw [hpfit]
      exact hfit i
    · rw [hqfit]
      exact hfit i
  · simpa [missing, List.ofFn_succ] using hpos
  · simpa [missing, List.ofFn_succ] using hneg

example := observed_fit_exists 1 (by norm_num)

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Unseen.Witness

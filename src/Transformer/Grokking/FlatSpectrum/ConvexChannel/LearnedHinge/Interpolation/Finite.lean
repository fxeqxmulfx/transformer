module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Interpolation.Basic
public import Transformer.SequenceModel.Resources.Parameters

/-!
# Stored-bit limits for preserving the complete learned family's predictions

Source: `review/flat_spectrum.md`, CM84, using the existing R1 information
bound. This permits any fixed nonlinear evaluation of a finite parameter
codebook. Exact coverage of every strict classification in the current
family needs enough bits for all full-word labels. Below that budget one
fixed feasible learned model has predictions no code can preserve. This
does not lower-bound one structured task, learning time or optimizer success.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Interpolation

open PatternReduction
open Transformer.SequenceModel.Resources (Realizes)

variable {O D : Type*} [Fintype O] [DecidableEq O] [Fintype D] {T : ℕ}

/-- Evaluate the original learned model on the complete words, CM84. -/
noncomputable def wordLogits (p : ReducedSequenceParameters O T) (word : Fin T → O) : O → ℝ :=
  sequenceLogits p (fullPrefix word)

/-- Every next-object label table has a strictly classifying feasible point
in the actual complete family, CM84, with unit target margin. -/
theorem classification_interpolation (labels : (Fin T → O) → O) :
    ∃ p : ReducedSequenceParameters O T,
      p ∈ complexDomain rawInput Set.univ diagonal ∧ Realizes wordLogits p labels := by
  obtain ⟨p, hp, hlogits⟩ := logit_interpolation
    (fun word c => if c = labels word then 1 else 0)
  refine ⟨p, hp, ?_⟩
  intro word c hc
  simp [wordLogits, hlogits, hc]

/-- Preserving all strict classifications of this whole family requires
at least |O|^T stored parameter bits, CM84. The code's forward may be nonlinear
and may use exact arithmetic after decoding. Architecture and codebook are fixed. -/
theorem classification_bit_bound (bits : ℕ) (hobjects : 2 ≤ Fintype.card O)
    (score : (D → Fin (2 ^ bits)) → (Fin T → O) → O → ℝ)
    (hcover : ∀ p : ReducedSequenceParameters O T,
      p ∈ complexDomain rawInput Set.univ diagonal →
      ∀ labels : (Fin T → O) → O, Realizes wordLogits p labels →
      ∃ code, Realizes score code labels) :
    Fintype.card O ^ T ≤ bits * Fintype.card D := by
  have h := Transformer.SequenceModel.Resources.universal_bit_budget bits hobjects score (by
    intro labels
    obtain ⟨p, hp, hlabels⟩ := classification_interpolation labels
    exact hcover p hp labels hlabels)
  simpa only [Fintype.card_fun, Fintype.card_fin] using h

/-- Below that information budget there is one fixed feasible learned model
and target table whose entire strict classification no finite code preserves,
CM84. The model and table are chosen before the competing code. -/
theorem exists_classification_not_preserved (bits : ℕ) (hobjects : 2 ≤ Fintype.card O)
    (hbudget : bits * Fintype.card D < Fintype.card O ^ T)
    (score : (D → Fin (2 ^ bits)) → (Fin T → O) → O → ℝ) :
    ∃ p : ReducedSequenceParameters O T,
      p ∈ complexDomain rawInput Set.univ diagonal ∧
      ∃ labels : (Fin T → O) → O, Realizes wordLogits p labels ∧
        ∀ code, ¬ Realizes score code labels := by
  obtain ⟨labels, hmissing⟩ := Transformer.SequenceModel.Resources.exists_unrealizable
    bits hobjects (by simpa only [Fintype.card_fun, Fintype.card_fin] using hbudget) score
  obtain ⟨p, hp, hlabels⟩ := classification_interpolation labels
  exact ⟨p, hp, labels, hlabels, hmissing⟩

example := classification_bit_bound (O := Bool) (D := Fin 2 → Bool) (T := 2) 1
  (by decide) (fun code word c => if c = decide (code word = 1) then 1 else 0) (by
    intro p hp labels hlabels
    refine ⟨fun word => if labels word then 1 else 0, ?_⟩
    intro word c hc
    cases h : labels word <;> cases c <;> simp_all)

example := exists_classification_not_preserved (O := Bool) (D := Unit) (T := 2) 1
  (by decide) (by decide)
  (fun code word c => if c = (word 0 && decide (code () = 1)) then 1 else 0)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Interpolation

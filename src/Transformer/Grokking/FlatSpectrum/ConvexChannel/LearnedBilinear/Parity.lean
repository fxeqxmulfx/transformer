module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Sequence

/-!
# A learned pair/query interaction realizes three-object parity

Source: `review/flat_spectrum.md`, CM86. An explicit feasible joint point
strictly classifies all eight Boolean triples through the actual raw learned
encoders. Parity appears only in this witness's chosen weights and correctness
statement, never in the architecture or reader. This proves representational
capacity beyond CM85, not that CE training identifies any held-out labels.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Parity

open scoped BigOperators Matrix
open LearnedHinge (Prefix)

/-- The same three immutable input objects used for the CM85 obstruction,
CM86. The architecture reads the final object as query. -/
def triple (a b c : Bool) : Prefix Bool (2 + 1) := ⟨[a, b, c], by simp⟩

/-- An explicit feasible joint point witnessing a higher-order interaction,
CM86. Its coefficients are not a rule used by the forward implementation. -/
def parameters : SequenceParameters Bool 2 := fun out q d =>
  if out then
    match q, d with
    | some c, some (.inr (_, a, b)) => if a.xor (b.xor c) then 1 else -1
    | _, _ => 0
  else 0

/-- The actual learned path strictly separates parity in all eight cases,
CM86. Bias, both output objects and every coordinate remain represented. -/
theorem raw_logits (a b c : Bool) :
    sequenceLogits parameters (triple a b c) false = 0 ∧
    sequenceLogits parameters (triple a b c) true = if a.xor (b.xor c) then 1 else -1 := by
  simp only [sequenceLogits_eq, logits_eq]
  cases a <;> cases b <;> cases c <;>
    norm_num [parameters, queryRow, contextRow, pairAt, pairFlag, past, triple,
      Matrix.mulVec, dotProduct, Fintype.sum_option, Fintype.sum_sum_type,
      Fintype.sum_prod_type, Fin.sum_univ_two]

/-- Every triple has its ordinary original-object parity answer as strict
winner, CM86. There is no transformed or many-to-one output label. -/
theorem strict_winner (a b c : Bool) :
    sequenceLogits parameters (triple a b c) (!(a.xor (b.xor c))) <
      sequenceLogits parameters (triple a b c) (a.xor (b.xor c)) := by
  have h := raw_logits a b c
  cases hx : a.xor (b.xor c) <;> simp only [hx, Bool.false_eq_true, ↓reduceIte] at h ⊢ <;>
    simp only [Bool.not_false, Bool.not_true, h.1, h.2] <;> norm_num

/-- The complete original-object inference returns the parity object under
every tie selection, CM86. These inputs have no tie. -/
theorem prediction (a b c witness : Bool) :
    nextObject parameters (triple a b c) witness = a.xor (b.xor c) := by
  apply nextObject_of_unique
  intro o ho
  have he : o = !(a.xor (b.xor c)) := by
    cases hx : a.xor (b.xor c) <;> cases o <;> simp_all
  rw [he]
  exact strict_winner a b c

example : nextObject parameters (triple true true false) false = false :=
  prediction true true false false

/-- Overlapping adjacent occurrences retain multiplicity: false/true
appears twice in this four-object context, CM86. -/
theorem repeated_pair_count :
    contextRow (⟨[false, true, false, true], by simp⟩ : Prefix Bool 4)
      (some (.inr (⟨0, by decide⟩, false, true))) = 2 ∧
    contextRow (⟨[false, true, false, true], by simp⟩ : Prefix Bool 4)
      (some (.inr (⟨0, by decide⟩, true, false))) = 1 := by
  simp only [contextRow]
  rw [Fin.sum_univ_three, Fin.sum_univ_three]
  norm_num [Fin.sum_univ_succ, pairAt, pairFlag]

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Parity

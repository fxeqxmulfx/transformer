module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Sequence

/-!
# The pair/query model still loses four-object parity

Source: `review/convex/moments.md`, CM86. At a fixed final object, the context
reader has only singleton and adjacent-pair dependencies. Its third mixed
difference is zero. The identity holds for the actual learned logits, so even
the eight four-object prefixes with fixed final false cannot all be strictly
classified by parity. The positive three-object witness is not universality,
unseen-answer identification or empirical credit on any stress task.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Restriction

open scoped BigOperators Matrix
open LearnedHinge (Prefix)

/-- Three varying context objects with one fixed query object, CM86. -/
def cube (a b c : Bool) : Prefix Bool (3 + 1) := ⟨[a, b, c, false], by simp⟩

/-- The query is fixed while the three earlier original objects vary, CM86. -/
theorem cube_query (a b c : Bool) : queryRow (cube a b c) = queryRow (cube false false false) := by
  funext q
  cases q <;> simp [queryRow, cube]

/-- The complete context-address vector has zero third mixed difference,
CM86. All position entries, pair entries, repetitions and bias are included. -/
theorem context_third_difference :
    contextRow (past (cube true true true)) - contextRow (past (cube true true false)) -
      contextRow (past (cube true false true)) - contextRow (past (cube false true true)) +
      contextRow (past (cube true false false)) + contextRow (past (cube false true false)) +
      contextRow (past (cube false false true)) - contextRow (past (cube false false false)) = 0 := by
  funext d
  have hs (f : Fin (3 - 1) → ℂ) :
      (∑ k, f k) = f ⟨0, by norm_num⟩ + f ⟨1, by norm_num⟩ := by
    change (∑ k : Fin 2, f k) = _
    exact Fin.sum_univ_two f
  fin_cases d <;> simp only [Pi.sub_apply, Pi.add_apply, Pi.zero_apply, contextRow, hs]
  all_goals norm_num [past, cube, pairAt, pairFlag]

/-- A zero context mixed difference is preserved by the entire recovered
complex network, CM86. No rank or positivity hypothesis is used. -/
theorem logit_third_difference (p : SequenceParameters Bool 3) (c : Bool) :
    sequenceLogits p (cube true true true) c - sequenceLogits p (cube true true false) c -
      sequenceLogits p (cube true false true) c - sequenceLogits p (cube false true true) c +
      sequenceLogits p (cube true false false) c + sequenceLogits p (cube false true false) c +
      sequenceLogits p (cube false false true) c - sequenceLogits p (cube false false false) c = 0 := by
  have h := congrArg
    (fun y => (forward (p c) (queryRow (cube false false false)) y).re) context_third_difference
  simp only [forward_eq, Matrix.mulVec_add, Matrix.mulVec_sub, dotProduct_add, dotProduct_sub,
    mul_add, mul_sub, Complex.add_re, Complex.sub_re, Matrix.mulVec_zero, dotProduct_zero,
    mul_zero, Complex.zero_re] at h
  simpa only [sequenceLogits_eq, logits_eq, cube_query] using h

/-- No joint parameter point strictly classifies even this fixed-query
slice of four-bit parity, CM86. Further learned structure is necessary. -/
theorem parity_not_realizable (p : SequenceParameters Bool 3) :
    ¬ ∀ a b c : Bool,
      sequenceLogits p (cube a b c) (!(a.xor (b.xor c))) <
        sequenceLogits p (cube a b c) (a.xor (b.xor c)) := by
  intro h
  have h000 := h false false false
  have h001 := h false false true
  have h010 := h false true false
  have h011 := h false true true
  have h100 := h true false false
  have h101 := h true false true
  have h110 := h true true false
  have h111 := h true true true
  have ht := logit_third_difference p true
  have hf := logit_third_difference p false
  simp only [Bool.xor_false, Bool.xor_true, Bool.not_false, Bool.not_true] at h000 h001 h010 h011
  simp only [Bool.xor_false, Bool.xor_true, Bool.not_false, Bool.not_true] at h100 h101 h110 h111
  linarith

end Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Restriction

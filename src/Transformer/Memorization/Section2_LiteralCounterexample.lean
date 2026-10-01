import Transformer.Memorization.Section2_Kolmogorov

/-!
# An allowed decoder need not exploit side information

arXiv:2505.24832v3, Section 2.2, Definitions 2 and 3, Proposition 4.
Definition 2 allows an arbitrary computational model. This total,
computable decoder uses the side-information count to choose a reversible
literal encoding. All descriptions have the same length as their output.
It is not a universal interpreter: universality is a missing hypothesis
if an algorithmic/Shannon comparison is intended.
-/

namespace Transformer.Memorization

/-- Section 2.2, Definition 2, counterexample interpreter: an even number
of side strings selects literal order; an odd number selects reverse order. -/
def reversibleLiteralDecoder (p : BitString) (side : List BitString) : Option BitString :=
  some (if side.length % 2 = 0 then p else p.reverse)

/-- Section 2.2, Definition 2: each successful program preserves length. -/
theorem reversibleLiteralDecoder_length (p x : BitString) (side : List BitString)
    (h : reversibleLiteralDecoder p side = some x) : p.length = x.length := by
  have he := Option.some.inj h
  have hl := congrArg List.length he
  by_cases hs : side.length % 2 = 0 <;> simpa [reversibleLiteralDecoder, hs] using hl

/-- Section 2.2: a one-bit description witnesses the decoding premise. -/
example : reversibleLiteralDecoder [true] [[]] = some [true] := rfl

/-- Section 2.2, Definition 2: every string has precisely its literal
length as complexity, under all side-information configurations. -/
theorem reversibleLiteral_complexity (x : BitString) (side : List BitString) :
    descriptionComplexity reversibleLiteralDecoder x side = some x.length := by
  by_cases hs : side.length % 2 = 0
  · apply descriptionComplexity_eq_minimal _ x x side
    · simp [reversibleLiteralDecoder, hs]
    · intro q hq
      exact (reversibleLiteralDecoder_length q x side hq).symm.le
  · have hdec : reversibleLiteralDecoder x.reverse side = some x := by
      simp [reversibleLiteralDecoder, hs]
    simpa using descriptionComplexity_eq_minimal reversibleLiteralDecoder x x.reverse side
      hdec (fun q hq => by simpa using (reversibleLiteralDecoder_length q x side hq).symm.le)

/-- Section 2.2, Definition 3: this genuine interpreter stores no
algorithmic information through the reference/trained side strings. -/
theorem reversibleLiteral_unintended (x reference trained : BitString) :
    kolmogorovUnintended reversibleLiteralDecoder x reference trained = some 0 := by
  simp [kolmogorovUnintended, reversibleLiteral_complexity]

end Transformer.Memorization

module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedQuadratic.Sequence

/-!
# The polynomial-cost quadratic family loses higher-order interactions

Source: `review/convex/moments.md`, CM85. Every quadratic reading has zero
third mixed difference. This applies to the actual recovered raw-sequence
logits, not just an auxiliary feature head. Therefore no point in this family
strictly classifies three-bit parity on all eight original Boolean prefixes.
This is an expressivity control, not a claim about any particular MQAR data
split, a proof against deeper models, or an empirical score.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedQuadratic.Restriction

open scoped BigOperators Matrix ComplexOrder MatrixOrder Matrix.Norms.Elementwise
open LearnedHinge (Coordinate Prefix)

/-- The third mixed difference vanishes for every full complex quadratic
form, CM85. Hermiticity and positivity are not needed for this identity. -/
theorem third_difference {D : Type*} [Fintype D] (M : Matrix D D ℂ)
    (a u v w : D → ℂ) :
    quadratic M (a + u + v + w) - quadratic M (a + u + v) -
      quadratic M (a + u + w) - quadratic M (a + v + w) +
      quadratic M (a + u) + quadratic M (a + v) + quadratic M (a + w) -
      quadratic M a = 0 := by
  simp only [quadratic, Matrix.mulVec_add, star_add, add_dotProduct, dotProduct_add,
    Complex.add_re]
  ring

/-- Eight distinct constant-object prefixes in the common input type, CM85. -/
def cube (a b c : Bool) : Prefix Bool 3 := ⟨[a, b, c], by simp⟩

/-- Each visible Boolean object changes only its own learned address row,
CM85. This expresses the raw interface, not a task-specific feature program. -/
theorem cube_row (a b c : Bool) :
    inputRow (cube a b c) = inputRow (cube false false false) +
      (if a then inputRow (cube true false false) - inputRow (cube false false false) else 0) +
      (if b then inputRow (cube false true false) - inputRow (cube false false false) else 0) +
      (if c then inputRow (cube false false true) - inputRow (cube false false false) else 0) := by
  funext d
  fin_cases d <;> cases a <;> cases b <;> cases c <;>
    norm_num [inputRow, LearnedHinge.Projection.rawRow, cube]

/-- The actual raw-input quadratic readings have zero third difference,
CM85. All seven addresses, including the bias, remain present. -/
theorem raw_third_difference (M : Matrix (Coordinate Bool 3) (Coordinate Bool 3) ℂ) :
    quadratic M (inputRow (cube true true true)) -
      quadratic M (inputRow (cube true true false)) -
      quadratic M (inputRow (cube true false true)) -
      quadratic M (inputRow (cube false true true)) +
      quadratic M (inputRow (cube true false false)) +
      quadratic M (inputRow (cube false true false)) +
      quadratic M (inputRow (cube false false true)) -
      quadratic M (inputRow (cube false false false)) = 0 := by
  have h := third_difference M (inputRow (cube false false false))
    (inputRow (cube true false false) - inputRow (cube false false false))
    (inputRow (cube false true false) - inputRow (cube false false false))
    (inputRow (cube false false true) - inputRow (cube false false false))
  rw [cube_row true true true, cube_row true true false,
    cube_row true false true, cube_row false true true]
  simpa only [Bool.false_eq_true, ↓reduceIte, add_zero, add_sub_cancel] using h

/-- The actual learned original-object logits inherit that limitation on
the entire PSD domain, CM85. Both signs and both complex factors are used. -/
theorem logit_third_difference {p : Parameters (Coordinate Bool 3) Bool}
    (hp : p ∈ domain) (c : Bool) :
    sequenceLogits p (cube true true true) c - sequenceLogits p (cube true true false) c -
      sequenceLogits p (cube true false true) c - sequenceLogits p (cube false true true) c +
      sequenceLogits p (cube true false false) c + sequenceLogits p (cube false true false) c +
      sequenceLogits p (cube false false true) c - sequenceLogits p (cube false false false) c = 0 := by
  simp only [sequenceLogits_eq, logits_eq hp]
  have ht := raw_third_difference (p c true)
  have hf := raw_third_difference (p c false)
  linarith

/-- No parameter point correctly strictly ranks all three-bit parity
answers, CM85. This is a concrete lost function of the smaller family. -/
theorem parity_not_realizable {p : Parameters (Coordinate Bool 3) Bool} (hp : p ∈ domain) :
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
  have ht := logit_third_difference hp true
  have hf := logit_third_difference hp false
  simp only [Bool.xor_false, Bool.xor_true, Bool.not_false, Bool.not_true] at h000 h001 h010 h011
  simp only [Bool.xor_false, Bool.xor_true, Bool.not_false, Bool.not_true] at h100 h101 h110 h111
  linarith

example := logit_third_difference (p := 0) zero_mem true
example := parity_not_realizable (p := 0) zero_mem

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedQuadratic.Restriction

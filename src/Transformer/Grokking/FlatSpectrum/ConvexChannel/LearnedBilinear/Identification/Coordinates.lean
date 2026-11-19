module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification.Basic

/-!
# Exact finite real coordinates of the actual learned complex margins

Source: `review/convex/moments.md`, CM87. Both complex coordinates of every
joint matrix are retained. Encoding and decoding are inverse. The fixed raw
address rows have zero imaginary part, while the complete learned complex
weights enter the real output through the proved recovered-forward identity.
The coordinate reading certifies inference; it does not redefine inference.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification

open scoped BigOperators Matrix
open LearnedHinge (Prefix)

variable {O : Type*} [Fintype O] [DecidableEq O] {T : ℕ}

/-- Both components of every actual joint parameter, CM87. -/
def encode (p : SequenceParameters O T) : RealCoordinate O T → ℝ := fun z =>
  if z.1 then (p z.2.1 z.2.2.1 z.2.2.2).im else (p z.2.1 z.2.2.1 z.2.2.2).re

/-- Every complete real coordinate vector gives an actual complex joint
parameter, CM87. There is no rank, label or observed-coordinate restriction. -/
def decode (r : RealCoordinate O T → ℝ) : SequenceParameters O T :=
  fun c q d => ⟨r (false, c, q, d), r (true, c, q, d)⟩

omit [Fintype O] [DecidableEq O] in
/-- Real coordinates are recovered without loss, CM87. -/
theorem encode_decode (r : RealCoordinate O T → ℝ) : encode (decode r) = r := by
  funext z
  rcases z with ⟨b, c, q, d⟩
  cases b <;> rfl

omit [Fintype O] [DecidableEq O] in
/-- Every actual joint parameter is preserved by coordinate reconstruction,
CM87. Both complex components are covered. -/
theorem decode_encode (p : SequenceParameters O T) : decode (encode p) = p := by
  funext c q d
  exact Complex.ext rfl rfl

omit [Fintype O] in
/-- The query's raw address indicators have no imaginary part, CM87. -/
theorem query_cast (tokens : Prefix O (T + 1)) (q : QueryCoordinate O) :
    ((queryRow tokens q).re : ℂ) = queryRow tokens q := by
  cases q <;> simp [queryRow, apply_ite]

omit [Fintype O] in
/-- Context indicators and pair multiplicities are real, CM87. This does
not restrict the learned weights or drop their imaginary coordinates. -/
theorem context_cast (tokens : Prefix O T) (d : ContextCoordinate O T) :
    ((contextRow tokens d).re : ℂ) = contextRow tokens d := by
  cases d with
  | none => simp [contextRow]
  | some d => cases d <;> simp [contextRow, apply_ite]

/-- The full complex bilinear reading at real address rows, CM87. Its
real output contains the real and imaginary joint weights with opposite signs. -/
theorem reading_real {Q D : Type*} [Fintype Q] [Fintype D]
    (M : Matrix Q D ℂ) (x : Q → ℝ) (y : D → ℝ) :
    ((1 + Complex.I) * (star (fun q => (x q : ℂ)) ⬝ᵥ
      (M *ᵥ (fun d => (y d : ℂ))))).re =
      ∑ q, ∑ d, ((M q d).re - (M q d).im) * x q * y d := by
  simp only [Matrix.mulVec, dotProduct, Finset.mul_sum, Complex.re_sum,
    Pi.star_apply, Complex.star_def, Complex.conj_ofReal]
  apply Finset.sum_congr rfl
  intro q hq
  apply Finset.sum_congr rfl
  intro d hd
  simp
  ring

/-- Expansion of the actual learned raw logit, CM87. The learned path and
its exact factor recovery remain the forward definition. -/
theorem logit_expansion (p : SequenceParameters O T) (tokens : Prefix O (T + 1)) (c : O) :
    sequenceLogits p tokens c =
      ∑ q, ∑ d, ((p c q d).re - (p c q d).im) *
        (queryRow tokens q).re * (contextRow (past tokens) d).re := by
  rw [sequenceLogits_eq, logits_eq]
  have h := reading_real (p c) (fun q => (queryRow tokens q).re)
    (fun d => (contextRow (past tokens) d).re)
  simpa only [query_cast, context_cast] using h

omit [Fintype O] in
/-- The query diagnostic row's imaginary component is zero, CM87. -/
theorem query_im (tokens : Prefix O (T + 1)) (q : QueryCoordinate O) :
    (queryRow tokens q).im = 0 := by
  have h := congrArg Complex.im (query_cast tokens q)
  simpa using h.symm

omit [Fintype O] in
/-- Pair multiplicities also have zero imaginary component, CM87. -/
theorem context_im (tokens : Prefix O T) (d : ContextCoordinate O T) :
    (contextRow tokens d).im = 0 := by
  have h := congrArg Complex.im (context_cast tokens d)
  simpa using h.symm

/-- The actual original-object margin equals its finite complete real
coordinate reading, CM87. This is the required bridge from inference to the
certificate's explicitly checkable coefficient identities. -/
theorem margin_expansion (p : SequenceParameters O T) (tokens : Prefix O (T + 1)) (a b : O) :
    margin p tokens a b = ∑ z, coefficient tokens a b z * encode p z := by
  simp only [margin, logit_expansion, Fintype.sum_prod_type, Fintype.sum_bool,
    coefficient, encode, Bool.false_eq_true, ↓reduceIte, Complex.mul_re, query_im, context_im,
    mul_zero, sub_zero, mul_assoc, sub_mul, ite_mul, one_mul, zero_mul,
    Finset.sum_sub_distrib, Finset.sum_ite_irrel, Finset.sum_const_zero,
    Fintype.sum_ite_eq', neg_mul, Finset.sum_neg_distrib]
  simp only [mul_assoc, mul_comm]
  ring

/-- Every real coordinate vector evaluates the same actual learned margin,
CM87. Thus duality ranges over the entire complex parameter space. -/
theorem margin_decode (r : RealCoordinate O T → ℝ) (tokens : Prefix O (T + 1)) (a b : O) :
    margin (decode r) tokens a b = ∑ z, coefficient tokens a b z * r z := by
  rw [margin_expansion, encode_decode]

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification

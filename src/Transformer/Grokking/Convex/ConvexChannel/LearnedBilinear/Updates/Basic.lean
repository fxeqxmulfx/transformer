module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Identification.Coordinates
public import Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Identification.Forms

/-!
# Complete-coordinate changes of actual learned logits

Source: `review/convex/moments.md`, CM88. Each scalar joint coordinate changes
exactly one original-object logit, with an input-only signed multiplicity.
The expansion is proved from the actual recovered complex path. It supports
exact reuse of forward loss queries without dropping any trainable coordinate.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Updates

open scoped BigOperators
open LearnedHinge (Prefix)
open Identification

variable {O : Type*} [Fintype O] [DecidableEq O] {T : ℕ}

/-- Signed raw-input response of a real or imaginary parameter coordinate,
CM88. Repeated context pairs retain their complete multiplicity. -/
def response (tokens : Prefix O (T + 1)) (z : RealCoordinate O T) : ℝ :=
  (if z.1 then -1 else 1) * (queryRow tokens z.2.2.1).re *
    (contextRow (past tokens) z.2.2.2).re

/-- Complete coefficient vector of one original-object logit, CM88. -/
def logitCoefficient (tokens : Prefix O (T + 1)) (c : O) (z : RealCoordinate O T) : ℝ :=
  if z.2.1 = c then response tokens z else 0

/-- Exact coordinate reading of the actual raw logit, CM88. Both complex
components and all classes remain present. -/
theorem logit_read (tokens : Prefix O (T + 1)) (c : O) (p : SequenceParameters O T) :
    form (logitCoefficient tokens c) (encode p) = sequenceLogits p tokens c := by
  simp only [form_apply, logitCoefficient, response, encode, Fintype.sum_prod_type,
    Fintype.sum_bool, Bool.false_eq_true, ↓reduceIte, ite_mul, zero_mul,
    Finset.sum_ite_irrel, Finset.sum_const_zero, Fintype.sum_ite_eq',
    one_mul, neg_mul, Finset.sum_neg_distrib, logit_expansion,
    sub_mul, Finset.sum_sub_distrib]
  simp only [mul_left_comm, mul_comm]
  ring

/-- Full real-coordinate perturbation, CM88. Every original coordinate
remains in the parameter domain. -/
noncomputable def shift (r : RealCoordinate O T → ℝ) (z : RealCoordinate O T) (t : ℝ) :
    RealCoordinate O T → ℝ := by
  classical
  exact r + t • Pi.single z 1

omit [DecidableEq O] in
/-- A complete finite form evaluates a coordinate perturbation exactly,
CM88. This is an algebraic identity, not a finite-difference approximation. -/
theorem form_shift (v r : RealCoordinate O T → ℝ) (z : RealCoordinate O T) (t : ℝ) :
    form v (shift r z t) = form v r + t * v z := by
  classical
  simp only [shift, map_add, map_smul, form_single, smul_eq_mul]

/-- The actual learned raw logits after a scalar parameter perturbation,
CM88. Only the coordinate's own original class receives its signed response. -/
theorem logit_shift (r : RealCoordinate O T → ℝ) (tokens : Prefix O (T + 1))
    (z : RealCoordinate O T) (t : ℝ) (c : O) :
    sequenceLogits (decode (shift r z t)) tokens c = sequenceLogits (decode r) tokens c +
      if c = z.2.1 then t * response tokens z else 0 := by
  rw [← logit_read, encode_decode, form_shift, ← encode_decode r, logit_read]
  by_cases h : c = z.2.1
  · simp [logitCoefficient, h, decode_encode]
  · simp [logitCoefficient, h, Ne.symm h, decode_encode]

/-- The real class-bias coordinate, CM88. It is an ordinary trainable
coordinate and has unit response on every raw prefix. -/
def biasCoordinate (c : O) : RealCoordinate O T := (false, c, none, none)

omit [Fintype O] in
/-- Every prefix, including an empty prefix, reads the learned bias once,
CM88. -/
theorem bias_response (tokens : Prefix O (T + 1)) (c : O) :
    response tokens (biasCoordinate c) = 1 := by
  simp [response, biasCoordinate, queryRow, contextRow]

/-- Every coordinate query is exactly a scaled class-bias query for that
record, CM88. This proves reuse over all coordinates, not coordinate sampling. -/
theorem coordinate_query_eq_bias (r : RealCoordinate O T → ℝ) (tokens : Prefix O (T + 1))
    (z : RealCoordinate O T) (t : ℝ) :
    sequenceLogits (decode (shift r z t)) tokens =
      sequenceLogits (decode (shift r (biasCoordinate z.2.1) (t * response tokens z))) tokens := by
  funext c
  rw [logit_shift, logit_shift, bias_response tokens z.2.1, mul_one]
  rfl

end Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Updates

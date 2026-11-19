module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification.Duality

/-!
# Exact rational checking of finite observed-margin certificates

Source: `review/convex/moments.md`, CM87. Raw coefficient rows are rational
integer counts with signs. This checker accepts rational weights and a rational
bound, and checks every real/imaginary, class, query and context coordinate.
Its correctness connects to the actual real-valued learned margins. It checks
a supplied certificate; it neither discovers a task family nor runs a solver.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification

open scoped BigOperators
open LearnedHinge (Prefix)

variable {O : Type*} [Fintype O] [DecidableEq O] {T N : ℕ}

/-- Exact rational query-address counts, CM87. -/
def rationalQuery (tokens : Prefix O (T + 1)) : QueryCoordinate O → ℚ
  | none => 1
  | some o => if tokens.val.getLast? = some o then 1 else 0

/-- Exact rational context counts, including repeated adjacent occurrences,
CM87. The tested pair address includes its flag. -/
def rationalContext (tokens : Prefix O T) : ContextCoordinate O T → ℚ
  | none => 1
  | some (.inl (i, o)) => if tokens.val[i.val]? = some o then 1 else 0
  | some (.inr p) => ∑ k : Fin (T - 1), if pairAt tokens k = some p then 1 else 0

/-- The complete raw margin coefficient in exact rational arithmetic, CM87. -/
def rationalCoefficient (tokens : Prefix O (T + 1)) (a b : O)
    (z : RealCoordinate O T) : ℚ :=
  ((if z.2.1 = a then 1 else 0) - (if z.2.1 = b then 1 else 0)) *
    (if z.1 then -1 else 1) * rationalQuery tokens z.2.2.1 *
      rationalContext (past tokens) z.2.2.2

omit [Fintype O] in
/-- Rational query counts embed into the actual complex address row, CM87. -/
theorem rationalQuery_cast (tokens : Prefix O (T + 1)) (q : QueryCoordinate O) :
    (rationalQuery tokens q : ℝ) = (queryRow tokens q).re := by
  cases q <;> simp [rationalQuery, queryRow, apply_ite]

omit [Fintype O] in
/-- Rational context counts preserve every actual occurrence, CM87. -/
theorem rationalContext_cast (tokens : Prefix O T) (d : ContextCoordinate O T) :
    (rationalContext tokens d : ℝ) = (contextRow tokens d).re := by
  cases d with
  | none => simp [rationalContext, contextRow]
  | some d => cases d <;> simp [rationalContext, contextRow, apply_ite]

omit [Fintype O] in
/-- Rational coefficient checking is exact for the complete actual learned
margin, CM87. It introduces no numerical tolerance or omitted coordinate. -/
theorem rationalCoefficient_cast (tokens : Prefix O (T + 1)) (a b : O)
    (z : RealCoordinate O T) :
    (rationalCoefficient tokens a b z : ℝ) = coefficient tokens a b z := by
  rcases z with ⟨flag, c, q, d⟩
  cases flag <;> by_cases ha : c = a <;> by_cases hb : c = b <;>
    simp_all [rationalCoefficient, coefficient, rationalQuery_cast, rationalContext_cast,
      Complex.mul_re, query_im, context_im]

/-- Finite executable exact-rational certificate checker, CM87. Every
coordinate and every indexed training-label comparison is tested. -/
def checkCertificate (tokens : Fin N → Prefix O (T + 1)) (labels : Fin N → O)
    (query : Prefix O (T + 1)) (a b : O) (bound : ℚ) (w : Fin N → O → ℚ) : Bool :=
  decide ((∀ i c, 0 ≤ w i c) ∧ (∀ i, w i (labels i) = 0) ∧
    bound ≤ ∑ i, ∑ c, w i c ∧
    ∀ z : RealCoordinate O T, rationalCoefficient query a b z =
      ∑ i, ∑ c, w i c * rationalCoefficient (tokens i) (labels i) c z)

/-- Acceptance is equivalent to the original real certificate for the
supplied rational weights and bound, CM87. This is checker correctness,
not a claim that this implementation searches for certificates. -/
theorem checkCertificate_iff (tokens : Fin N → Prefix O (T + 1)) (labels : Fin N → O)
    (query : Prefix O (T + 1)) (a b : O) (bound : ℚ) (w : Fin N → O → ℚ) :
    checkCertificate tokens labels query a b bound w = true ↔
      Certificate tokens labels query a b (bound : ℝ) (fun i c => (w i c : ℝ)) := by
  simp only [checkCertificate, decide_eq_true_eq, Certificate,
    ← rationalCoefficient_cast, ← Rat.cast_mul, ← Rat.cast_sum,
    ← Rat.cast_zero (α := ℝ), Rat.cast_inj, Rat.cast_le]

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification

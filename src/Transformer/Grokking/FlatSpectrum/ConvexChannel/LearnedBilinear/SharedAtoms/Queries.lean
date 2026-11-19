module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Model

/-!
# Feasible forward-only class queries in the bounded learned domain

Source: `review/convex/moments.md`, CM92. Shrinking the current joint state by
1-eta leaves room for either signed class-bias atom. The two actual bounded
parameter queries are exactly finite CE logit queries of size eta*B around
the shrunken state. Mini and forward-only Mini must use the same centre.
No analytically computed CE derivative is needed for the latter construction.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms

open scoped BigOperators
open Identification
open LearnedHinge (Prefix)

variable {O : Type*} [Fintype O] [DecidableEq O] {T : ℕ}

/-- A signed bias for exactly one original class, CM92. It is an ordinary
learned factor point, used here only to construct complete forward queries. -/
def bias (target : O) (s : ℝ) : Atom O T where
  output c := if c = target then (s : ℂ) else 0
  query q := if q = none then 1 else 0
  context d := if d = none then 1 else 0

/-- Both signs of a unit class bias are admissible components, CM92. -/
theorem bias_unit (target : O) (s : ℝ) (hs : s ^ 2 ≤ 1) : Unit (bias target s : Atom O T) := by
  simpa [Unit, size, bias, apply_ite, Complex.normSq_apply, sq, Fintype.sum_option] using hs

omit [Fintype O] in
/-- The bias component is exactly the complete real class-bias coordinate,
CM92. Every other real or imaginary coordinate is retained with value zero. -/
theorem bias_encode [DecidableEq (RealCoordinate O T)] (target : O) (s : ℝ) :
    encode (joint (bias target s : Atom O T)) = s • Pi.single (Updates.biasCoordinate target) 1 := by
  funext z
  rcases z with ⟨b, c, q, d⟩
  cases b <;> cases q <;> cases d <;> by_cases hc : c = target <;>
    simp [encode, joint, bias, Updates.biasCoordinate, hc]

/-- A feasible signed loss-query parameter, CM92. The original state and
the selected unit component remain explicit, rather than perturbing logits
without a corresponding admissible learned parameter. -/
noncomputable def queryPoint (B eta : ℝ) (r : RealCoordinate O T → ℝ) (target : O) (s : ℝ) :
    RealCoordinate O T → ℝ :=
  (1 - eta) • r + eta • (B • encode (joint (bias target s)))

/-- The two signed query points remain in the same bounded domain, CM92. -/
theorem query_feasible (B eta : ℝ) (r : RealCoordinate O T → ℝ) (target : O) (s : ℝ)
    (hr : r ∈ domain B) (he : 0 ≤ eta) (he1 : eta ≤ 1) (hs : s ^ 2 ≤ 1) :
    queryPoint B eta r target s ∈ domain B :=
  step_feasible B eta r _ hr (bias_unit target s hs) he he1

omit [Fintype O] in
/-- The exact complete joint perturbation at the shrunken centre, CM92. -/
theorem query_eq_shift (B eta : ℝ) (r : RealCoordinate O T → ℝ) (target : O) (s : ℝ) :
    queryPoint B eta r target s =
      Updates.shift ((1 - eta) • r) (Updates.biasCoordinate target) (eta * B * s) := by
  classical
  simp only [queryPoint, bias_encode, Updates.shift, smul_smul, mul_assoc]

/-- Every original object score of the actual feasible query, CM92.
The plus/minus step is eta*B, independent of prefix length or label identity. -/
theorem query_logits (B eta : ℝ) (r : RealCoordinate O T → ℝ) (target c : O) (s : ℝ)
    (tokens : Prefix O (T + 1)) :
    sequenceLogits (decode (queryPoint B eta r target s)) tokens c =
      sequenceLogits (decode ((1 - eta) • r)) tokens c +
        if c = target then eta * B * s else 0 := by
  rw [query_eq_shift, Updates.logit_shift, Updates.bias_response]
  simp only [Updates.biasCoordinate, mul_one]

/-- Two feasible model evaluations supply the usual central class-bias
CE difference at their common centre, CM92. CM88 reuses these per-record
quantities for the complete real/imaginary coordinate direction. No equality
between a finite difference and an analytic derivative is asserted. -/
theorem feasible_difference (B eta : ℝ) (r : RealCoordinate O T → ℝ)
    (tokens : Prefix O (T + 1)) (target c : O) :
    (Updates.recordLoss (queryPoint B eta r c 1) tokens target -
      Updates.recordLoss (queryPoint B eta r c (-1)) tokens target) / (2 * (eta * B)) =
        Updates.coordinateDifference ((1 - eta) • r) tokens target
          (Updates.biasCoordinate c) (eta * B) := by
  simp only [query_eq_shift, mul_one, mul_neg_one, Updates.coordinateDifference]

example : Unit (bias true (-1) : Atom Bool 1) := bias_unit true (-1) (by norm_num)

example : queryPoint 2 (1 / 2) (0 : RealCoordinate Bool 1 → ℝ) true (-1) ∈ domain 2 :=
  query_feasible 2 (1 / 2) 0 true (-1) (zero_mem 2)
    (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms

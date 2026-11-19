module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Sequence

/-!
# Observed margins and finite original-object certificates

Source: `review/convex/moments.md`, CM87, initially requested as one open Lean
statement on 2026-11-16 and subsequently proved. The bridge is a finite certificate for forced
class margins in the actual learned model. The certificate uses only observed
prefixes/labels and the query prefix; it contains no query label, unknown target
parameters, successful optimizer, or assumption of correct unseen predictions.
Its finite coefficient equalities are connected to the actual recovered
forward in the subsequent coordinate and duality modules.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Identification

open scoped BigOperators Matrix
open LearnedHinge (Prefix)

variable {O : Type*} [Fintype O] [DecidableEq O] {T N : ℕ}

/-- Margin between two original objects after the actual learned path,
CM87. Candidate objects are prediction queries, not supplied true answers. -/
noncomputable def margin (p : SequenceParameters O T) (tokens : Prefix O (T + 1))
    (a b : O) : ℝ := sequenceLogits p tokens a - sequenceLogits p tokens b

/-- Unit-margin consistency with every original observed label, CM87.
This is a predicate of its data and parameters, not an identification claim. -/
noncomputable def FitsObserved (tokens : Fin N → Prefix O (T + 1))
    (labels : Fin N → O) (p : SequenceParameters O T) : Prop :=
  ∀ i c, c ≠ labels i → 1 ≤ margin p (tokens i) (labels i) c

/-- Every joint real coordinate: imaginary flag, class, query and context
address. No class, complex component or unobserved pair is removed, CM87. -/
abbrev RealCoordinate (O : Type*) (T : ℕ) :=
  Bool × O × QueryCoordinate O × ContextCoordinate O T

/-- Finite coefficients of an actual raw margin in the complete joint real
coordinates, CM87. Their identity with `margin` is a required proof obligation,
not the definition of inference. False selects real, true imaginary entries. -/
def coefficient (tokens : Prefix O (T + 1)) (a b : O) : RealCoordinate O T → ℝ :=
  fun z => ((if z.2.1 = a then 1 else 0) - (if z.2.1 = b then 1 else 0)) *
    (if z.1 then -1 else 1) *
    (queryRow tokens z.2.2.1 * contextRow (past tokens) z.2.2.2).re

/-- A finite observed-label certificate for one query margin, CM87. Weights
on a training label against itself are zero; all other weights are nonnegative.
The coefficient identity can be checked on the entire finite coordinate set.
It does not quantify over agreeing completions or assume the desired answer. -/
def Certificate (tokens : Fin N → Prefix O (T + 1)) (labels : Fin N → O)
    (query : Prefix O (T + 1)) (a b : O) (bound : ℝ) (w : Fin N → O → ℝ) : Prop :=
  (∀ i c, 0 ≤ w i c) ∧
  (∀ i, w i (labels i) = 0) ∧
  bound ≤ ∑ i, ∑ c, w i c ∧
  ∀ z : RealCoordinate O T, coefficient query a b z =
    ∑ i, ∑ c, w i c * coefficient (tokens i) (labels i) c z

/-- A two-class feasible parameter point for the theorem's consistency
hypothesis, CM87. The full original vocabulary is present. -/
def feasiblePoint : SequenceParameters Bool 0 := fun c q d =>
  if c then (if q = none ∧ d = none then 1 else 0) else 0

example : ∃ p : SequenceParameters Bool 0,
    FitsObserved (fun _ : Fin 1 => ⟨[], by simp⟩) (fun _ => true) p := by
  refine ⟨feasiblePoint, ?_⟩
  intro i c hc
  cases c
  · norm_num [margin, sequenceLogits_eq, logits_eq, feasiblePoint, queryRow,
      contextRow, past, Matrix.mulVec, dotProduct, Fintype.sum_option,
      Fintype.sum_sum_type, Fintype.sum_prod_type]
  · exact False.elim (hc rfl)

end Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Identification

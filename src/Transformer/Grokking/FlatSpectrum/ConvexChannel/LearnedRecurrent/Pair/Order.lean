module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Pair.Basic

/-!
# Noncommuting learned transitions retain sequence order

Source: `review/convex/moments.md`, CM97. A nonconstant parameter witness in the
full two-state family distinguishes the swapped prefixes forbidden by CM96's
scalar obstruction. Both transitions preserve the state norm. This is an
expressivity proof, never a supplied learning rule or trained stress result.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Pair

/-- One possible assignment of learned matrix entries, CM97. The generic
family does not prescribe these matrices or object interpretations. -/
def orderTransition (o : Bool) : Matrix :=
  if o then ((0, 1), (1, 0)) else ((1, 0), (0, -1))

/-- A parameter witness whose imaginary state/readouts contribute to
the actual real scores, CM97. It is not supplied to a native learner. -/
noncomputable def orderWitness : Parameters Bool State :=
  parameters (Complex.I, Complex.I) orderTransition
    (fun c => if c then (-Complex.I, 0) else (Complex.I, 0))

/-- Every explicit operator constraint is inhabited by this nonconstant
parameter point, CM97. -/
theorem order_unit : Unit orderWitness :=
  unit _ _ _ (by simp [Prod.norm_def])
    (by intro o; cases o <;> simp [orderTransition])
    (by intro c; cases c <;> simp)

/-- Both learned witness transitions preserve the maximum state norm,
CM97, at every state rather than only on an observed prefix. -/
theorem order_transition_isometry (o : Bool) (z : State) :
    ‖orderWitness.transition o z‖ = ‖z‖ := by
  cases o <;> simp [orderWitness, parameters, orderTransition, operator_apply,
    Prod.norm_def, max_comm]

/-- The same witness retains its nonzero state size at every sequence
length, CM97. This norm statement does not specify general target labels. -/
theorem order_flat_run (xs : List Bool) :
    ‖runFrom orderWitness orderWitness.initial xs‖ = 1 := by
  rw [flat_run orderWitness order_transition_isometry]
  simp [orderWitness, parameters, Prod.norm_def]

/-- The actual learned operators do not commute, CM97. A scalar
representation would erase this distinction. -/
theorem order_noncommuting : ∃ z : State,
    operator (orderTransition true) (operator (orderTransition false) z) ≠
      operator (orderTransition false) (operator (orderTransition true) z) := by
  refine ⟨(Complex.I, Complex.I), ?_⟩
  intro h
  have hi := congrArg (fun z : State => z.1.im) h
  norm_num [operator_apply, orderTransition] at hi

/-- Executing the two opposite orders yields different actual states,
CM97. The parameter assignment is identical for both input sequences. -/
theorem order_execution (b : Bool) :
    runFrom orderWitness orderWitness.initial [b, !b] =
      if b then (Complex.I, -Complex.I) else (-Complex.I, Complex.I) := by
  cases b <;> simp [orderWitness, parameters, runFrom, orderTransition, operator_apply]

/-- Complete class scores of the actual recurrence distinguish the
original first object on the swapped-prefix witness, CM97. -/
theorem order_scores (b c : Bool) :
    score orderWitness [b, !b] c = if c = b then 1 else -1 := by
  change (row (if c then (-Complex.I, 0) else (Complex.I, 0))
    (runFrom orderWitness orderWitness.initial [b, !b])).re = _
  rw [order_execution]
  cases b <;> cases c <;> norm_num [row_apply]

/-- Both opposite orders have strictly correct margin two under the
same learned parameters, CM97. -/
theorem order_margin (b : Bool) :
    score orderWitness [b, !b] b - score orderWitness [b, !b] (!b) = 2 := by
  rw [order_scores, order_scores]
  cases b <;> norm_num

/-- This order-sensitive score function belongs to the same bounded
convex family used for next-object CE, CM97, now with two-state execution. -/
theorem order_mem_domain : score orderWitness ∈ domain Bool State 1 := by
  simpa using generator_mem 1 orderWitness order_unit

/-- The actual output is the first original object in each of the two
swapped prefixes, CM97. No separate decoder is substituted. -/
theorem order_prediction (b fallback : Bool) :
    nextObject (score orderWitness) [b, !b] fallback = b := by
  by_contra hn
  have h := nextObject_max (score orderWitness) [b, !b] fallback b
  simp only [order_scores, ite_eq_right hn] at h
  norm_num at h

/-- The full learned operator domain can make the order distinction
which the entire scalar domain cannot, CM97. -/
theorem order_nonconstant :
    nextObject (score orderWitness) [false, true] true = false ∧
    nextObject (score orderWitness) [true, false] false = true := by
  exact ⟨order_prediction false true, order_prediction true false⟩

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Pair

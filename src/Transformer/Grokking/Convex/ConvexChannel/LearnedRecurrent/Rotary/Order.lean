module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Queries
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-!
# Rotary positions distinguish order even with commuting content operators

Source: `review/convex/moments.md`, CM98. An actual common parameter point
predicts different original objects on reversed prefixes. Removing its learned
frequency loses that distinction. The witness is not supplied to training.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary

/-- Content matrices for an existence witness only, CM98. These
matrices commute; position-dependent conjugation supplies the distinction. -/
def positionTransition (o : Bool) : Pair.Matrix :=
  if o then ((1, 0), (0, 1)) else ((0, 1), (1, 0))

/-- Fully complex content parameters of the witness, CM98. -/
noncomputable def positionContent : LearnedRecurrent.Parameters Bool Pair.State :=
  Pair.parameters (1, 0) positionTransition
    (fun c => if c then (0, Complex.I) else (0, 1))

/-- Both frequencies are model parameters; these displayed values only
inhabit a nonconstant expressivity example, CM98. -/
noncomputable def positionWitness : Parameters Bool :=
  ⟨(Real.pi / 2, 0), positionContent⟩

/-- Every actual witness matrix/readout and initial state obeys the
explicit row-sum capacity, CM98. -/
theorem position_unit : Unit positionWitness :=
  Pair.unit _ _ _ (by simp [Prod.norm_def])
    (by intro o; cases o <;> simp [positionTransition])
    (by intro c; cases c <;> simp)

/-- The unrotated content transitions commute on every state, CM98.
The following order result therefore uses rotary position interactions. -/
theorem position_content_commutes (a b : Bool) (z : Pair.State) :
    positionContent.transition a (positionContent.transition b z) =
      positionContent.transition b (positionContent.transition a z) := by
  cases a <;> cases b <;>
    simp [positionContent, Pair.parameters, positionTransition, Pair.operator_apply]

/-- Exact quarter-turn phase at the second sequence position, CM98. -/
theorem phase_quarter : phase (Real.pi / 2) 1 = Complex.I := by
  simp [phase]

/-- Its negative-position inverse is the opposite imaginary phase,
CM98. -/
theorem phase_negative_quarter : phase (Real.pi / 2) (-1) = -Complex.I := by
  simpa [phase, neg_div] using Complex.exp_neg_pi_div_two_mul_I

/-- Actual states for the two original object orders, CM98. -/
theorem position_execution (b : Bool) :
    runFrom positionWitness 0 positionWitness.content.initial [b, !b] =
      if b then (0, -Complex.I) else (0, 1) := by
  cases b <;> norm_num [runFrom, step, transport, positionWitness, positionContent,
    Pair.parameters, positionTransition, Pair.operator_apply, turn_apply, phase_zero,
    zero_frequency, phase_quarter, phase_negative_quarter]

/-- Complete actual scores distinguish the first original object with
the same frequency/content parameters, CM98. -/
theorem position_scores (b c : Bool) :
    score positionWitness [b, !b] c = if c = b then 1 else 0 := by
  change (Pair.row (if c then (0, Complex.I) else (0, 1))
    (runFrom positionWitness 0 positionWitness.content.initial [b, !b])).re = _
  rw [position_execution]
  cases b <;> cases c <;> norm_num [Pair.row_apply]

/-- Strict original-class margin one on both swapped prefixes, CM98. -/
theorem position_margin (b : Bool) :
    score positionWitness [b, !b] b - score positionWitness [b, !b] (!b) = 1 := by
  rw [position_scores, position_scores]
  cases b <;> norm_num

/-- The nonconstant rotary witness is in the bounded convex domain,
CM98. -/
theorem position_mem_domain : score positionWitness ∈ domain Bool 1 := by
  simpa using generator_mem 1 positionWitness position_unit

/-- Actual next-object decoding, without a substitute answer procedure,
CM98. -/
theorem position_prediction (b fallback : Bool) :
    nextObject (score positionWitness) [b, !b] fallback = b := by
  by_contra hn
  have h := nextObject_max (score positionWitness) [b, !b] fallback b
  simp only [position_scores, ite_eq_right hn] at h
  norm_num at h

/-- Both opposite orders yield different original next objects, CM98. -/
theorem position_nonconstant :
    nextObject (score positionWitness) [false, true] true = false ∧
    nextObject (score positionWitness) [true, false] false = true :=
  ⟨position_prediction false true, position_prediction true false⟩

/-- Setting the frequencies to zero erases this witness's order
distinction under the same content parameters, CM98. -/
theorem position_zero_equal (c : Bool) :
    score ⟨(0, 0), positionContent⟩ [false, true] c =
      score ⟨(0, 0), positionContent⟩ [true, false] c := by
  rw [score_zero, score_zero]
  simp [LearnedRecurrent.score, LearnedRecurrent.complex, LearnedRecurrent.runFrom,
    positionContent, Pair.parameters, positionTransition, Pair.operator_apply]

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary

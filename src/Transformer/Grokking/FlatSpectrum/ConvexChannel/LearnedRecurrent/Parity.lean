module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Scalar

/-!
# A nonconstant arbitrary-length parity witness in the learned family

Source: `review/flat_spectrum.md`, CM95. This is an existence witness, never
a supplied runtime feature or learning algorithm. It removes CM86's degree
obstruction: one bounded learned recurrence strictly represents parity at
every length, with nonzero imaginary state/readouts. Neither identification
from incomplete observations nor successful optimizer discovery is asserted.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Parity

/-- Desired labels for this expressivity witness only, CM95. -/
def parity : List Bool → Bool
  | [] => false
  | o :: xs => o.xor (parity xs)

/-- One possible learned transition parameter assignment, CM95. It is
not hard-coded in the generic learned recurrence or a native learner. -/
def phase (o : Bool) : ℂ := if o then -1 else 1

/-- Every witness transition has unit magnitude, CM95. -/
theorem phase_norm (o : Bool) : ‖phase o‖ = 1 := by
  cases o <;> simp [phase]

/-- The witness's complex multiplication realizes Boolean composition,
CM95. This identity is specific to the displayed parameter assignment. -/
theorem phase_xor (a b : Bool) : phase (a.xor b) = phase a * phase b := by
  cases a <;> cases b <;> norm_num [phase]

/-- The same parameters compose at every input length, CM95. -/
theorem phase_product (xs : List Bool) : (xs.map phase).prod = phase (parity xs) := by
  induction xs with
  | nil => simp [phase, parity]
  | cons o xs ih =>
    simp only [List.map_cons, List.prod_cons, parity]
    rw [ih, phase_xor]

/-- A concrete fully complex parameter point, CM95. The imaginary initial
state and readouts are essential to its actual real scores. -/
noncomputable def witness : Parameters Bool ℂ :=
  Scalar.parameters Complex.I phase (fun c => if c then Complex.I else -Complex.I)

/-- All learned parameter operators obey the declared capacities,
CM95, on a nonconstant witness. -/
theorem unit : Unit witness :=
  Scalar.unit _ _ _ (by simp) (fun o => (phase_norm o).le)
    (by intro c; cases c <;> simp)

/-- The witness's one-dimensional transition spectrum is flat, CM95. -/
theorem transition_norm (o : Bool) : ‖witness.transition o‖ = 1 := by
  simpa only [witness, Scalar.parameters, Scalar.multiply_norm] using phase_norm o

/-- Actual complete complex class scores are separated at every length,
CM95; this calculation uses the learned recurrence, not a separate decoder. -/
theorem complex_eq (xs : List Bool) (c : Bool) :
    complex witness xs c = if c = parity xs then 1 else -1 := by
  rw [witness, Scalar.complex_product, phase_product]
  cases c <;> cases parity xs <;> simp [phase]

/-- Real scores inherit the same strict separation for every length,
CM95. -/
theorem score_eq (xs : List Bool) (c : Bool) :
    score witness xs c = if c = parity xs then 1 else -1 := by
  rw [score, complex_eq]
  by_cases hc : c = parity xs <;> simp [hc]

/-- The correct original object has margin two on every sequence,
CM95. The old fixed-degree restriction therefore does not apply to this family. -/
theorem margin (xs : List Bool) :
    score witness xs (parity xs) - score witness xs (!(parity xs)) = 2 := by
  cases hp : parity xs <;> norm_num [score_eq, hp]

/-- The same bounded parameter point realizes all lengths, CM95. This
is expressivity, not a claim that the learner is given these parameters. -/
theorem realizable : ∃ f : Parameters Bool ℂ, Unit f ∧
    ∀ xs, 2 ≤ score f xs (parity xs) - score f xs (!(parity xs)) :=
  ⟨witness, unit, fun xs => (margin xs).ge⟩

/-- The complete witness is feasible at a length-independent budget one,
CM95. -/
theorem mem_domain : score witness ∈ domain Bool ℂ 1 := by
  simpa using generator_mem 1 witness unit

/-- Actual next-object prediction equals the desired parity object at
every length, CM95. Its fallback object only fixes nonemptiness. -/
theorem prediction (xs : List Bool) (fallback : Bool) :
    nextObject (score witness) xs fallback = parity xs := by
  by_contra hn
  have h := nextObject_max (score witness) xs fallback (parity xs)
  simp only [score_eq, ite_eq_right hn] at h
  norm_num at h

/-- Two different original sequences receive different original labels,
CM95, so the positive witness is nonconstant. -/
theorem nonconstant : nextObject (score witness) [true] false = true ∧
    nextObject (score witness) [false] true = false := by
  simp [prediction, parity]

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Parity

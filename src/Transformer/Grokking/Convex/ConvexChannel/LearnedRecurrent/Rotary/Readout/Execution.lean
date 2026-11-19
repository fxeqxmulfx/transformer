module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Basic

/-!
# Replacing every readout on the actual learned recurrent execution

Source: `review/convex/moments.md`, CM100. Holding already learned content fixed
defines a conditional optimization step, not a fixed-feature architecture. All
class readouts vary, and execution retains every learned frequency and transition.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout

open Pair
variable {O : Type*}

/-- Replace every complete class row while retaining learned content, CM100. -/
noncomputable def replace (f : Parameters O) (u : O → State) : Parameters O :=
  ⟨f.frequency, {f.content with readout := fun c => Pair.row (u c)}⟩

/-- Readout replacement preserves the complete ordered state execution, CM100. -/
theorem replace_run (f : Parameters O) (u : O → State) (position : ℕ)
    (z : State) (xs : List O) :
    runFrom (replace f u) position z xs = runFrom f position z xs := by
  induction xs generalizing position z with
  | nil => rfl
  | cons o xs ih =>
    change runFrom (replace f u) (position + 1) (step f position o z) xs =
      runFrom f (position + 1) (step f position o z) xs
    exact ih _ _

/-- The actual output uses both new complex coefficients, CM100. -/
theorem replace_complex (f : Parameters O) (u : O → State) (xs : List O) (c : O) :
    complex (replace f u) xs c = Pair.row (u c) (runFrom f 0 f.content.initial xs) := by
  change Pair.row (u c) (runFrom (replace f u) 0 f.content.initial xs) = _
  rw [replace_run]

/-- Readout coordinates reconstruct every existing class operator, CM100. -/
theorem replace_self (f : Parameters O) :
    replace f (fun c => (f.content.readout c (1, 0), f.content.readout c (0, 1))) = f := by
  have h : (fun c => Pair.row (f.content.readout c (1, 0), f.content.readout c (0, 1))) =
      f.content.readout := by
    funext c
    exact Pair.row_representation _
  unfold replace
  rw [h]

/-- Every bounded complete replacement remains an admissible learned model, CM100. -/
theorem replace_unit (f : Parameters O) (hf : Unit f) (u : O → State)
    (hu : ∀ c, rowSize (u c) ≤ 1) : Unit (replace f u) :=
  ⟨hf.1, hf.2.1, fun c => Pair.row_norm_bound (u c) (hu c)⟩

/-- Actual complete logits add in all readout coordinates, CM100. -/
theorem score_add (f : Parameters O) (u v : O → State) (xs : List O) (c : O) :
    score (replace f (u + v)) xs c = score (replace f u) xs c + score (replace f v) xs c := by
  simp [score, replace_complex, Pair.row_apply, add_mul]
  ring

/-- Actual complete logits respect real scaling of complex rows, CM100. -/
theorem score_smul (f : Parameters O) (u : O → State) (a : ℝ) (xs : List O) (c : O) :
    score (replace f (a • u)) xs c = a * score (replace f u) xs c := by
  simp only [score, replace_complex, Pair.row_apply]
  change ((a • (u c).1) * _ + (a • (u c).2) * _).re = _
  rw [smul_mul_assoc, smul_mul_assoc, ← smul_add, Complex.smul_re, smul_eq_mul]

/-- The concrete radial replacement preserves every model capacity, CM100. -/
theorem projected_unit (f : Parameters O) (hf : Unit f) (u : O → State) :
    Unit (replace f (fun c => projectRow (u c))) :=
  replace_unit f hf _ (fun c => projectRow_bound (u c))

example : Unit (replace positionWitness (fun c => if c then (Complex.I, 0)
    else (0, Complex.I))) :=
  replace_unit _ position_unit _ (by intro c; cases c <;> simp [rowSize])

example : Unit (replace positionWitness (fun c => projectRow
    (if c then (2 * Complex.I, 0) else (0, 2 * Complex.I)))) :=
  projected_unit _ position_unit _

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout

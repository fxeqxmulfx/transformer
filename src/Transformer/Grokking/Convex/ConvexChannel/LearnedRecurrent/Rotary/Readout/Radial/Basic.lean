module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Domain
public import Transformer.Grokking.FlatSpectrum.Training.Logits

/-!
# An actual nonconstant bounded recurrent CE witness

Source: `review/convex/moments.md`, CM101. Opposite original objects execute to
opposite complete states. The displayed parameters are proof witnesses only;
no task features or answer rule are supplied to benchmark training.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Radial

open Pair

/-- CM101: radial normalization discards every outward displacement on a ray. -/
theorem project_ray (u : State) (hu : rowSize u = 1) (a : ℝ) (ha : 1 ≤ a) :
    projectRow (a • u) = u := by
  have hpos : 0 < a := lt_of_lt_of_le (by norm_num) ha
  have hs : rowSize (a • u) = a := by
    rw [rowSize_smul, abs_of_pos hpos, hu, mul_one]
  by_cases he : a = 1
  · subst a
    simpa using projectRow_eq u hu.le
  · have hn : ¬a ≤ 1 := by intro h; exact he (le_antisymm h ha)
    rw [projectRow, hs, ite_eq_right hn, smul_smul, inv_mul_cancel₀ (ne_of_gt hpos),
      one_smul]

/-- CM101: both complete complex coefficients of the stalled boundary row. -/
noncomputable def boundaryRow : State := ((2 / 3 : ℂ), (1 / 3 : ℂ))

/-- CM101: both original output classes have learned rows. -/
noncomputable def stalledRows (c : Bool) : State :=
  if c then boundaryRow else -boundaryRow

/-- CM101: strictly better rows in the same complete row domain. -/
def betterRows (c : Bool) : State := if c then (1, 0) else (-1, 0)

/-- CM101: complete content matrices distinguish the two object prefixes. -/
def transitions (b : Bool) : Matrix :=
  if b then ((1, 0), (0, 1)) else ((-1, 0), (0, -1))

/-- CM101: actual learned recurrent parameters, including both class readouts. -/
noncomputable def witness : Parameters Bool :=
  ⟨(0, 0), Pair.parameters (1, (1 / 2 : ℂ)) transitions stalledRows⟩

/-- CM101: the two complete complex hidden states. -/
noncomputable def hidden (b : Bool) : State :=
  if b then (1, (1 / 2 : ℂ)) else (-1, (-1 / 2 : ℂ))

/-- CM101: every stalled class row saturates the original capacity. -/
theorem stalled_size (c : Bool) : rowSize (stalledRows c) = 1 := by
  cases c <;> norm_num [stalledRows, boundaryRow, rowSize, norm_div]

/-- CM101: the better class rows obey exactly the same budget. -/
theorem better_size (c : Bool) : rowSize (betterRows c) = 1 := by
  cases c <;> norm_num [betterRows, rowSize]

/-- CM101: all content, initial and readout bounds hold for the actual model. -/
theorem witness_unit : Unit witness :=
  Pair.unit _ _ _ (by norm_num [Prod.norm_def, norm_div])
    (by intro b; cases b <;> norm_num [transitions])
    (fun c => (stalled_size c).le)

/-- CM101: better readouts preserve the actual learned content and capacity. -/
theorem better_unit : Unit (replace witness betterRows) :=
  replace_unit witness witness_unit betterRows (fun c => (better_size c).le)

/-- CM101: actual ordered execution, not an assigned feature value. -/
theorem execution (b : Bool) :
    runFrom witness 0 witness.content.initial [b] = hidden b := by
  cases b <;> norm_num [runFrom, step, transport, witness, Pair.parameters,
    transitions, Pair.operator_apply, turn_apply, zero_frequency, hidden]

/-- CM101: arbitrary complete replacement rows execute the same actual states. -/
theorem actual_score (u : Bool → State) (b c : Bool) :
    score (replace witness u) [b] c = (Pair.row (u c) (hidden b)).re := by
  rw [score, replace_complex, execution]

/-- CM101: full original-class logits of the stalled rows. -/
theorem stalled_scores (b c : Bool) :
    score (replace witness stalledRows) [b] c = if c = b then 5 / 6 else -5 / 6 := by
  rw [actual_score]
  cases b <;> cases c <;>
    norm_num [Pair.row_apply, hidden, stalledRows, boundaryRow]

/-- CM101: full original-class logits of the better feasible readouts. -/
theorem better_scores (b c : Bool) :
    score (replace witness betterRows) [b] c = if c = b then 1 else -1 := by
  rw [actual_score]
  cases b <;> cases c <;> norm_num [Pair.row_apply, hidden, betterRows]

example : projectRow ((3 : ℝ) • (Complex.I, 0)) = (Complex.I, 0) :=
  project_ray _ (by simp [rowSize]) _ (by norm_num)

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Radial

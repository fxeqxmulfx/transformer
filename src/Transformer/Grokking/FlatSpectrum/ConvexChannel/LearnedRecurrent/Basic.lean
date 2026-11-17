module

public import Mathlib.Analysis.Normed.Operator.Basic
public import Mathlib.Analysis.Complex.Basic

/-!
# Learned complex recurrence on opaque object sequences

Source: `review/flat_spectrum.md`, CM95. Each original object indexes a learned
complex linear transition. Initial state and every original-object readout are
learned too. The transitions execute in sequence order at arbitrary length.
The norm bound is a capacity restriction, not an answer program or a claim of
joint convexity in independently varied transition/readout coordinates.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent

variable (O E : Type*) [NormedAddCommGroup E] [NormedSpace ℂ E]

/-- Learned numerical parameters; `E` may be a finite complex state space,
CM95. The input itself remains a list of original immutable objects. -/
structure Parameters where
  initial : E
  transition : O → E →L[ℂ] E
  readout : O → E →L[ℂ] ℂ

variable {O E}

/-- Execute each learned transition in input order, CM95. -/
def runFrom (f : Parameters O E) (v : E) : List O → E
  | [] => v
  | o :: xs => runFrom f (f.transition o v) xs

/-- Actual complete complex next-object scores, CM95. -/
def complex (f : Parameters O E) (xs : List O) (c : O) : ℂ :=
  f.readout c (runFrom f f.initial xs)

/-- Ordinary real class scores read from the learned complex execution,
CM95. Both real and imaginary numerical parameters contribute. -/
def score (f : Parameters O E) (xs : List O) (c : O) : ℝ := (complex f xs c).re

/-- Uniform operator capacity; no labels or task semantics occur, CM95. -/
def Unit (f : Parameters O E) : Prop :=
  ‖f.initial‖ ≤ 1 ∧ (∀ o, ‖f.transition o‖ ≤ 1) ∧ ∀ c, ‖f.readout c‖ ≤ 1

/-- Zero execution can hold unused convex mixture mass, CM95. -/
noncomputable def zeroParameters : Parameters O E := ⟨0, fun _ => 0, fun _ => 0⟩

/-- The complete zero component is within capacity, CM95. -/
theorem zero_unit : Unit (zeroParameters : Parameters O E) := by
  simp [Unit, zeroParameters]

/-- Streaming composition preserves the actual sequence order, CM95. -/
theorem runFrom_append (f : Parameters O E) (v : E) (xs ys : List O) :
    runFrom f v (xs ++ ys) = runFrom f (runFrom f v xs) ys := by
  induction xs generalizing v with
  | nil => rfl
  | cons o xs ih => exact ih (f.transition o v)

/-- A product of learned contractive operators does not amplify the
initial state, at any sequence length, CM95. -/
theorem runFrom_norm_le (f : Parameters O E) (h : ∀ o, ‖f.transition o‖ ≤ 1)
    (v : E) (xs : List O) : ‖runFrom f v xs‖ ≤ ‖v‖ := by
  induction xs generalizing v with
  | nil => exact le_rfl
  | cons o xs ih =>
    calc
      ‖runFrom f v (o :: xs)‖ ≤ ‖f.transition o v‖ := ih _
      _ ≤ ‖f.transition o‖ * ‖v‖ := (f.transition o).le_opNorm v
      _ ≤ 1 * ‖v‖ := mul_le_mul_of_nonneg_right (h o) (norm_nonneg v)
      _ = ‖v‖ := one_mul _

/-- Isometric (flat) learned transitions preserve state size exactly at
every depth, CM95. Flatness bounds execution; it does not specify labels. -/
theorem flat_run (f : Parameters O E)
    (h : ∀ o v, ‖f.transition o v‖ = ‖v‖) (v : E) (xs : List O) :
    ‖runFrom f v xs‖ = ‖v‖ := by
  induction xs generalizing v with
  | nil => rfl
  | cons o xs ih => exact (ih _).trans (h o v)

/-- The actual complex output has a depth-independent bound, CM95. -/
theorem complex_bound (f : Parameters O E) (hf : Unit f) (xs : List O) (c : O) :
    ‖complex f xs c‖ ≤ 1 := by
  calc
    ‖complex f xs c‖ ≤ ‖f.readout c‖ * ‖runFrom f f.initial xs‖ :=
      (f.readout c).le_opNorm _
    _ ≤ 1 * ‖runFrom f f.initial xs‖ :=
      mul_le_mul_of_nonneg_right (hf.2.2 c) (norm_nonneg _)
    _ ≤ 1 := by simpa using (runFrom_norm_le f hf.2.1 f.initial xs).trans hf.1

/-- Every original-class real score obeys the same capacity bound, CM95. -/
theorem score_bound (f : Parameters O E) (hf : Unit f) (xs : List O) (c : O) :
    |score f xs c| ≤ 1 :=
  (Complex.abs_re_le_norm _).trans (complex_bound f hf xs c)

/-- All-zero initial and transition parameters execute zero, CM95. -/
theorem zero_run (xs : List O) : runFrom (zeroParameters : Parameters O E) 0 xs = 0 := by
  induction xs with
  | nil => rfl
  | cons o xs ih => simpa [runFrom, zeroParameters] using ih

/-- The zero component has exactly zero score for every original object,
CM95, regardless of input length. -/
theorem zero_score (xs : List O) (c : O) : score (zeroParameters : Parameters O E) xs c = 0 := by
  simp [score, complex, zeroParameters]

example : ‖runFrom (zeroParameters : Parameters Bool ℂ) 1 [true, false]‖ ≤ ‖(1 : ℂ)‖ :=
  runFrom_norm_le _ (by simp [zeroParameters]) _ _

example : ‖complex (zeroParameters : Parameters Bool ℂ) [true, false] true‖ ≤ 1 :=
  complex_bound _ zero_unit _ _

example : |score (zeroParameters : Parameters Bool ℂ) [true, false] true| ≤ 1 :=
  score_bound _ zero_unit _ _

example (xs : List Bool) :
    ‖runFrom (⟨Complex.I, fun _ => ContinuousLinearMap.id ℂ ℂ,
      fun _ => ContinuousLinearMap.id ℂ ℂ⟩ : Parameters Bool ℂ) Complex.I xs‖ = ‖Complex.I‖ :=
  flat_run _ (by intro o v; simp) _ _

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent

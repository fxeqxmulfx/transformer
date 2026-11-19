module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Basic

/-!
# Actual position-dependent recurrent execution on original objects

Source: `review/convex/moments.md`, CM98. Positions are internal counters, not
extra input objects. All frequencies, content matrices, initial coordinates
and readouts remain numerical parameters. Bounds follow from actual rotary
transport at each occurrence and are independent of sequence length.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary

open Pair

/-- Full learned content parameters and two shared trainable frequencies,
CM98. There is no assertion or unproved claim hidden in this structure. -/
structure Parameters (O : Type*) where
  frequency : ℝ × ℝ
  content : LearnedRecurrent.Parameters O State

variable {O : Type*}

/-- The actual operator at the current internal position, CM98. -/
noncomputable def step (f : Parameters O) (position : ℕ) (o : O) : State →L[ℂ] State :=
  transport f.frequency position (f.content.transition o)

/-- Complete ordered execution, incrementing position for each original
object occurrence, CM98. Arbitrary lengths use the same learned parameters. -/
noncomputable def runFrom (f : Parameters O) : ℕ → State → List O → State
  | _, v, [] => v
  | position, v, o :: xs => runFrom f (position + 1) (step f position o v) xs

/-- Actual complete complex class score, CM98. -/
noncomputable def complex (f : Parameters O) (xs : List O) (c : O) : ℂ :=
  f.content.readout c (runFrom f 0 f.content.initial xs)

/-- Ordinary real logit for every original class, CM98. -/
noncomputable def score (f : Parameters O) (xs : List O) (c : O) : ℝ := (complex f xs c).re

/-- Explicit content-operator capacity. Frequencies need no norm cutoff
because their actual phases are unit magnitude, CM98. -/
def Unit (f : Parameters O) : Prop := LearnedRecurrent.Unit f.content

/-- Actual position-dependent transition capacity is exactly the learned
content transition capacity, CM98. -/
theorem step_norm (f : Parameters O) (position : ℕ) (o : O) :
    ‖step f position o‖ = ‖f.content.transition o‖ := transport_norm _ _ _

/-- Every actual ordered execution remains bounded at any starting
position, CM98; no observed-label premise is used. -/
theorem run_norm_le (f : Parameters O) (h : ∀ o, ‖f.content.transition o‖ ≤ 1)
    (position : ℕ) (v : State) (xs : List O) : ‖runFrom f position v xs‖ ≤ ‖v‖ := by
  induction xs generalizing position v with
  | nil => exact le_rfl
  | cons o xs ih =>
    calc
      ‖runFrom f position v (o :: xs)‖ ≤ ‖step f position o v‖ := ih _ _
      _ ≤ ‖step f position o‖ * ‖v‖ := (step f position o).le_opNorm v
      _ ≤ 1 * ‖v‖ := mul_le_mul_of_nonneg_right (by rw [step_norm]; exact h o) (norm_nonneg _)
      _ = ‖v‖ := one_mul _

/-- Original-class complex scores obey the same length-independent
bound under learned rotary positions, CM98. -/
theorem complex_bound (f : Parameters O) (hf : Unit f) (xs : List O) (c : O) :
    ‖complex f xs c‖ ≤ 1 := by
  calc
    ‖complex f xs c‖ ≤ ‖f.content.readout c‖ * ‖runFrom f 0 f.content.initial xs‖ :=
      (f.content.readout c).le_opNorm _
    _ ≤ 1 * ‖runFrom f 0 f.content.initial xs‖ :=
      mul_le_mul_of_nonneg_right (hf.2.2 c) (norm_nonneg _)
    _ ≤ 1 := by simpa using (run_norm_le f hf.2.1 0 _ xs).trans hf.1

/-- The complete actual real logits are bounded, CM98. -/
theorem score_bound (f : Parameters O) (hf : Unit f) (xs : List O) (c : O) :
    |score f xs c| ≤ 1 :=
  (Complex.abs_re_le_norm _).trans (complex_bound f hf xs c)

/-- Zero-frequency transport recovers every unchanged content operator,
CM98. It is a permitted parameter value, not a fixed feature. -/
theorem step_zero (f : LearnedRecurrent.Parameters O State) (position : ℕ) (o : O) (v : State) :
    step ⟨(0, 0), f⟩ position o v = f.transition o v := by
  simp [step, transport, turn_zero_frequency]

/-- Zero frequencies preserve the entire nonrotary execution, CM98. -/
theorem run_zero (f : LearnedRecurrent.Parameters O State) (position : ℕ)
    (v : State) (xs : List O) :
    runFrom ⟨(0, 0), f⟩ position v xs = LearnedRecurrent.runFrom f v xs := by
  induction xs generalizing position v with
  | nil => rfl
  | cons o xs ih =>
    change runFrom ⟨(0, 0), f⟩ (position + 1) (step ⟨(0, 0), f⟩ position o v) xs = _
    rw [step_zero, ih]
    rfl

/-- Zero-frequency scores are exactly the parent family's scores,
CM98, including the actual constant class-bias components. -/
theorem score_zero (f : LearnedRecurrent.Parameters O State) (xs : List O) (c : O) :
    score ⟨(0, 0), f⟩ xs c = LearnedRecurrent.score f xs c := by
  unfold score complex
  rw [run_zero]
  rfl

/-- A genuine zero component carries unused convex mixture mass, CM98. -/
noncomputable def zeroParameters : Parameters O := ⟨(0, 0), LearnedRecurrent.zeroParameters⟩

/-- The zero component inhabits the explicit capacity constraints, CM98. -/
theorem zero_unit : Unit (zeroParameters : Parameters O) := LearnedRecurrent.zero_unit

/-- The actual rotary zero component has zero class scores, CM98. -/
theorem zero_score (xs : List O) (c : O) : score (zeroParameters : Parameters O) xs c = 0 := by
  rw [zeroParameters, score_zero, LearnedRecurrent.zero_score]

/-- Exact agreement of every responding transition gives exact state
agreement, CM98. No small-response threshold is allowed. -/
theorem run_congr (f g : Parameters O) (xs : List O)
    (h : ∀ o ∈ xs, ∀ position v, step f position o v = step g position o v)
    (position : ℕ) (v : State) : runFrom f position v xs = runFrom g position v xs := by
  induction xs generalizing position v with
  | nil => rfl
  | cons o xs ih =>
    change runFrom f (position + 1) (step f position o v) xs =
      runFrom g (position + 1) (step g position o v) xs
    rw [h o (by simp) position v]
    exact ih (fun x hx => h x (by simp [hx])) _ _

/-- Replacing an absent object's full matrix changes no executed state,
CM98. Every matrix entry remains stored and available to future batches. -/
theorem absent_run [DecidableEq O] (frequency : ℝ × ℝ) (a : State)
    (t : O → Pair.Matrix) (u : O → State) (o : O) (matrix : Pair.Matrix)
    (xs : List O) (ho : o ∉ xs) (position : ℕ) (v : State) :
    runFrom ⟨frequency, Pair.parameters a (Function.update t o matrix) u⟩ position v xs =
      runFrom ⟨frequency, Pair.parameters a t u⟩ position v xs := by
  apply run_congr
  intro x hx n z
  have hne : x ≠ o := by intro he; exact ho (he ▸ hx)
  simp [step, Pair.parameters, Function.update_apply, hne]

/-- The exact absent-object identity reaches every actual class score,
CM98. It justifies zero-response skips, never magnitude or class filtering. -/
theorem absent_score [DecidableEq O] (frequency : ℝ × ℝ) (a : State)
    (t : O → Pair.Matrix) (u : O → State) (o : O) (matrix : Pair.Matrix)
    (xs : List O) (ho : o ∉ xs) (c : O) :
    score ⟨frequency, Pair.parameters a (Function.update t o matrix) u⟩ xs c =
      score ⟨frequency, Pair.parameters a t u⟩ xs c := by
  unfold score complex
  change (Pair.row (u c) (runFrom _ 0 a xs)).re = (Pair.row (u c) (runFrom _ 0 a xs)).re
  rw [absent_run frequency a t u o matrix xs ho]

example : Unit (⟨(Real.pi, 1), orderWitness⟩ : Parameters Bool) := order_unit

example : ‖runFrom (⟨(Real.pi, 1), orderWitness⟩ : Parameters Bool) 3
    (Complex.I, Complex.I) [false, true]‖ ≤ ‖(Complex.I, Complex.I)‖ :=
  run_norm_le (⟨(Real.pi, 1), orderWitness⟩ : Parameters Bool) order_unit.2.1 3 _ _

example : ‖complex (⟨(Real.pi, 1), orderWitness⟩ : Parameters Bool) [false, true] true‖ ≤ 1 :=
  complex_bound (⟨(Real.pi, 1), orderWitness⟩ : Parameters Bool) order_unit _ _

example : |score (⟨(Real.pi, 1), orderWitness⟩ : Parameters Bool) [false, true] true| ≤ 1 :=
  score_bound (⟨(Real.pi, 1), orderWitness⟩ : Parameters Bool) order_unit _ _

example : runFrom (⟨(Real.pi, 1), orderWitness⟩ : Parameters Bool) 0
    (Complex.I, Complex.I) [false, true] =
      runFrom (⟨(Real.pi, 1), orderWitness⟩ : Parameters Bool) 0
        (Complex.I, Complex.I) [false, true] := run_congr _ _ _ (by intros; rfl) _ _

example : runFrom ⟨(1, 2), Pair.parameters (1, 0)
    (Function.update Pair.orderTransition true ((0, Complex.I), (Complex.I, 0)))
    (fun _ => (Complex.I, 0))⟩ 0 (1, 0) [false] =
      runFrom ⟨(1, 2), Pair.parameters (1, 0) Pair.orderTransition
        (fun _ => (Complex.I, 0))⟩ 0 (1, 0) [false] :=
  absent_run _ _ _ _ true _ _ (by simp) _ _

example : score ⟨(1, 2), Pair.parameters (1, 0)
    (Function.update Pair.orderTransition true ((0, Complex.I), (Complex.I, 0)))
    (fun _ => (Complex.I, 0))⟩ [false] true =
      score ⟨(1, 2), Pair.parameters (1, 0) Pair.orderTransition
        (fun _ => (Complex.I, 0))⟩ [false] true :=
  absent_score _ _ _ _ true _ _ (by simp) _

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary

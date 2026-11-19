module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Order

/-!
# Actual ordered state error after learned content changes

Source: `review/convex/moments.md`, CM106. Every transported operator contains
the current learned transition and frequencies. Bounds retain all occurrences
and positions, and do not use a label or a prescribed state feature.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained

open Pair
variable {O : Type*}

/-- Sum of actual operator changes along every ordered occurrence, CM106. -/
noncomputable def operatorDrift (f g : Parameters O) : ℕ → List O → ℝ
  | _, [] => 0
  | p, o :: xs => ‖step f p o - step g p o‖ + operatorDrift f g (p + 1) xs

/-- Complete initial-state and transported-operator displacement, CM106. -/
noncomputable def stateDrift (f g : Parameters O) (xs : List O) : ℝ :=
  ‖f.content.initial - g.content.initial‖ + operatorDrift f g 0 xs

/-- Every occurrence contributes a nonnegative operator error, CM106. -/
theorem operator_drift_nonneg (f g : Parameters O) (p : ℕ) (xs : List O) :
    0 ≤ operatorDrift f g p xs := by
  induction xs generalizing p with
  | nil => exact le_rfl
  | cons o xs ih => exact add_nonneg (norm_nonneg _) (ih _)

/-- Changing no operator gives zero transport error at every position, CM106. -/
theorem operator_drift_self (f : Parameters O) (p : ℕ) (xs : List O) :
    operatorDrift f f p xs = 0 := by
  induction xs generalizing p with
  | nil => rfl
  | cons o xs ih => simp [operatorDrift, ih]

/-- Identical full learned content has zero displacement budget, CM106. -/
theorem state_drift_self (f : Parameters O) (xs : List O) : stateDrift f f xs = 0 := by
  rw [stateDrift, sub_self, norm_zero, operator_drift_self, add_zero]

/-- Contractive execution propagates state error plus each actual operator
change, including changed learned frequencies, CM106. -/
theorem run_change_le (f g : Parameters O)
    (hf : ∀ o, ‖f.content.transition o‖ ≤ 1)
    (hg : ∀ o, ‖g.content.transition o‖ ≤ 1)
    (p : ℕ) (v w : State) (xs : List O) (hw : ‖w‖ ≤ 1) :
    ‖runFrom f p v xs - runFrom g p w xs‖ ≤ ‖v - w‖ + operatorDrift f g p xs := by
  induction xs generalizing p v w with
  | nil => simp only [runFrom, operatorDrift, add_zero, le_refl]
  | cons o xs ih =>
    have hfn : ‖step f p o‖ ≤ 1 := by rw [step_norm]; exact hf o
    have hgn : ‖step g p o‖ ≤ 1 := by rw [step_norm]; exact hg o
    have hw' : ‖step g p o w‖ ≤ 1 := calc
      _ ≤ ‖step g p o‖ * ‖w‖ := (step g p o).le_opNorm w
      _ ≤ 1 * ‖w‖ := mul_le_mul_of_nonneg_right hgn (norm_nonneg _)
      _ ≤ 1 := by simpa using hw
    have he : step f p o v - step g p o w =
        step f p o (v - w) + (step f p o - step g p o) w := by
      simp only [map_sub, sub_apply]
      abel
    have hs : ‖step f p o v - step g p o w‖ ≤
        ‖v - w‖ + ‖step f p o - step g p o‖ := calc
      _ ≤ ‖step f p o (v - w)‖ + ‖(step f p o - step g p o) w‖ := by
        rw [he]; exact norm_add_le _ _
      _ ≤ ‖step f p o‖ * ‖v - w‖ + ‖step f p o - step g p o‖ * ‖w‖ :=
        add_le_add ((step f p o).le_opNorm _) ((step f p o - step g p o).le_opNorm _)
      _ ≤ 1 * ‖v - w‖ + ‖step f p o - step g p o‖ * 1 :=
        add_le_add (mul_le_mul_of_nonneg_right hfn (norm_nonneg _))
          (mul_le_mul_of_nonneg_left hw (norm_nonneg _))
      _ = _ := by ring
    have ht := ih (p + 1) (step f p o v) (step g p o w) hw'
    change ‖runFrom f (p + 1) (step f p o v) xs -
      runFrom g (p + 1) (step g p o w) xs‖ ≤ _
    change _ ≤ ‖v - w‖ + (‖step f p o - step g p o‖ + operatorDrift f g (p + 1) xs)
    linarith

/-- The state error bound starts from both actual learned initial states,
CM106; no fixed encoding is substituted. -/
theorem state_change_le (f g : Parameters O) (hf : Unit f) (hg : Unit g) (xs : List O) :
    ‖runFrom f 0 f.content.initial xs - runFrom g 0 g.content.initial xs‖ ≤
      stateDrift f g xs :=
  run_change_le f g hf.2.1 hg.2.1 0 _ _ xs hg.1

example : ‖runFrom positionWitness 0 (Complex.I, Complex.I) [true, false] -
    runFrom (⟨(0, 0), positionContent⟩ : Parameters Bool) 0 (1, 0) [true, false]‖ ≤
      ‖((Complex.I, Complex.I) : State) - (1, 0)‖ +
        operatorDrift positionWitness ⟨(0, 0), positionContent⟩ 0 [true, false] :=
  run_change_le positionWitness ⟨(0, 0), positionContent⟩ position_unit.2.1
    position_unit.2.1 0 (Complex.I, Complex.I) (1, 0) [true, false]
    (by simp [Prod.norm_def])

example : ‖runFrom positionWitness 0 positionWitness.content.initial [true, false] -
    runFrom (⟨(0, 0), positionContent⟩ : Parameters Bool) 0 positionContent.initial
      [true, false]‖ ≤ stateDrift positionWitness ⟨(0, 0), positionContent⟩ [true, false] :=
  state_change_le positionWitness ⟨(0, 0), positionContent⟩ position_unit position_unit
    [true, false]

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained

module

public import Mathlib.Data.ZMod.Basic

/-!
# Testing row and column class shifts in held-out predictions

Source: the class translations in CM121 of `review/convex/moments.md` and the
NV11 prediction-shift diagnostic in `experiments/phasor_value_transport/README.md`.
For an additive class group, the difference between a prediction and its
target is a class residual. A row plus a column shift cancels around every
rectangle, including a rectangle contained entirely in the held-out set.
Counting equal column differences for each row pair tests this property.
-/

@[expose] public section

namespace Transformer.Phasor

variable {I J C : Type*} [AddCommGroup C]

/-- A residual table compares unmodified checkpoint predictions with the
targets in the outer evaluator, in their common additive class group. -/
def predictionResidual (prediction target : I → J → C) (i : I) (j : J) : C :=
  prediction i j - target i j

/-- Alternating sum of the four class residuals, in row-major order. -/
def rectangleResidual (e : I → J → C) (i k : I) (j l : J) : C :=
  (e i j - e k j) - (e i l - e k l)

/-- For two fixed rows, a rectangle passes precisely when its two columns
fall in the same class-difference bucket. This gives the native counter. -/
theorem rectangleResidual_zero_iff (e : I → J → C) (i k : I) (j l : J) :
    rectangleResidual e i k j l = 0 ↔ e i j - e k j = e i l - e k l :=
  sub_eq_zero

/-- Any CM121 translation of the true classes passes every rectangle,
even when its predicted classes individually differ from the targets. -/
theorem predictionResidual_shift_rectangle (target : I → J → C) (u : I → C) (v : J → C)
    (i k : I) (j l : J) :
    rectangleResidual (predictionResidual (fun a b => target a b + u a + v b) target) i k j l = 0 := by
  have cell (a : I) (b : J) :
      predictionResidual (fun x y => target x y + u x + v y) target a b = u a + v b := by
    change (target a b + u a + v b) - target a b = _
    rw [add_assoc, add_sub_cancel_left]
  simp only [rectangleResidual, cell, add_sub_add_right_eq_sub, sub_self]

/-- On a complete table, all rectangle residuals vanish exactly when the
predictions differ from the targets by a row shift and a column shift.
An anchor row and column suffice to construct the two shifts. -/
theorem all_prediction_rectangles_iff_shift (prediction target : I → J → C) (i₀ : I) (j₀ : J) :
    (∀ i k j l, rectangleResidual (predictionResidual prediction target) i k j l = 0) ↔
    ∃ (u : I → C) (v : J → C), ∀ i j, prediction i j = target i j + u i + v j := by
  let e := predictionResidual prediction target
  constructor
  · intro h
    refine ⟨fun i => e i j₀ - e i₀ j₀, fun j => e i₀ j, ?_⟩
    intro i j
    have row := (rectangleResidual_zero_iff e i i₀ j j₀).mp (h i i₀ j j₀)
    have splits := sub_eq_iff_eq_add.mp row
    calc prediction i j = target i j + e i j := by
          rw [add_comm]; exact (sub_add_cancel (prediction i j) (target i j)).symm
      _ = target i j + ((e i j₀ - e i₀ j₀) + e i₀ j) := by rw [splits]
      _ = _ := (add_assoc ..).symm
  · rintro ⟨u,v,h⟩ i k j l
    have eq : prediction = fun a b => target a b + u a + v b := funext fun a => funext (h a)
    rw [eq]
    exact predictionResidual_shift_rectangle target u v i k j l

/-- NV11 and multiplication differ by a row translation. Their top-one
class tables therefore agree on every residual rectangle over `ZMod 97`. -/
theorem nv11_mul_prediction_rectangles (a a' b b' : ZMod 97) :
    rectangleResidual (predictionResidual (fun i j : ZMod 97 => i^3+i*j) (fun i j => i*j))
      a a' b b' = 0 := by
  simp only [rectangleResidual, predictionResidual, add_sub_cancel_right, sub_self]

end Transformer.Phasor

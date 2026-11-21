module

public import Mathlib.Algebra.Field.ZMod
public import Transformer.GainMatroid.Section4_Small

/-!
# Gain functions satisfying the hypotheses of Lemmas 4.3 to 4.10

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §4.2. Over the field `ZMod 3`, these gain functions
give `2C₃` and the tube no balanced 2-cycle and `K₄` no balanced triangle, so the hypotheses of
Lemmas 4.3 to 4.10 and of the row theorems behind them can be met.
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

/-- A gain function on `2C₃` with no balanced 2-cycle. -/
def twoC3Gain : Fin 6 → (ZMod 3)ˣ := ![1, -1, 1, -1, 1, -1]

/-- An additive gain function on `2C₃` with no balanced 2-cycle. -/
def twoC3AddGain : Fin 6 → ZMod 3 := ![0, 1, 0, 1, 0, 1]

/-- A gain function on `K₄` with no balanced triangle. -/
def K4Gain : Fin 6 → (ZMod 3)ˣ := ![1, 1, 1, -1, -1, -1]

/-- An additive gain function on `K₄` with no balanced triangle. -/
def K4AddGain : Fin 6 → ZMod 3 := ![0, 0, 0, 1, 1, 1]

/-- A gain function on the tube with no balanced 2-cycle. -/
def tubeGain : Fin 6 → (ZMod 3)ˣ := ![1, -1, 1, 1, 1, -1]

/-- An additive gain function on the tube with no balanced 2-cycle. -/
def tubeAddGain : Fin 6 → ZMod 3 := ![0, 1, 0, 0, 0, 1]

theorem twoC3Gain_noPair : ∀ C ∈ twoC3.balancedCycles twoC3Gain, C.encard ≠ 2 :=
  encard_ne_two_of_parallel (by
    simp only [Joins, dart, twoC3, twoC3Gain]
    decide)

theorem twoC3AddGain_noPair : ∀ C ∈ twoC3.addBalancedCycles twoC3AddGain, C.encard ≠ 2 :=
  encard_ne_two_of_addParallel (by
    simp only [Joins, addDart, twoC3, twoC3AddGain]
    decide)

theorem tubeGain_noPair : ∀ C ∈ tube.balancedCycles tubeGain, C.encard ≠ 2 :=
  encard_ne_two_of_parallel (by
    simp only [Joins, dart, tube, tubeGain]
    decide)

theorem tubeAddGain_noPair : ∀ C ∈ tube.addBalancedCycles tubeAddGain, C.encard ≠ 2 :=
  encard_ne_two_of_addParallel (by
    simp only [Joins, addDart, tube, tubeAddGain]
    decide)

set_option maxRecDepth 100000 in
theorem K4Gain_noTriangle : ∀ C ∈ K4.balancedCycles K4Gain, C.encard ≠ 3 :=
  encard_ne_three_of_triangle (by
    simp only [Joins, dart, K4, K4Gain]
    decide)

set_option maxRecDepth 100000 in
theorem K4AddGain_noTriangle : ∀ C ∈ K4.addBalancedCycles K4AddGain, C.encard ≠ 3 :=
  encard_ne_three_of_addTriangle (by
    simp only [Joins, addDart, K4, K4AddGain]
    decide)

private theorem twoC3_refl : twoC3.IsReorientationOf twoC3 := IsReorientationOf.refl _

private theorem K4_refl : K4.IsReorientationOf K4 := IsReorientationOf.refl _

private theorem tube_refl : tube.IsReorientationOf tube := IsReorientationOf.refl _

private theorem twoC3_connectedOff : twoC3.ConnectedOff := connectedOff_of_star twoC3_star

example (ψ : Fin 6 → (ZMod 3)ˣ) : SwitchEquiv twoC3 twoC3Gain ψ ↔
    ProjEquiv (frameMatrix twoC3 twoC3Gain) (frameMatrix twoC3 ψ) :=
  twoC3_switchEquiv_iff twoC3_refl twoC3Gain_noPair

example (ψ : Fin 6 → ZMod 3) : SwitchScaleEquiv twoC3 twoC3AddGain ψ ↔
    ProjEquiv (liftMatrix twoC3 twoC3AddGain) (liftMatrix twoC3 ψ) :=
  twoC3_switchScaleEquiv_iff twoC3_refl twoC3AddGain_noPair

example (ψ : Fin 6 → ZMod 3) : ¬ProjEquiv (frameMatrix twoC3 twoC3Gain) (liftMatrix twoC3 ψ) :=
  twoC3_not_projEquiv twoC3_refl twoC3Gain_noPair ψ

example (ψ : Fin 6 → (ZMod 3)ˣ) : SwitchEquiv K4 K4Gain ψ ↔
    ProjEquiv (frameMatrix K4 K4Gain) (frameMatrix K4 ψ) :=
  K4_switchEquiv_iff K4_refl K4Gain_noTriangle

example (ψ : Fin 6 → ZMod 3) : SwitchScaleEquiv K4 K4AddGain ψ ↔
    ProjEquiv (liftMatrix K4 K4AddGain) (liftMatrix K4 ψ) :=
  K4_switchScaleEquiv_iff K4_refl K4AddGain_noTriangle

example (ψ : Fin 6 → ZMod 3) : ¬ProjEquiv (frameMatrix K4 K4Gain) (liftMatrix K4 ψ) :=
  K4_not_projEquiv K4_refl K4Gain_noTriangle ψ

example (ψ : Fin 6 → (ZMod 3)ˣ) : SwitchEquiv tube tubeGain ψ ↔
    ProjEquiv (frameMatrix tube tubeGain) (frameMatrix tube ψ) :=
  tube_switchEquiv_iff tube_refl tubeGain_noPair

example (ψ : Fin 6 → ZMod 3) : SwitchScaleEquiv tube tubeAddGain ψ ↔
    ProjEquiv (liftMatrix tube tubeAddGain) (liftMatrix tube ψ) :=
  tube_switchScaleEquiv_iff tube_refl tubeAddGain_noPair

example : SwitchEquiv twoC3 twoC3Gain twoC3Gain :=
  switchEquiv_of_projEquiv twoC3_loopless twoC3_connectedOff
    (hasNoBalancingVertex_of_twoC3 twoC3_refl twoC3Gain_noPair) (ProjEquiv.refl _)

example : SwitchScaleEquiv twoC3 twoC3AddGain twoC3AddGain :=
  switchScaleEquiv_of_projEquiv twoC3_loopless twoC3_connectedOff
    (hasNoAddBalancingVertex_of_twoC3 twoC3_refl twoC3AddGain_noPair) (ProjEquiv.refl _)

example (ψ : Fin 6 → ZMod 3) : ¬ProjEquiv (frameMatrix twoC3 twoC3Gain) (liftMatrix twoC3 ψ) :=
  not_projEquiv_frame_lift twoC3_loopless twoC3_connectedOff
    (hasNoBalancingVertex_of_twoC3 twoC3_refl twoC3Gain_noPair) ψ

example : (0 : Fin 3 → ZMod 3) 1 = 0 :=
  eq_zero_of_isUnbalancedOff (G := twoC3) (φ := twoC3Gain) (w := 0) (fun _ _ _ => by simp)
    ((hasNoBalancingVertex_of_twoC3 twoC3_refl twoC3Gain_noPair).isUnbalancedOff twoC3_connectedOff
      (Joins.right_mem_verts (show twoC3.Joins 0 0 1 from Or.inl ⟨rfl, rfl⟩)) (by decide))

example (p : twoC3.Walk 0 1) : (1 : ZMod 3) = 1 * (p.gain (1 : Fin 6 → (ZMod 3)ˣ) : (ZMod 3)ˣ) :=
  p.eq_mul_gain 1 (fun _ => 1) fun e _ => by simp

example (p : twoC3.Walk 0 1) :
    (0 : ZMod 3) * p.addGain (0 : Fin 6 → ZMod 3) + (0 : Fin 3 → ZMod 3) 0 -
      (0 : Fin 3 → ZMod 3) 1 = 0 :=
  p.add_relation 0 0 0 fun _ _ => by simp

example : Nonempty (twoC3.Walk 0 1) :=
  twoC3_connectedOff.nonempty_walk twoC3_loopless
    (hasNoBalancingVertex_of_twoC3 twoC3_refl twoC3Gain_noPair).exists_edge_off
    (tail_mem_verts (e := 0)) (head_mem_verts (e := 0))

example : (0 : Fin 3) ≠ 1 :=
  Joins.ne_of_loopless twoC3_loopless (show twoC3.Joins 0 0 1 from Or.inl ⟨rfl, rfl⟩)

end OrientedGraph

end Transformer.GainMatroid

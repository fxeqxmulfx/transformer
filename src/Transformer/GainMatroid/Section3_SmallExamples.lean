module

public import Transformer.GainMatroid.Section3_SmallGraphs

/-!
# Concrete properly unbalanced `2C₃`, tubes and `K₄`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §3.1. The gain functions of `Section4_SmallExamples`
realize properly unbalanced biased graphs, by the classification in `Section3_SmallGraphs`. They
supply the hypotheses of Theorem 5.1. The other examples apply the cycle lemmas of
`Section3_Classes` to cycles of `2C₃`, `K₄` and the tube.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph BiasedGraph

/-- `2C₃` is 2-connected. -/
theorem twoC3_isKConnected : twoC3.IsKConnected 2 := by
  refine Or.inr ⟨by simp [Set.encard_univ]; norm_num, ?_⟩
  simp only [Joins, twoC3]
  decide

/-- The biased `2C₃` of `twoC3Gain` is properly unbalanced. -/
theorem twoC3Gain_isProperlyUnbalanced : (twoC3.gainBiased twoC3Gain).IsProperlyUnbalanced :=
  (isProperlyUnbalanced_iff_of_twoC3 (IsReorientationOf.refl _)).2 twoC3Gain_noPair

/-- The biased `2C₃` of `twoC3AddGain` is properly unbalanced. -/
theorem twoC3AddGain_isProperlyUnbalanced :
    (twoC3.addGainBiased twoC3AddGain).IsProperlyUnbalanced :=
  (isProperlyUnbalanced_iff_of_twoC3 (IsReorientationOf.refl _)).2 twoC3AddGain_noPair

example : (tube.gainBiased tubeGain).IsProperlyUnbalanced :=
  (isProperlyUnbalanced_iff_of_tube (IsReorientationOf.refl _)).2 tubeGain_noPair

example : (tube.addGainBiased tubeAddGain).IsProperlyUnbalanced :=
  (isProperlyUnbalanced_iff_of_tube (IsReorientationOf.refl _)).2 tubeAddGain_noPair

example : (K4.gainBiased K4Gain).IsProperlyUnbalanced :=
  (isProperlyUnbalanced_iff_of_K4 (IsReorientationOf.refl _)).2 K4Gain_noTriangle

example : (K4.addGainBiased K4AddGain).IsProperlyUnbalanced :=
  (isProperlyUnbalanced_iff_of_K4 (IsReorientationOf.refl _)).2 K4AddGain_noTriangle

/-- The 2-cycle on the first two links of `2C₃`. -/
private def twoC3Pair : twoC3.Walk 0 0 :=
  pairWalk (e := 0) (f := 1) (v := 1) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩)

private theorem twoC3Pair_isCycleWalk : twoC3Pair.IsCycleWalk :=
  pairWalk_isCycleWalk _ _ (by decide) (by decide)

example : ∃ e f u v, e ≠ f ∧ u ≠ v ∧ twoC3.Joins e u v ∧ twoC3.Joins f u v ∧
    twoC3Pair.edgeSet = {e, f} :=
  IsCycle.exists_pair ⟨0, twoC3Pair, twoC3Pair_isCycleWalk, rfl⟩
    (by rw [twoC3Pair_isCycleWalk.encard_edgeSet]; simp [twoC3Pair, pairWalk, Walk.edges])

/-- The triangle of `K₄` on the vertices `0`, `1`, `2`. -/
private def K4Triangle : K4.Walk 0 0 :=
  triangleWalk (a := 0) (b := 3) (c := 1) (y := 1) (z := 2) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩)
    (Or.inr ⟨rfl, rfl⟩)

private theorem K4Triangle_isCycleWalk : K4Triangle.IsCycleWalk :=
  triangleWalk_isCycleWalk _ _ _ (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide)

example : 3 ≤ K4Triangle.edgeSet.encard :=
  three_le_encard_of_K4 (IsReorientationOf.refl K4) ⟨0, K4Triangle, K4Triangle_isCycleWalk, rfl⟩

example : ∃ a b c x y z, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ K4.Joins a x y ∧
    K4.Joins b y z ∧ K4.Joins c z x ∧ K4Triangle.edgeSet = {a, b, c} :=
  IsCycle.exists_triangle ⟨0, K4Triangle, K4Triangle_isCycleWalk, rfl⟩
    (by rw [K4Triangle_isCycleWalk.encard_edgeSet]; simp [K4Triangle, triangleWalk, Walk.edges])

/-- The 2-cycle on the first two links of the tube. -/
private def tubePair : tube.Walk 0 0 :=
  pairWalk (e := 0) (f := 1) (v := 1) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩)

/-- Link `3` ends at vertex `3`, which the 2-cycle misses. -/
example : (3 : Fin 6) ∉ tubePair.edgeSet :=
  IsCycle.not_mem_of_pendant (p := 3)
    ⟨0, tubePair, pairWalk_isCycleWalk _ _ (by decide) (by decide), rfl⟩ (Or.inr rfl)
    (tube_loopless 3) fun f (hf : f ∈ tubePair.edges) _ => by
      have hf' : f = 0 ∨ f = 1 := by simpa [tubePair, pairWalk, Walk.edges] using hf
      rcases hf' with rfl | rfl <;> simp [tube]

example : twoC3.tail 0 ≠ 2 ∧ twoC3.head 0 ≠ 2 :=
  (IsReorientationOf.refl twoC3).ends_ne (by decide) (by decide)

example : twoC3.tail 0 = 0 ∨ twoC3.head 0 = 0 :=
  (IsReorientationOf.refl twoC3).has_end (Or.inl rfl)

example : twoC3.IsReorientationOf twoC3 :=
  (IsReorientationOf.refl twoC3).symm

end Transformer.GainMatroid

module

public import Transformer.GainMatroid.Section2_DeltaBiased
public import Transformer.GainMatroid.Section2_NablaExamples

/-!
# Examples of thetas of `∇_Y G` matched by thetas of `G`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, Proposition 2.7. The examples apply the
lemmas on matching thetas, the corner and cover cases and the theta property of `∇_Y B` to the
thetas of `clawPlusNabla`, with every cycle of `clawPlus` balanced, and the permutations of a match
to the theta of `clawGraph`, which matches itself.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

open scoped symmDiff

/-- The theta of `clawGraph` matches itself under its balanced cycles. -/
theorem claw_matchesTheta : MatchesTheta (clawGraph.gainBiased (1 : Fin 5 → ℚˣ)).balanced
    (clawGraph.gainBiased (1 : Fin 5 → ℚˣ)) clawSpoke1 clawSpoke2 clawSpoke3 :=
  ⟨0, 1, clawSpoke1, clawSpoke2, clawSpoke3, claw_isTheta, Iff.rfl, Iff.rfl, Iff.rfl⟩

/-- A tripod of `clawPlusNabla` off `Y` from `1`, `2`, `3` to `2`. -/
theorem plus_isTripod : IsTripod {0, 1, 2} plusE3 (.nil 2) plusE5 :=
  ⟨by unfold Walk.IsPath; decide, by unfold Walk.IsPath; decide, by unfold Walk.IsPath; decide,
    by simp [plusE3, Walk.edges], by simp [Walk.edges], by simp [plusE5, Walk.edges], by decide,
    by decide, by decide, by decide, by decide, by decide⟩

theorem plus_corner_ne : plusB5.edgeSet ∪ plusC.edgeSet ≠ {0, 1, 2} := fun h => by
  simpa [plusB5, plusC, Walk.edgeSet, Walk.edges] using Set.ext_iff.1 h 5

theorem plus_single_ne₁₂ : plusC.edgeSet ∪ plusE3.edgeSet ≠ {0, 1, 2} := fun h => by
  simpa [plusC, plusE3, Walk.edgeSet, Walk.edges] using Set.ext_iff.1 h 3

theorem plus_single_ne₁₃ : plusC.edgeSet ∪ plusE4.edgeSet ≠ {0, 1, 2} := fun h => by
  simpa [plusC, plusE4, Walk.edgeSet, Walk.edges] using Set.ext_iff.1 h 4

theorem plus_single_ne₂₃ : plusE3.edgeSet ∪ plusE4.edgeSet ≠ {0, 1, 2} := fun h => by
  simpa [plusE3, plusE4, Walk.edgeSet, Walk.edges] using Set.ext_iff.1 h 3

/-- The digon through `c` and `3` is in `∇_Y B` for `B` all cycles of `clawPlus`: its symmetric
difference with `Y` is the triangle `{0, 3, 1}` through the centre. -/
theorem plus_mem_nablaBalanced₃ : plusC.edgeSet ∪ plusE3.edgeSet ∈
    nablaBalanced clawPlusNabla (clawPlus.gainBiased (1 : Fin 7 → ℚˣ)).balanced {0, 1, 2} := by
  rw [clawPlusNabla_isExchange.mem_nablaBalanced_iff_of_singleton (Ω := clawPlus.gainBiased 1)
    plus_isTheta_single.isCycle.1 (e := 2) (by simp)
    (by ext g; fin_cases g <;> simp [plusC, plusE3, Walk.edgeSet, Walk.edges])]
  have : (plusC.edgeSet ∪ plusE3.edgeSet) ∆ {0, 1, 2} = ({0, 3, 1} : Set (Fin 7)) := by
    ext g
    fin_cases g <;> simp [plusC, plusE3, Walk.edgeSet, Walk.edges, Set.mem_symmDiff]
  rw [this]
  exact gainBiased_one_isBalanced clawPlus _ (isCycle_of_three (x := 0) (y := 1) (z := 2)
    (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide))

/-- The digon through `c` and `4` is in `∇_Y B` for `B` all cycles of `clawPlus`: its symmetric
difference with `Y` is the triangle `{0, 4, 1}` through the centre. -/
theorem plus_mem_nablaBalanced₄ : plusC.edgeSet ∪ plusE4.edgeSet ∈
    nablaBalanced clawPlusNabla (clawPlus.gainBiased (1 : Fin 7 → ℚˣ)).balanced {0, 1, 2} := by
  rw [clawPlusNabla_isExchange.mem_nablaBalanced_iff_of_singleton (Ω := clawPlus.gainBiased 1)
    plus_isTheta_single.isCycle.2.1 (e := 2) (by simp)
    (by ext g; fin_cases g <;> simp [plusC, plusE4, Walk.edgeSet, Walk.edges])]
  have : (plusC.edgeSet ∪ plusE4.edgeSet) ∆ {0, 1, 2} = ({0, 4, 1} : Set (Fin 7)) := by
    ext g
    fin_cases g <;> simp [plusC, plusE4, Walk.edgeSet, Walk.edges, Set.mem_symmDiff]
  rw [this]
  exact gainBiased_one_isBalanced clawPlus _ (isCycle_of_three (x := 0) (y := 1) (z := 2)
    (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide))

example := claw_matchesTheta.theta (gainBiased_one_isBalanced clawGraph _ claw_isTheta.isCycle.1)
  (gainBiased_one_isBalanced clawGraph _ claw_isTheta.isCycle.2.1)

example := claw_matchesTheta.swap₁₂

example := claw_matchesTheta.swap₂₃

example := MatchesTheta.of_reverse (B' := (clawGraph.gainBiased (1 : Fin 5 → ℚˣ)).balanced)
  (Ω := clawGraph.gainBiased (1 : Fin 5 → ℚˣ))
  ⟨1, 0, clawSpoke1.reverse, clawSpoke2.reverse, clawSpoke3.reverse, claw_isTheta.reverse,
    Iff.rfl, Iff.rfl, Iff.rfl⟩

example := clawPlusNabla_isExchange.isPath_cons_of_map (by unfold Walk.IsPath; decide :
  plusE3.IsPath) (by decide) (e := 0) (he := Or.inl ⟨rfl, rfl⟩) (P' := plusS3) rfl

example := clawPlusNabla_isExchange.openly_cons_of_map (P := plusE3) (Q := plusE5) (e := 0)
  (f := 2) (he := Or.inl ⟨rfl, rfl⟩) (hf := Or.inl ⟨rfl, rfl⟩) (P' := plusS3) (Q' := plusS5)
  rfl rfl rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)

example := clawPlusNabla_isExchange.exists_isTheta_of_isTripod plus_isTripod

example := Walk.IsPath.exists_eq_cons (by unfold Walk.IsPath; decide : plusB5.IsPath) (e := 1)
  (by decide) (Or.inl ⟨rfl, rfl⟩ : clawPlusNabla.Joins 1 1 3)

example := insert_symmDiff_triple_b (a := (0 : Fin 7)) (b := 1) (c := 2) (X := {3}) (by simp)
  (by decide) (by decide)

example := insert_symmDiff_triple_c (a := (0 : Fin 7)) (b := 1) (c := 2) (X := {3}) (by simp)
  (by decide) (by decide)

example := clawPlusNabla_isExchange.mem_nablaBalanced_iff_insert
  (Ω := clawPlus.gainBiased (1 : Fin 7 → ℚˣ)) (X := {3}) (by simp) (e := 2) (by simp)
  plusNabla_isCycle₂₃

example := clawPlusNabla_isExchange.mem_nablaBalanced_iff_insert_insert
  (Ω := clawPlus.gainBiased (1 : Fin 7 → ℚˣ)) (X := {5}) (by simp) (e₁ := 1) (e₂ := 2) (by simp)
  (by simp) (by decide) plusNabla_isCycle₁₂₅

example := clawPlusNabla_isExchange.matchesTheta_of_corner
  (Ω := clawPlus.gainBiased (1 : Fin 7 → ℚˣ)) plus_isTheta_corner (by decide) (by decide)
  plus_corner_ne

example := clawPlusNabla_isExchange.matchesTheta_of_mem_of_mem
  (Ω := clawPlus.gainBiased (1 : Fin 7 → ℚˣ)) plus_isTheta_corner (e₁ := 1) (by decide) (by simp)
  (e₂ := 2) (by decide) (by simp) plus_corner_ne

example := union_inter_of_disjoint (P := ({0} : Set (Fin 7))) (D := {3}) (Y := {0, 1, 2})
  (by simp)

example := clawPlusNabla_isExchange.exists_isDetour (Ω := clawPlus.gainBiased (1 : Fin 7 → ℚˣ))
  (by unfold Walk.IsPath; decide : plusC.IsPath) fun hsub => by
    simpa [plusC, Walk.edgeSet, Walk.edges] using hsub (show (0 : Fin 7) ∈ ({0, 1, 2} : Set _)
      by simp)

example := clawPlusNabla_isExchange.matchesTheta_of_off (Ω := clawPlus.gainBiased (1 : Fin 7 → ℚˣ))
  plus_isTheta_single (by simp [plusE3, Walk.edges]) (by simp [plusE4, Walk.edges])

example := plus_isTheta_single.symmDiff_union

example := clawPlusNabla_isExchange.matchesTheta (Ω := clawPlus.gainBiased (1 : Fin 7 → ℚˣ))
  plus_isTheta_single plus_single_ne₁₂ plus_single_ne₁₃ plus_single_ne₂₃

example := clawPlusNabla_isExchange.theta_nablaBalanced
  (Ω := clawPlus.gainBiased (1 : Fin 7 → ℚˣ)) plus_isTheta_single plus_mem_nablaBalanced₃
  plus_mem_nablaBalanced₄

example := clawPlusNabla_isExchange.exists_nablaBiasedGraph
  (Ω := clawPlus.gainBiased (1 : Fin 7 → ℚˣ))

end Transformer.GainMatroid

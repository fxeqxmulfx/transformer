module

public import Transformer.GainMatroid.Section1_Particular
public import Transformer.GainMatroid.Section1_Theorem1
public import Transformer.GainMatroid.Section2_RollUps
public import Transformer.GainMatroid.Section3_FanTheta

/-!
# The almost-balanced fan and its unrolling

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5 and §5. The contrabalanced `fanGraph` of
`Section1_Fan` meets the hypotheses of the almost-balanced statements of §5: the vertex `2` is its
unique balancing vertex after deleting joints (`fan_isRollable`), it has no joint, so it is its own
unrolling to `2` (`fan_isUnrolling`), and each of its edges at `2` is an unbalancing class by
itself. The derived gains `fanDerived₁` realize it over `𝔽₇⁺`. A graph on three vertices has no
vertical 2-separation (`OrientedGraph.not_isVerticalSeparation_two`), and rolling up no edge changes
nothing. -/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph BiasedGraph

local instance instFactPrimeSevenFanFacts : Fact (Nat.Prime 7) := ⟨by decide⟩

namespace OrientedGraph

variable {V E : Type*} {G : OrientedGraph V E}

/-- Rolling up no edge changes nothing. -/
theorem rollUpGraph_empty (G : OrientedGraph V E) (u : V) : G.rollUpGraph u ∅ = G := by
  simp only [rollUpGraph, Set.mem_empty_iff_false, ite_false]

/-- A graph on three vertices has no vertical 2-separation: it would need four vertices. -/
theorem not_isVerticalSeparation_two {G : OrientedGraph (Fin 3) E} (A B : Set E) :
    ¬G.IsVerticalSeparation 2 A B := by
  rintro ⟨hsep, ⟨x, hxA, hxB⟩, ⟨y, hyB, hyA⟩⟩
  have hxy : x ≠ y := fun h => hxB (h ▸ hyB)
  have hdisj : Disjoint ({x, y} : Set (Fin 3)) (G.verts A ∩ G.verts B) := by
    rw [Set.disjoint_left]
    rintro v (rfl | rfl) ⟨hA, hB⟩
    · exact hxB hB
    · exact hyA hA
  have h4 : ({x, y} ∪ (G.verts A ∩ G.verts B)).encard = 4 := by
    rw [Set.encard_union_eq hdisj, Set.encard_pair hxy, hsep.inter]
    norm_num
  have hle := Set.encard_le_encard (Set.subset_univ ({x, y} ∪ (G.verts A ∩ G.verts B)))
  rw [h4, Set.encard_univ, ENat.card_eq_coe_fintype_card, Fintype.card_fin] at hle
  norm_num at hle

end OrientedGraph

namespace BiasedGraph

variable {V E : Type*}

/-- Rolling up no edge changes nothing. -/
theorem rollUp_empty (Ω : BiasedGraph V E) (u : V) : Ω.rollUp u ∅ = Ω := by
  cases Ω
  simp only [rollUp, rollUpGraph_empty, Set.disjoint_empty, and_true, Set.ofPred_mem_eq]

end BiasedGraph

/-- In the contrabalanced `fanGraph`, every cycle of two or more edges passes through `2`. -/
theorem fan_isLinkBalancingVertex : fanGraph.contrabalanced.IsLinkBalancingVertex 2 := by
  have hhead : ∀ e : Fin 5, e ≠ 0 → fanGraph.head e = 2 := by
    simp only [fanGraph]
    decide
  intro C _ h2
  by_contra hv
  have hsub : C ⊆ {0} := fun e he => by
    by_contra hne
    exact hv (mem_verts.2 ⟨e, he, Or.inr (hhead e hne)⟩)
  have hle := Set.encard_le_encard hsub
  rw [Set.encard_singleton] at hle
  exact absurd (h2.trans hle) (by norm_num)

/-- `2` is a balancing vertex of the contrabalanced `fanGraph`, which is loopless. -/
theorem fan_isBalancingVertex : fanGraph.contrabalanced.IsBalancingVertex 2 := fun C hC =>
  fan_isLinkBalancingVertex C hC (by
    rcases hC.1.encard_eq_two_or_three fanGraph_loopless with h | h
    · exact h.ge
    · rw [h]; norm_num)

/-- The contrabalanced `fanGraph` is almost balanced, with `2` its unique balancing vertex. -/
theorem fan_isRollable : fanGraph.contrabalanced.IsRollable 2 :=
  ⟨fan_isAlmostBalanced, fun _ => ⟨fun hv =>
    fan_existsUnique_isLinkBalancingVertex.unique hv fan_isLinkBalancingVertex,
    fun h => h ▸ fan_isLinkBalancingVertex⟩⟩

/-- The loopless `fanGraph` has no joint. -/
theorem fan_not_isJoint (e : Fin 5) : ¬fanGraph.contrabalanced.IsJoint e := fun h =>
  fanGraph_loopless e h.1

theorem fan_offJoints : fanGraph.contrabalanced.offJoints 2 = ∅ :=
  Set.eq_empty_of_forall_notMem fun e he => fan_not_isJoint e he.1

theorem fan_unrollGraph : fanGraph.contrabalanced.unrollGraph 2 = fanGraph := by
  simp only [unrollGraph, fan_offJoints, Set.mem_empty_iff_false, ite_false]
  rfl

/-- Each edge at `2` is an unbalancing class by itself: no cycle is balanced. -/
theorem fan_unbalancingClass {e : Fin 5} (he : e ∈ fanGraph.contrabalanced.sigmaStar 2) :
    fanGraph.contrabalanced.unbalancingClass 2 e = {e} := by
  ext f
  simp only [unbalancingClass, UnbalancingRel, fan_offJoints, Set.mem_empty_iff_false, and_false,
    or_false, Set.mem_ofPred_eq, Set.mem_singleton_iff]
  constructor
  · rintro ⟨-, h | ⟨C, hC, -⟩⟩
    · exact h.symm
    · exact absurd hC (Set.notMem_empty _)
  · rintro rfl
    exact ⟨he, Or.inl rfl⟩

theorem fan_mem_sigmaStar {e : Fin 5} (he : e ≠ 0) : e ∈ fanGraph.contrabalanced.sigmaStar 2 := by
  refine Or.inl ⟨fanGraph_loopless e, Or.inr ?_⟩
  revert e
  simp only [contrabalanced, fanGraph]
  decide

theorem fan_mem_unbalancingClasses : ({1} : Set (Fin 5)) ∈
    fanGraph.contrabalanced.unbalancingClasses 2 :=
  ⟨1, fan_mem_sigmaStar (by decide), (fan_unbalancingClass (fan_mem_sigmaStar (by decide))).symm⟩

/-- No cycle of `fanGraph` meets each unbalancing class at `2` in zero or two edges. -/
theorem fan_unrollBalanced : fanGraph.contrabalanced.unrollBalanced 2 = ∅ := by
  refine Set.eq_empty_of_forall_notMem fun C hC' => ?_
  obtain ⟨⟨hC, hcls⟩, -⟩ := hC'
  rw [fan_unrollGraph] at hC
  have hsub : C ⊆ {0} := fun e he => by
    by_contra hne
    have hσ := fan_mem_sigmaStar hne
    rcases hcls e hσ with h | h <;> rw [fan_unbalancingClass hσ] at h
    · have hmem : e ∈ C ∩ {e} := ⟨he, rfl⟩
      rw [Set.encard_eq_zero.1 h] at hmem
      exact hmem
    · have hle := Set.encard_le_encard (Set.inter_subset_right (s := C) (t := {e}))
      rw [h, Set.encard_singleton] at hle
      norm_num at hle
  have hle := Set.encard_le_encard hsub
  rw [Set.encard_singleton] at hle
  rcases hC.encard_eq_two_or_three fanGraph_loopless with h | h <;> rw [h] at hle <;>
    norm_num at hle

/-- The contrabalanced `fanGraph` is its own unrolling to `2`. -/
theorem fan_isUnrolling :
    fanGraph.contrabalanced.IsUnrolling 2 fanGraph.contrabalanced :=
  ⟨fan_unrollGraph.symm, fan_unrollBalanced.symm⟩

/-- The derived gains `(0, 2, 3, 4, 5)` realize the contrabalanced `fanGraph` over `𝔽₇⁺`: no 2-cycle
or triangle has gain zero. -/
theorem fanDerived₁_addRealizes : fanGraph.contrabalanced.AddRealizes fanDerived₁ := by
  refine Set.eq_empty_of_forall_notMem fun C hC => ?_
  have hC' : C ∈ fanGraph.addBalancedCycles fanDerived₁ := hC
  rcases (mem_addBalancedCycles_iff.1 hC').1.encard_eq_two_or_three fanGraph_loopless with h | h
  · exact encard_ne_two_of_addParallel
      (by simp only [Joins, addDart, fanGraph, fanDerived₁]; decide +kernel) C hC' h
  · exact encard_ne_three_of_addTriangle
      (by simp only [Joins, addDart, fanGraph, fanDerived₁]; decide +kernel) C hC' h

example := fan_unbalancingClass (e := 1) (fan_mem_sigmaStar (by decide))

end Transformer.GainMatroid

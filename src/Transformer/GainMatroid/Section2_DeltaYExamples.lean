module

public import Transformer.GainMatroid.Section1_ThreeVertices
public import Transformer.GainMatroid.Section2_DeltaY
public import Transformer.GainMatroid.Section5_ProperlyUnbalanced

/-!
# Examples of Δ-Y and Y-Δ exchanges

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. The source calls `∇_Y B` a "collection of
cycles of `∇_Y G`" made of `Y` and the minimal nonempty members of a family. On `K₄` minus an edge,
with one balanced four-cycle through the star, that four-cycle is a minimal member of the family
and is not a cycle of `∇_Y G`: there it is two digons sharing a vertex.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

open scoped symmDiff

/-- The labels of the triangle graph as in §2.4: `a = 1`, `b = 2`, `c = 0` on `x = 0`, `y = 1`,
`z = 2`. -/
theorem triangle_isLabelledTriangle : triangle.IsLabelledTriangle 1 2 0 0 1 2 :=
  ⟨Or.inl ⟨rfl, rfl⟩, Or.inr ⟨rfl, rfl⟩, Or.inl ⟨rfl, rfl⟩, by decide, by decide, by decide,
    by decide, by decide, by decide⟩

/-- The triangle is balanced for the identity gain function. -/
theorem triangle_mem_gainBiased :
    ({1, 2, 0} : Set (Fin 3)) ∈ (triangle.gainBiased (1 : Fin 3 → ℚˣ)).balanced :=
  mem_balancedCycles_iff.2 ⟨triangle_isLabelledTriangle.isCycle, 1, fun e _ => by
    rw [switch_one]
    rfl⟩

/-- In `∇_Y` of `clawGraph`, the four edges of the cycle through the star have no vertex `0`, so
they cannot form a cycle on four vertices. -/
theorem clawGraph_nabla_not_isCycle :
    ¬(clawGraph.nablaGraph 0 1 2 1 2 3).IsCycle {1, 2, 3, 4} := by
  rintro ⟨u, W, hW, hC⟩
  have hlen : W.edges.length = 4 := by
    have h := hW.encard_edgeSet
    rw [hC] at h
    have h4 : ({1, 2, 3, 4} : Set (Fin 5)).encard = 4 := by
      rw [Set.encard_insert_of_notMem (by simp), Set.encard_insert_of_notMem (by simp),
        Set.encard_pair (by decide)]
      norm_num
    rw [h4] at h
    exact_mod_cast h.symm
  have hoff : ∀ w ∈ W.support, w ≠ 0 := by
    intro w hw
    have hv := Walk.mem_verts_of_mem_support W hW.edges_ne_nil hw
    rw [hC, mem_verts] at hv
    obtain ⟨e, he, h⟩ := hv
    rintro rfl
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at he
    rcases he with rfl | rfl | rfl | rfl <;> simp [nablaGraph, clawGraph] at h
  have hcard : W.support.tail.toFinset.card = 4 := by
    rw [List.toFinset_card_of_nodup hW.support_tail_nodup, List.length_tail, Walk.length_support,
      hlen]
  have hsub : W.support.tail.toFinset ⊆ Finset.univ.erase (0 : Fin 4) := fun w hw =>
    Finset.mem_erase.2 ⟨hoff w (List.mem_of_mem_tail (List.mem_toFinset.1 hw)), Finset.mem_univ w⟩
  have hle := Finset.card_le_card hsub
  rw [hcard] at hle
  simp at hle

/-- A minimal nonempty member of the family defining `∇_Y B` (§2.4) need not be a cycle of
`∇_Y G`. So the source's "collection of cycles" is read as the cycles among those members. -/
theorem nablaFamily_minimal_not_isCycle :
    ∃ (Ω : BiasedGraph (Fin 4) (Fin 5)) (C : Set (Fin 5)),
      Ω.graph.IsLabelledClaw 0 0 1 2 1 2 3 ∧
        Minimal (fun D => D ∈ nablaFamily Ω.balanced {0, 1, 2} ∧ D.Nonempty) C ∧
          ¬(Ω.graph.nablaGraph 0 1 2 1 2 3).IsCycle C := by
  refine ⟨clawGraph.singleCycleBiased _ clawGraph_isCycle, {1, 2, 3, 4},
    clawGraph_isLabelledClaw, ⟨⟨Or.inl ⟨rfl, Or.inr ?_⟩, ⟨1, by simp⟩⟩, ?_⟩,
    clawGraph_nabla_not_isCycle⟩
  · have hinter : ({1, 2, 3, 4} ∩ {0, 1, 2} : Set (Fin 5)) = {1, 2} := by
      ext i
      fin_cases i <;> simp
    rw [hinter, Set.ncard_pair (by decide)]
  · rintro D ⟨⟨hD, -⟩ | ⟨C', hC', -, rfl⟩, -⟩ hle
    · exact le_of_eq (Eq.symm hD)
    · obtain rfl : C' = {1, 2, 3, 4} := hC'
      have h0 : (0 : Fin 5) ∈ ({1, 2, 3, 4} : Set (Fin 5)) ∆ {0, 1, 2} :=
        Set.mem_symmDiff.2 (Or.inr ⟨by simp, by simp⟩)
      have := hle h0
      simp at this

example := triangle_isLabelledTriangle.isCycle

example := BiasedGraph.exists_isNablaExchange clawGraph.contrabalanced clawGraph_isLabelledClaw

end Transformer.GainMatroid

module

public import Transformer.GainMatroid.Section2_UnrollingLinks

/-!
# The unrolling is a biased graph

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5, calls `(Ĝ, B̂)` a biased graph without proof.
`IsLinkBalancingVertex.exists_isUnrolling` proves it for the corrected `B̂` of `unrollBalanced`, at
any balancing vertex `u` after deleting the joints. In a theta whose two cycles through `P` are in
`B̂`, the two edges at `u` of the third cycle lie on `Q`, on `R`, or one on each. In the last case
`u` is an end of the theta, and the edge of `P` at `u` is related to both
(`IsLinkBalancingVertex.unrollBalanced_theta`). The third cycle has at least two edges, so it is not
a joint.
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

namespace OrientedGraph

variable {G : OrientedGraph V E}

/-- A walk between distinct vertices has an edge at its start. -/
theorem Walk.exists_mem_edges_start {x y : V} (P : G.Walk x y) (hxy : x ≠ y) :
    ∃ p ∈ P.edges, G.tail p = x ∨ G.head p = x := by
  cases P with
  | nil => exact absurd rfl hxy
  | cons p hp P' => exact ⟨p, List.mem_cons_self, hp.tail_or_head⟩

end OrientedGraph

open OrientedGraph

namespace BiasedGraph

variable {Ω : BiasedGraph V E} {u : V}

/-- The theta property of `B̂`: if the two cycles of a theta through `P` are in `B̂`, so is the
third. -/
theorem IsLinkBalancingVertex.unrollBalanced_theta (hΩ : Ω.IsLinkBalancingVertex u) {x y : V}
    {P Q R : (Ω.unrollGraph u).Walk x y} (hT : IsTheta P Q R)
    (hPQ : P.edgeSet ∪ Q.edgeSet ∈ Ω.unrollBalanced u)
    (hPR : P.edgeSet ∪ R.edgeSet ∈ Ω.unrollBalanced u) :
    Q.edgeSet ∪ R.edgeSet ∈ Ω.unrollBalanced u := by
  have hC := hT.isCycle.2.2
  by_cases hS : ∃ a ∈ Q.edgeSet ∪ R.edgeSet, a ∈ Ω.sigmaStar u
  · obtain ⟨a, haC, haS⟩ := hS
    obtain ⟨haL, hau⟩ := unrollGraph_isLink_of_mem_sigmaStar haS
    obtain ⟨b, hbC, hba, hbu, -⟩ := hC.exists_partner haC haL hau
    have hbS : b ∈ Ω.sigmaStar u := mem_sigmaStar_of_unrollGraph (fun hll => hba (by
      rw [hC.eq_singleton_of_loop hbC hll] at haC
      exact (Set.mem_singleton_iff.1 haC).symm)) hbu
    refine mem_unrollBalanced_of_rel hΩ hC haS hbS hba.symm haC hbC ?_
    have hmix : ∀ {c d}, c ∈ Q.edges → d ∈ R.edges → c ∈ Ω.sigmaStar u → d ∈ Ω.sigmaStar u →
        ((Ω.unrollGraph u).tail c = u ∨ (Ω.unrollGraph u).head c = u) →
        ((Ω.unrollGraph u).tail d = u ∨ (Ω.unrollGraph u).head d = u) →
        Ω.UnbalancingRel u c d := by
      intro c d hcQ hdR hcS hdS hcu hdu
      obtain ⟨p, hpP, hpu⟩ : ∃ p ∈ P.edges,
          (Ω.unrollGraph u).tail p = u ∨ (Ω.unrollGraph u).head p = u := by
        rcases hT.openQR.verts u (Walk.mem_support_of_mem_edges hcQ hcu)
          (Walk.mem_support_of_mem_edges hdR hdu) with rfl | rfl
        · exact P.exists_mem_edges_start hT.ne
        · obtain ⟨p, hp, hpu⟩ := P.reverse.exists_mem_edges_start hT.ne.symm
          exact ⟨p, Walk.mem_edges_reverse.1 hp, hpu⟩
      have hpS := mem_sigmaStar_of_unrollGraph (hT.pathP.isLink_of_mem hpP) hpu
      exact hΩ.unbalancingRel_trans hcS hpS hdS
        (unbalancingRel_of_mem_unrollBalanced hPQ hpS hcS (Or.inl hpP) (Or.inr hcQ)).symm
        (unbalancingRel_of_mem_unrollBalanced hPR hpS hdS (Or.inl hpP) (Or.inr hdR))
    rcases haC with haQ | haR <;> rcases hbC with hbQ | hbR
    · exact unbalancingRel_of_mem_unrollBalanced hPQ haS hbS (Or.inr haQ) (Or.inr hbQ)
    · exact hmix haQ hbR haS hbS hau hbu
    · exact (hmix hbQ haR hbS haS hbu hau).symm
    · exact unbalancingRel_of_mem_unrollBalanced hPR haS hbS (Or.inr haR) (Or.inr hbR)
  · push Not at hS
    obtain ⟨q, hq⟩ := List.exists_mem_of_ne_nil _ (Walk.edges_ne_nil_of_ne hT.ne Q)
    obtain ⟨r, hr⟩ := List.exists_mem_of_ne_nil _ (Walk.edges_ne_nil_of_ne hT.ne R)
    refine ⟨⟨hC, fun k _ => Or.inl (Set.encard_eq_zero.2 (Set.eq_empty_of_forall_notMem
      fun z hz => hS z hz.1 hz.2.1))⟩, fun c hc _ => hT.openQR.edges q hq ?_⟩
    have hq' : q ∈ ({c} : Set E) := by
      rw [← hc]
      exact Or.inl hq
    have hr' : r ∈ ({c} : Set E) := by
      rw [← hc]
      exact Or.inr hr
    rw [Set.mem_singleton_iff.1 hq', ← Set.mem_singleton_iff.1 hr']
    exact hr

/-- §2.5: `(Ĝ, B̂)` is a biased graph, for the corrected `B̂` of `unrollBalanced`. The source
unrolls to the unique balancing vertex after deleting the joints; any such vertex will do. -/
theorem IsLinkBalancingVertex.exists_isUnrolling (h : Ω.IsLinkBalancingVertex u) :
    ∃ Ω', Ω.IsUnrolling u Ω' :=
  ⟨⟨Ω.unrollGraph u, Ω.unrollBalanced u, fun _ hC => hC.1.1, h.unrollBalanced_theta⟩, rfl, rfl⟩

end BiasedGraph

open BiasedGraph

/-- The three links from `1` to `0` of `Ĝ` for `threeJoints` form a theta. -/
theorem threeJoints_isTheta : IsTheta (threeJointsWalk 4 (by decide))
    (threeJointsWalk 5 (by decide)) (threeJointsWalk 6 (by decide)) :=
  ⟨by decide, threeJointsWalk_isPath _ _, threeJointsWalk_isPath _ _, threeJointsWalk_isPath _ _,
    threeJointsWalk_openly _ _ (by decide), threeJointsWalk_openly _ _ (by decide),
    threeJointsWalk_openly _ _ (by decide)⟩

example := (pumpkinWalk 0).exists_mem_edges_start (by decide)

example := threeJoints_isRollable.balancing.unrollBalanced_theta threeJoints_isTheta
  (threeJoints_mem_unrollBalanced _ _ (by decide)) (threeJoints_mem_unrollBalanced _ _ (by decide))

example := threeJoints_isRollable.balancing.exists_isUnrolling

end Transformer.GainMatroid

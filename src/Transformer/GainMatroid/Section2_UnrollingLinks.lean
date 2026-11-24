module

public import Transformer.GainMatroid.Section2_UnrollingEmptyJ

/-!
# The cycles of `B̂`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5:

> "Now consider the biased graph `(Ĝ, B̂)` obtained from `(G, B)` by replacing each joint `e ∈ J`
> incident to a vertex `v ≠ u` with a `uv`-link. Define `B̂` to be the set of those cycles having
> intersection of size 0 or 2 with each unbalancing class of `Σ_G(u)`."

The edges of `Σ(u)` are the links at `u` in `Ĝ` (`unrollGraph_isLink_of_mem_sigmaStar`,
`mem_sigmaStar_of_unrollGraph`), so a cycle of `Ĝ` has none of them or exactly its two edges at
`u`. Two edges of `Σ(u)` on a cycle of the corrected `B̂` of `unrollBalanced` are related in
`(G, B)` (`unbalancingRel_of_mem_unrollBalanced`), and a cycle of `Ĝ` through two related edges of
`Σ(u)` is in `B̂` (`mem_unrollBalanced_of_rel`). The examples use `threeJoints`, whose three joints
off `0` become a theta of `Ĝ`.
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

open OrientedGraph

namespace BiasedGraph

variable {Ω : BiasedGraph V E} {u : V}

/-- An edge of `Σ(u)` is a link at `u` in `Ĝ`: a link at `u` keeps its ends, and a joint of `J`
becomes a link to `u`. -/
theorem unrollGraph_isLink_of_mem_sigmaStar {e : E} (he : e ∈ Ω.sigmaStar u) :
    (Ω.unrollGraph u).IsLink e ∧ ((Ω.unrollGraph u).tail e = u ∨ (Ω.unrollGraph u).head e = u) := by
  by_cases hJ : e ∈ Ω.offJoints u
  · have hh : (Ω.unrollGraph u).head e = u := ite_eq_left hJ
    exact ⟨fun h => hJ.2 (h.trans hh), Or.inr hh⟩
  · have hh : (Ω.unrollGraph u).head e = Ω.graph.head e := ite_eq_right hJ
    obtain ⟨hl, hend⟩ := Or.resolve_right he hJ
    exact ⟨fun h => hl (h.trans hh), hend.imp id fun h => hh.trans h⟩

/-- A link at `u` in `Ĝ` is in `Σ(u)`. -/
theorem mem_sigmaStar_of_unrollGraph {e : E} (hl : (Ω.unrollGraph u).IsLink e)
    (hend : (Ω.unrollGraph u).tail e = u ∨ (Ω.unrollGraph u).head e = u) :
    e ∈ Ω.sigmaStar u := by
  by_cases hJ : e ∈ Ω.offJoints u
  · exact Or.inr hJ
  · have hh : (Ω.unrollGraph u).head e = Ω.graph.head e := ite_eq_right hJ
    exact Or.inl ⟨fun h => hl (h.trans hh.symm), hend.imp id fun h => hh.symm.trans h⟩

/-- Two edges of `Σ(u)` on a cycle of `B̂` are related in `(G, B)`: the class of the first meets
the cycle in a second edge, which is the other edge of the cycle at `u`. -/
theorem unbalancingRel_of_mem_unrollBalanced {C : Set E} (hC : C ∈ Ω.unrollBalanced u) {e f : E}
    (he : e ∈ Ω.sigmaStar u) (hf : f ∈ Ω.sigmaStar u) (heC : e ∈ C) (hfC : f ∈ C) :
    Ω.UnbalancingRel u e f := by
  by_cases hef : e = f
  · exact Or.inl hef
  obtain ⟨⟨hC', hcls⟩, -⟩ := hC
  obtain ⟨g, ⟨hgC, hge⟩, hgne⟩ : ∃ g ∈ C ∩ Ω.unbalancingClass u e, g ≠ e := by
    refine Set.exists_ne_of_one_lt_encard ?_ e
    rcases hcls e he with h0 | h2
    · exact absurd h0 (Set.encard_ne_zero.2 ⟨e, heC, he, Or.inl rfl⟩)
    · rw [h2]
      norm_num
  obtain ⟨heL, heu⟩ := unrollGraph_isLink_of_mem_sigmaStar he
  obtain ⟨p, -, -, -, huniq⟩ := hC'.exists_partner heC heL heu
  have hgp := (huniq g hgC (unrollGraph_isLink_of_mem_sigmaStar hge.1).2).resolve_left hgne
  have hfp := (huniq f hfC (unrollGraph_isLink_of_mem_sigmaStar hf).2).resolve_left (Ne.symm hef)
  exact hgp.trans hfp.symm ▸ hge.2

/-- A cycle of `Ĝ` through two related edges of `Σ(u)` is in `B̂`: they are its only edges in
`Σ(u)`, and each class contains both or neither. -/
theorem mem_unrollBalanced_of_rel (hΩ : Ω.IsLinkBalancingVertex u) {C : Set E}
    (hC : (Ω.unrollGraph u).IsCycle C) {e f : E} (he : e ∈ Ω.sigmaStar u) (hf : f ∈ Ω.sigmaStar u)
    (hne : e ≠ f) (heC : e ∈ C) (hfC : f ∈ C) (hef : Ω.UnbalancingRel u e f) :
    C ∈ Ω.unrollBalanced u := by
  obtain ⟨heL, heu⟩ := unrollGraph_isLink_of_mem_sigmaStar he
  obtain ⟨p, -, -, -, huniq⟩ := hC.exists_partner heC heL heu
  have hfp := (huniq f hfC (unrollGraph_isLink_of_mem_sigmaStar hf).2).resolve_left (Ne.symm hne)
  have hS : ∀ x ∈ C, x ∈ Ω.sigmaStar u → x = e ∨ x = f := fun x hx hxS =>
    (huniq x hx (unrollGraph_isLink_of_mem_sigmaStar hxS).2).imp_right
      fun hxp => hxp.trans hfp.symm
  refine ⟨⟨hC, fun k hk => ?_⟩, fun c hc _ => hne ?_⟩
  · by_cases hke : e ∈ Ω.unbalancingClass u k
    · have hkf : f ∈ Ω.unbalancingClass u k := ⟨hf, hΩ.unbalancingRel_trans hk he hf hke.2 hef⟩
      refine Or.inr ((congrArg Set.encard (Set.ext fun x => ⟨fun hx => ?_, ?_⟩)).trans
        (Set.encard_pair hne))
      · exact hS x hx.1 hx.2.1
      · rintro (rfl | rfl)
        exacts [⟨heC, hke⟩, ⟨hfC, hkf⟩]
    · refine Or.inl (Set.encard_eq_zero.2 (Set.eq_empty_of_forall_notMem fun x hx => ?_))
      rcases hS x hx.1 hx.2.1 with rfl | rfl
      · exact hke hx.2
      · exact hke ⟨he, hΩ.unbalancingRel_trans hk hx.2.1 he hx.2.2 hef.symm⟩
  · have he' : e ∈ ({c} : Set E) := by
      rw [← hc]
      exact heC
    have hf' : f ∈ ({c} : Set E) := by
      rw [← hc]
      exact hfC
    exact (Set.mem_singleton_iff.1 he').trans (Set.mem_singleton_iff.1 hf').symm

end BiasedGraph

open BiasedGraph

/-- Two digons at `0` and three joints at `1`. `J` is one class, and in `Ĝ` the joints are three
links from `1` to `0`, a theta whose cycles are in `B̂`. -/
def threeJoints : OrientedGraph (Fin 3) (Fin 7) := ⟨![0, 0, 0, 0, 1, 1, 1], ![1, 1, 2, 2, 1, 1, 1]⟩

theorem threeJoints_isRollable : threeJoints.contrabalanced.IsRollable 0 :=
  isRollable_of_digons (a := 0) (a' := 1) (b := 2) (b' := 3) (Or.inl ⟨rfl, rfl⟩)
    (Or.inl ⟨rfl, rfl⟩) (by decide) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide)
    (by decide)

theorem threeJoints_mem_offJoints {e : Fin 7} (he : 4 ≤ e) :
    e ∈ threeJoints.contrabalanced.offJoints 0 := by
  have h : ∀ e : Fin 7, 4 ≤ e → threeJoints.tail e = 1 ∧ threeJoints.head e = 1 := by decide
  exact ⟨⟨(h e he).1.trans (h e he).2.symm, Set.notMem_empty _⟩, (h e he).1.trans_ne (by decide)⟩

/-- The one-edge walk of `Ĝ` along the joint `e` of `threeJoints`, now a link from `1` to `0`. -/
def threeJointsWalk (e : Fin 7) (he : 4 ≤ e) :
    (threeJoints.contrabalanced.unrollGraph 0).Walk 1 0 :=
  .cons e (Or.inl ⟨by revert he; revert e; decide, ite_eq_left (threeJoints_mem_offJoints he)⟩)
    (.nil 0)

theorem threeJointsWalk_isPath (e : Fin 7) (he : 4 ≤ e) : (threeJointsWalk e he).IsPath := by
  simp [Walk.IsPath, threeJointsWalk, Walk.support]

theorem threeJointsWalk_openly {e f : Fin 7} (he : 4 ≤ e) (hf : 4 ≤ f) (hef : e ≠ f) :
    Walk.Openly (threeJointsWalk e he) (threeJointsWalk f hf) := by
  refine ⟨fun g hg hg' => ?_, fun z hz _ => ?_⟩
  · simp only [threeJointsWalk, Walk.edges, List.mem_singleton] at hg hg'
    exact hef (hg.symm.trans hg')
  · simpa [threeJointsWalk, Walk.support] using hz

/-- The cycle of two of the links from `1` to `0` is in `B̂`: both are joints of `J`. -/
theorem threeJoints_mem_unrollBalanced {e f : Fin 7} (he : 4 ≤ e) (hf : 4 ≤ f) (hef : e ≠ f) :
    (threeJointsWalk e he).edgeSet ∪ (threeJointsWalk f hf).edgeSet ∈
      threeJoints.contrabalanced.unrollBalanced 0 :=
  mem_unrollBalanced_of_rel threeJoints_isRollable.balancing
    ⟨1, _, (threeJointsWalk_openly he hf hef).isCycleWalk (by decide) (threeJointsWalk_isPath e he)
      (threeJointsWalk_isPath f hf), Walk.edgeSet_append_reverse _ _⟩
    (Or.inr (threeJoints_mem_offJoints he)) (Or.inr (threeJoints_mem_offJoints hf)) hef
    (Or.inl List.mem_cons_self) (Or.inr List.mem_cons_self)
    (Or.inr (Or.inr ⟨threeJoints_mem_offJoints he, threeJoints_mem_offJoints hf⟩))

example := unrollGraph_isLink_of_mem_sigmaStar (Or.inr (threeJoints_mem_offJoints (e := 4)
  (by decide)))

example := mem_sigmaStar_of_unrollGraph
  ((threeJointsWalk_isPath 4 (by decide)).isLink_of_mem List.mem_cons_self)
  (Or.inr (ite_eq_left (threeJoints_mem_offJoints (e := 4) (by decide))))

example := unbalancingRel_of_mem_unrollBalanced (threeJoints_mem_unrollBalanced (e := 4) (f := 5)
  (by decide) (by decide) (by decide)) (Or.inr (threeJoints_mem_offJoints (e := 4) (by decide)))
  (Or.inr (threeJoints_mem_offJoints (e := 5) (by decide))) (Or.inl List.mem_cons_self)
  (Or.inr List.mem_cons_self)

example := threeJointsWalk_openly (e := 4) (f := 6) (by decide) (by decide) (by decide)

example := threeJoints_mem_unrollBalanced (e := 5) (f := 6) (by decide) (by decide) (by decide)

end Transformer.GainMatroid

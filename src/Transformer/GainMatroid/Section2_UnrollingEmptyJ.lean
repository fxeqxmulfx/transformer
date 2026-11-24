module

public import Transformer.GainMatroid.Section2_UnrollingFailures

/-!
# The unrolling when `J` is empty

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5:

> "If `J` is empty, then set `(Ĝ, B̂) = (G, B)`."

This agrees with the definition of the unrolling, for the corrected `B̂` of `unrollBalanced`: when
`J` is empty, `(G, B)` is its own unrolling to `u` (`IsRollable.isUnrolling_self`), so every
unrolling has the graph `G` (`IsUnrolling.graph_eq_of_offJoints`) and the balanced cycles `B`
(`IsUnrolling.balanced_eq_of_offJoints`). A link at `u` on a cycle meets exactly one other edge of
the cycle at `u` (`IsCycle.exists_partner`). So a balanced cycle, which has no joint, meets each
unbalancing class in zero or two edges: `B ⊆ B̂` for every `J` (`IsRollable.balanced_subset`). When
`J` is empty, a cycle through `u` that is not a loop meets `Σ(u) = δ(u)` in two links, and these
lie in one unbalancing class exactly when the cycle is balanced (`IsRollable.balanced_iff`). A cycle
of at least two edges that misses `u` is balanced, and a loop leaves `B̂` exactly when it is a joint.
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

namespace OrientedGraph

variable {G : OrientedGraph V E}

open Walk

/-- A link `f` at `u` on a cycle meets exactly one other edge of the cycle at `u`. -/
theorem IsCycle.exists_partner {C : Set E} (hC : G.IsCycle C) {u : V} {f : E} (hf : f ∈ C)
    (hl : G.IsLink f) (hfu : G.tail f = u ∨ G.head f = u) :
    ∃ g ∈ C, g ≠ f ∧ (G.tail g = u ∨ G.head g = u) ∧
      ∀ c ∈ C, (G.tail c = u ∨ G.head c = u) → c = f ∨ c = g := by
  obtain ⟨a, W, hW, rfl⟩ := hC
  obtain ⟨s, t, hj, R, hR, hfR, hWR⟩ := hW.exists_isPath_insert hf
  have hst : s ≠ t := fun hst => by
    subst hst
    rcases hj with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩ <;> exact hl (h₁.trans h₂.symm)
  obtain ⟨x, P, hP, hux, hfP, hPR⟩ : ∃ (x : V) (P : G.Walk u x), P.IsPath ∧ u ≠ x ∧
      f ∉ P.edges ∧ P.edgeSet = R.edgeSet := by
    rcases hj.eq_or_eq hfu with rfl | rfl
    · exact ⟨t, R.reverse, hR.reverse, hst, fun h => hfR (mem_edges_reverse.1 h),
        R.edgeSet_reverse⟩
    · exact ⟨s, R, hR, hst.symm, hfR, rfl⟩
  rw [hWR, ← hPR]
  cases P with
  | nil => exact absurd rfl hux
  | cons g hg P' =>
    refine ⟨g, Or.inr List.mem_cons_self, fun h => hfP (by subst h; exact List.mem_cons_self),
      hg.tail_or_head, fun c hc hcu => ?_⟩
    rcases hc with rfl | hc
    · exact Or.inl rfl
    · exact Or.inr (hP.eq_of_mem_start hc hcu)

end OrientedGraph

open OrientedGraph

namespace BiasedGraph

variable {Ω Ω' : BiasedGraph V E} {u : V}

/-- With `J` empty no joint moves: `Ĝ = G`. -/
theorem unrollGraph_of_offJoints (hJ : Ω.offJoints u = ∅) : Ω.unrollGraph u = Ω.graph :=
  eq_of_tail_head rfl (funext fun e => ite_eq_right (hJ ▸ Set.notMem_empty e))

/-- `B ⊆ B̂`: a balanced cycle has no joint, so it is a cycle of `Ĝ`, and its edges in `Σ(u)` are
two links at `u` in one class or none, so it meets each class in zero or two edges. -/
theorem IsRollable.balanced_subset (hΩ : Ω.IsRollable u) : Ω.balanced ⊆ Ω.unrollBalanced u := by
  intro C hCB
  have hC := Ω.isCycle_of_mem C hCB
  have hJ : ∀ {x}, x ∈ C → x ∉ Ω.offJoints u := fun {x} hxC hxJ =>
    hxJ.1.not_mem_of_balanced hCB hxC
  have hS : ∀ {x}, x ∈ C → x ∈ Ω.sigmaStar u → x ∈ Ω.linkStar u := fun {x} hxC hx =>
    Or.resolve_right hx (hJ hxC)
  have hC' : (Ω.unrollGraph u).IsCycle C := by
    simpa using hC.map (G := Ω.unrollGraph u) id Function.injective_id
      fun x hx => ⟨rfl, ite_eq_right (hJ hx)⟩
  refine ⟨⟨hC', fun e he => ?_⟩, fun e hCe hj => hj.2 (hCe ▸ hCB)⟩
  rcases (C ∩ Ω.unbalancingClass u e).eq_empty_or_nonempty with h0 | ⟨f, hfC, hfe⟩
  · exact Or.inl (Set.encard_eq_zero.2 h0)
  have hfL := hS hfC hfe.1
  obtain ⟨g, hgC, hgf, hgu, huniq⟩ := hC.exists_partner hfC hfL.1 hfL.2
  have hgL : g ∈ Ω.linkStar u := by
    refine ⟨fun hll => hgf ?_, hgu⟩
    rw [hC.eq_singleton_of_loop hgC hll] at hfC
    exact (Set.mem_singleton_iff.1 hfC).symm
  have hge : g ∈ Ω.unbalancingClass u e := ⟨Or.inl hgL, hΩ.unbalancingRel_trans he hfe.1
    (Or.inl hgL) hfe.2 (Or.inr (Or.inl ⟨C, hCB, hfC, hgC⟩))⟩
  refine Or.inr ((congrArg Set.encard (Set.ext fun x => ⟨fun hx => ?_, ?_⟩)).trans
    (Set.encard_pair hgf.symm))
  · exact huniq x hx.1 (hS hx.1 hx.2.1).2
  · rintro (rfl | rfl)
    exacts [⟨hfC, hfe⟩, ⟨hgC, hge⟩]

/-- With `J` empty the corrected `B̂` is `B`: a cycle of `G` is balanced if and only if it meets
each unbalancing class in zero or two edges and is not a joint. -/
theorem IsRollable.unrollBalanced_eq (hΩ : Ω.IsRollable u) (hJ : Ω.offJoints u = ∅) :
    Ω.unrollBalanced u = Ω.balanced := by
  have hS : ∀ {e}, e ∈ Ω.sigmaStar u → e ∈ Ω.linkStar u := fun {e} he =>
    Or.resolve_right he fun hJe => Set.notMem_empty e (hJ ▸ hJe)
  refine Set.Subset.antisymm ?_ hΩ.balanced_subset
  · rintro C ⟨⟨hC, hcls⟩, hjoint⟩
    rw [unrollGraph_of_offJoints hJ] at hC
    by_contra hCB
    by_cases hloop : ∃ l ∈ C, Ω.graph.tail l = Ω.graph.head l
    · obtain ⟨l, hl, hll⟩ := hloop
      have hCl := hC.eq_singleton_of_loop hl hll
      exact hjoint l hCl ⟨hll, hCl ▸ hCB⟩
    push Not at hloop
    obtain ⟨c, hc⟩ := hC.nonempty
    obtain ⟨d, hd, hdc, -⟩ := hC.exists_partner hc (hloop c hc) (Or.inl rfl)
    obtain ⟨e, heC, heu⟩ := mem_verts.1 ((hΩ.unique u).2 rfl C ⟨hC, hCB⟩
      ((Set.encard_pair hdc.symm).ge.trans (Set.encard_le_encard (Set.pair_subset hc hd))))
    have heL : e ∈ Ω.linkStar u := ⟨hloop e heC, heu⟩
    obtain ⟨g, ⟨hgC, hgS, hrel⟩, hge⟩ : ∃ g ∈ C ∩ Ω.unbalancingClass u e, g ≠ e := by
      refine Set.exists_ne_of_one_lt_encard ?_ e
      rcases hcls e (Or.inl heL) with h0 | h2
      · exact absurd h0 (Set.encard_ne_zero.2 ⟨e, heC, Or.inl heL, Or.inl rfl⟩)
      · rw [h2]
        norm_num
    rcases hrel with rfl | ⟨D, hD, heD, hgD⟩ | ⟨heJ, -⟩
    · exact hge rfl
    · exact hCB ((hΩ.balanced_iff heL (hS hgS) hge.symm hC (Ω.isCycle_of_mem D hD) heC hgC heD
        hgD).2 hD)
    · exact Set.notMem_empty e (hJ ▸ heJ)

/-- §2.5: "If `J` is empty, then set `(Ĝ, B̂) = (G, B)`": then `(G, B)` is its own unrolling. -/
theorem IsRollable.isUnrolling_self (hΩ : Ω.IsRollable u) (hJ : Ω.offJoints u = ∅) :
    Ω.IsUnrolling u Ω :=
  ⟨(unrollGraph_of_offJoints hJ).symm, (hΩ.unrollBalanced_eq hJ).symm⟩

namespace IsUnrolling

/-- §2.5: if `J` is empty then `Ĝ = G`. -/
theorem graph_eq_of_offJoints (h : Ω.IsUnrolling u Ω') (hJ : Ω.offJoints u = ∅) :
    Ω'.graph = Ω.graph :=
  h.graph_eq.trans (unrollGraph_of_offJoints hJ)

/-- §2.5: "If `J` is empty, then set `(Ĝ, B̂) = (G, B)`": the corrected `B̂` is then `B`. -/
theorem balanced_eq_of_offJoints (h : Ω.IsUnrolling u Ω') (hΩ : Ω.IsRollable u)
    (hJ : Ω.offJoints u = ∅) : Ω'.balanced = Ω.balanced :=
  h.balanced_eq.trans (hΩ.unrollBalanced_eq hJ)

end IsUnrolling

end BiasedGraph

/-- Two digons at `0` and a joint at `0`: a rollable graph with `J` empty. -/
def centreLoop : OrientedGraph (Fin 3) (Fin 5) := ⟨![0, 0, 0, 0, 0], ![1, 1, 2, 2, 0]⟩

theorem centreLoop_isRollable : centreLoop.contrabalanced.IsRollable 0 :=
  isRollable_of_digons (a := 0) (a' := 1) (b := 2) (b' := 3) (Or.inl ⟨rfl, rfl⟩)
    (Or.inl ⟨rfl, rfl⟩) (by decide) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide)
    (by decide)

theorem centreLoop_offJoints : centreLoop.contrabalanced.offJoints 0 = ∅ := by
  have h0 : ∀ e : Fin 5, centreLoop.tail e = 0 := by decide
  exact Set.eq_empty_of_forall_notMem fun e he => he.2 (h0 e)

example := (isCycle_pair (G := pumpkin) (e := 0) (f := 1) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩)
  (by decide) (by decide)).exists_partner (u := 0) (f := 0) (by simp) (by decide) (Or.inl rfl)

example := BiasedGraph.unrollGraph_of_offJoints centreLoop_offJoints

example := centreLoop_isRollable.balanced_subset

example := centreLoop_isRollable.unrollBalanced_eq centreLoop_offJoints

example := centreLoop_isRollable.isUnrolling_self centreLoop_offJoints

example := (centreLoop_isRollable.isUnrolling_self centreLoop_offJoints).graph_eq_of_offJoints
  centreLoop_offJoints

example := (centreLoop_isRollable.isUnrolling_self centreLoop_offJoints).balanced_eq_of_offJoints
  centreLoop_isRollable centreLoop_offJoints

end Transformer.GainMatroid

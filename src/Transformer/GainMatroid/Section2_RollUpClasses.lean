module

public import Transformer.GainMatroid.Section2_RollUpClaims

/-!
# The roll-ups keep the unbalancing classes

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5:

> "It is a straightforward check that for each unbalancing class `U ∈ Σ_Ĝ(u)/∼`,
> `Σ_{G_U}(u)/∼ = Σ_Ĝ(u)/∼`."

`IsUnrolling.unbalancingClass_rollUp` proves it for the class of each edge, and
`IsUnrolling.sigmaStar_rollUp` for the edges. A cycle of `Ĝ` through two edges of `Σ(u)` meets
`Σ(u)` in these two, so it is balanced exactly when they are related in `(G, B)`
(`IsUnrolling.mem_balanced_iff`). Hence `∼` is transitive on `Σ_Ĝ(u)` with no assumption on the
joints (`IsUnrolling.unbalancingRel_trans`, through a theta). In `G_U` the joints off `u` are the
edges of `U` (`IsUnrolling.offJoints_rollUp`) and the balanced cycles are those of `Ĝ` that avoid
`U`; a balanced cycle of `Ĝ` through an edge of `U` has both its edges at `u` in `U`.
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

open OrientedGraph

namespace BiasedGraph

variable {Ω Ω' : BiasedGraph V E} {u : V}

namespace IsUnrolling

/-- An edge of `Σ(u)` is a link at `u` in `Ĝ`. -/
theorem mem_linkStar (h : Ω.IsUnrolling u Ω') {e : E} (he : e ∈ Ω.sigmaStar u) :
    e ∈ Ω'.linkStar u := by
  rw [← h.sigmaStar_eq] at he
  exact Or.resolve_right he fun hJ => by rwa [h.offJoints_eq_empty] at hJ

/-- A cycle of `Ĝ` through two edges of `Σ(u)` is balanced exactly when they are related in
`(G, B)`: they are its only edges in `Σ(u)`, so it meets each class in both of them or in none. -/
theorem mem_balanced_iff (h : Ω.IsUnrolling u Ω') (hΩ : Ω.IsLinkBalancingVertex u) {e f : E}
    (he : e ∈ Ω.sigmaStar u) (hf : f ∈ Ω.sigmaStar u) (hne : e ≠ f) {C : Set E}
    (hC : Ω'.graph.IsCycle C) (heC : e ∈ C) (hfC : f ∈ C) :
    C ∈ Ω'.balanced ↔ Ω.UnbalancingRel u e f := by
  refine ⟨fun hCB => h.unbalancingRel_of_rel he hf (Or.inr (Or.inl ⟨C, hCB, heC, hfC⟩)),
    fun hef => ?_⟩
  have heL := h.mem_linkStar he
  obtain ⟨p, -, -, -, huniq⟩ := hC.exists_partner heC heL.1 heL.2
  have hfp := (huniq f hfC (h.end_of_mem_sigmaStar hf)).resolve_left (Ne.symm hne)
  have hS : ∀ x ∈ C, x ∈ Ω.sigmaStar u → x = e ∨ x = f := fun x hx hxS =>
    (huniq x hx (h.end_of_mem_sigmaStar hxS)).imp_right fun hxp => hxp.trans hfp.symm
  rw [h.balanced_eq]
  refine ⟨⟨h.graph_eq ▸ hC, fun k hk => ?_⟩, fun c hc _ => hne ?_⟩
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

/-- `∼` is transitive on `Σ_Ĝ(u)`: pairs related in `Ĝ` are related in `(G, B)`, so every cycle of
`Ĝ` through either pair is balanced, and a theta gives a balanced cycle through the outer two edges
(`exists_balanced_of_links`). -/
theorem unbalancingRel_trans (h : Ω.IsUnrolling u Ω') (hΩ : Ω.IsLinkBalancingVertex u) {e f g : E}
    (he : e ∈ Ω.sigmaStar u) (hf : f ∈ Ω.sigmaStar u) (hg : g ∈ Ω.sigmaStar u)
    (hef : Ω'.UnbalancingRel u e f) (hfg : Ω'.UnbalancingRel u f g) : Ω'.UnbalancingRel u e g := by
  have hJ : ∀ {x}, x ∉ Ω'.offJoints u := fun hx => by
    rw [h.offJoints_eq_empty] at hx
    exact hx
  by_cases heg : e = g
  · exact Or.inl heg
  rcases hef with rfl | ⟨C, hC, heC, hfC⟩ | ⟨heJ, -⟩
  · exact hfg
  · rcases hfg with rfl | ⟨D, hD, hfD, hgD⟩ | ⟨hfJ, -⟩
    · exact Or.inr (Or.inl ⟨C, hC, heC, hfC⟩)
    · by_cases hef : e = f
      · subst hef
        exact Or.inr (Or.inl ⟨D, hD, hfD, hgD⟩)
      by_cases hfg : f = g
      · subst hfg
        exact Or.inr (Or.inl ⟨C, hC, heC, hfC⟩)
      have hC' := Ω'.isCycle_of_mem _ hC
      have hD' := Ω'.isCycle_of_mem _ hD
      have hEF := (h.mem_balanced_iff hΩ he hf hef hC' heC hfC).1 hC
      have hFG := (h.mem_balanced_iff hΩ hf hg hfg hD' hfD hgD).1 hD
      exact Or.inr (Or.inl (exists_balanced_of_links hef hfg heg (h.end_of_mem_sigmaStar he)
        (h.end_of_mem_sigmaStar hf) (h.end_of_mem_sigmaStar hg) hC' hD' heC hfC hfD hgD
        (fun B hB heB hfB => (h.mem_balanced_iff hΩ he hf hef hB heB hfB).2 hEF)
        fun B hB hfB hgB => (h.mem_balanced_iff hΩ hf hg hfg hB hfB hgB).2 hFG))
    · exact (hJ hfJ).elim
  · exact (hJ heJ).elim

/-- In `G_U` the joints off `u` are the edges of `U`: each becomes an unbalanced loop at its other
end, and `Ĝ` has no joint off `u`. -/
theorem offJoints_rollUp (h : Ω.IsUnrolling u Ω') {U : Set E} (hU : U ∈ Ω'.unbalancingClasses u) :
    (Ω'.rollUp u U).offJoints u = U := by
  ext e
  constructor
  · rintro ⟨⟨hloop, hnot⟩, hoff⟩
    by_contra heU
    have hmem : e ∈ Ω'.offJoints u := ⟨⟨(rollUpGraph_tail_of_notMem heU).symm.trans
      (hloop.trans (rollUpGraph_head_of_notMem heU)),
      fun hb => hnot ⟨hb, Set.disjoint_singleton_left.2 heU⟩⟩,
      fun ht => hoff ((rollUpGraph_tail_of_notMem heU).trans ht)⟩
    rw [h.offJoints_eq_empty] at hmem
    exact hmem
  · intro heU
    exact ⟨⟨(rollUpGraph_tail_of_mem heU).trans (rollUpGraph_head_of_mem heU).symm,
      fun hb => Set.disjoint_left.1 hb.2 (Set.mem_singleton e) heU⟩,
      (rollUpGraph_tail_of_mem heU).trans_ne (otherEnd_ne (h.subset_linkStar hU heU).1)⟩

/-- §2.5: "`Σ_{G_U}(u)/∼ = Σ_Ĝ(u)/∼`", for each edge of `Σ(u)`; `sigmaStar_rollUp` says the two
sets of edges agree. -/
theorem unbalancingClass_rollUp (h : Ω.IsUnrolling u Ω') (hΩ : Ω.IsLinkBalancingVertex u)
    {U : Set E} (hU : U ∈ Ω'.unbalancingClasses u) {f : E} (hf : f ∈ Ω'.sigmaStar u) :
    (Ω'.rollUp u U).unbalancingClass u f = Ω'.unbalancingClass u f := by
  have hS : ∀ {x}, x ∈ Ω'.sigmaStar u → x ∈ Ω.sigmaStar u := fun hx => by
    rw [← h.sigmaStar_eq]
    exact hx
  have hJU := h.offJoints_rollUp hU
  have hsub := h.subset_linkStar hU
  ext g
  change g ∈ (Ω'.rollUp u U).sigmaStar u ∧ (Ω'.rollUp u U).UnbalancingRel u f g ↔
    g ∈ Ω'.sigmaStar u ∧ Ω'.UnbalancingRel u f g
  rw [h.sigmaStar_rollUp hU]
  refine and_congr_right fun hg => ?_
  obtain ⟨e₀, he₀, rfl⟩ := hU
  constructor
  · rintro (rfl | ⟨C, ⟨hC, -⟩, hfC, hgC⟩ | ⟨hfJ, hgJ⟩)
    · exact Or.inl rfl
    · exact Or.inr (Or.inl ⟨C, hC, hfC, hgC⟩)
    · rw [hJU] at hfJ hgJ
      exact h.unbalancingRel_trans hΩ (hS hf) (hS he₀) (hS hg) hfJ.2.symm hgJ.2
  · rintro (rfl | ⟨C, hC, hfC, hgC⟩ | ⟨hfJ, -⟩)
    · exact Or.inl rfl
    · by_cases hCU : Disjoint C (Ω'.unbalancingClass u e₀)
      · exact Or.inr (Or.inl ⟨C, ⟨hC, hCU⟩, hfC, hgC⟩)
      by_cases hfg : f = g
      · exact Or.inl hfg
      obtain ⟨x, hxC, hxU⟩ := Set.not_disjoint_iff.1 hCU
      have hfL := h.mem_linkStar (hS hf)
      obtain ⟨p, -, -, -, huniq⟩ := (Ω'.isCycle_of_mem C hC).exists_partner hfC hfL.1 hfL.2
      have hgp := (huniq g hgC (h.end_of_mem_sigmaStar (hS hg))).resolve_left (Ne.symm hfg)
      have hrel : Ω'.UnbalancingRel u f g := Or.inr (Or.inl ⟨C, hC, hfC, hgC⟩)
      rcases huniq x hxC (hsub hxU).2 with rfl | hxp
      · have hgU : g ∈ Ω'.unbalancingClass u e₀ :=
          ⟨hg, h.unbalancingRel_trans hΩ (hS he₀) (hS hf) (hS hg) hxU.2 hrel⟩
        exact Or.inr (Or.inr ⟨hJU.symm ▸ hxU, hJU.symm ▸ hgU⟩)
      · have hgU : g ∈ Ω'.unbalancingClass u e₀ := by
          rw [hgp, ← hxp]
          exact hxU
        have hfU : f ∈ Ω'.unbalancingClass u e₀ :=
          ⟨hf, h.unbalancingRel_trans hΩ (hS he₀) (hS hg) (hS hf) hgU.2 hrel.symm⟩
        exact Or.inr (Or.inr ⟨hJU.symm ▸ hfU, hJU.symm ▸ hgU⟩)
    · rw [h.offJoints_eq_empty] at hfJ
      exact hfJ.elim

end IsUnrolling

end BiasedGraph

open BiasedGraph

/-- The edge `0` of `twoJoints` is in `Σ(0)`. -/
theorem twoJoints_zero_mem : (0 : Fin 6) ∈ twoJoints.contrabalanced.sigmaStar 0 :=
  Or.inl ⟨by decide, Or.inl rfl⟩

example := twoJoints_isUnrolling.mem_linkStar twoJoints_zero_mem

example := twoJoints_isUnrolling.mem_balanced_iff twoJoints_isRollable.balancing twoJoints_zero_mem
  (f := 1) (Or.inl ⟨by decide, Or.inl rfl⟩) (by decide) (isCycle_pair (G := twoJointsHat)
    (e := 0) (f := 1) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)) (by simp)
  (by simp)

example := twoJoints_isUnrolling.unbalancingRel_trans twoJoints_isRollable.balancing
  twoJoints_zero_mem twoJoints_zero_mem twoJoints_zero_mem (Or.inl rfl) (Or.inl rfl)

example := twoJoints_isUnrolling.offJoints_rollUp twoJointsHat_class_mem

example := twoJoints_isUnrolling.unbalancingClass_rollUp twoJoints_isRollable.balancing
  twoJointsHat_class_mem twoJointsHat_zero_mem

end Transformer.GainMatroid

module

public import Transformer.GainMatroid.Section2_UnrollingClasses
public import Transformer.GainMatroid.Section2_UnrollingEmptyJ

/-!
# The unrolling keeps the unbalancing classes

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5:

> "`Σ_Ĝ(u)/∼ = Σ_G(u)/∼`; that is, the unbalancing classes of `Σ_G(u)` and of `Σ_Ĝ(u)` are the
> same."

The claim fails as printed (`unbalancingClass_printed_fails`); `IsUnrolling.unbalancingClass_eq`
proves it when any two joints of `J` lie in one component of `G − u` (`JointsConnected`). Two edges
of `Σ(u)` on a balanced cycle of `(Ĝ, B̂)` are its two edges at `u`, so they lie in one class of
`(G, B)` (`IsUnrolling.unbalancingRel_of_rel`). Conversely a balanced cycle of `(G, B)` is one of
`(Ĝ, B̂)` (`IsRollable.balanced_subset`). Two joints of `J` joined by a path off `u` close, with
their new links to `u`, a cycle of `Ĝ` that meets `J` in them and no other class
(`IsUnrolling.unbalancingRel_of_offJoints`).
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

namespace OrientedGraph

variable {G : OrientedGraph V E}

/-- A walk from a vertex other than `u` whose edges miss `u` misses `u`. -/
theorem Walk.notMem_support_of_forall {a b u : V} (p : G.Walk a b) (ha : a ≠ u)
    (hp : ∀ g ∈ p.edges, G.tail g ≠ u ∧ G.head g ≠ u) : u ∉ p.support := by
  induction p with
  | nil a => exact fun h => ha (List.mem_singleton.1 h).symm
  | @cons x y z g hg p ih =>
    have hy : y ≠ u := by
      obtain ⟨h₁, h₂⟩ := hp g List.mem_cons_self
      rcases hg with ⟨-, hb⟩ | ⟨hb, -⟩
      · exact fun h => h₂ (hb.trans h)
      · exact fun h => h₁ (hb.trans h)
    intro hu
    rcases List.mem_cons.1 hu with rfl | hu
    · exact ha rfl
    · exact ih hy (fun g' hg' => hp g' (List.mem_cons_of_mem _ hg')) hu

end OrientedGraph

open OrientedGraph

namespace BiasedGraph

variable {Ω Ω' : BiasedGraph V E} {u : V}

namespace IsUnrolling

/-- Two edges of `Σ(u)` on a balanced cycle of the unrolling are its two edges at `u`, so they lie
in one unbalancing class of `(G, B)`. -/
theorem unbalancingRel_of_rel (h : Ω.IsUnrolling u Ω') {e f : E} (he : e ∈ Ω.sigmaStar u)
    (hf : f ∈ Ω.sigmaStar u) (hr : Ω'.UnbalancingRel u e f) : Ω.UnbalancingRel u e f := by
  rcases hr with rfl | ⟨C, hC, heC, hfC⟩ | ⟨heJ, -⟩
  · exact Or.inl rfl
  · by_cases hef : e = f
    · exact Or.inl hef
    have hC' : C ∈ Ω.unrollBalanced u := h.balanced_eq ▸ hC
    obtain ⟨⟨-, hcls⟩, -⟩ := hC'
    obtain ⟨g, ⟨hgC, hge⟩, hgne⟩ : ∃ g ∈ C ∩ Ω.unbalancingClass u e, g ≠ e := by
      refine Set.exists_ne_of_one_lt_encard ?_ e
      rcases hcls e he with h0 | h2
      · exact absurd h0 (Set.encard_ne_zero.2 ⟨e, heC, he, Or.inl rfl⟩)
      · rw [h2]
        norm_num
    have he' : e ∈ Ω'.sigmaStar u := by
      rw [h.sigmaStar_eq]
      exact he
    have heL : e ∈ Ω'.linkStar u := Or.resolve_right he' fun hJ => by
      rw [h.offJoints_eq_empty] at hJ
      exact hJ
    obtain ⟨p, -, -, -, huniq⟩ := (Ω'.isCycle_of_mem C hC).exists_partner heC heL.1 heL.2
    have hgp := (huniq g hgC (h.end_of_mem_sigmaStar hge.1)).resolve_left hgne
    have hfp := (huniq f hfC (h.end_of_mem_sigmaStar hf)).resolve_left (Ne.symm hef)
    exact hgp.trans hfp.symm ▸ hge.2
  · rw [h.offJoints_eq_empty] at heJ
    exact (Set.notMem_empty e heJ).elim

/-- Two joints of `J` joined by a path off `u` lie on a balanced cycle of the unrolling: with their
new links to `u` the path closes a cycle of `Ĝ` that meets `J` in them and no other class. -/
theorem unbalancingRel_of_offJoints (h : Ω.IsUnrolling u Ω') (hJ : Ω.JointsConnected u)
    {e f : E} (he : e ∈ Ω.offJoints u) (hf : f ∈ Ω.offJoints u) : Ω'.UnbalancingRel u e f := by
  by_cases hef : e = f
  · exact Or.inl hef
  obtain ⟨p, hpu⟩ := hJ e he f hf
  obtain ⟨q, hq, hqp⟩ := p.exists_isPath
  have hqe : ∀ x ∈ q.edges, Ω.graph.IsLink x ∧ Ω.graph.tail x ≠ u ∧ Ω.graph.head x ≠ u :=
    fun x hx => ⟨hq.isLink_of_mem hx,
      fun ht => hpu (Walk.mem_support_of_mem_edges (hqp x hx) (Or.inl ht)),
      fun hh => hpu (Walk.mem_support_of_mem_edges (hqp x hx) (Or.inr hh))⟩
  obtain ⟨r, hr₁, hr₂⟩ := q.exists_map (G := Ω'.graph) id fun x hx =>
    ⟨h.tail_eq x, h.head_of_notMem fun hxJ => (hqe x hx).1 hxJ.1.1⟩
  rw [List.map_id] at hr₁
  have hr : r.IsPath := by
    unfold Walk.IsPath
    rw [hr₂]
    exact hq
  have hru : u ∉ r.support := by
    rw [hr₂]
    exact q.notMem_support_of_forall he.2 fun x hx => (hqe x hx).2
  have hC := isCycle_pair_union (G := Ω'.graph) (Or.inl ⟨h.tail_eq e, h.head_of_mem he⟩)
    (Or.inl ⟨h.tail_eq f, h.head_of_mem hf⟩) hef hr hru
  have hS : ∀ x ∈ ({e, f} ∪ r.edgeSet : Set E), x ∈ Ω.sigmaStar u → x = e ∨ x = f := by
    rintro x (hx | hx) hxS
    · exact hx
    · have hx' : x ∈ q.edges := by
        rw [← hr₁]
        exact hx
      obtain ⟨hl, ht, hh⟩ := hqe x hx'
      rcases hxS with ⟨-, hu⟩ | hxJ
      · exact (hu.elim ht hh).elim
      · exact (hl hxJ.1.1).elim
  refine Or.inr (Or.inl ⟨{e, f} ∪ r.edgeSet, ?_, Or.inl (Or.inl rfl), Or.inl (Or.inr rfl)⟩)
  rw [h.balanced_eq]
  refine ⟨⟨h.graph_eq ▸ hC, fun g hg => ?_⟩, fun c hc _ => hef ?_⟩
  · by_cases hgJ : g ∈ Ω.offJoints u
    · rw [unbalancingClass_of_mem_offJoints hgJ]
      refine Or.inr ((congrArg Set.encard (Set.ext fun x => ⟨fun hx => ?_, ?_⟩)).trans
        (Set.encard_pair hef))
      · exact hS x hx.1 (Or.inr hx.2)
      · rintro (rfl | rfl)
        exacts [⟨Or.inl (Or.inl rfl), he⟩, ⟨Or.inl (Or.inr rfl), hf⟩]
    · refine Or.inl (Set.encard_eq_zero.2 (Set.eq_empty_of_forall_notMem fun x hx => ?_))
      obtain ⟨hxC, hxS, hxr⟩ := hx
      have hxJ : x ∈ Ω.offJoints u := by
        rcases hS x hxC hxS with rfl | rfl
        exacts [he, hf]
      rcases hxr with rfl | ⟨D, hD, -, hxD⟩ | ⟨hgJ', -⟩
      · exact hgJ hxJ
      · exact hxJ.1.not_mem_of_balanced hD hxD
      · exact hgJ hgJ'
  · have he' : e ∈ ({c} : Set E) := by
      rw [← hc]
      exact Or.inl (Or.inl rfl)
    have hf' : f ∈ ({c} : Set E) := by
      rw [← hc]
      exact Or.inl (Or.inr rfl)
    exact (Set.mem_singleton_iff.1 he').trans (Set.mem_singleton_iff.1 hf').symm

/-- §2.5: "the unbalancing classes of `Σ_G(u)` and of `Σ_Ĝ(u)` are the same", for joints of `J` in
one component of `G − u`. -/
theorem unbalancingClass_eq (h : Ω.IsUnrolling u Ω') (hΩ : Ω.IsRollable u)
    (hJ : Ω.JointsConnected u) {e : E} (he : e ∈ Ω.sigmaStar u) :
    Ω'.unbalancingClass u e = Ω.unbalancingClass u e := by
  ext f
  change f ∈ Ω'.sigmaStar u ∧ Ω'.UnbalancingRel u e f ↔ f ∈ Ω.sigmaStar u ∧ Ω.UnbalancingRel u e f
  rw [h.sigmaStar_eq]
  refine and_congr_right fun hf => ⟨h.unbalancingRel_of_rel he hf, ?_⟩
  rintro (rfl | ⟨D, hD, heD, hfD⟩ | ⟨heJ, hfJ⟩)
  · exact Or.inl rfl
  · exact Or.inr (Or.inl ⟨D, h.balanced_eq ▸ hΩ.balanced_subset hD, heD, hfD⟩)
  · exact h.unbalancingRel_of_offJoints hJ heJ hfJ

end IsUnrolling

end BiasedGraph

/-- The single joint of `centreJoint` off `0` is connected to itself off `0`. -/
theorem centreJoint_jointsConnected : centreJoint.contrabalanced.JointsConnected 0 := by
  intro e he f hf
  obtain rfl := (centreJoint_mem_offJoints e).1 he
  obtain rfl := (centreJoint_mem_offJoints f).1 hf
  exact ⟨.nil _, fun h => he.2 (List.mem_singleton.1 h).symm⟩

example := Walk.notMem_support_of_forall (G := triangle) (u := 1) triangleLast (by decide)
  (by decide)

example := centreJoint_isUnrolling.unbalancingRel_of_rel (centreJoint_mem_sigmaStar (c := 0)
  (by decide)) (centreJoint_mem_sigmaStar (c := 0) (by decide)) (Or.inl rfl)

example := centreJoint_isUnrolling.unbalancingRel_of_offJoints centreJoint_jointsConnected
  ((centreJoint_mem_offJoints 5).2 rfl) ((centreJoint_mem_offJoints 5).2 rfl)

example := centreJoint_isUnrolling.unbalancingClass_eq centreJoint_isRollable
  centreJoint_jointsConnected (centreJoint_mem_sigmaStar (c := 5) (by decide))

end Transformer.GainMatroid

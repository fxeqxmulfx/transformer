module

public import Transformer.GainMatroid.Section2_RollUps
public import Transformer.GainMatroid.Section2_UnrollingSameClasses

/-!
# What §2.5 claims of the roll-ups

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5. The source proves these by "comparing
circuits" or calls them straightforward, and cites [AlmostBalancedReps, Proposition 2.2], which is
not among the local sources, for the first:

> "`F(G_U, B_U) = F(G, B) = F(Ĝ, B̂)`" (`IsUnrolling.isFrameCircuit_rollUp`, in
> `Section2_RollUpCircuits`);
> "for each unbalancing class `U`, `Σ_{G_U}(u)/∼ = Σ_Ĝ(u)/∼`" (`IsUnrolling.sigmaStar_rollUp`, and
> `IsUnrolling.unbalancingClass_rollUp` in `Section2_RollUpClasses`);
> "`J` is an unbalancing class of `Σ_Ĝ(u)`, and […] `(G_J, B_J) = (G, B)`"
> (`IsUnrolling.offJoints_mem_unbalancingClasses`, `IsUnrolling.graph_rollUp_offJoints`,
> `IsUnrolling.balanced_rollUp_offJoints`).

That `J` is a class needs its joints in one component of `G − u`, by
`unbalancingClass_printed_fails`, so the statement assumes `JointsConnected`, and that `J` is
nonempty, as no class is empty (`empty_notMem_unbalancingClasses`).

Matroid equalities are stated on circuits, which determine a matroid (`Matroid.ext_isCircuit`).
The source's special case, a biased graph balanced after removing its joints, is in
`Section2_LoopUnrolling` and `Section2_LoopCircuits`.
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

open OrientedGraph

namespace BiasedGraph

variable {Ω Ω' : BiasedGraph V E} {u : V}

/-- No unbalancing class is empty, so an empty `J` is not one. -/
theorem empty_notMem_unbalancingClasses : ∅ ∉ Ω.unbalancingClasses u := by
  rintro ⟨e, he, h0⟩
  have hmem : e ∈ Ω.unbalancingClass u e := ⟨he, Or.inl rfl⟩
  rw [← h0] at hmem
  exact hmem

namespace IsUnrolling

/-- §2.5: the edges of `Σ_{G_U}(u)` and of `Σ_Ĝ(u)` are the same. -/
theorem sigmaStar_rollUp (h : Ω.IsUnrolling u Ω') {U : Set E} (hU : U ∈ Ω'.unbalancingClasses u) :
    (Ω'.rollUp u U).sigmaStar u = Ω'.sigmaStar u := by
  have hsub := h.subset_linkStar hU
  have hJ := h.offJoints_eq_empty
  ext f
  constructor
  · rintro (⟨hl, hend⟩ | ⟨⟨hloop, hnot⟩, hoff⟩)
    · by_cases hfU : f ∈ U
      · exact Or.inl (hsub hfU)
      · refine Or.inl ⟨fun hc => hl ((rollUpGraph_tail_of_notMem hfU).trans
          (hc.trans (rollUpGraph_head_of_notMem hfU).symm)), ?_⟩
        exact hend.imp (fun ht => (rollUpGraph_tail_of_notMem hfU).symm.trans ht)
          fun hh => (rollUpGraph_head_of_notMem hfU).symm.trans hh
    · by_cases hfU : f ∈ U
      · exact Or.inl (hsub hfU)
      · exfalso
        have hmem : f ∈ Ω'.offJoints u := by
          refine ⟨⟨(rollUpGraph_tail_of_notMem hfU).symm.trans
            (hloop.trans (rollUpGraph_head_of_notMem hfU)),
            fun hb => hnot ⟨hb, Set.disjoint_singleton_left.2 hfU⟩⟩, ?_⟩
          exact fun ht => hoff ((rollUpGraph_tail_of_notMem hfU).trans ht)
        rw [hJ] at hmem
        exact hmem
  · rintro (hl | hj)
    · by_cases hfU : f ∈ U
      · refine Or.inr ⟨⟨(rollUpGraph_tail_of_mem hfU).trans (rollUpGraph_head_of_mem hfU).symm,
          fun hb => Set.disjoint_left.1 hb.2 (Set.mem_singleton f) hfU⟩, ?_⟩
        exact (rollUpGraph_tail_of_mem hfU).trans_ne (otherEnd_ne hl.1)
      · refine Or.inl ⟨fun hc => hl.1 ((rollUpGraph_tail_of_notMem hfU).symm.trans
          (hc.trans (rollUpGraph_head_of_notMem hfU))), ?_⟩
        exact hl.2.imp (fun ht => (rollUpGraph_tail_of_notMem hfU).trans ht)
          fun hh => (rollUpGraph_head_of_notMem hfU).trans hh
    · rw [hJ] at hj
      exact hj.elim

/-- §2.5: "`J` is an unbalancing class of `Σ_Ĝ(u)`", for `J` nonempty in one component of
`G − u`; the printed claim fails otherwise (`unbalancingClass_printed_fails`). `J` is the class in
`(G, B)` of each of its joints, and the unrolling keeps the classes (`unbalancingClass_eq`). -/
theorem offJoints_mem_unbalancingClasses (h : Ω.IsUnrolling u Ω') (hΩ : Ω.IsRollable u)
    (hJ : Ω.JointsConnected u) (hne : (Ω.offJoints u).Nonempty) :
    Ω.offJoints u ∈ Ω'.unbalancingClasses u := by
  obtain ⟨e, he⟩ := hne
  refine ⟨e, by rw [h.sigmaStar_eq]; exact Or.inr he, ?_⟩
  rw [h.unbalancingClass_eq hΩ hJ (Or.inr he), unbalancingClass_of_mem_offJoints he]

/-- §2.5: the graph of `(G_J, B_J)` is `G`. -/
theorem graph_rollUp_offJoints (h : Ω.IsUnrolling u Ω') :
    (Ω'.rollUp u (Ω.offJoints u)).graph = Ω.graph := by
  change Ω'.graph.rollUpGraph u (Ω.offJoints u) = Ω.graph
  rw [h.graph_eq]
  exact Ω.rollUpGraph_unrollGraph u

/-- §2.5: the balanced cycles of `(G_J, B_J)` are `B`, the cycles of `B̂` that avoid `J`. -/
theorem balanced_rollUp_offJoints (h : Ω.IsUnrolling u Ω') (hΩ : Ω.IsRollable u) :
    (Ω'.rollUp u (Ω.offJoints u)).balanced = Ω.balanced :=
  (congrArg (fun B : Set (Set E) => {C ∈ B | Disjoint C (Ω.offJoints u)}) h.balanced_eq).trans
    hΩ.unrollBalanced_sep_disjoint

end IsUnrolling

end BiasedGraph

open BiasedGraph

/-- The edge `0` of `twoJointsHat` is a link at `0`. -/
theorem twoJointsHat_zero_mem : (0 : Fin 6) ∈ twoJointsHat.contrabalanced.sigmaStar 0 :=
  Or.inl ⟨by decide, Or.inl rfl⟩

/-- The class of the edge `0` of `twoJointsHat`. -/
theorem twoJointsHat_class_mem : twoJointsHat.contrabalanced.unbalancingClass 0 0 ∈
    twoJointsHat.contrabalanced.unbalancingClasses 0 :=
  ⟨0, twoJointsHat_zero_mem, rfl⟩

example := rollUpGraph_tail_of_mem (G := triangle) (u := 0) (U := {0}) (e := 0) rfl
example := rollUpGraph_head_of_mem (G := triangle) (u := 0) (U := {0}) (e := 0) rfl
example := rollUpGraph_tail_of_notMem (G := triangle) (u := 0) (U := ∅) (e := 0)
  (Set.notMem_empty _)
example := rollUpGraph_head_of_notMem (G := triangle) (u := 0) (U := ∅) (e := 0)
  (Set.notMem_empty _)
example := otherEnd_ne (G := triangle) (u := 0) (e := 0) (by decide)
example := pumpkin_isTheta.exists_of_agree (G := pumpkin) (fun _ _ => ⟨rfl, rfl⟩)
  (fun _ _ => ⟨rfl, rfl⟩) fun _ _ => ⟨rfl, rfl⟩
example := twoJoints_isUnrolling.subset_linkStar twoJointsHat_class_mem
example := twoJoints_isUnrolling.rollUp_injOn
example := twoJoints_isUnrolling.rollUp_ne twoJointsHat_class_mem
example := twoJoints_isUnrolling.ncard_rollUpFamily
example := twoJoints_isUnrolling.sigmaStar_rollUp twoJointsHat_class_mem
example := twoJoints_isUnrolling.graph_rollUp_offJoints
example := twoJoints_isUnrolling.balanced_rollUp_offJoints twoJoints_isRollable
example := centreJoint_isUnrolling.offJoints_mem_unbalancingClasses centreJoint_isRollable
  centreJoint_jointsConnected ⟨5, (centreJoint_mem_offJoints 5).2 rfl⟩

end Transformer.GainMatroid

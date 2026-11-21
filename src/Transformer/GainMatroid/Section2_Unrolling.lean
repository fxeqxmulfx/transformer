module

public import Transformer.GainMatroid.Section2_UnbalancingRelation

/-!
# The unrolling of an almost-balanced biased graph

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5:

> "Now consider the biased graph `(Ĝ, B̂)` obtained from `(G, B)` by replacing each joint `e ∈ J`
> incident to a vertex `v ≠ u` with a `uv`-link. Define `B̂` to be the set of those cycles having
> intersection of size 0 or 2 with each unbalancing class of `Σ_G(u)`. It is straightforward to
> check by comparing circuits that `F(Ĝ, B̂) = F(G, B)`."

Each new link is oriented here from the joint's vertex to `u`; the source does not orient it.

**Two corrections.** A joint at `u` is not in `J`, stays a loop of `Ĝ`, and meets every class in no
edge. When `J` is nonempty the printed `B̂` (`printedUnrollBalanced`) therefore makes it balanced,
and then `F(Ĝ, B̂) ≠ F(G, B)` (`printedUnrolling_fails`, in `Section2_UnrollingFailures`). When `J`
is empty the source sets `(Ĝ, B̂) = (G, B)` instead. The source's own derived gain gives such a
joint the value `-1` in `(Ĝ, B̂)`, so `unrollBalanced` keeps it unbalanced in both cases.

The source also says that `Σ_Ĝ(u)/∼ = Σ_G(u)/∼` and that `J` is an unbalancing class of `Σ_Ĝ(u)`.
This fails when two joints of `J` lie in different components of `G − u`, since then no cycle of `Ĝ`
contains both (`unbalancingClass_printed_fails`, in `Section2_UnrollingClasses`), where
`IsUnrolling.unbalancingClass_eq` assumes `JointsConnected`. That holds whenever `G − u` is
connected, as in the 2-connected graphs of §5.
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

open OrientedGraph

namespace BiasedGraph

section Definitions

variable (Ω : BiasedGraph V E) (u : V)

open Classical in
/-- `Ĝ` (§2.5): each joint of `J` becomes a link from its vertex to `u`. -/
noncomputable def unrollGraph : OrientedGraph V E :=
  ⟨Ω.graph.tail, fun e => if e ∈ Ω.offJoints u then u else Ω.graph.head e⟩

/-- `B̂` as printed in §2.5: the cycles of `Ĝ` meeting each unbalancing class in 0 or 2 edges. -/
def printedUnrollBalanced : Set (Set E) :=
  {C | (Ω.unrollGraph u).IsCycle C ∧ ∀ e ∈ Ω.sigmaStar u,
    (C ∩ Ω.unbalancingClass u e).encard = 0 ∨ (C ∩ Ω.unbalancingClass u e).encard = 2}

/-- `B̂` corrected: the printed set without the joints of `Ω` at `u`. -/
def unrollBalanced : Set (Set E) :=
  {C | C ∈ Ω.printedUnrollBalanced u ∧ ∀ e, C = {e} → ¬Ω.IsJoint e}

/-- `Ω'` is the **unrolling** of `Ω` to `u` (§2.5). A relation, because that `B̂` has the theta
property is the claim `IsRollable.exists_isUnrolling`. -/
structure IsUnrolling (Ω' : BiasedGraph V E) : Prop where
  /-- The graph of the unrolling is `Ĝ`. -/
  graph_eq : Ω'.graph = Ω.unrollGraph u
  /-- Its balanced cycles are `B̂`. -/
  balanced_eq : Ω'.balanced = Ω.unrollBalanced u

/-- Any two joints of `J` lie in one component of `G − u`. -/
def JointsConnected : Prop :=
  ∀ e ∈ Ω.offJoints u, ∀ f ∈ Ω.offJoints u,
    ∃ P : Ω.graph.Walk (Ω.graph.tail e) (Ω.graph.tail f), u ∉ P.support

end Definitions

variable {Ω Ω' : BiasedGraph V E} {u : V}

/-- A graph that stays connected off each vertex has its joints connected off `u`. -/
theorem jointsConnected_of_connectedOff (hG : Ω.graph.ConnectedOff) : Ω.JointsConnected u :=
  fun e he f hf => hG u _ _ (mem_verts.2 ⟨e, trivial, Or.inl rfl⟩)
    (mem_verts.2 ⟨f, trivial, Or.inl rfl⟩) he.2 hf.2

/-- The class of a joint of `J` is `J`. -/
theorem unbalancingClass_of_mem_offJoints {e : E} (he : e ∈ Ω.offJoints u) :
    Ω.unbalancingClass u e = Ω.offJoints u := by
  ext f
  constructor
  · rintro ⟨-, rfl | ⟨C, hC, heC, -⟩ | ⟨-, hf⟩⟩
    · exact he
    · exact (he.1.not_mem_of_balanced hC heC).elim
    · exact hf
  · exact fun hf => ⟨Or.inr hf, Or.inr (Or.inr ⟨he, hf⟩)⟩

namespace IsUnrolling

theorem tail_eq (h : Ω.IsUnrolling u Ω') (e : E) : Ω'.graph.tail e = Ω.graph.tail e := by
  rw [h.graph_eq]
  rfl

theorem head_of_mem (h : Ω.IsUnrolling u Ω') {e : E} (he : e ∈ Ω.offJoints u) :
    Ω'.graph.head e = u := by
  rw [h.graph_eq]
  exact ite_eq_left he

theorem head_of_notMem (h : Ω.IsUnrolling u Ω') {e : E} (he : e ∉ Ω.offJoints u) :
    Ω'.graph.head e = Ω.graph.head e := by
  rw [h.graph_eq]
  exact ite_eq_right he

/-- Every edge of `Σ(u)` has `u` as an end in `Ĝ`. -/
theorem end_of_mem_sigmaStar (h : Ω.IsUnrolling u Ω') {e : E} (he : e ∈ Ω.sigmaStar u) :
    Ω'.graph.tail e = u ∨ Ω'.graph.head e = u := by
  rcases he with ⟨hlink, ht | hh⟩ | hJ
  · exact Or.inl (by rwa [h.tail_eq])
  · have hJ : e ∉ Ω.offJoints u := fun hJ => hlink hJ.1.1
    exact Or.inr (by rwa [h.head_of_notMem hJ])
  · exact Or.inr (h.head_of_mem hJ)

/-- §2.5: "`u` is a balancing vertex of `(Ĝ, B̂)`". -/
theorem isBalancingVertex (h : Ω.IsUnrolling u Ω') : Ω'.IsBalancingVertex u := by
  intro C hC
  by_contra hu
  refine hC.2 ?_
  rw [h.balanced_eq]
  refine ⟨⟨h.graph_eq ▸ hC.1, fun e _ => Or.inl ?_⟩, fun e hCe hJ => hu ?_⟩
  · refine Set.encard_eq_zero.2 (Set.eq_empty_of_forall_notMem fun f hf => hu ?_)
    exact mem_verts.2 ⟨f, hf.1, h.end_of_mem_sigmaStar hf.2.1⟩
  · subst hCe
    by_cases ht : Ω.graph.tail e = u
    · exact mem_verts.2 ⟨e, rfl, Or.inl ((h.tail_eq e).trans ht)⟩
    · exact mem_verts.2 ⟨e, rfl, Or.inr (h.head_of_mem ⟨hJ, ht⟩)⟩

/-- `(Ĝ, B̂)` has no joint off `u`. -/
theorem offJoints_eq_empty (h : Ω.IsUnrolling u Ω') : Ω'.offJoints u = ∅ := by
  refine Set.eq_empty_of_forall_notMem fun e ⟨⟨hloop, hnot⟩, htu⟩ => hnot ?_
  have htu' : Ω.graph.tail e ≠ u := by rwa [h.tail_eq] at htu
  have hJ : e ∉ Ω.offJoints u := fun he => htu (hloop.trans (h.head_of_mem he))
  have hG : Ω.graph.tail e = Ω.graph.head e := by
    rw [← h.tail_eq, hloop, h.head_of_notMem hJ]
  rw [h.balanced_eq]
  refine ⟨⟨h.graph_eq ▸ isCycle_singleton_of_loop hloop, fun f _ => Or.inl ?_⟩, ?_⟩
  · refine Set.encard_eq_zero.2 (Set.eq_empty_of_forall_notMem fun g hg => ?_)
    obtain rfl : g = e := hg.1
    rcases hg.2.1 with hl | hg'
    · exact hl.1 hG
    · exact hJ hg'
  · intro f hf hj
    obtain rfl : e = f := Set.singleton_eq_singleton_iff.1 hf
    exact hJ ⟨hj, htu'⟩

/-- The links at `u` in `Ĝ` are those of `G` and the joints of `J`. -/
theorem linkStar_eq (h : Ω.IsUnrolling u Ω') : Ω'.linkStar u = Ω.linkStar u ∪ Ω.offJoints u := by
  ext e
  by_cases hJ : e ∈ Ω.offJoints u
  · refine iff_of_true ⟨?_, Or.inr (h.head_of_mem hJ)⟩ (Or.inr hJ)
    change Ω'.graph.tail e ≠ Ω'.graph.head e
    rw [h.tail_eq, h.head_of_mem hJ]
    exact hJ.2
  · simp only [linkStar, Set.mem_union, hJ, or_false, Set.mem_ofPred_eq, IsLink, h.tail_eq,
      h.head_of_notMem hJ]

/-- §2.5: "`Σ_Ĝ(u) = Σ_G(u)`". -/
theorem sigmaStar_eq (h : Ω.IsUnrolling u Ω') : Ω'.sigmaStar u = Ω.sigmaStar u := by
  rw [sigmaStar, h.offJoints_eq_empty, Set.union_empty, h.linkStar_eq]
  rfl

/-- §2.5: "since `(Ĝ, B̂)` has no pair of vertex-disjoint unbalanced cycles,
`L(Ĝ, B̂) = F(Ĝ, B̂)`". -/
theorem isFrameMatroid_iff_isLiftMatroid (h : Ω.IsUnrolling u Ω') {M : Matroid E} :
    Ω'.IsFrameMatroid M ↔ Ω'.IsLiftMatroid M :=
  BiasedGraph.isFrameMatroid_iff_isLiftMatroid fun C D hC hD hd =>
    Set.disjoint_left.1 hd (h.isBalancingVertex C hC) (h.isBalancingVertex D hD)

end IsUnrolling

end BiasedGraph

end Transformer.GainMatroid

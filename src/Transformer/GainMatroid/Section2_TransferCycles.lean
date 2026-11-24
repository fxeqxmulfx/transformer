module

public import Transformer.GainMatroid.Section2_ExchangeTransfer

/-!
# Cycles across an exchange pair

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, the proof of Proposition 2.6. Let `T`, `S` be
an exchange pair on `X = {a, b, c}` (`IsExchange`), with the star `X` at `v` in `S`. A walk of `S`
through `v` that does not end there has an edge of `X`
(`IsExchange.exists_mem_edges_of_mem_support`), so a path passes through `v` when its transfer
(`IsTransfer`) has an edge of `X` (`IsExchange.mem_support_of_isTransfer`), and transfers of openly
disjoint paths between vertices other than `v` are openly disjoint
(`IsExchange.openly_of_isTransfer`).

Let `X` be a balanced triangle of a biased graph `Ω` on `T`. A cycle of `S` through an edge of `X`
is in `Δ_X B` exactly when its symmetric difference with `X` is in `B`
(`IsExchange.mem_deltaBalanced_iff_of_isCycle`), so the cycle of two openly disjoint paths of `S`
between vertices other than `v` is in `Δ_X B` exactly when the cycle of their transfers is in `B`
(`IsExchange.mem_deltaBalanced_iff_of_isTransfer`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open scoped symmDiff

/-- `(P ∪ Q) ∆ X = (P ∆ X) ∪ Q` for `Q` disjoint from `X`. -/
theorem union_symmDiff_of_disjoint {E : Type*} {P Q X : Set E} (hQ : Disjoint Q X) :
    (P ∪ Q) ∆ X = P ∆ X ∪ Q := by
  ext g
  have hg : g ∈ Q → g ∉ X := fun h => Set.disjoint_left.1 hQ h
  simp only [Set.mem_symmDiff, Set.mem_union]
  tauto

namespace OrientedGraph.IsExchange

variable {V W E : Type*} {T : OrientedGraph V E} {S : OrientedGraph W E} {ι : V → W} {v : W}
  {a b c : E} {x y z : V}

/-- A walk of `S` through `v` that does not end at `v` has an edge of `X`. -/
theorem exists_mem_edges_of_mem_support (h : IsExchange T S ι v a b c x y z) {s t : W}
    {P : S.Walk s t} (hv : v ∈ P.support) (ht : t ≠ v) :
    ∃ g ∈ P.edges, g ∈ ({a, b, c} : Set E) := by
  obtain ⟨A, B, rfl⟩ := P.exists_eq_append_of_mem_support hv
  obtain ⟨g, hg, hgv⟩ := B.exists_mem_edges_start ht.symm
  exact ⟨g, Walk.mem_edges_append.2 (Or.inr hg), h.mem_of_incident hgv⟩

/-- An edge of `X` on a transfer means that the path passes through `v`. -/
theorem mem_support_of_isTransfer (h : IsExchange T S ι v a b c x y z) {s t : V}
    {P : S.Walk (ι s) (ι t)} {P' : T.Walk s t} (hP : IsTransfer {a, b, c} ι v P P') {g : E}
    (hg : g ∈ P'.edges) (hgX : g ∈ ({a, b, c} : Set E)) : v ∈ P.support := by
  by_contra hv
  have hg' : g ∈ P'.edgeSet := hg
  rw [hP.edgeSet_of_not_mem hv] at hg'
  exact hv (Walk.mem_support_of_mem_edges hg' (h.incident_of_mem hgX))

/-- Transfers of openly disjoint paths between vertices other than `v` are openly disjoint. -/
theorem openly_of_isTransfer (h : IsExchange T S ι v a b c x y z) {s t : V} (hs : ι s ≠ v)
    (ht : ι t ≠ v) {P Q : S.Walk (ι s) (ι t)} {P' Q' : T.Walk s t} (hPQ : Walk.Openly P Q)
    (hP : IsTransfer {a, b, c} ι v P P') (hQ : IsTransfer {a, b, c} ι v Q Q') :
    Walk.Openly P' Q' := by
  refine ⟨fun g hg hg' => ?_, fun u hu hu' =>
    (hPQ.verts _ (hP.support u hu) (hQ.support u hu')).imp (fun e => h.inj e) fun e => h.inj e⟩
  by_cases hgX : g ∈ ({a, b, c} : Set E)
  · rcases hPQ.verts v (h.mem_support_of_isTransfer hP hg hgX)
      (h.mem_support_of_isTransfer hQ hg' hgX) with hv | hv
    · exact hs hv.symm
    · exact ht hv.symm
  · exact hPQ.edges g (hP.mem_edges hg hgX) (hQ.mem_edges hg' hgX)

/-- A cycle of `S` through an edge of `X` is in `Δ_X B` exactly when its symmetric difference with
`X` is in `B`, for `X` balanced in `Ω`. -/
theorem mem_deltaBalanced_iff_of_isCycle {Ω : BiasedGraph V E}
    (h : IsExchange Ω.graph S ι v a b c x y z) (hbal : ({a, b, c} : Set E) ∈ Ω.balanced)
    {D : Set E} (hD : S.IsCycle D) {g : E} (hg : g ∈ D) (hgX : g ∈ ({a, b, c} : Set E)) :
    D ∈ deltaBalanced Ω.balanced {a, b, c} ↔ D ∆ {a, b, c} ∈ Ω.balanced := by
  rcases h.inter_cases_of_isCycle hD with h0 | ⟨a', b', c', x', y', z', h', hX, h2⟩
  · have hg' : g ∈ D ∩ {a, b, c} := ⟨hg, hgX⟩
    simp [h0] at hg'
  · rw [← hX] at hbal h2 ⊢
    exact Ω.mem_deltaBalanced_iff_of_inter_eq_pair h'.triangle hbal h2

/-- §2.4: for `X` balanced in `Ω`, the cycle of two openly disjoint paths of `S` between vertices
other than `v` is in `Δ_X B` exactly when the cycle of their transfers is in `B`. -/
theorem mem_deltaBalanced_iff_of_isTransfer {Ω : BiasedGraph V E}
    (h : IsExchange Ω.graph S ι v a b c x y z) (hbal : ({a, b, c} : Set E) ∈ Ω.balanced)
    {s t : V} (hs : ι s ≠ v) (ht : ι t ≠ v) {P Q : S.Walk (ι s) (ι t)}
    {P' Q' : Ω.graph.Walk s t} (hPQ : Walk.Openly P Q) (hC : S.IsCycle (P.edgeSet ∪ Q.edgeSet))
    (hP : IsTransfer {a, b, c} ι v P P') (hQ : IsTransfer {a, b, c} ι v Q Q') :
    P.edgeSet ∪ Q.edgeSet ∈ deltaBalanced Ω.balanced {a, b, c} ↔
      P'.edgeSet ∪ Q'.edgeSet ∈ Ω.balanced := by
  have hoff : ∀ {R : S.Walk (ι s) (ι t)}, v ∉ R.support → Disjoint R.edgeSet {a, b, c} :=
    fun hv => Set.disjoint_left.2 fun _ hg => h.not_mem_of_not_mem_support hv hg
  have hmeet : ∀ {R : S.Walk (ι s) (ι t)}, v ∈ R.support → (P.edgeSet ∪ Q.edgeSet ⊇ R.edgeSet) →
      (P.edgeSet ∪ Q.edgeSet ∈ deltaBalanced Ω.balanced {a, b, c} ↔
        (P.edgeSet ∪ Q.edgeSet) ∆ {a, b, c} ∈ Ω.balanced) := fun hv hR => by
    obtain ⟨g, hg, hgX⟩ := h.exists_mem_edges_of_mem_support hv ht
    exact h.mem_deltaBalanced_iff_of_isCycle hbal hC (hR hg) hgX
  by_cases hvP : v ∈ P.support
  · have hvQ : v ∉ Q.support := fun hvQ =>
      (hPQ.verts v hvP hvQ).elim (fun e => hs e.symm) fun e => ht e.symm
    rw [hmeet hvP Set.subset_union_left, hP.edgeSet_of_mem hvP, hQ.edgeSet_of_not_mem hvQ,
      union_symmDiff_of_disjoint (hoff hvQ)]
  · rw [hP.edgeSet_of_not_mem hvP]
    by_cases hvQ : v ∈ Q.support
    · rw [hmeet hvQ Set.subset_union_right, hQ.edgeSet_of_mem hvQ, Set.union_comm,
        union_symmDiff_of_disjoint (hoff hvP), Set.union_comm]
    · rw [hQ.edgeSet_of_not_mem hvQ, mem_deltaBalanced_iff_of_inter_eq_empty h.triangle.ne_ab
        (Set.disjoint_iff_inter_eq_empty.1 (Set.disjoint_union_left.2 ⟨hoff hvP, hoff hvQ⟩))]

end OrientedGraph.IsExchange

open OrientedGraph

/-- The triangle `{0, 1, 2}` of `clawNabla` is balanced for the trivial gains. -/
theorem clawNabla_mem_balanced :
    ({0, 1, 2} : Set (Fin 5)) ∈ (clawNabla.gainBiased (1 : Fin 5 → ℚˣ)).balanced :=
  gainBiased_one_isBalanced clawNabla _ clawNabla_isExchange.triangle.isCycle

/-- The edge `3` and the path `1, 0, 2` of `clawGraph` are openly disjoint. -/
theorem clawEdge12_openly : Walk.Openly clawEdge12 clawStar12 := ⟨by decide, by decide⟩

/-- The edge `3` and the path `1, 0, 2` of `clawGraph` form the triangle `{0, 1, 3}`. -/
theorem clawEdge12_isCycle : clawGraph.IsCycle (clawEdge12.edgeSet ∪ clawStar12.edgeSet) :=
  ⟨1, _, clawEdge12_openly.isCycleWalk (by decide) clawEdge12_isPath clawStar12_isPath,
    Walk.edgeSet_append_reverse _ _⟩

example := union_symmDiff_of_disjoint (P := ({0} : Set (Fin 5))) (Q := {3}) (X := {0, 1, 2})
  (by simp)

example := clawNabla_isExchange.exists_mem_edges_of_mem_support (P := clawStar12) (by decide)
  (by decide)

example := clawNabla_isExchange.mem_support_of_isTransfer clawStar12_isTransfer (g := 2)
  (by simp [nablaTriangle12, Walk.edges]) (by simp)

example := clawNabla_isExchange.openly_of_isTransfer (by decide) (by decide) clawEdge12_openly
  clawEdge12_isTransfer clawStar12_isTransfer

example := clawNabla_isExchange.mem_deltaBalanced_iff_of_isCycle
  (Ω := clawNabla.gainBiased (1 : Fin 5 → ℚˣ)) clawNabla_mem_balanced clawEdge12_isCycle (g := 0)
  (by simp [clawStar12, Walk.edgeSet, Walk.edges]) (by simp)

example := clawNabla_isExchange.mem_deltaBalanced_iff_of_isTransfer
  (Ω := clawNabla.gainBiased (1 : Fin 5 → ℚˣ)) clawNabla_mem_balanced (by decide) (by decide)
  clawEdge12_openly clawEdge12_isCycle clawEdge12_isTransfer clawStar12_isTransfer

end Transformer.GainMatroid

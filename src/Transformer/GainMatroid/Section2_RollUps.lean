module

public import Transformer.GainMatroid.Section2_UnrollingClasses

/-!
# The roll-ups of an unrolling

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5:

> "For each unbalancing class `U` of `Σ_Ĝ(u)` there is a biased graph `(G_U, B_U)` for which
> `F(G_U, B_U) = F(G, B) = F(Ĝ, B̂)`, obtained from `(Ĝ, B̂)` by replacing each link `e = uv ∈ U`
> with a joint incident to `v` […]. Call each such biased graph a *roll-up of `(Ĝ, B̂)` from `u`*."

The source does not write out `B_U`. An edge of `U` becomes an unbalanced loop, so a cycle of `G_U`
through it is that loop, and every other cycle of `G_U` is a cycle of `Ĝ`: `BiasedGraph.rollUp`
keeps the balanced cycles of `Ĝ` that avoid `U`.

`R_{(G, B)}` is `(Ĝ, B̂)` together with its roll-ups (`BiasedGraph.rollUpFamily`). Its size
`|Σ(u)/∼| + 1` is `IsUnrolling.ncard_rollUpFamily`. What the source says of the roll-ups by
comparing circuits, and its special case of a graph balanced off its joints, are stated in
`Section2_RollUpClaims`.
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

open OrientedGraph

namespace OrientedGraph

variable {G : OrientedGraph V E}

open Classical in
/-- `G_U` (§2.5): each edge of `U` becomes a loop at its end away from `u`. -/
noncomputable def rollUpGraph (G : OrientedGraph V E) (u : V) (U : Set E) : OrientedGraph V E :=
  ⟨fun e => if e ∈ U then G.otherEnd u e else G.tail e,
    fun e => if e ∈ U then G.otherEnd u e else G.head e⟩

theorem rollUpGraph_tail_of_mem {u : V} {U : Set E} {e : E} (he : e ∈ U) :
    (G.rollUpGraph u U).tail e = G.otherEnd u e := ite_eq_left he

theorem rollUpGraph_head_of_mem {u : V} {U : Set E} {e : E} (he : e ∈ U) :
    (G.rollUpGraph u U).head e = G.otherEnd u e := ite_eq_left he

theorem rollUpGraph_tail_of_notMem {u : V} {U : Set E} {e : E} (he : e ∉ U) :
    (G.rollUpGraph u U).tail e = G.tail e := ite_eq_right he

theorem rollUpGraph_head_of_notMem {u : V} {U : Set E} {e : E} (he : e ∉ U) :
    (G.rollUpGraph u U).head e = G.head e := ite_eq_right he

/-- A theta whose edges have the same ends in `G` is a theta of `G`. -/
theorem IsTheta.exists_of_agree {H : OrientedGraph V E} {x y : V} {P Q R : H.Walk x y}
    (hT : IsTheta P Q R) (hP : ∀ g ∈ P.edges, G.tail g = H.tail g ∧ G.head g = H.head g)
    (hQ : ∀ g ∈ Q.edges, G.tail g = H.tail g ∧ G.head g = H.head g)
    (hR : ∀ g ∈ R.edges, G.tail g = H.tail g ∧ G.head g = H.head g) :
    ∃ P' Q' R' : G.Walk x y, IsTheta P' Q' R' ∧ P'.edgeSet = P.edgeSet ∧ Q'.edgeSet = Q.edgeSet ∧
      R'.edgeSet = R.edgeSet := by
  obtain ⟨P', hP₁, hP₂⟩ := P.exists_map (G := G) id hP
  obtain ⟨Q', hQ₁, hQ₂⟩ := Q.exists_map (G := G) id hQ
  obtain ⟨R', hR₁, hR₂⟩ := R.exists_map (G := G) id hR
  rw [List.map_id] at hP₁ hQ₁ hR₁
  have hopen {A B : H.Walk x y} {A' B' : G.Walk x y} (h₁ : A'.edges = A.edges)
      (h₂ : A'.support = A.support) (h₃ : B'.edges = B.edges) (h₄ : B'.support = B.support)
      (h : Walk.Openly A B) : Walk.Openly A' B' :=
    ⟨fun e he => h₃ ▸ h.edges e (h₁ ▸ he), fun z hz hz' => h.verts z (h₂ ▸ hz) (h₄ ▸ hz')⟩
  have hset {A : H.Walk x y} {A' : G.Walk x y} (h₁ : A'.edges = A.edges) :
      A'.edgeSet = A.edgeSet := by
    ext e
    simp [Walk.edgeSet, h₁]
  refine ⟨P', Q', R', ⟨hT.ne, ?_, ?_, ?_, hopen hP₁ hP₂ hQ₁ hQ₂ hT.openPQ,
    hopen hP₁ hP₂ hR₁ hR₂ hT.openPR, hopen hQ₁ hQ₂ hR₁ hR₂ hT.openQR⟩, hset hP₁, hset hQ₁, hset hR₁⟩
  · change P'.support.Nodup
    rw [hP₂]
    exact hT.pathP
  · change Q'.support.Nodup
    rw [hQ₂]
    exact hT.pathQ
  · change R'.support.Nodup
    rw [hR₂]
    exact hT.pathR

end OrientedGraph

namespace BiasedGraph

variable {Ω Ω' : BiasedGraph V E} {u : V}

/-- `(G_U, B_U)`: the roll-up from `u` on `U` (§2.5). Its balanced cycles are those of `Ω` that
avoid `U`. -/
noncomputable def rollUp (Ω : BiasedGraph V E) (u : V) (U : Set E) : BiasedGraph V E where
  graph := Ω.graph.rollUpGraph u U
  balanced := {C | C ∈ Ω.balanced ∧ Disjoint C U}
  isCycle_of_mem := by
    rintro C ⟨hC, hd⟩
    simpa using (Ω.isCycle_of_mem _ hC).map (G := Ω.graph.rollUpGraph u U) id Function.injective_id
      fun g hg => ⟨rollUpGraph_tail_of_notMem (Set.disjoint_left.1 hd hg),
        rollUpGraph_head_of_notMem (Set.disjoint_left.1 hd hg)⟩
  theta := by
    rintro x y P Q R hT ⟨hb₁, hd₁⟩ ⟨hb₂, hd₂⟩
    have hoff {A : (Ω.graph.rollUpGraph u U).Walk x y} (hA : ∀ g ∈ A.edges, g ∉ U) :
        ∀ g ∈ A.edges, Ω.graph.tail g = (Ω.graph.rollUpGraph u U).tail g ∧
          Ω.graph.head g = (Ω.graph.rollUpGraph u U).head g := fun g hg =>
      ⟨(rollUpGraph_tail_of_notMem (hA g hg)).symm, (rollUpGraph_head_of_notMem (hA g hg)).symm⟩
    obtain ⟨P', Q', R', hT', hP, hQ, hR⟩ := hT.exists_of_agree
      (hoff fun g hg => Set.disjoint_left.1 hd₁ (Or.inl hg))
      (hoff fun g hg => Set.disjoint_left.1 hd₁ (Or.inr hg))
      (hoff fun g hg => Set.disjoint_left.1 hd₂ (Or.inr hg))
    refine ⟨?_, Set.disjoint_union_left.2 ⟨hd₁.mono_left Set.subset_union_right,
      hd₂.mono_left Set.subset_union_right⟩⟩
    have h := Ω.theta hT' (by rwa [hP, hQ]) (by rwa [hP, hR])
    rwa [hQ, hR] at h

/-- `Σ(u)/∼` (§2.5): the set of unbalancing classes. -/
def unbalancingClasses (Ω : BiasedGraph V E) (u : V) : Set (Set E) :=
  {U | ∃ e ∈ Ω.sigmaStar u, U = Ω.unbalancingClass u e}

/-- `R_{(G, B)}` (§2.5), for `Ω = (Ĝ, B̂)`: `Ω` together with its roll-ups from `u`. -/
noncomputable def rollUpFamily (Ω : BiasedGraph V E) (u : V) : Set (BiasedGraph V E) :=
  insert Ω (Ω.rollUp u '' Ω.unbalancingClasses u)

/-- Rolling up `J` undoes the unrolling on the graph: the graph half of `(G_J, B_J) = (G, B)`. -/
theorem rollUpGraph_unrollGraph (Ω : BiasedGraph V E) (u : V) :
    (Ω.unrollGraph u).rollUpGraph u (Ω.offJoints u) = Ω.graph := by
  refine eq_of_tail_head (funext fun e => ?_) (funext fun e => ?_)
  · by_cases he : e ∈ Ω.offJoints u
    · rw [rollUpGraph_tail_of_mem he]
      exact ite_eq_right he.2
    · exact rollUpGraph_tail_of_notMem he
  · by_cases he : e ∈ Ω.offJoints u
    · rw [rollUpGraph_head_of_mem he]
      exact (ite_eq_right he.2).trans he.1.1
    · rw [rollUpGraph_head_of_notMem he]
      exact ite_eq_right he

namespace IsUnrolling

/-- Each unbalancing class of `(Ĝ, B̂)` consists of links at `u`. -/
theorem subset_linkStar (h : Ω.IsUnrolling u Ω') {U : Set E} (hU : U ∈ Ω'.unbalancingClasses u) :
    U ⊆ Ω'.linkStar u := by
  obtain ⟨e, -, rfl⟩ := hU
  rintro f ⟨hf | hf, -⟩
  · exact hf
  · rw [h.offJoints_eq_empty] at hf
    exact hf.elim

/-- Distinct unbalancing classes give distinct roll-ups: their edges are loops in one only. -/
theorem rollUp_injOn (h : Ω.IsUnrolling u Ω') :
    Set.InjOn (Ω'.rollUp u) (Ω'.unbalancingClasses u) := by
  have key {A B : Set E} (hA : A ⊆ Ω'.linkStar u) (hAB : Ω'.rollUp u A = Ω'.rollUp u B) :
      A ⊆ B := by
    intro e he
    by_contra heB
    have hloop : (Ω'.rollUp u A).graph.tail e = (Ω'.rollUp u A).graph.head e :=
      (rollUpGraph_tail_of_mem he).trans (rollUpGraph_head_of_mem he).symm
    rw [hAB] at hloop
    exact (hA he).1 ((rollUpGraph_tail_of_notMem heB).symm.trans
      (hloop.trans (rollUpGraph_head_of_notMem heB)))
  exact fun U hU W hW hUW => (key (h.subset_linkStar hU) hUW).antisymm
    (key (h.subset_linkStar hW) hUW.symm)

/-- A roll-up differs from `(Ĝ, B̂)`: an edge of the class is a link of one and a loop of the
other. -/
theorem rollUp_ne (h : Ω.IsUnrolling u Ω') {U : Set E} (hU : U ∈ Ω'.unbalancingClasses u) :
    Ω'.rollUp u U ≠ Ω' := by
  intro heq
  obtain ⟨e, he, hUe⟩ := hU
  have hmem : e ∈ U := hUe ▸ ⟨he, Or.inl rfl⟩
  have hloop : (Ω'.rollUp u U).graph.tail e = (Ω'.rollUp u U).graph.head e :=
    (rollUpGraph_tail_of_mem hmem).trans (rollUpGraph_head_of_mem hmem).symm
  rw [heq] at hloop
  exact (h.subset_linkStar ⟨e, he, hUe⟩ hmem).1 hloop

/-- §2.5: "`|R_{(G, B)}| = |Σ(u)/∼| + 1`". -/
theorem ncard_rollUpFamily [Finite E] (h : Ω.IsUnrolling u Ω') :
    (Ω'.rollUpFamily u).ncard = (Ω'.unbalancingClasses u).ncard + 1 := by
  have hfin : (Ω'.unbalancingClasses u).Finite := Set.toFinite _
  rw [rollUpFamily, Set.ncard_insert_of_notMem ?_ (hfin.image _), h.rollUp_injOn.ncard_image]
  rintro ⟨U, hU, heq⟩
  exact h.rollUp_ne hU heq

end IsUnrolling

end BiasedGraph

end Transformer.GainMatroid

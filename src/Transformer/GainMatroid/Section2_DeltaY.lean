module

public import Transformer.GainMatroid.Section2_MatrixEquivalence

/-!
# Δ-Y and Y-Δ exchanges of graphs and biased graphs

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. A Δ-Y exchange replaces the triangle
`X = {a, b, c}`, with `a = yz`, `b = xz`, `c = xy`, by a star at a new vertex `v`, with `a = vx`,
`b = vy`, `c = vz`. A Y-Δ exchange is the reverse, at a vertex of degree three.

The new vertex of `Δ_X G` is `none : Option V`. The vertex `v` of a Y-Δ exchange stays in `∇_Y G` as
an isolated vertex, which §2.1 permits to delete without comment. The reference orientations are
those of the first and the last three columns of the matrix `I(K₄)` of §2.4: `x → v`, `y → v`,
`z → v`, and `y → z`, `x → z`, `x → y`.

The source defines `∇_Y B` as "the collection of cycles of `∇_Y G` consisting of `Y` together with
the minimal nonempty members" of a family. A minimal member of that family need not be a cycle of
`∇_Y G` (`nablaFamily_minimal_not_isCycle`), so `nablaBalanced` keeps the cycles among them.
-/

@[expose] public section

open scoped symmDiff

namespace Transformer.GainMatroid

variable {V E : Type*}

namespace OrientedGraph

/-- The triangle `X = {a, b, c}` of §2.4, with `a = yz`, `b = xz` and `c = xy`. -/
structure IsLabelledTriangle (G : OrientedGraph V E) (a b c : E) (x y z : V) : Prop where
  joins_a : G.Joins a y z
  joins_b : G.Joins b x z
  joins_c : G.Joins c x y
  ne_ab : a ≠ b
  ne_ac : a ≠ c
  ne_bc : b ≠ c
  ne_xy : x ≠ y
  ne_xz : x ≠ z
  ne_yz : y ≠ z

theorem IsLabelledTriangle.isCycle {G : OrientedGraph V E} {a b c : E} {x y z : V}
    (h : G.IsLabelledTriangle a b c x y z) : G.IsCycle {a, b, c} := by
  refine ⟨x, triangleWalk h.joins_c h.joins_a h.joins_b.symm,
    triangleWalk_isCycleWalk _ _ _ h.ne_ac.symm h.ne_bc.symm h.ne_ab h.ne_xy h.ne_xz h.ne_yz, ?_⟩
  ext e
  simp only [triangleWalk, Walk.edgeSet, Walk.edges, Set.mem_insert_iff, Set.mem_singleton_iff,
    Set.mem_ofPred_eq, List.mem_cons, List.not_mem_nil, or_false]
  tauto

open Classical in
/-- **`Δ_X G`** (§2.4): the edges `a`, `b`, `c` of the triangle become `xv`, `yv`, `zv` for a new
vertex `v = none`. -/
noncomputable def deltaGraph (G : OrientedGraph V E) (a b c : E) (x y z : V) :
    OrientedGraph (Option V) E where
  tail e := if e = a then some x else if e = b then some y else if e = c then some z
    else some (G.tail e)
  head e := if e = a ∨ e = b ∨ e = c then none else some (G.head e)

/-- The star `Y = {a, b, c}` of §2.4 at a vertex `v` of degree three, with `a = vx`, `b = vy` and
`c = vz` for distinct neighbours `x`, `y`, `z`. -/
structure IsLabelledClaw (G : OrientedGraph V E) (v : V) (a b c : E) (x y z : V) : Prop where
  joins_a : G.Joins a v x
  joins_b : G.Joins b v y
  joins_c : G.Joins c v z
  ne_ab : a ≠ b
  ne_ac : a ≠ c
  ne_bc : b ≠ c
  ne_xy : x ≠ y
  ne_xz : x ≠ z
  ne_yz : y ≠ z
  ne_vx : v ≠ x
  ne_vy : v ≠ y
  ne_vz : v ≠ z
  /-- The vertex `v` has degree three. -/
  degree : ∀ e, G.tail e = v ∨ G.head e = v → e = a ∨ e = b ∨ e = c

open Classical in
/-- **`∇_Y G`** (§2.4): the edges `a`, `b`, `c` of the star become `yz`, `xz`, `xy`. The centre of
the star stays as an isolated vertex. -/
noncomputable def nablaGraph (G : OrientedGraph V E) (a b c : E) (x y z : V) :
    OrientedGraph V E where
  tail e := if e = a then y else if e = b then x else if e = c then x else G.tail e
  head e := if e = a then z else if e = b then z else if e = c then y else G.head e

end OrientedGraph

/-- **`Δ_X B`** (§2.4): the members of `B` meeting `X` in `0` or `2` edges, and `C △ X` for the
members `C` of `B` meeting `X` in one edge. -/
def deltaBalanced (B : Set (Set E)) (X : Set E) : Set (Set E) :=
  {C | C ∈ B ∧ ((C ∩ X).ncard = 0 ∨ (C ∩ X).ncard = 2)} ∪
    {D | ∃ C ∈ B, (C ∩ X).ncard = 1 ∧ D = C ∆ X}

/-- The family of §2.4 whose minimal nonempty members define `∇_Y B`. -/
def nablaFamily (B : Set (Set E)) (Y : Set E) : Set (Set E) :=
  {C | C ∈ B ∧ ((C ∩ Y).ncard = 0 ∨ (C ∩ Y).ncard = 2)} ∪
    {D | ∃ C ∈ B, (C ∩ Y).ncard = 2 ∧ D = C ∆ Y}

/-- **`∇_Y B`** (§2.4): the cycles of `∇_Y G` that are `Y` or minimal nonempty members of
`nablaFamily B Y`. -/
def nablaBalanced (G' : OrientedGraph V E) (B : Set (Set E)) (Y : Set E) : Set (Set E) :=
  {C | G'.IsCycle C ∧ (C = Y ∨ Minimal (fun D => D ∈ nablaFamily B Y ∧ D.Nonempty) C)}

namespace BiasedGraph

/-- `Ω'` is `Δ_X Ω` for a balanced triangle `X = {a, b, c}` (§2.4). -/
structure IsDeltaExchange (Ω : BiasedGraph V E) (a b c : E) (x y z : V)
    (Ω' : BiasedGraph (Option V) E) : Prop where
  triangle : Ω.graph.IsLabelledTriangle a b c x y z
  balanced : ({a, b, c} : Set E) ∈ Ω.balanced
  graph_eq : Ω'.graph = Ω.graph.deltaGraph a b c x y z
  balanced_eq : Ω'.balanced = deltaBalanced Ω.balanced {a, b, c}

/-- `Ω'` is `∇_Y Ω` for a star `Y = {a, b, c}` at a vertex `v` of degree three (§2.4). -/
structure IsNablaExchange (Ω : BiasedGraph V E) (v : V) (a b c : E) (x y z : V)
    (Ω' : BiasedGraph V E) : Prop where
  claw : Ω.graph.IsLabelledClaw v a b c x y z
  graph_eq : Ω'.graph = Ω.graph.nablaGraph a b c x y z
  balanced_eq : Ω'.balanced =
    nablaBalanced (Ω.graph.nablaGraph a b c x y z) Ω.balanced {a, b, c}

/-- **Proposition 2.7** (§2.4). If `Y` is a `K_{1,3}`-subgraph of `(G, B)`, then
`(∇_Y G, ∇_Y B)` is a biased graph. -/
theorem exists_isNablaExchange (Ω : BiasedGraph V E) {v : V} {a b c : E} {x y z : V}
    (hY : Ω.graph.IsLabelledClaw v a b c x y z) :
    ∃ Ω' : BiasedGraph V E, Ω.IsNablaExchange v a b c x y z Ω' := by
  sorry

end BiasedGraph

namespace OrientedGraph

/-- A graph with one cycle in its biased collection: no theta has two of its cycles equal. -/
def singleCycleBiased (G : OrientedGraph V E) (C : Set E) (hC : G.IsCycle C) :
    BiasedGraph V E where
  graph := G
  balanced := {C}
  isCycle_of_mem := by
    rintro _ rfl
    exact hC
  theta := by
    intro x y P Q R h hPQ hPR
    exfalso
    obtain ⟨f, hf⟩ := List.exists_mem_of_ne_nil _ (Walk.edges_ne_nil_of_ne h.ne Q)
    have heq : P.edgeSet ∪ Q.edgeSet = P.edgeSet ∪ R.edgeSet := hPQ.trans hPR.symm
    have hmem : f ∈ P.edgeSet ∪ R.edgeSet := heq ▸ Or.inr hf
    rcases hmem with hp | hr
    · exact h.openPQ.edges f hp hf
    · exact h.openQR.edges f hf hr

/-- `K₄` minus the edge `yz`: the star at `0` with leaves `1`, `2`, `3`, and the edges `2 → 1` and
`1 → 3`. -/
def clawGraph : OrientedGraph (Fin 4) (Fin 5) := ⟨![0, 0, 0, 2, 1], ![1, 2, 3, 1, 3]⟩

theorem clawGraph_isLabelledClaw : clawGraph.IsLabelledClaw 0 0 1 2 1 2 3 := by
  refine ⟨Or.inl ⟨rfl, rfl⟩, Or.inl ⟨rfl, rfl⟩, Or.inl ⟨rfl, rfl⟩, by decide, by decide, by decide,
    by decide, by decide, by decide, by decide, by decide, by decide, ?_⟩
  simp only [clawGraph]
  decide

/-- The four-cycle `0, 2, 1, 3` of `clawGraph` through the star. -/
theorem clawGraph_isCycle : clawGraph.IsCycle {1, 2, 3, 4} := by
  have h₁ : clawGraph.Joins 1 0 2 := Or.inl ⟨rfl, rfl⟩
  have h₂ : clawGraph.Joins 3 2 1 := Or.inl ⟨rfl, rfl⟩
  have h₃ : clawGraph.Joins 4 1 3 := Or.inl ⟨rfl, rfl⟩
  have h₄ : clawGraph.Joins 2 3 0 := Or.inr ⟨rfl, rfl⟩
  refine ⟨0, .cons 1 h₁ (.cons 3 h₂ (.cons 4 h₃ (.cons 2 h₄ (.nil 0)))),
    ⟨by simp [Walk.edges], by simp [Walk.edges], by simp [Walk.support]⟩, ?_⟩
  ext e
  simp only [Walk.edgeSet, Walk.edges, Set.mem_insert_iff, Set.mem_singleton_iff,
    Set.mem_ofPred_eq, List.mem_cons, List.not_mem_nil, or_false]
  tauto

end OrientedGraph

end Transformer.GainMatroid

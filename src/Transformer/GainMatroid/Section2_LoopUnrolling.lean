module

public import Transformer.GainMatroid.Section2_WalkMaps
public import Transformer.GainMatroid.Section2_UnrollingEmptyJ
public import Transformer.GainMatroid.Section2_RollUps

/-!
# Unrolling loops to one vertex

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5, unroll the joints of a biased graph that is
balanced after removing them to a new vertex `x`, and roll up the edges at a vertex `x` of a graph
`H`. Both are one setting: `IsLoopUnrolling Ω H ι z U` says that a graph `H` keeps the edges of `Ω`
outside `U` along an injective vertex map `ι`, that the edges of `U` are loops of `Ω` which `H`
turns into its links at a vertex `z`, and that a cycle of `Ω` is balanced exactly when it avoids
`U`. Then the edges of `U` are joints (`IsLoopUnrolling.isJoint`), and an unbalanced cycle of `Ω` is
one of them (`IsLoopUnrolling.eq_singleton_of_notMem`).

`unrollToNew` is the source's `H`: each joint becomes a link to the new vertex `none` of `Option V`.
A biased graph whose unbalanced cycles have one edge unrolls to it (`isLoopUnrolling_unrollToNew`),
and a balanced biased graph rolled up at a vertex unrolls back to its graph
(`IsBalanced.isLoopUnrolling_rollUp`). `Section2_LoopCircuits` proves that the frame circuits of `Ω`
are then the cycles of `H`.
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V W E : Type*}

namespace OrientedGraph

/-- A cycle of one edge is a loop. -/
theorem IsCycle.tail_eq_head_of_singleton {G : OrientedGraph V E} {e : E} (hC : G.IsCycle {e}) :
    G.tail e = G.head e := by
  by_contra hl
  obtain ⟨g, hg, hge, -⟩ := hC.exists_partner rfl hl (Or.inl rfl)
  exact hge hg

end OrientedGraph

open OrientedGraph

namespace BiasedGraph

/-- `H` **unrolls** the loops `U` of `Ω` to the vertex `z` along `ι`: the other edges keep their
ends, the edges of `U` become the links of `H` at `z`, and the balanced cycles of `Ω` are those
avoiding `U`. -/
structure IsLoopUnrolling (Ω : BiasedGraph V E) (H : OrientedGraph W E) (ι : V → W) (z : W)
    (U : Set E) : Prop where
  /-- The vertex map is injective. -/
  inj : Function.Injective ι
  /-- An edge outside `U` keeps its ends. -/
  keep : ∀ e ∉ U, H.tail e = ι (Ω.graph.tail e) ∧ H.head e = ι (Ω.graph.head e)
  /-- The edges of `U` are loops of `Ω`. -/
  loop : ∀ e ∈ U, Ω.graph.tail e = Ω.graph.head e
  /-- An edge of `U` joins the image of its vertex to `z`. -/
  joins : ∀ e ∈ U, H.Joins e (ι (Ω.graph.tail e)) z
  /-- The vertex of a loop of `U` does not go to `z`. -/
  ne : ∀ e ∈ U, ι (Ω.graph.tail e) ≠ z
  /-- Every link of `H` at `z` is in `U`. -/
  star : ∀ e, H.IsLink e → (H.tail e = z ∨ H.head e = z) → e ∈ U
  /-- A cycle of `Ω` is balanced exactly when it avoids `U`. -/
  balanced : ∀ C, Ω.graph.IsCycle C → (C ∈ Ω.balanced ↔ Disjoint C U)

namespace IsLoopUnrolling

variable {Ω : BiasedGraph V E} {H : OrientedGraph W E} {ι : V → W} {z : W} {U : Set E}

/-- An edge of `U` is a joint of `Ω`. -/
theorem isJoint (h : IsLoopUnrolling Ω H ι z U) {e : E} (he : e ∈ U) : Ω.IsJoint e :=
  ⟨h.loop e he, fun hb => Set.disjoint_left.1
    ((h.balanced _ (isCycle_singleton_of_loop (h.loop e he))).1 hb) rfl he⟩

/-- An edge of `U` is a link of `H` at `z`. -/
theorem isLink (h : IsLoopUnrolling Ω H ι z U) {e : E} (he : e ∈ U) :
    H.IsLink e ∧ (H.tail e = z ∨ H.head e = z) := by
  refine ⟨fun ht => h.ne e he ?_, (h.joins e he).symm.tail_or_head⟩
  rcases h.joins e he with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
  · exact h₁.symm.trans (ht.trans h₂)
  · exact h₂.symm.trans (ht.symm.trans h₁)

/-- A link of `Ω` keeps its ends. -/
theorem keep_of_isLink (h : IsLoopUnrolling Ω H ι z U) {e : E} (hl : Ω.graph.IsLink e) :
    H.tail e = ι (Ω.graph.tail e) ∧ H.head e = ι (Ω.graph.head e) :=
  h.keep e fun he => hl (h.loop e he)

/-- An unbalanced cycle of `Ω` is one loop of `U`. -/
theorem eq_singleton_of_notMem (h : IsLoopUnrolling Ω H ι z U) {C : Set E}
    (hC : Ω.graph.IsCycle C) (hb : C ∉ Ω.balanced) : ∃ e ∈ U, C = {e} := by
  obtain ⟨e, heC, heU⟩ := Set.not_disjoint_iff.1 fun hd => hb ((h.balanced C hC).2 hd)
  exact ⟨e, heU, hC.eq_singleton_of_loop heC (h.loop e heU)⟩

/-- The one vertex of a loop of `U`. -/
theorem tail_eq_of_mem_verts (h : IsLoopUnrolling Ω H ι z U) {e : E} (he : e ∈ U) {v : V}
    (hv : v ∈ Ω.graph.verts {e}) : Ω.graph.tail e = v := by
  obtain ⟨g, hg, hgv⟩ := mem_verts.1 hv
  obtain rfl := Set.mem_singleton_iff.1 hg
  exact hgv.elim id fun hh => (h.loop _ he).trans hh

end IsLoopUnrolling

open Classical in
/-- `H` (§2.5): each joint of `Ω` becomes a link to the new vertex `none`. -/
noncomputable def unrollToNew (Ω : BiasedGraph V E) : OrientedGraph (Option V) E where
  tail e := some (Ω.graph.tail e)
  head e := if Ω.IsJoint e then none else some (Ω.graph.head e)

/-- A biased graph whose unbalanced cycles have one edge unrolls its joints to the new vertex. -/
theorem isLoopUnrolling_unrollToNew {Ω : BiasedGraph V E}
    (hbal : ∀ C, Ω.IsUnbalancedCycle C → ∃ e, C = {e}) :
    IsLoopUnrolling Ω Ω.unrollToNew some none {e | Ω.IsJoint e} where
  inj := Option.some_injective V
  keep _ he := ⟨rfl, ite_eq_right he⟩
  loop _ he := he.1
  joins _ he := Or.inl ⟨rfl, ite_eq_left he⟩
  ne _ _ := Option.some_ne_none _
  star e _ hz := by
    by_contra hj
    rcases hz with hz | hz
    · exact Option.some_ne_none _ hz
    · exact Option.some_ne_none _ ((ite_eq_right hj).symm.trans hz)
  balanced C hC := by
    refine ⟨fun hb => Set.disjoint_left.2 fun e heC hj => IsJoint.not_mem_of_balanced hj hb heC,
      fun hd => ?_⟩
    by_contra hb
    obtain ⟨e, rfl⟩ := hbal C ⟨hC, hb⟩
    exact Set.disjoint_left.1 hd rfl ⟨hC.tail_eq_head_of_singleton, hb⟩

/-- A balanced biased graph rolled up at `x` unrolls back to its graph: the links at `x` became
loops at their other ends. -/
theorem IsBalanced.isLoopUnrolling_rollUp {Ω : BiasedGraph V E} (hΩ : Ω.IsBalanced) (x : V) :
    IsLoopUnrolling (Ω.rollUp x (Ω.linkStar x)) Ω.graph id x (Ω.linkStar x) where
  inj := Function.injective_id
  keep _ he := ⟨(rollUpGraph_tail_of_notMem he).symm, (rollUpGraph_head_of_notMem he).symm⟩
  loop _ he := (rollUpGraph_tail_of_mem he).trans (rollUpGraph_head_of_mem he).symm
  joins e he := by
    change Ω.graph.Joins e ((Ω.graph.rollUpGraph x (Ω.linkStar x)).tail e) x
    rw [rollUpGraph_tail_of_mem he]
    exact joins_otherEnd he.2
  ne e he := by
    change (Ω.graph.rollUpGraph x (Ω.linkStar x)).tail e ≠ x
    rw [rollUpGraph_tail_of_mem he]
    exact otherEnd_ne he.1
  star _ hl hx := ⟨hl, hx⟩
  balanced C hC := ⟨fun hb => hb.2, fun hd => ⟨hΩ C (hC.map_vertex Function.injective_id
    fun _ hg => ⟨(rollUpGraph_tail_of_notMem (Set.disjoint_left.1 hd hg)).symm,
      (rollUpGraph_head_of_notMem (Set.disjoint_left.1 hd hg)).symm⟩), hd⟩⟩

end BiasedGraph

open BiasedGraph

/-- Every unbalanced cycle of `twoLoops` is one loop. -/
theorem twoLoops_unbalanced_singleton (C : Set Bool)
    (hC : twoLoops.contrabalanced.IsUnbalancedCycle C) : ∃ e, C = {e} := by
  obtain ⟨c, hc⟩ := hC.1.nonempty
  exact ⟨c, hC.1.eq_singleton_of_loop hc rfl⟩

/-- The edge `0` of `triangle` is a link at `0`. -/
theorem triangle_zero_mem_linkStar :
    (0 : Fin 3) ∈ (triangle.gainBiased (1 : Fin 3 → ℚˣ)).linkStar 0 :=
  ⟨by decide, Or.inl rfl⟩

/-- The edge `1` of `triangle` misses `0`. -/
theorem triangle_one_notMem_linkStar :
    (1 : Fin 3) ∉ (triangle.gainBiased (1 : Fin 3 → ℚˣ)).linkStar 0 := fun h => by
  rcases h.2 with h | h <;> exact absurd h (by decide)

/-- `triangle` with its links at `0` rolled up, for the examples. -/
theorem triangle_isLoopUnrolling :
    IsLoopUnrolling ((triangle.gainBiased (1 : Fin 3 → ℚˣ)).rollUp 0
      ((triangle.gainBiased (1 : Fin 3 → ℚˣ)).linkStar 0)) triangle id 0
      ((triangle.gainBiased (1 : Fin 3 → ℚˣ)).linkStar 0) :=
  (gainBiased_one_isBalanced triangle).isLoopUnrolling_rollUp 0

example := (isCycle_singleton_of_loop (G := twoLoops) (e := true) rfl).tail_eq_head_of_singleton

example := triangle_isLoopUnrolling.isJoint triangle_zero_mem_linkStar

example := triangle_isLoopUnrolling.isLink triangle_zero_mem_linkStar

example := triangle_isLoopUnrolling.keep_of_isLink (e := 1) fun h => absurd
  ((rollUpGraph_tail_of_notMem triangle_one_notMem_linkStar).symm.trans
    (h.trans (rollUpGraph_head_of_notMem triangle_one_notMem_linkStar))) (by decide)

example := triangle_isLoopUnrolling.eq_singleton_of_notMem
  (isCycle_singleton_of_loop (triangle_isLoopUnrolling.loop 0 triangle_zero_mem_linkStar))
  (triangle_isLoopUnrolling.isJoint triangle_zero_mem_linkStar).2

example := triangle_isLoopUnrolling.tail_eq_of_mem_verts triangle_zero_mem_linkStar
  (mem_verts.2 ⟨0, rfl, Or.inl rfl⟩)

example := isLoopUnrolling_unrollToNew twoLoops_unbalanced_singleton

example := (gainBiased_one_isBalanced (Γ := ℚˣ) triangle).isLoopUnrolling_rollUp 1

example := twoLoops_unbalanced_singleton {true} ⟨isCycle_singleton_of_loop rfl, Set.notMem_empty _⟩

end Transformer.GainMatroid

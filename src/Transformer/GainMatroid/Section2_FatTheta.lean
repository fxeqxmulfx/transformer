module

public import Transformer.GainMatroid.Section2_ExchangeGraphs
public import Transformer.GainMatroid.Section3_Families

/-!
# Proposition 2.12: two balancing vertices

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5, Proposition 2.12, credited to Zaslavsky: a
2-connected unbalanced biased graph with two distinct balancing vertices `x` and `y` is a union of
subgraphs meeting pairwise in `{x, y}`, whose balanced cycles are those inside one subgraph
(`BiasedGraph.exists_fatTheta`).

The subgraph `G_i` is given by the edges of class `i`, and its vertices are their ends. A
2-connected graph in the sense of §2.1 may have isolated vertices, which lie on no cycle and in no
`G_i`.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

namespace BiasedGraph

variable {V E : Type*} {Ω : BiasedGraph V E}

/-- **Proposition 2.12** (Zaslavsky). "Let `(G, B)` be a 2-connected unbalanced biased graph with
two distinct balancing vertices `x` and `y`. Then `G` is a union of subgraphs `G₁ ∪ ⋯ ∪ G_m` where
for each pair `i ≠ j`, `G_i ∩ G_j = {x, y}`, and a cycle is in `B` if and only if it is contained in
a single subgraph `G_i`. If `m ≥ 3` then `x` and `y` are the only balancing vertices of `(G, B)`."
Here `G_i` is spanned by the edges `e` with `c e = i`. The graph is finite, as in §2.1. -/
theorem exists_fatTheta [Finite V] [Finite E] (h2 : Ω.graph.IsKConnected 2) (hΩ : ¬Ω.IsBalanced)
    {x y : V} (hxy : x ≠ y) (hx : Ω.IsBalancingVertex x) (hy : Ω.IsBalancingVertex y) :
    ∃ (m : ℕ) (c : E → Fin m),
      (∀ i j, i ≠ j → Ω.graph.verts {e | c e = i} ∩ Ω.graph.verts {e | c e = j} = {x, y}) ∧
      (∀ C, Ω.graph.IsCycle C → (C ∈ Ω.balanced ↔ ∃ i, C ⊆ {e | c e = i})) ∧
      (3 ≤ m → ∀ w, Ω.IsBalancingVertex w → w = x ∨ w = y) := by
  sorry

end BiasedGraph

/-- The hypotheses of `BiasedGraph.exists_fatTheta` hold for a triangle with its cycle unbalanced:
each vertex lies on the only cycle. -/
example : triangle.contrabalanced.graph.IsKConnected 2 ∧ ¬triangle.contrabalanced.IsBalanced ∧
    (0 : Fin 3) ≠ 1 ∧ triangle.contrabalanced.IsBalancingVertex 0 ∧
    triangle.contrabalanced.IsBalancingVertex 1 := by
  have hall : ∀ C, triangle.IsCycle C → ∀ v : Fin 3, v ∈ triangle.verts C := by
    intro C hC v
    have hC' : triangle.IsCycle ((C.toFinite.toFinset : Finset (Fin 3)) : Set (Fin 3)) := by
      simpa using hC
    have hv : v ∈ C.toFinite.toFinset := by
      rw [(triangle_isCycle_iff _).1 hC']
      fin_cases v <;> simp
    exact mem_verts.2 ⟨v, by simpa using hv, Or.inl (by fin_cases v <;> rfl)⟩
  refine ⟨triangle_isKConnected, fun h => ?_, by decide, fun C hC => hall C hC.1 0,
    fun C hC => hall C hC.1 1⟩
  exact Set.notMem_empty _ (h _ triangle_isLabelledTriangle.isCycle)

end Transformer.GainMatroid

module

public import Transformer.GainMatroid.Section2_DeltaY

/-!
# The prism `Pr` and its contractions `Pr₂`, `Pr₁`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §1.2 and §3.2. `Pr` is "the biased graph consisting
of the triangular prism with just its two triangles balanced" (`prismBiased`). Let `L` be the
matching of three edges linking the two triangles; `Pr₂` and `Pr₁` are "obtained from `Pr` by
contracting 1 and 2 edges, respectively" of `L` (`prismTwoBiased`, `prismOneBiased`), so that
`Pr_k` keeps `k` edges of `L`. In each of them the two triangles share no edge, and nothing else is
balanced.

A family of pairwise edge-disjoint cycles has the theta property: two cycles of a theta share the
edges of one of its paths, which has an edge because its ends are distinct
(`OrientedGraph.disjointCyclesBiased`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

variable {V E : Type*}

namespace OrientedGraph

/-- A graph with a family of pairwise edge-disjoint cycles as its balanced cycles. Two cycles of a
theta share the edges of a path between distinct vertices, so no theta has two of them. -/
def disjointCyclesBiased (G : OrientedGraph V E) (B : Set (Set E)) (hB : ∀ C ∈ B, G.IsCycle C)
    (hdisj : B.Pairwise Disjoint) : BiasedGraph V E where
  graph := G
  balanced := B
  isCycle_of_mem := hB
  theta := by
    intro x y P Q R h hPQ hPR
    exfalso
    obtain ⟨f, hf⟩ := List.exists_mem_of_ne_nil _ (Walk.edges_ne_nil_of_ne h.ne P)
    by_cases heq : P.edgeSet ∪ Q.edgeSet = P.edgeSet ∪ R.edgeSet
    · obtain ⟨g, hg⟩ := List.exists_mem_of_ne_nil _ (Walk.edges_ne_nil_of_ne h.ne Q)
      have hmem : g ∈ P.edgeSet ∪ R.edgeSet := heq ▸ Or.inr hg
      rcases hmem with hp | hr
      · exact h.openPQ.edges g hp hg
      · exact h.openQR.edges g hg hr
    · exact Set.disjoint_left.1 (hdisj hPQ hPR heq) (Or.inl hf) (Or.inl hf)

/-- The triangular prism: the triangles `x₁x₂x₃ = 0, 1, 2` and `y₁y₂y₃ = 3, 4, 5`, with edges
`0, 1, 2` and `3, 4, 5`, and the matching `L = {x₁y₁, x₂y₂, x₃y₃}` of edges `6, 7, 8`. -/
def prism : OrientedGraph (Fin 6) (Fin 9) :=
  ⟨![0, 1, 2, 3, 4, 5, 0, 1, 2], ![1, 2, 0, 4, 5, 3, 3, 4, 5]⟩

/-- The prism with `x₁y₁` contracted: `0 = x₁ = y₁`, `1 = x₂`, `2 = x₃`, `3 = y₂`, `4 = y₃`. The
triangles are the edges `0, 1, 2` and `3, 4, 5`, and `6, 7` are the edges `x₂y₂`, `x₃y₃` left of
`L`. -/
def prismTwo : OrientedGraph (Fin 5) (Fin 8) :=
  ⟨![0, 1, 2, 0, 3, 4, 1, 2], ![1, 2, 0, 3, 4, 0, 3, 4]⟩

/-- The prism with `x₁y₁` and `x₂y₂` contracted: `0 = x₁ = y₁`, `1 = x₂ = y₂`, `2 = x₃`, `3 = y₃`.
The triangles are the edges `0, 1, 2` and `3, 4, 5`, and `6` is the edge `x₃y₃` left of `L`. -/
def prismOne : OrientedGraph (Fin 4) (Fin 7) :=
  ⟨![0, 1, 2, 0, 1, 3, 2], ![1, 2, 0, 1, 3, 0, 3]⟩

theorem prism_isCycle_left : prism.IsCycle {0, 1, 2} := by
  have h : prism.IsLabelledTriangle 1 2 0 0 1 2 := by
    refine ⟨Or.inl ⟨rfl, rfl⟩, Or.inr ⟨rfl, rfl⟩, Or.inl ⟨rfl, rfl⟩, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
      decide
  have hs : ({1, 2, 0} : Set (Fin 9)) = {0, 1, 2} := by grind
  exact hs ▸ h.isCycle

theorem prism_isCycle_right : prism.IsCycle {3, 4, 5} := by
  have h : prism.IsLabelledTriangle 4 5 3 3 4 5 := by
    refine ⟨Or.inl ⟨rfl, rfl⟩, Or.inr ⟨rfl, rfl⟩, Or.inl ⟨rfl, rfl⟩, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
      decide
  have hs : ({4, 5, 3} : Set (Fin 9)) = {3, 4, 5} := by grind
  exact hs ▸ h.isCycle

theorem prismTwo_isCycle_left : prismTwo.IsCycle {0, 1, 2} := by
  have h : prismTwo.IsLabelledTriangle 1 2 0 0 1 2 := by
    refine ⟨Or.inl ⟨rfl, rfl⟩, Or.inr ⟨rfl, rfl⟩, Or.inl ⟨rfl, rfl⟩, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
      decide
  have hs : ({1, 2, 0} : Set (Fin 8)) = {0, 1, 2} := by grind
  exact hs ▸ h.isCycle

theorem prismTwo_isCycle_right : prismTwo.IsCycle {3, 4, 5} := by
  have h : prismTwo.IsLabelledTriangle 4 5 3 0 3 4 := by
    refine ⟨Or.inl ⟨rfl, rfl⟩, Or.inr ⟨rfl, rfl⟩, Or.inl ⟨rfl, rfl⟩, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
      decide
  have hs : ({4, 5, 3} : Set (Fin 8)) = {3, 4, 5} := by grind
  exact hs ▸ h.isCycle

theorem prismOne_isCycle_left : prismOne.IsCycle {0, 1, 2} := by
  have h : prismOne.IsLabelledTriangle 1 2 0 0 1 2 := by
    refine ⟨Or.inl ⟨rfl, rfl⟩, Or.inr ⟨rfl, rfl⟩, Or.inl ⟨rfl, rfl⟩, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
      decide
  have hs : ({1, 2, 0} : Set (Fin 7)) = {0, 1, 2} := by grind
  exact hs ▸ h.isCycle

theorem prismOne_isCycle_right : prismOne.IsCycle {3, 4, 5} := by
  have h : prismOne.IsLabelledTriangle 4 5 3 0 1 3 := by
    refine ⟨Or.inl ⟨rfl, rfl⟩, Or.inr ⟨rfl, rfl⟩, Or.inl ⟨rfl, rfl⟩, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
      decide
  have hs : ({4, 5, 3} : Set (Fin 7)) = {3, 4, 5} := by grind
  exact hs ▸ h.isCycle

end OrientedGraph

/-- Two disjoint triangles, as a family of pairwise disjoint sets. -/
theorem pairwise_disjoint_triangles {n : ℕ} (a b c d e f : Fin n)
    (h : Disjoint ({a, b, c} : Set (Fin n)) {d, e, f}) :
    ({{a, b, c}, {d, e, f}} : Set (Set (Fin n))).Pairwise Disjoint := by
  rintro X (rfl | rfl) Y (rfl | rfl) hXY
  · exact (hXY rfl).elim
  · exact h
  · exact h.symm
  · exact (hXY rfl).elim

/-- **`Pr`** (§3.2): the triangular prism with just its two triangles balanced. -/
def prismBiased : BiasedGraph (Fin 6) (Fin 9) :=
  prism.disjointCyclesBiased {{0, 1, 2}, {3, 4, 5}}
    (by rintro C (rfl | rfl); exacts [prism_isCycle_left, prism_isCycle_right])
    (pairwise_disjoint_triangles _ _ _ _ _ _ (by simp))

/-- **`Pr₂`** (§3.2): `Pr` with one edge of `L` contracted. -/
def prismTwoBiased : BiasedGraph (Fin 5) (Fin 8) :=
  prismTwo.disjointCyclesBiased {{0, 1, 2}, {3, 4, 5}}
    (by rintro C (rfl | rfl); exacts [prismTwo_isCycle_left, prismTwo_isCycle_right])
    (pairwise_disjoint_triangles _ _ _ _ _ _ (by simp))

/-- **`Pr₁`** (§3.2): `Pr` with two edges of `L` contracted. -/
def prismOneBiased : BiasedGraph (Fin 4) (Fin 7) :=
  prismOne.disjointCyclesBiased {{0, 1, 2}, {3, 4, 5}}
    (by rintro C (rfl | rfl); exacts [prismOne_isCycle_left, prismOne_isCycle_right])
    (pairwise_disjoint_triangles _ _ _ _ _ _ (by simp))

example : prismBiased.balanced = {{0, 1, 2}, {3, 4, 5}} := rfl

example : prismOneBiased.graph = prismOne := rfl

example : prismTwoBiased.graph = prismTwo := rfl

example := pairwise_disjoint_triangles (0 : Fin 6) 1 2 3 4 5 (by simp)

end Transformer.GainMatroid

module

public import Mathlib.Data.Set.Card
public import Mathlib.Order.Zorn
public import Mathlib.Data.Set.Finite.Lattice
public import Transformer.GainMatroid.Section2_Minors

/-!
# Separations, connectivity and maximal forests

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.1. `G.verts X` is the vertex set `V(X)` of the
subgraph induced by a set of edges. A `k`-separation is a partition `(A, B)` of the edges with
`|A|, |B| ≥ k` and `|V(A) ∩ V(B)| = k`, and it is vertical when both `V(A) - V(B)` and
`V(B) - V(A)` are nonempty. A graph on at least `k + 2` vertices is `k`-connected if it has no
vertical `l`-separation for `l < k`, and a graph on `k + 1` vertices is `k`-connected if it has a
spanning complete subgraph. The source leaves smaller graphs out, so they are not `k`-connected.

The module also extends any forest to a maximal one (`exists_maximal_isForest`), the step that the
source uses each time it normalizes a gain function on a maximal forest containing a given one.
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE

variable {V : Type uV} {E : Type uE}

namespace OrientedGraph

variable {G : OrientedGraph V E}

/-- `V(X)`: the vertices of the subgraph induced by a set of edges (§2.1). -/
def verts (G : OrientedGraph V E) (X : Set E) : Set V := G.tail '' X ∪ G.head '' X

theorem mem_verts {X : Set E} {v : V} :
    v ∈ G.verts X ↔ ∃ e ∈ X, G.tail e = v ∨ G.head e = v := by
  simp only [verts, Set.mem_union, Set.mem_image]
  constructor
  · rintro (⟨e, he, h⟩ | ⟨e, he, h⟩)
    · exact ⟨e, he, Or.inl h⟩
    · exact ⟨e, he, Or.inr h⟩
  · rintro ⟨e, he, h | h⟩
    · exact Or.inl ⟨e, he, h⟩
    · exact Or.inr ⟨e, he, h⟩

theorem verts_mono {X Y : Set E} (h : X ⊆ Y) : G.verts X ⊆ G.verts Y :=
  Set.union_subset_union (Set.image_mono h) (Set.image_mono h)

/-- A **`k`-separation** of `G` (§2.1): a partition `(A, B)` of the edges with `|A| ≥ k`, `|B| ≥ k`
and `|V(A) ∩ V(B)| = k`. -/
structure IsSeparation (G : OrientedGraph V E) (k : ℕ) (A B : Set E) : Prop where
  /-- The two sides are disjoint. -/
  disjoint : Disjoint A B
  /-- The two sides cover the edges. -/
  union : A ∪ B = Set.univ
  /-- The first side has at least `k` edges. -/
  left : (k : ℕ∞) ≤ A.encard
  /-- The second side has at least `k` edges. -/
  right : (k : ℕ∞) ≤ B.encard
  /-- The two sides share exactly `k` vertices. -/
  inter : (G.verts A ∩ G.verts B).encard = k

/-- A **vertical `k`-separation** (§2.1): a `k`-separation with a vertex on each side only. -/
def IsVerticalSeparation (G : OrientedGraph V E) (k : ℕ) (A B : Set E) : Prop :=
  G.IsSeparation k A B ∧ (G.verts A \ G.verts B).Nonempty ∧ (G.verts B \ G.verts A).Nonempty

/-- `G` is **`k`-connected** (§2.1): on at least `k + 2` vertices it has no vertical `l`-separation
for any `l < k`, and on exactly `k + 1` vertices it has a spanning complete subgraph. -/
def IsKConnected (G : OrientedGraph V E) (k : ℕ) : Prop :=
  ((k : ℕ∞) + 2 ≤ (Set.univ : Set V).encard ∧ ∀ l < k, ∀ A B, ¬G.IsVerticalSeparation l A B) ∨
    ((Set.univ : Set V).encard = (k : ℕ∞) + 1 ∧ ∀ u v, u ≠ v → ∃ e, G.Joins e u v)

/-- The empty set of edges is a forest. -/
theorem isForest_empty : G.IsForest ∅ := by
  rintro C ⟨u, W, hW, rfl⟩ hle
  obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hW.edges_ne_nil
  exact hle he

/-- Every forest lies in a maximal forest. -/
theorem exists_maximal_isForest {F : Set E} (hF : G.IsForest F) :
    ∃ F', F ⊆ F' ∧ Maximal G.IsForest F' := by
  obtain ⟨m, hsub, hmax⟩ := zorn_subset_nonempty {S | G.IsForest S} (fun c hc hch hne => by
    refine ⟨⋃₀ c, ?_, fun s hs => Set.subset_sUnion_of_mem hs⟩
    rintro C ⟨u, W, hW, rfl⟩ hle
    obtain ⟨t, htc, hsubt⟩ := DirectedOn.exists_mem_subset_of_finite_of_subset_sUnion hne
      hch.directedOn (List.finite_toSet W.edges) hle
    exact hc htc _ ⟨u, W, hW, rfl⟩ hsubt) F hF
  exact ⟨m, hsub, hmax⟩

/-- A path on three vertices, for the examples. -/
def pathThree : OrientedGraph (Fin 3) (Fin 2) := ⟨![0, 1], ![1, 2]⟩

/-- A triangle, for the examples. -/
def triangle : OrientedGraph (Fin 3) (Fin 3) := ⟨![0, 1, 2], ![1, 2, 0]⟩

/-- The two edges of the path form a vertical 1-separation at the middle vertex. -/
theorem pathThree_isVerticalSeparation : pathThree.IsVerticalSeparation 1 {0} {1} := by
  have hA : pathThree.verts {0} = {0, 1} := by
    ext v
    fin_cases v <;> simp [verts, pathThree]
  have hB : pathThree.verts {1} = {1, 2} := by
    ext v
    fin_cases v <;> simp [verts, pathThree]
  have hAB : pathThree.verts {0} ∩ pathThree.verts {1} = {1} := by
    rw [hA, hB]
    ext v
    fin_cases v <;> simp
  refine ⟨⟨by simp, ?_, by simp, by simp, by simp [hAB]⟩, ⟨0, ?_⟩, ⟨2, ?_⟩⟩
  · ext e
    fin_cases e <;> simp
  · rw [hA, hB]
    simp
  · rw [hA, hB]
    simp

/-- A triangle is 2-connected: it has three vertices and a spanning complete subgraph. -/
theorem triangle_isKConnected : triangle.IsKConnected 2 := by
  refine Or.inr ⟨by simp [Set.encard_univ]; norm_num, ?_⟩
  simp only [Joins, triangle]
  decide

/-- The path on three vertices is not 2-connected: its ends are not adjacent. -/
theorem pathThree_not_isKConnected : ¬pathThree.IsKConnected 2 := by
  rintro (⟨h, -⟩ | ⟨-, h⟩)
  · have hcard : (Set.univ : Set (Fin 3)).encard = 3 := by simp [Set.encard_univ]
    rw [hcard] at h
    norm_num at h
  · obtain ⟨e, he⟩ := h 0 2 (by decide)
    fin_cases e <;> simp [Joins, pathThree] at he

example : pumpkin.IsForest ∅ := isForest_empty

example : ∃ F', ({0} : Set (Fin 3)) ⊆ F' ∧ Maximal pumpkin.IsForest F' :=
  exists_maximal_isForest pumpkin_isForest

example : pathThree.verts {0} ⊆ pathThree.verts Set.univ :=
  verts_mono (Set.subset_univ _)

example : (1 : Fin 3) ∈ pathThree.verts {0} :=
  mem_verts.2 ⟨0, rfl, Or.inr rfl⟩

example : ∃ (u : Fin 2) (W : pumpkin.Walk u u), W.IsCycleWalk ∧ (1 : Fin 3) ∈ W.edges ∧
    ∀ f ∈ W.edges, f ≠ 1 → f ∈ ({0} : Set (Fin 3)) :=
  exists_cycleWalk_of_maximal pumpkin_maximal (by simp)

end OrientedGraph

end Transformer.GainMatroid

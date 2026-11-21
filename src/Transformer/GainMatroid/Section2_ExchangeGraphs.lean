module

public import Transformer.GainMatroid.Section2_DeltaYExamples
public import Transformer.GainMatroid.Section2_MatroidExchange

/-!
# The cycles of the small graphs of the exchanges

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. The smallest graphs on which the exchanges of
§2.4 can be tested. A triangle with its cycle balanced becomes a star, which has no cycles. The
graph `clawGraph`, which is `K₄` minus the edge `yz`, is the smallest with a star that is a triad,
and its Y-Δ exchange is `clawNabla`: a triangle with two doubled edges.

The cycles are listed through the degree lemma `IsCycle.card_incident`, and `clawNabla_balanced_eq`
evaluates `∇_Y B` for the balanced `clawGraph`: all six cycles of `clawNabla` are balanced.
-/

@[expose] public section

namespace Transformer.GainMatroid

open scoped symmDiff

namespace OrientedGraph

variable {V E : Type*}

/-- Two oriented graphs with the same tails and heads are equal. -/
theorem eq_of_tail_head {G H : OrientedGraph V E} (ht : G.tail = H.tail) (hh : G.head = H.head) :
    G = H := by
  cases G
  cases H
  simp_all

/-- A cycle has an edge. -/
theorem IsCycle.nonempty {G : OrientedGraph V E} {C : Set E} (hC : G.IsCycle C) : C.Nonempty := by
  obtain ⟨u, W, hW, rfl⟩ := hC
  obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hW.edges_ne_nil
  exact ⟨e, he⟩

/-- `Δ_X` of the triangle: a star at the new vertex `none`. -/
def triangleStar : OrientedGraph (Option (Fin 3)) (Fin 3) :=
  ⟨![some 2, some 0, some 1], fun _ => none⟩

theorem triangle_deltaGraph : triangle.deltaGraph 1 2 0 0 1 2 = triangleStar := by
  refine eq_of_tail_head ?_ ?_ <;> funext e <;> fin_cases e <;> simp [deltaGraph, triangleStar]

theorem triangleStar_loopless : triangleStar.Loopless := by decide

theorem triangle_loopless : triangle.Loopless := by decide

set_option maxRecDepth 100000 in
private theorem triangleStar_degree : ∀ C : Finset (Fin 3), C.Nonempty →
    ¬∀ w : Option (Fin 3),
      (C.filter fun e => triangleStar.tail e = w ∨ triangleStar.head e = w).card = 0 ∨
        (C.filter fun e => triangleStar.tail e = w ∨ triangleStar.head e = w).card = 2 := by
  decide +kernel

/-- A star has no cycles. -/
theorem triangleStar_not_isCycle (C : Set (Fin 3)) : ¬triangleStar.IsCycle C := by
  intro hC
  obtain ⟨C, rfl⟩ := (Set.toFinite C).exists_finset_coe
  exact triangleStar_degree C (by exact_mod_cast hC.nonempty)
    (IsCycle.card_incident triangleStar_loopless hC)

set_option maxRecDepth 100000 in
private theorem triangle_degree : ∀ C : Finset (Fin 3), C.Nonempty →
    (∀ w : Fin 3, (C.filter fun e => triangle.tail e = w ∨ triangle.head e = w).card = 0 ∨
      (C.filter fun e => triangle.tail e = w ∨ triangle.head e = w).card = 2) →
    C = {1, 2, 0} := by
  decide +kernel

/-- The only cycle of `triangle` is the triangle. -/
theorem triangle_isCycle_iff (C : Finset (Fin 3)) :
    triangle.IsCycle (C : Set (Fin 3)) ↔ C = {1, 2, 0} := by
  constructor
  · intro hC
    exact triangle_degree C (by exact_mod_cast hC.nonempty)
      (IsCycle.card_incident triangle_loopless hC)
  · rintro rfl
    have h := triangle_isLabelledTriangle.isCycle
    simpa using h

/-- `∇_Y` of `clawGraph`, on the vertices `1`, `2`, `3`: the edges `0`, `{1, 4}` and `{2, 3}` join
`2` to `3`, `1` to `3` and `1` to `2`. -/
def clawNabla : OrientedGraph (Fin 4) (Fin 5) := ⟨![2, 1, 1, 2, 1], ![3, 3, 2, 1, 3]⟩

theorem clawGraph_nablaGraph : clawGraph.nablaGraph 0 1 2 1 2 3 = clawNabla := by
  refine eq_of_tail_head ?_ ?_ <;> funext e <;> fin_cases e <;>
    simp [nablaGraph, clawGraph, clawNabla]

theorem clawGraph_loopless : clawGraph.Loopless := by decide

theorem clawNabla_loopless : clawNabla.Loopless := by decide

/-- The three cycles of `clawGraph`, with the four-cycle written along its walk. -/
def clawCycles : Finset (Finset (Fin 5)) := {{0, 1, 3}, {0, 2, 4}, {1, 3, 4, 2}}

/-- The six cycles of `clawNabla`: two digons and four triangles. -/
def nablaCycles : Finset (Finset (Fin 5)) :=
  {{2, 3}, {1, 4}, {0, 1, 2}, {0, 1, 3}, {0, 2, 4}, {0, 3, 4}}

set_option maxRecDepth 100000 in
set_option synthInstance.maxSize 4096 in
set_option synthInstance.maxHeartbeats 400000 in
private theorem claw_degree : ∀ C : Finset (Fin 5), C.Nonempty →
    (∀ w : Fin 4, (C.filter fun e => clawGraph.tail e = w ∨ clawGraph.head e = w).card = 0 ∨
      (C.filter fun e => clawGraph.tail e = w ∨ clawGraph.head e = w).card = 2) →
    C ∈ clawCycles := by
  decide +kernel

set_option maxRecDepth 100000 in
set_option synthInstance.maxSize 4096 in
set_option synthInstance.maxHeartbeats 400000 in
private theorem nabla_degree : ∀ C : Finset (Fin 5), C.Nonempty →
    (∀ w : Fin 4, (C.filter fun e => clawNabla.tail e = w ∨ clawNabla.head e = w).card = 0 ∨
      (C.filter fun e => clawNabla.tail e = w ∨ clawNabla.head e = w).card = 2) →
    C ∈ nablaCycles := by
  decide +kernel

theorem clawGraph_isCycle_iff (C : Finset (Fin 5)) :
    clawGraph.IsCycle (C : Set (Fin 5)) ↔ C ∈ clawCycles := by
  constructor
  · intro hC
    exact claw_degree C (by exact_mod_cast hC.nonempty)
      (IsCycle.card_incident clawGraph_loopless hC)
  · intro h
    simp only [clawCycles, Finset.mem_insert, Finset.mem_singleton] at h
    rcases h with rfl | rfl | rfl <;> push_cast
    · exact isCycle_of_three (G := clawGraph) (x := 1) (y := 0) (z := 2) (Or.inr ⟨rfl, rfl⟩)
        (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)
    · exact isCycle_of_three (G := clawGraph) (x := 1) (y := 0) (z := 3) (Or.inr ⟨rfl, rfl⟩)
        (Or.inl ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide)
    · exact isCycle_of_four (G := clawGraph) (x := 0) (y := 2) (z := 1) (t := 3) (Or.inl ⟨rfl, rfl⟩)
        (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide)

theorem clawNabla_isCycle_iff (C : Finset (Fin 5)) :
    clawNabla.IsCycle (C : Set (Fin 5)) ↔ C ∈ nablaCycles := by
  constructor
  · intro hC
    exact nabla_degree C (by exact_mod_cast hC.nonempty)
      (IsCycle.card_incident clawNabla_loopless hC)
  · intro h
    simp only [nablaCycles, Finset.mem_insert, Finset.mem_singleton] at h
    rcases h with rfl | rfl | rfl | rfl | rfl | rfl <;> push_cast
    · exact isCycle_pair (G := clawNabla) (u := 1) (v := 2) (Or.inl ⟨rfl, rfl⟩)
        (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide)
    · exact isCycle_pair (G := clawNabla) (u := 1) (v := 3) (Or.inl ⟨rfl, rfl⟩)
        (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)
    · exact isCycle_of_three (G := clawNabla) (x := 2) (y := 3) (z := 1) (Or.inl ⟨rfl, rfl⟩)
        (Or.inr ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)
    · exact isCycle_of_three (G := clawNabla) (x := 2) (y := 3) (z := 1) (Or.inl ⟨rfl, rfl⟩)
        (Or.inr ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide)
    · exact isCycle_of_three (G := clawNabla) (x := 3) (y := 2) (z := 1) (Or.inr ⟨rfl, rfl⟩)
        (Or.inr ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)
    · exact isCycle_of_three (G := clawNabla) (x := 3) (y := 2) (z := 1) (Or.inr ⟨rfl, rfl⟩)
        (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)

end OrientedGraph

end Transformer.GainMatroid

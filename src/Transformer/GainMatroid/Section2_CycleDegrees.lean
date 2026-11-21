module

public import Transformer.GainMatroid.Section4_Graphs

/-!
# Degrees in a cycle, and the cycles of `K₄`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §3: "The set of cycles of the graph `K₄` consists of
four triangles and three quadrilaterals" (`K4_isCycle_iff`). In a loopless graph every vertex meets
a cycle in zero or two edges (`IsCycle.card_incident`), and on `K₄` this decides which edge sets are
cycles.
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

namespace OrientedGraph

variable {G : OrientedGraph V E}

section Degree

variable [DecidableEq V]

/-- In a loopless graph each step of a walk meets `w` at exactly one end or not at all, so the
edges at `w` and the two ends of the walk together count every visit to `w` twice. -/
theorem Walk.countP_incident (hG : G.Loopless) {a b : V} (p : G.Walk a b) (w : V) :
    p.edges.countP (fun e => decide (G.tail e = w ∨ G.head e = w)) +
        (if w = a then 1 else 0) + (if w = b then 1 else 0) =
      2 * p.support.count w := by
  induction p with
  | nil u =>
    by_cases h : w = u
    · subst h
      simp [Walk.edges, Walk.support]
    · have h' : ¬u = w := fun hu => h hu.symm
      simp [Walk.edges, Walk.support, h, h']
  | cons e he p ih =>
    rename_i u t b
    have hut : u ≠ t := by
      intro h
      rcases he with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> exact hG e (by rw [h1, h2, h])
    have hP : (G.tail e = w ∨ G.head e = w) ↔ (w = u ∨ w = t) := by
      rcases he with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> rw [h1, h2] <;> constructor <;>
        rintro (h | h) <;> simp [h]
    simp only [Walk.edges, Walk.support, List.countP_cons, List.count_cons, hP, decide_eq_true_eq,
      beq_iff_eq]
    by_cases hu : w = u
    · subst hu
      simp only [hut, ↓reduceIte, true_or] at ih ⊢
      omega
    · have hu' : ¬u = w := fun h => hu h.symm
      by_cases ht : w = t
      · subst ht
        simp only [hu, hu', ↓reduceIte, or_true] at ih ⊢
        omega
      · simp only [hu, hu', ht, ↓reduceIte, or_self] at ih ⊢
        omega

/-- A vertex meets a cycle of a loopless graph in zero or two edges, counted along a cycle walk. -/
theorem Walk.IsCycleWalk.countP_incident (hG : G.Loopless) {u : V} {W : G.Walk u u}
    (hW : W.IsCycleWalk) (w : V) :
    W.edges.countP (fun e => decide (G.tail e = w ∨ G.head e = w)) = 0 ∨
      W.edges.countP (fun e => decide (G.tail e = w ∨ G.head e = w)) = 2 := by
  have h := Walk.countP_incident hG W w
  have hu : u ∈ W.support.tail := by
    cases W with
    | nil => exact absurd rfl hW.edges_ne_nil
    | cons e he p => exact p.end_mem_support
  have hc : W.support.tail.count w = if w ∈ W.support.tail then 1 else 0 :=
    hW.support_tail_nodup.count
  have hs : W.support.count w = W.support.tail.count w + if u = w then 1 else 0 := by
    conv_lhs => rw [W.support_eq_cons, List.count_cons]
    simp
  rw [hs] at h
  by_cases hwu : w = u
  · subst hwu
    rw [ite_eq_left hu] at hc
    simp only [↓reduceIte] at h
    omega
  · have hwu' : ¬u = w := fun h => hwu h.symm
    simp only [hwu, hwu', ↓reduceIte] at h
    split_ifs at hc <;> omega

/-- **Degrees in a cycle**: a vertex of a loopless graph meets a cycle in zero or two edges. -/
theorem IsCycle.card_incident [DecidableEq E] (hG : G.Loopless) {C : Finset E}
    (hC : G.IsCycle (C : Set E)) (w : V) :
    (C.filter fun e => G.tail e = w ∨ G.head e = w).card = 0 ∨
      (C.filter fun e => G.tail e = w ∨ G.head e = w).card = 2 := by
  obtain ⟨u, W, hW, hWC⟩ := hC
  have hset : W.edges.toFinset = C := by
    ext e
    simpa [Walk.edgeSet] using Set.ext_iff.1 hWC e
  have hcard : (C.filter fun e => G.tail e = w ∨ G.head e = w).card =
      W.edges.countP (fun e => decide (G.tail e = w ∨ G.head e = w)) := by
    rw [List.countP_eq_length_filter, ← List.toFinset_card_of_nodup (hW.edges_nodup.filter _),
      List.toFinset_filter, hset]
    simp
  rw [hcard]
  exact hW.countP_incident hG w

end Degree

/-- A closed walk of three edges on three distinct vertices is a cycle. -/
theorem isCycle_of_three {a b c : E} {x y z : V} (ha : G.Joins a x y) (hb : G.Joins b y z)
    (hc : G.Joins c z x) (he : [a, b, c].Nodup) (hv : [y, z, x].Nodup) :
    G.IsCycle {a, b, c} := by
  refine ⟨x, .cons a ha (.cons b hb (.cons c hc (.nil x))),
    ⟨by simp [Walk.edges], he, hv⟩, ?_⟩
  ext f
  simp only [Walk.edgeSet, Walk.edges, Set.mem_insert_iff, Set.mem_singleton_iff,
    Set.mem_ofPred_eq, List.mem_cons, List.not_mem_nil, or_false]

/-- A closed walk of four edges on four distinct vertices is a cycle. -/
theorem isCycle_of_four {a b c d : E} {x y z t : V} (ha : G.Joins a x y) (hb : G.Joins b y z)
    (hc : G.Joins c z t) (hd : G.Joins d t x) (he : [a, b, c, d].Nodup)
    (hv : [y, z, t, x].Nodup) : G.IsCycle {a, b, c, d} := by
  refine ⟨x, .cons a ha (.cons b hb (.cons c hc (.cons d hd (.nil x)))),
    ⟨by simp [Walk.edges], he, hv⟩, ?_⟩
  ext f
  simp only [Walk.edgeSet, Walk.edges, Set.mem_insert_iff, Set.mem_singleton_iff,
    Set.mem_ofPred_eq, List.mem_cons, List.not_mem_nil, or_false]

/-- The four triangles and the three quadrilaterals of `K4`. -/
def K4Cycles : Finset (Finset (Fin 6)) :=
  {{0, 1, 3}, {0, 2, 4}, {1, 2, 5}, {3, 4, 5}, {1, 2, 4, 3}, {0, 2, 5, 3}, {0, 1, 5, 4}}

set_option maxRecDepth 100000 in
set_option synthInstance.maxSize 4096 in
set_option synthInstance.maxHeartbeats 400000 in
private theorem K4_mem_of_degree : ∀ C : Finset (Fin 6), C.Nonempty →
    (∀ w : Fin 4, (C.filter fun e => K4.tail e = w ∨ K4.head e = w).card = 0 ∨
      (C.filter fun e => K4.tail e = w ∨ K4.head e = w).card = 2) → C ∈ K4Cycles := by
  decide +kernel

/-- **§3**: "The set of cycles of the graph `K₄` consists of four triangles and three
quadrilaterals." -/
theorem K4_isCycle_iff (C : Finset (Fin 6)) : K4.IsCycle (C : Set (Fin 6)) ↔ C ∈ K4Cycles := by
  constructor
  · intro hC
    refine K4_mem_of_degree C ?_ (IsCycle.card_incident K4_loopless hC)
    obtain ⟨u, W, hW, hWC⟩ := hC
    obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hW.edges_ne_nil
    have heC : e ∈ (C : Set (Fin 6)) := hWC ▸ (he : e ∈ W.edgeSet)
    exact ⟨e, Finset.mem_coe.1 heC⟩
  · intro h
    simp only [K4Cycles, Finset.mem_insert, Finset.mem_singleton] at h
    rcases h with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> push_cast
    · exact isCycle_of_three (G := K4) (x := 1) (y := 0) (z := 2) (Or.inr ⟨rfl, rfl⟩)
        (Or.inl ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide)
    · exact isCycle_of_three (G := K4) (x := 1) (y := 0) (z := 3) (Or.inr ⟨rfl, rfl⟩)
        (Or.inl ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide)
    · exact isCycle_of_three (G := K4) (x := 2) (y := 0) (z := 3) (Or.inr ⟨rfl, rfl⟩)
        (Or.inl ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide)
    · exact isCycle_of_three (G := K4) (x := 2) (y := 1) (z := 3) (Or.inr ⟨rfl, rfl⟩)
        (Or.inl ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide)
    · exact isCycle_of_four (G := K4) (x := 2) (y := 0) (z := 3) (t := 1) (Or.inr ⟨rfl, rfl⟩)
        (Or.inl ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)
    · exact isCycle_of_four (G := K4) (x := 1) (y := 0) (z := 3) (t := 2) (Or.inr ⟨rfl, rfl⟩)
        (Or.inl ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide)
    · exact isCycle_of_four (G := K4) (x := 1) (y := 0) (z := 2) (t := 3) (Or.inr ⟨rfl, rfl⟩)
        (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide)

example := Walk.countP_incident K4_loopless (Walk.nil (0 : Fin 4)) (0 : Fin 4)

example := Walk.IsCycleWalk.countP_incident K4_loopless
  (triangleWalk_isCycleWalk (G := K4) (a := 0) (b := 3) (c := 1) (x := 0) (y := 1) (z := 2)
    (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide)) 0

example := isCycle_of_three (G := K4) (a := 3) (b := 4) (c := 5) (x := 2) (y := 1) (z := 3)
  (Or.inr ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide)

example := isCycle_of_four (G := K4) (a := 1) (b := 2) (c := 4) (d := 3) (x := 2) (y := 0) (z := 3)
  (t := 1) (Or.inr ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩)
  (by decide) (by decide)

example := IsCycle.card_incident K4_loopless ((K4_isCycle_iff {3, 4, 5}).2 (by decide)) 1

end OrientedGraph

end Transformer.GainMatroid

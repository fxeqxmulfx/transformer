module

public import Transformer.GainMatroid.Section2_ExchangeLabels

/-!
# Balanced cycles through a balanced triangle

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. Let `X = {a, b, c}` be a balanced triangle of
a biased graph `(G, B)`, with `a = yz`, `b = xz` and `c = xy`. A cycle `C` meeting `X` in `b` and
`c` alone is `b`, `c` and a path `P` from `z` to `y` avoiding `x`, and `C ∆ X` is `a` and `P`. With
`X`, they are the three cycles of the theta formed by `cb`, `a` and `P` between `y` and `z`, so `C`
is balanced exactly when `C ∆ X` is (`BiasedGraph.mem_balanced_iff_symmDiff`).

So a set of edges meeting `X` in `b` and `c` alone is in `Δ_X B` exactly when its symmetric
difference with `X` is in `B` (`BiasedGraph.mem_deltaBalanced_iff_of_inter_eq_pair`), and a set of
edges off `X` is in `Δ_X B` exactly when it is in `B` (`mem_deltaBalanced_iff_of_inter_eq_empty`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open scoped symmDiff

open OrientedGraph

variable {V E : Type*}

/-- A one-edge path is openly disjoint from a path with the same ends that misses its edge. -/
theorem OrientedGraph.Walk.openly_single {G : OrientedGraph V E} {s t : V} {e : E}
    (he : G.Joins e s t) {P : G.Walk s t} (heP : e ∉ P.edges) :
    Walk.Openly (Walk.cons e he (Walk.nil t)) P := by
  refine ⟨fun g hg hg' => ?_, fun u hu _ => ?_⟩
  · simp only [Walk.edges, List.mem_cons, List.not_mem_nil, or_false] at hg
    subst hg
    exact heP hg'
  · simpa only [Walk.support, List.mem_cons, List.not_mem_nil, or_false] using hu

/-- A set of edges off `X = {a, b, c}`, with `a ≠ b`, is in `Δ_X B` exactly when it is in `B`. -/
theorem mem_deltaBalanced_iff_of_inter_eq_empty {B : Set (Set E)} {a b c : E} (hab : a ≠ b)
    {D : Set E} (hD : D ∩ {a, b, c} = ∅) : D ∈ deltaBalanced B {a, b, c} ↔ D ∈ B := by
  simp only [deltaBalanced, Set.mem_union, Set.mem_ofPred_eq]
  rw [hD, Set.ncard_empty]
  refine ⟨?_, fun hDB => Or.inl ⟨hDB, Or.inl rfl⟩⟩
  rintro (⟨hDB, -⟩ | ⟨C, -, hC, rfl⟩)
  · exact hDB
  · rw [Set.inter_symmDiff_distrib_right, Set.inter_self, Set.symmDiff_eq_empty] at hD
    rw [hD] at hC
    obtain ⟨g, hg⟩ := Set.ncard_eq_one.1 hC
    have ha : a ∈ ({g} : Set E) := by
      rw [← hg]
      exact Or.inl rfl
    have hb : b ∈ ({g} : Set E) := by
      rw [← hg]
      exact Or.inr (Or.inl rfl)
    exact (hab ((Set.mem_singleton_iff.1 ha).trans (Set.mem_singleton_iff.1 hb).symm)).elim

namespace BiasedGraph

variable {Ω : BiasedGraph V E} {a b c : E} {x y z : V}

/-- §2.4: in a biased graph with a balanced triangle `X = {a, b, c}`, a cycle `C` meeting `X` in
`b` and `c` alone is balanced exactly when `C ∆ X` is: `X`, `C` and `C ∆ X` are the cycles of a
theta. -/
theorem mem_balanced_iff_symmDiff (hX : Ω.graph.IsLabelledTriangle a b c x y z)
    (hbal : ({a, b, c} : Set E) ∈ Ω.balanced) {C : Set E} (hC : Ω.graph.IsCycle C)
    (hCX : C ∩ {a, b, c} = {b, c}) : C ∈ Ω.balanced ↔ C ∆ {a, b, c} ∈ Ω.balanced := by
  obtain ⟨P, hP, hxP, hPX, rfl⟩ := hX.exists_isPath_of_inter_eq_pair hC hCX
  rw [insert_insert_symmDiff_triple hPX hX.ne_ab hX.ne_ac hX.ne_bc]
  set Pa : Ω.graph.Walk y z := Walk.cons a hX.joins_a (Walk.nil z)
  set Pcb : Ω.graph.Walk y z := Walk.cons c hX.joins_c.symm (Walk.cons b hX.joins_b (Walk.nil z))
  have haP : a ∉ P.reverse.edges := fun ha => hPX a (Walk.mem_edges_reverse.1 ha) (Or.inl rfl)
  have hxP' : x ∉ P.reverse.support := fun hx => hxP (Walk.mem_support_reverse.1 hx)
  have hθ : IsTheta Pcb Pa P.reverse :=
    ⟨hX.ne_yz, by simp [Pcb, Walk.IsPath, Walk.support, hX.ne_xy.symm, hX.ne_xz, hX.ne_yz],
      by simp [Pa, Walk.IsPath, Walk.support, hX.ne_yz], hP.reverse,
      (Walk.openly_of_not_mem hX.joins_c.symm hX.joins_b (P := Pa)
        (by simp [Pa, Walk.support, hX.ne_xy, hX.ne_xz])).symm,
      (Walk.openly_of_not_mem hX.joins_c.symm hX.joins_b hxP').symm,
      Walk.openly_single hX.joins_a haP⟩
  have hθ' : IsTheta Pa Pcb P.reverse :=
    ⟨hθ.ne, hθ.pathQ, hθ.pathP, hθ.pathR, hθ.openPQ.symm, hθ.openQR, hθ.openPR⟩
  have e₁ : Pcb.edgeSet ∪ Pa.edgeSet = {a, b, c} := by
    ext g
    simp only [Pcb, Pa, Walk.edgeSet, Walk.edges, Set.mem_union, Set.mem_ofPred_eq,
      List.mem_cons, List.not_mem_nil, or_false, Set.mem_insert_iff, Set.mem_singleton_iff]
    tauto
  have e₂ : Pcb.edgeSet ∪ P.reverse.edgeSet = insert b (insert c P.edgeSet) := by
    ext g
    simp only [Pcb, Walk.edgeSet, Walk.mem_edges_reverse, Walk.edges, Set.mem_union,
      Set.mem_ofPred_eq, List.mem_cons, List.not_mem_nil, or_false, Set.mem_insert_iff]
    tauto
  have e₃ : Pa.edgeSet ∪ P.reverse.edgeSet = insert a P.edgeSet := by
    ext g
    simp only [Pa, Walk.edgeSet, Walk.mem_edges_reverse, Walk.edges, Set.mem_union,
      Set.mem_ofPred_eq, List.mem_cons, List.not_mem_nil, or_false, Set.mem_insert_iff]
  constructor
  · intro hC'
    have := Ω.theta hθ (by rwa [e₁]) (by rwa [e₂])
    rwa [e₃] at this
  · intro hD
    have := Ω.theta hθ' (by rwa [Set.union_comm, e₁]) (by rwa [e₃])
    rwa [e₂] at this

/-- §2.4: in a biased graph with a balanced triangle `X = {a, b, c}`, a set of edges meeting `X` in
`b` and `c` alone is in `Δ_X B` exactly when its symmetric difference with `X` is in `B`. -/
theorem mem_deltaBalanced_iff_of_inter_eq_pair (hX : Ω.graph.IsLabelledTriangle a b c x y z)
    (hbal : ({a, b, c} : Set E) ∈ Ω.balanced) {D : Set E} (hDX : D ∩ {a, b, c} = {b, c}) :
    D ∈ deltaBalanced Ω.balanced {a, b, c} ↔ D ∆ {a, b, c} ∈ Ω.balanced := by
  simp only [deltaBalanced, Set.mem_union, Set.mem_ofPred_eq]
  rw [hDX, Set.ncard_pair hX.ne_bc]
  constructor
  · rintro (⟨hDB, -⟩ | ⟨C, hCB, -, rfl⟩)
    · exact (mem_balanced_iff_symmDiff hX hbal (Ω.isCycle_of_mem D hDB) hDX).1 hDB
    · rwa [symmDiff_symmDiff_cancel_right]
  · intro hD
    refine Or.inr ⟨D ∆ {a, b, c}, hD, ?_, (symmDiff_symmDiff_cancel_right _ _).symm⟩
    rw [symmDiff_inter_of_inter_eq_pair hX.ne_ab hX.ne_ac hDX, Set.ncard_singleton]

end BiasedGraph

example := Walk.openly_single (G := pumpkin) (e := 0) (Or.inl ⟨rfl, rfl⟩) (P := pumpkinWalk 1)
  (by simp [pumpkinWalk, Walk.edges])

example := mem_deltaBalanced_iff_of_inter_eq_empty (B := ∅) (a := (5 : Fin 6)) (b := 4) (c := 3)
  (D := ∅) (by decide) (Set.empty_inter _)

example := (K4.gainBiased (1 : Fin 6 → ℚˣ)).mem_balanced_iff_symmDiff K4_isLabelledTriangle
  (gainBiased_one_isBalanced K4 _ K4_isLabelledTriangle.isCycle) K4_isCycle_1243 K4_inter_1243

example := (K4.gainBiased (1 : Fin 6 → ℚˣ)).mem_deltaBalanced_iff_of_inter_eq_pair
  K4_isLabelledTriangle (gainBiased_one_isBalanced K4 _ K4_isLabelledTriangle.isCycle)
  K4_inter_1243

end Transformer.GainMatroid

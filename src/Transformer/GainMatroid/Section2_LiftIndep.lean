module

public import Transformer.GainMatroid.Section2_BalancedSpan
public import Transformer.GainMatroid.Section2_BicycleNullity

/-!
# The independent sets of the lift matroid

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2: a circuit of `L(G, B)` is a balanced cycle,
or a tight handcuff, two vertex-disjoint cycles or a theta containing no balanced cycle
(`BiasedGraph.IsLiftCircuit`). The sets containing none of them (`BiasedGraph.IsLiftIndep`) are
those with no balanced cycle and nullity at most one (`BiasedGraph.isLiftIndep_iff`). From this:

* no lift circuit lies in another (`BiasedGraph.IsLiftCircuit.eq_of_subset`), because the circuits
  other than balanced cycles have nullity two and each of their edges lies on a cycle in them;
* these sets satisfy the exchange axiom (`BiasedGraph.IsLiftIndep.exists_insert`). If no edge of a
  larger such set `J` can be added to `I`, then every edge of `J` has its ends linked along `I`, so
  `J` has larger nullity. With `I` a forest, every edge of `J` closes a balanced cycle with `I`, so
  every cycle of `J` is balanced (`BiasedGraph.mem_balanced_of_fundamental`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

variable {V E : Type*}

namespace OrientedGraph

variable {G : OrientedGraph V E}

/-- A tight handcuff, two vertex-disjoint cycles or a theta contains two distinct cycles, and each
of its edges lies on a cycle inside it. -/
theorem exists_two_cycles_of_bicycle {X : Set E}
    (h : G.IsTightHandcuff X ∨ G.IsDisjointCycles X ∨ G.IsThetaSet X) :
    (∃ C D, G.IsCycle C ∧ G.IsCycle D ∧ C ≠ D ∧ C ⊆ X ∧ D ⊆ X) ∧
      ∀ e ∈ X, ∃ C ⊆ X, G.IsCycle C ∧ e ∈ C := by
  rcases h with ⟨C, D, v, hC, hD, hCD, -, rfl⟩ | ⟨C, D, hC, hD, hCD, rfl⟩ |
    ⟨x, y, P, Q, R, hT, rfl⟩
  · refine ⟨⟨C, D, hC, hD, fun h => ?_, Set.subset_union_left, Set.subset_union_right⟩, ?_⟩
    · obtain ⟨e, he⟩ := hC.nonempty
      exact Set.disjoint_left.1 hCD he (h ▸ he)
    · rintro e (he | he)
      · exact ⟨C, Set.subset_union_left, hC, he⟩
      · exact ⟨D, Set.subset_union_right, hD, he⟩
  · refine ⟨⟨C, D, hC, hD, fun h => ?_, Set.subset_union_left, Set.subset_union_right⟩, ?_⟩
    · obtain ⟨e, he⟩ := hC.nonempty
      have hv : G.tail e ∈ G.verts C := mem_verts.2 ⟨e, he, Or.inl rfl⟩
      exact Set.disjoint_left.1 hCD hv (h ▸ hv)
    · rintro e (he | he)
      · exact ⟨C, Set.subset_union_left, hC, he⟩
      · exact ⟨D, Set.subset_union_right, hD, he⟩
  · obtain ⟨hPQ, hPR, -⟩ := hT.isCycle
    refine ⟨⟨_, _, hPQ, hPR, hT.union_ne, Set.subset_union_left,
      Set.union_subset (Set.subset_union_left.trans Set.subset_union_left)
        Set.subset_union_right⟩, ?_⟩
    rintro e ((he | he) | he)
    · exact ⟨_, Set.subset_union_left, hPQ, Or.inl he⟩
    · exact ⟨_, Set.subset_union_left, hPQ, Or.inr he⟩
    · exact ⟨_, Set.union_subset (Set.subset_union_left.trans Set.subset_union_left)
        Set.subset_union_right, hPR, Or.inr he⟩

end OrientedGraph

namespace BiasedGraph

variable (Ω : BiasedGraph V E)

/-- The sets of edges containing no circuit of `L(G, B)`. -/
def IsLiftIndep (I : Set E) : Prop :=
  ∀ X ⊆ I, ¬Ω.IsLiftCircuit X

variable {Ω} [Finite V] [Finite E]

/-- A set contains no lift circuit exactly when it contains no balanced cycle and has nullity at
most one. -/
theorem isLiftIndep_iff {I : Set E} : Ω.IsLiftIndep I ↔
    (∀ C ⊆ I, Ω.graph.IsCycle C → C ∉ Ω.balanced) ∧ Ω.graph.nullity I ≤ 1 := by
  constructor
  · intro hI
    refine ⟨fun C hCI _ hCB => hI C hCI (Or.inl hCB), not_lt.1 fun hlt => ?_⟩
    obtain ⟨C, D, hC, hD, hCD, hCI, hDI⟩ := exists_two_cycles hlt
    obtain ⟨X, hXCD, hX⟩ := hC.exists_lift_bicycle hD hCD
    have hXI : X ⊆ I := hXCD.trans (Set.union_subset hCI hDI)
    by_cases hB : ∃ C' ⊆ X, Ω.graph.IsCycle C' ∧ C' ∈ Ω.balanced
    · obtain ⟨C', hC'X, -, hC'B⟩ := hB
      exact hI C' (hC'X.trans hXI) (Or.inl hC'B)
    · push Not at hB
      exact hI X hXI (Or.inr ⟨hX, hB⟩)
  · rintro ⟨hIB, hIn⟩ X hXI (hXB | ⟨hX, -⟩)
    · exact hIB X hXI (Ω.isCycle_of_mem X hXB) hXB
    · obtain ⟨C, D, hC, hD, hCD, hCX, hDX⟩ := (exists_two_cycles_of_bicycle hX).1
      have := two_le_nullity hC hD hCD (hCX.trans hXI) (hDX.trans hXI)
      omega

/-- No circuit of `L(G, B)` lies in another. -/
theorem IsLiftCircuit.eq_of_subset {X Y : Set E} (hX : Ω.IsLiftCircuit X)
    (hY : Ω.IsLiftCircuit Y) (hYX : Y ⊆ X) : Y = X := by
  rcases hX with hXB | ⟨hX, hXB⟩
  · obtain ⟨C, hCY, hC⟩ := hY.exists_isCycle
    have hCX : C = X := (Ω.isCycle_of_mem X hXB).eq_of_subset hC (hCY.trans hYX)
    exact hYX.antisymm (hCX ▸ hCY)
  · rcases hY with hYB | ⟨hY, -⟩
    · exact absurd hYB (hXB Y hYX (Ω.isCycle_of_mem Y hYB))
    by_contra hne
    obtain ⟨e, heX, heY⟩ := Set.exists_of_ssubset (hYX.ssubset_of_ne hne)
    obtain ⟨C, hCX, hC, heC⟩ := (exists_two_cycles_of_bicycle hX).2 e heX
    have h₁ := nullity_diff_add_one_of_isCycle hC hCX heC
    have h₂ := hX.elim IsTightHandcuff.nullity_le fun h => h.elim IsDisjointCycles.nullity_le
      IsThetaSet.nullity_le
    obtain ⟨C₁, D₁, hC₁, hD₁, hCD₁, hC₁Y, hD₁Y⟩ := (exists_two_cycles_of_bicycle hY).1
    have h₃ := two_le_nullity hC₁ hD₁ hCD₁ hC₁Y hD₁Y
    have h₄ : Ω.graph.nullity Y ≤ Ω.graph.nullity (X \ {e}) :=
      nullity_mono fun f hf => ⟨hYX hf, fun h => heY (h ▸ hf)⟩
    omega

/-- **The exchange axiom** for the sets containing no lift circuit. -/
theorem IsLiftIndep.exists_insert {I J : Set E} (hI : Ω.IsLiftIndep I) (hJ : Ω.IsLiftIndep J)
    (hlt : I.ncard < J.ncard) : ∃ e ∈ J, e ∉ I ∧ Ω.IsLiftIndep (insert e I) := by
  by_contra! hno
  obtain ⟨hIB, hIn⟩ := isLiftIndep_iff.1 hI
  obtain ⟨hJB, hJn⟩ := isLiftIndep_iff.1 hJ
  have hR : ∀ e ∈ J, Ω.graph.ReachableIn I (Ω.graph.tail e) (Ω.graph.head e) := by
    intro e heJ
    by_cases heI : e ∈ I
    · exact reachableIn_of_mem heI
    by_contra hr
    refine hno e heJ heI (isLiftIndep_iff.2 ⟨fun C hCI hC hCB => ?_, ?_⟩)
    · by_cases heC : e ∈ C
      · exact hr (hC.reachableIn_of_subset_insert heC hCI)
      · exact hIB C (fun f hf => (hCI hf).resolve_left fun h => heC (h ▸ hf)) hC hCB
    · rwa [nullity_insert_of_not_reachableIn hr]
  have hc := compCount_le_of_forall hR
  have hnI := nullity_add_card (G := Ω.graph) I
  have hnJ := nullity_add_card (G := Ω.graph) J
  have hIf : Ω.graph.nullity I = 0 := by omega
  obtain ⟨C, hC, hCJ⟩ : ∃ C, Ω.graph.IsCycle C ∧ C ⊆ J := by
    by_contra! hno'
    have := (nullity_eq_zero_iff (G := Ω.graph)).2 fun C hC hCJ => hno' C hC hCJ
    omega
  refine hJB C hCJ hC (Ω.mem_balanced_of_fundamental (nullity_eq_zero_iff.1 hIf) hR ?_ hC
    (hCJ.trans Set.subset_union_right))
  intro e heJ D hD hDI
  by_cases heI : e ∈ I
  · rw [Set.insert_eq_of_mem heI] at hDI
    exact absurd hDI (nullity_eq_zero_iff.1 hIf D hD)
  have hle : Ω.graph.nullity (insert e I) ≤ 1 := (nullity_insert_le I e).trans (by omega)
  have hnot := hno e heJ heI
  obtain ⟨D', hD'I, hD', hD'B⟩ : ∃ D' ⊆ insert e I, Ω.graph.IsCycle D' ∧ D' ∈ Ω.balanced := by
    by_contra! h
    exact hnot (isLiftIndep_iff.2 ⟨fun C hC hC' hCB => h C hC hC' hCB, hle⟩)
  rwa [eq_of_nullity_le_one hle hD hD' hDI hD'I]

/-- The first link of `triangle` is a forest. -/
theorem triangle_isForest_singleton : triangle.IsForest {0} := by
  rw [Set.singleton_def]
  exact isForest_empty.insert fun h =>
    absurd (h.eq_of_notMem_verts (by simp [verts])) (by decide)

example := exists_two_cycles_of_bicycle (G := pumpkin)
  (Or.inr (Or.inr ⟨0, 1, pumpkinWalk 0, pumpkinWalk 1, pumpkinWalk 2, pumpkin_isTheta, rfl⟩))

example := IsLiftCircuit.eq_of_subset (Ω := triangleBiased) (X := {1, 2, 0}) (Y := {1, 2, 0})
  (Or.inl rfl) (Or.inl rfl) le_rfl

example := IsLiftIndep.exists_insert (Ω := triangleBiased) (I := ∅) (J := {0})
  (fun X hX hXc => by
    obtain ⟨C, hCX, hC⟩ := hXc.exists_isCycle
    obtain ⟨e, he⟩ := hC.nonempty
    exact hX (hCX he))
  (isLiftIndep_iff.2 ⟨fun C hC hCyc _ => triangle_isForest_singleton C hCyc hC,
    (nullity_eq_zero_iff.2 triangle_isForest_singleton).le.trans zero_le_one⟩)
  (by simp)

end BiasedGraph

end Transformer.GainMatroid

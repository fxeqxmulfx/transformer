module

public import Transformer.GainMatroid.Section2_DeltaDual

/-!
# Examples for the generalized parallel connection

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. The lemmas on `natRk`, `connRk`,
`deltaConnection` and `deltaRk`, applied to `U_{2,3}` (`twoOfThree`) and its triangle `{0, 1, 2}`
(`twoOfThree_isCircuit_012`).
-/

@[expose] public section

namespace Transformer.GainMatroid

/-- The triangle of `U_{2,3}`. -/
theorem twoOfThree_isCircuit_012 : twoOfThree.IsCircuit {0, 1, 2} := by
  have h : ({0, 1, 2} : Set (Fin 3)) = (({0, 1, 2} : Finset (Fin 3)) : Set (Fin 3)) := by simp
  rw [h]
  exact (matroidOfFinset_isCircuit_iff _).2 (by decide +kernel)

theorem twoOfThree_indep_01 : twoOfThree.Indep {0, 1} := by
  have h : ({0, 1} : Set (Fin 3)) = (({0, 1} : Finset (Fin 3)) : Set (Fin 3)) := by simp
  rw [h]
  exact (matroidOfFinset_indep _).2 (by decide)

theorem twoOfThree_zero_notMem_closure : (0 : Fin 3) ∉ twoOfThree.closure ∅ := by
  have := (twoOfThree_indep_01.subset (Set.singleton_subset_iff.2 (Set.mem_insert 0 {1}))
    ).notMem_closure_sdiff_of_mem (Set.mem_singleton 0)
  rwa [Set.sdiff_self] at this

example := eq_of_forall_insert_eq (r := natRk twoOfThree) (fun _ _ h => natRk_mono twoOfThree h)
  (natRk_submod twoOfThree) (Y := ∅) Set.finite_empty fun _ he => absurd he (Set.notMem_empty _)

example := natRk_mono twoOfThree (Set.empty_subset ({0} : Set (Fin 3)))

example := natRk_union_of_subset_closure twoOfThree (X := {0}) (Set.empty_subset _)

example := natRk_insert_of_notMem_closure (M := twoOfThree)
  ⟨Set.mem_univ (0 : Fin 3), twoOfThree_zero_notMem_closure⟩

example := mem_closure_iff_natRk (M := twoOfThree) (X := ∅) (Set.mem_univ (0 : Fin 3))

example := natRk_eq_ncard_of_isBasis twoOfThree_indep_01.isBasis_self

example := natRk_add_one_of_isCircuit twoOfThree_isCircuit_012

example := spanIdx_mono (Finset.empty_subset ({0} : Finset (Fin 3)))

example := nonemptyInd_mono (Finset.empty_subset ({0} : Finset (Fin 3)))

example := exists_triple_eq (a := (0 : Fin 3)) (b := 1) (c := 2) (e := 1) (by simp)

example := image_triClosure_subset twoOfThree_isCircuit_012 {0, 1}

example := newIdx_mono (Set.empty_subset ({Sum.inr 0} : Set (Fin 3 ⊕ Fin 3)))

example := connRk_mono (M := twoOfThree) (a := 0) (b := 1) (c := 2)
  (Set.empty_subset ({Sum.inr 0} : Set (Fin 3 ⊕ Fin 3)))

example := connRk_le_ncard twoOfThree_isCircuit_012 {Sum.inr 0}

example := natRk_union_spanIdx_union twoOfThree_isCircuit_012 ∅ {0} {1}

example := connRk_submod twoOfThree_isCircuit_012 {Sum.inr 0} {Sum.inl 1}

example := connRk_eq_of_forall_insert twoOfThree_isCircuit_012 (Y := ∅) (D := ∅)
  fun _ he => absurd he (Set.notMem_empty _)

example := connRk_insert_le twoOfThree_isCircuit_012 (Sum.inr 0) ∅

/-- The empty set is independent in the parallel connection. -/
theorem connRk_empty : connRk twoOfThree 0 1 2 ∅ = (∅ : Set (Fin 3 ⊕ Fin 3)).ncard :=
  ((deltaConnection_indep_iff twoOfThree_isCircuit_012).1
    (deltaConnection twoOfThree 0 1 2 twoOfThree_isCircuit_012).empty_indep).2

example := connRk_eq_ncard_of_subset twoOfThree_isCircuit_012 connRk_empty (Set.empty_subset _)

example := ncard_le_of_forall_insert twoOfThree_isCircuit_012 connRk_empty connRk_empty
  fun _ he => absurd he (Set.notMem_empty _)

example := deltaConnection_E twoOfThree_isCircuit_012

example := deltaConnection_indep_iff twoOfThree_isCircuit_012 (Z := ∅)

example := natRk_deltaConnection twoOfThree_isCircuit_012 (Set.empty_subset _)

example := deltaConnection_comap_inl twoOfThree_isCircuit_012

example := isFlat_comap (Matroid.ground_isFlat twoOfThree) (id : Fin 3 → Fin 3)

example := triple_injective (a := (0 : Fin 3)) (b := 1) (c := 2) (by decide) (by decide) (by decide)

example := natRk_image_triple twoOfThree_isCircuit_012 (by decide) (by decide) (by decide) {0}

example := deltaConnection_comap_deltaEmbed twoOfThree_isCircuit_012 (by decide) (by decide)
  (by decide)

example := k4_isFlat_spanIdx (a := (0 : Fin 3)) (b := 1) (c := 2) (S := Set.univ)
  (by rw [Set.preimage_univ]; exact Matroid.ground_isFlat k4Matroid)

example := isFlat_deltaConnection_iff twoOfThree_isCircuit_012 (by decide) (by decide) (by decide)
  ∅

example := deltaRelabel_eq_inr_iff (a := (0 : Fin 3)) (b := 1) (c := 2) (by decide) (by decide)
  (by decide) (f := 0) (k := 0)

example := connRk_image_deltaRelabel (M := twoOfThree) (a := 0) (b := 1) (c := 2) (by decide)
  (by decide) (by decide) {0}

example := deltaRk_of_disjoint (M := twoOfThree) (a := 0) (b := 1) (c := 2) (by decide)
  (Set.empty_disjoint _)

example := deltaRk_insert_swap twoOfThree_isCircuit_012 (by decide) (by decide) (by decide)
  (Set.notMem_empty 2)

example := exists_eq_insert_swap (I := ({2} : Set (Fin 3))) (u := 0) (w := 1) (Set.mem_singleton 2)
  (by decide) (by decide)

example := ncard_insert_swap (J := (∅ : Set (Fin 3))) (s := 2) (u := 0) (w := 1)
  (Set.notMem_empty 2) (by decide) (by decide)

example := deltaRelabel_preimage_connGround twoOfThree_isCircuit_012

example := eq_of_isFlat_iff (P := twoOfThree) (Q := twoOfThree) rfl fun _ => Iff.rfl

example := swap_mem_iff (S := Set.univ) (Set.mem_univ (0 : Fin 3)) (Set.mem_univ 1) (e := 2)

example := sdiff_eq_insert_swap (S := Set.univ) (I := ∅) (Set.mem_univ (0 : Fin 3))
  (Set.mem_univ 1) (Set.mem_univ 2) (by decide) (by decide) (Set.notMem_empty 2)

example := deltaRk_of_subset (N := twoOfThree) (subset_refl ({0} : Set (Fin 3)))
  (Set.singleton_nonempty 0)

example := dual_indep_iff_natRk (M := twoOfThree) (Set.empty_subset _)

end Transformer.GainMatroid

module

public import Mathlib.Combinatorics.Matroid.Minor.Contract
public import Transformer.GainMatroid.Section2_MatrixMatroids

/-!
# Flats and independent sets of the dual of a represented matroid

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, which defines `∇_Y M` as `(Δ_Y M*)*`. A set
`S` is a flat of `N*` exactly when every element outside `S` lies in the closure in `N` of the other
elements outside `S` (`isFlat_dual_iff`): its complement is a union of circuits of `N`. For a
matroid that `A` represents, the column of every element outside `S` then lies in the span of the
columns of the other elements outside `S` (`IsMatrixRepresentation.isFlat_dual_iff`), and a set `I`
is independent in the dual exactly when the columns outside `I` span every column
(`IsMatrixRepresentation.dual_indep_iff`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open Submodule

variable {α E R F : Type*} [Field F]

/-- An element outside `S` is a coloop of `N \ S` exactly when it is not in the closure of the
other elements outside `S`. -/
theorem mem_coloops_delete_iff {N : Matroid α} {S : Set α} {e : α} (he : e ∈ N.E \ S) :
    e ∈ (N.delete S).coloops ↔ e ∉ N.closure ((N.E \ S) \ {e}) := by
  have hX : (N.E \ S) \ {e} ⊆ N.E := Set.sdiff_subset.trans Set.sdiff_subset
  rw [← Matroid.isColoop_iff_mem_coloops, Matroid.isColoop_iff_sdiff_closure, Matroid.delete_ground,
    Matroid.delete_closure_eq_of_disjoint _ (Set.disjoint_sdiff_left.mono_left Set.sdiff_subset)]
  constructor
  · intro hne hcl
    refine hne (Set.Subset.antisymm (Set.sdiff_subset_sdiff_left (N.closure_subset_ground _))
      fun f hf => ⟨?_, hf.2⟩)
    by_cases hfe : f = e
    · exact hfe ▸ hcl
    · exact N.subset_closure _ hX ⟨hf, hfe⟩
  · intro hcl heq
    exact hcl (heq ▸ he).1

/-- A set is a flat of the dual exactly when every element of the ground set outside it lies in the
closure of the other elements outside it. -/
theorem isFlat_dual_iff {N : Matroid α} {S : Set α} (hS : S ⊆ N.E) :
    N✶.IsFlat S ↔ ∀ e ∈ N.E \ S, e ∈ N.closure ((N.E \ S) \ {e}) := by
  have key : N✶.closure S \ S = (N.delete S).coloops := by
    rw [← Matroid.contract_loops_eq, ← Matroid.dual_delete, Matroid.dual_loops]
  rw [Matroid.isFlat_iff_closure_eq]
  constructor
  · intro h e he
    by_contra hcl
    have hmem := (mem_coloops_delete_iff he).2 hcl
    rw [← key, h, Set.sdiff_self] at hmem
    exact hmem
  · intro h
    refine Set.Subset.antisymm (fun e he => ?_) (N✶.subset_closure S hS)
    by_contra heS
    have heE : e ∈ N.E := N✶.closure_subset_ground S he
    have hmem : e ∈ (N.delete S).coloops := key ▸ ⟨he, heS⟩
    exact (mem_coloops_delete_iff ⟨heE, heS⟩).1 hmem (h e ⟨heE, heS⟩)

namespace IsMatrixRepresentation

variable {M : Matroid E} {A : Matrix R E F}

/-- In the dual of a matroid that `A` represents, the flats are the sets outside which every column
lies in the span of the other columns outside them. -/
theorem isFlat_dual_iff (hA : IsMatrixRepresentation M A) (S : Set E) :
    M✶.IsFlat S ↔ ∀ e, e ∉ S → (fun r => A r e) ∈ span F ((fun f r => A r f) '' (Sᶜ \ {e})) := by
  rw [Transformer.GainMatroid.isFlat_dual_iff (hA.1 ▸ Set.subset_univ S), hA.1,
    ← Set.compl_eq_univ_sdiff]
  exact forall_congr' fun e => imp_congr_right fun _ => hA.mem_closure_iff _ e

/-- In the dual of a matroid that `A` represents, a set is independent exactly when the columns
outside it span every column. -/
theorem dual_indep_iff (hA : IsMatrixRepresentation M A) (I : Set E) :
    M✶.Indep I ↔ ∀ e, (fun r => A r e) ∈ span F ((fun f r => A r f) '' Iᶜ) := by
  have hI : I ⊆ M.E := hA.1 ▸ Set.subset_univ I
  rw [← Matroid.coindep_def, Matroid.coindep_iff_compl_spanning hI,
    Matroid.spanning_iff_closure_eq, hA.1, ← Set.compl_eq_univ_sdiff]
  constructor
  · intro h e
    exact (hA.mem_closure_iff _ e).1 (h ▸ Set.mem_univ e)
  · intro h
    exact Set.eq_univ_of_forall fun e => (hA.mem_closure_iff _ e).2 (h e)

end IsMatrixRepresentation

/-- `U_{2,3}` over `ℚ` with nothing deleted: its first element is a coloop exactly when the other
two do not span it. -/
example := mem_coloops_delete_iff (N := matrixMatroid !![(1 : ℚ), 0, 1; 0, 1, 1]) (S := ∅) (e := 0)
  ⟨(matrixMatroid_isMatrixRepresentation !![(1 : ℚ), 0, 1; 0, 1, 1]).1 ▸ Set.mem_univ _,
    Set.notMem_empty _⟩

example := isFlat_dual_iff (N := (matrixMatroid !![(1 : ℚ), 0, 1; 0, 1, 1] : Matroid (Fin 3)))
  (Set.empty_subset _)

example (S : Set (Fin 3)) :=
  (matrixMatroid_isMatrixRepresentation !![(1 : ℚ), 0, 1; 0, 1, 1]).isFlat_dual_iff S

example (I : Set (Fin 3)) :=
  (matrixMatroid_isMatrixRepresentation !![(1 : ℚ), 0, 1; 0, 1, 1]).dual_indep_iff I

end Transformer.GainMatroid

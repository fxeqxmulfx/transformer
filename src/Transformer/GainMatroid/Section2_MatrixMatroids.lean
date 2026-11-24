module

public import Transformer.GainMatroid.Section2_Representable

/-!
# The matroid of a matrix, and its flats

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, which reads matroids off matrices: `M(A)` is
the matroid of the columns of `A`. A matrix with finitely many columns has such a matroid
(`matrixMatroid`): its independent sets are the sets of linearly independent columns. In a matroid
that `A` represents, the closure of a set is the set of columns in the span of its columns
(`IsMatrixRepresentation.mem_closure_iff`), so the flats are the sets of columns closed under spans
(`IsMatrixRepresentation.isFlat_iff`). Relabelling the columns along an injection pulls the matroid
back (`IsMatrixRepresentation.comap`), and a matrix determines the matroid it represents
(`IsMatrixRepresentation.eq`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open Submodule

variable {E E' R F : Type*} [Field F]

/-- Linearly independent vectors inside the span of finitely many linearly independent vectors are
no more numerous than they. -/
theorem card_le_of_forall_mem_span {v : E → R → F} {I J : Finset E}
    (hI : LinearIndepOn F v (I : Set E)) (hJ : LinearIndepOn F v (J : Set E))
    (h : ∀ e ∈ J, v e ∈ span F (v '' (I : Set E))) : J.card ≤ I.card := by
  have hI' := finrank_span_eq_card hI
  have hJ' := finrank_span_eq_card hJ
  rw [← Set.image_eq_range] at hI' hJ'
  simp only [Finset.coe_sort_coe, Fintype.card_coe] at hI' hJ'
  have : Module.Finite F (span F (v '' (I : Set E))) :=
    Module.Finite.span_of_finite F (I.finite_toSet.image v)
  have hle : span F (v '' (J : Set E)) ≤ span F (v '' (I : Set E)) :=
    span_le.2 (by rintro _ ⟨e, he, rfl⟩; exact h e he)
  have := Submodule.finrank_mono hle
  omega

/-- **`M(A)`**: the matroid of the columns of `A`, whose independent sets are the sets of linearly
independent columns. -/
noncomputable def matrixMatroid [Finite E] [DecidableEq E] (A : Matrix R E F) : Matroid E :=
  matroidOfFinset (fun I => LinearIndepOn F (fun e r => A r e) (I : Set E)) (by simp)
    (fun _ _ hJ hIJ => hJ.mono (by exact_mod_cast hIJ)) fun I J hI hJ hlt => by
      by_contra hne
      push Not at hne
      refine absurd (card_le_of_forall_mem_span hI hJ fun e he => ?_) (not_le.2 hlt)
      by_cases heI : e ∈ I
      · exact subset_span ⟨e, heI, rfl⟩
      · by_contra hspan
        refine hne e he heI ?_
        rw [Finset.coe_insert]
        exact (linearIndepOn_insert (by exact_mod_cast heI)).2 ⟨hI, hspan⟩

/-- `A` represents `M(A)`. -/
theorem matrixMatroid_isMatrixRepresentation [Finite E] [DecidableEq E] (A : Matrix R E F) :
    IsMatrixRepresentation (matrixMatroid A) A := by
  refine ⟨rfl, fun I => ?_⟩
  obtain ⟨I, rfl⟩ := I.toFinite.exists_finset_coe
  exact matroidOfFinset_indep I

namespace IsMatrixRepresentation

variable {M : Matroid E} {A : Matrix R E F}

/-- A matrix determines the matroid it represents. -/
theorem eq {M' : Matroid E} (hA : IsMatrixRepresentation M A) (hA' : IsMatrixRepresentation M' A) :
    M = M' :=
  Matroid.ext_indep (hA.1.trans hA'.1.symm) fun I _ => (hA.2 I).trans (hA'.2 I).symm

/-- In a matroid that `A` represents, an element lies in the closure of `S` exactly when its column
lies in the span of the columns of `S`. -/
theorem mem_closure_iff (hA : IsMatrixRepresentation M A) (S : Set E) (e : E) :
    e ∈ M.closure S ↔ (fun r => A r e) ∈ span F ((fun f r => A r f) '' S) := by
  have hsub (X : Set E) : X ⊆ M.E := hA.1 ▸ Set.subset_univ X
  obtain ⟨I, hI⟩ := M.exists_isBasis S (hsub S)
  have hIi : LinearIndepOn F (fun f r => A r f) I := (hA.2 I).1 hI.indep
  have hspan : span F ((fun f r => A r f) '' S) = span F ((fun f r => A r f) '' I) := by
    refine le_antisymm (span_le.2 ?_) (span_mono (Set.image_mono hI.subset))
    rintro _ ⟨f, hf, rfl⟩
    by_cases hfI : f ∈ I
    · exact subset_span ⟨f, hfI, rfl⟩
    · by_contra hn
      exact hfI (hI.mem_of_insert_indep hf ((hA.2 _).2 ((linearIndepOn_insert hfI).2 ⟨hIi, hn⟩)))
  rw [← hI.closure_eq_closure, hI.indep.mem_closure_iff, hspan]
  by_cases heI : e ∈ I
  · simp only [heI, or_true, true_iff]
    exact subset_span ⟨e, heI, rfl⟩
  · rw [← Matroid.not_indep_iff (hsub _), hA.2, linearIndepOn_insert heI]
    simp only [heI, or_false]
    tauto

/-- In a matroid that `A` represents, the flats are the sets containing every column in the span of
their columns. -/
theorem isFlat_iff (hA : IsMatrixRepresentation M A) (S : Set E) :
    M.IsFlat S ↔ ∀ e, (fun r => A r e) ∈ span F ((fun f r => A r f) '' S) → e ∈ S := by
  rw [Matroid.isFlat_iff_closure_eq]
  constructor
  · intro h e he
    rw [← h]
    exact (hA.mem_closure_iff S e).2 he
  · intro h
    exact Set.Subset.antisymm (fun e he => h e ((hA.mem_closure_iff S e).1 he))
      (M.subset_closure S (hA.1 ▸ Set.subset_univ S))

/-- Pulling a represented matroid back along an injection is represented by the relabelled
columns. -/
theorem comap {P : Matroid E'} {B : Matrix R E' F} (hB : IsMatrixRepresentation P B) {f : E → E'}
    (hf : Function.Injective f) {A : Matrix R E F} (hAB : ∀ r e, A r e = B r (f e)) :
    IsMatrixRepresentation (P.comap f) A := by
  have hcol : (fun e r => A r e) = (fun e' r => B r e') ∘ f :=
    funext fun e => funext fun r => hAB r e
  refine ⟨by rw [Matroid.comap_ground_eq, hB.1, Set.preimage_univ], fun I => ?_⟩
  rw [Matroid.comap_indep_iff, hB.2, hcol]
  exact ⟨fun h => h.1.comp_of_image h.2, fun h => ⟨h.image_of_comp, hf.injOn⟩⟩

end IsMatrixRepresentation

/-- The standard basis of `ℚ²`, inside its own span. -/
example : (Finset.univ : Finset (Fin 2)).card ≤ (Finset.univ : Finset (Fin 2)).card := by
  have h : LinearIndepOn ℚ (Pi.basisFun ℚ (Fin 2)) (Finset.univ : Finset (Fin 2)) := by
    rw [Finset.coe_univ]
    exact (Pi.basisFun ℚ (Fin 2)).linearIndependent.linearIndepOn_univ
  exact card_le_of_forall_mem_span h h fun e he => subset_span ⟨e, he, rfl⟩

/-- `U_{2,3}` over `ℚ`. -/
example : matrixMatroid !![(1 : ℚ), 0, 1; 0, 1, 1] = matrixMatroid !![(1 : ℚ), 0, 1; 0, 1, 1] :=
  (matrixMatroid_isMatrixRepresentation _).eq (matrixMatroid_isMatrixRepresentation _)

example (S : Set (Fin 3)) (e : Fin 3) :=
  (matrixMatroid_isMatrixRepresentation !![(1 : ℚ), 0, 1; 0, 1, 1]).mem_closure_iff S e

example (S : Set (Fin 3)) :=
  (matrixMatroid_isMatrixRepresentation !![(1 : ℚ), 0, 1; 0, 1, 1]).isFlat_iff S

/-- Deleting the last column of `U_{2,3}`. -/
example := (matrixMatroid_isMatrixRepresentation !![(1 : ℚ), 0, 1; 0, 1, 1]).comap
  (Fin.castSucc_injective 2) (A := !![(1 : ℚ), 0; 0, 1]) fun r e => by
    fin_cases r <;> fin_cases e <;> rfl

end Transformer.GainMatroid

module

public import Transformer.GainMatroid.Section2_DeltaYExamples
public import Transformer.GainMatroid.Section2_ExchangeForms
public import Transformer.GainMatroid.Section2_ExchangeMatroids

/-!
# The row vector of a cocircuit

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. A cocircuit of a represented matroid is the
support of a vector of the row space (`IsMatrixRepresentation.exists_mem_rowSpace_of_cocircuit`).
This vector is the fourth row of `I(K₄)` when a triad is brought to its first three columns.

`twoOfThreeMatrix` and `clawMatrix` are binary representations of `U_{2,3}` and of the matroid of
`K₄` minus an edge, on which these results and those of `Section2_ExchangeForms` are applied.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

variable {R E F : Type*} [Field F]

/-- A vector of the row space is a linear functional applied to the columns. -/
theorem exists_linearMap_col_eq {A : Matrix R E F} {w : E → F} (hw : w ∈ rowSpace A) :
    ∃ ℓ : (R → F) →ₗ[F] F, ∀ e, ℓ (fun r => A r e) = w e := by
  induction hw using Submodule.span_induction with
  | mem x hx =>
    obtain ⟨r, rfl⟩ := hx
    exact ⟨LinearMap.proj r, fun _ => rfl⟩
  | zero => exact ⟨0, fun _ => rfl⟩
  | add x y _ _ hx hy =>
    obtain ⟨ℓ₁, h₁⟩ := hx
    obtain ⟨ℓ₂, h₂⟩ := hy
    exact ⟨ℓ₁ + ℓ₂, fun e => by simp [h₁, h₂]⟩
  | smul c x _ hx =>
    obtain ⟨ℓ, h⟩ := hx
    exact ⟨c • ℓ, fun e => by simp [h]⟩

/-- A vector of the row space that vanishes on an independent set `I` vanishes on every element
spanned by `I`. -/
theorem IsMatrixRepresentation.eq_zero_of_not_indep_insert {M : Matroid E} {A : Matrix R E F}
    (hA : IsMatrixRepresentation M A) {I : Set E} {e : E} (hI : M.Indep I) (heI : e ∉ I)
    (hdep : ¬M.Indep (insert e I)) {w : E → F} (hw : w ∈ rowSpace A) (h0 : ∀ i ∈ I, w i = 0) :
    w e = 0 := by
  obtain ⟨ℓ, hℓ⟩ := exists_linearMap_col_eq hw
  have hn : ¬LinearIndepOn F (fun e r => A r e) (insert e I) := fun h => hdep ((hA.2 _).2 h)
  rw [linearIndepOn_insert heI] at hn
  have hspan : (fun r => A r e) ∈ Submodule.span F ((fun e r => A r e) '' I) := by
    by_contra h
    exact hn ⟨(hA.2 _).1 hI, h⟩
  have hle : Submodule.span F ((fun e r => A r e) '' I) ≤ LinearMap.ker ℓ :=
    Submodule.span_le.2 (by
      rintro _ ⟨i, hi, rfl⟩
      simp [hℓ, h0 i hi])
  rw [← hℓ]
  exact hle hspan

/-- A cocircuit of a represented matroid is the support of a vector of the row space. -/
theorem IsMatrixRepresentation.exists_mem_rowSpace_of_cocircuit [Finite E] {M : Matroid E}
    {A : Matrix R E F} (hA : IsMatrixRepresentation M A) {C : Set E} (hC : M✶.IsCircuit C) :
    ∃ w ∈ rowSpace A, (∀ e ∈ C, w e ≠ 0) ∧ ∀ e, e ∉ C → w e = 0 := by
  classical
  obtain ⟨hdep, hind⟩ := Matroid.isCircuit_iff_dep_forall_sdiff_singleton_indep.1 hC
  have hsub (X : Set E) : X ⊆ M.E := hA.1 ▸ Set.subset_univ X
  have hmeet : ∀ B, M.IsBase B → ¬Disjoint C B := fun B hB hd =>
    hdep.1 ((Matroid.dual_indep_iff_exists (hsub C)).2 ⟨B, hB, hd⟩)
  have hbase : ∀ d ∈ C, ∃ B, M.IsBase B ∧ d ∈ B ∧ ∀ u ∈ C, u ≠ d → u ∉ B := by
    intro d hd
    obtain ⟨B, hB, hdis⟩ := (Matroid.dual_indep_iff_exists (hsub _)).1 (hind d hd)
    refine ⟨B, hB, ?_, fun u hu hne huB => Set.disjoint_left.1 hdis ⟨hu, hne⟩ huB⟩
    by_contra hdB
    refine hmeet B hB (Set.disjoint_left.2 fun u hu huB => ?_)
    by_cases hud : u = d
    · exact hdB (hud ▸ huB)
    · exact Set.disjoint_left.1 hdis ⟨hu, hud⟩ huB
  obtain ⟨a, ha⟩ := hC.nonempty
  obtain ⟨Ba, hBa, haB, hBaC⟩ := hbase a ha
  have hlin : LinearIndepOn F (fun e r => A r e) ((Ba.toFinite.toFinset : Finset E) : Set E) := by
    rw [Set.Finite.coe_toFinset]
    exact (hA.2 _).1 hBa.indep
  obtain ⟨w, hw, hwv⟩ := exists_mem_rowSpace_eq_on hlin fun e => if e = a then 1 else 0
  have hwa : w a = 1 := by simpa using hwv a (by simpa using haB)
  have hwB : ∀ e ∈ Ba, e ≠ a → w e = 0 := fun e he hne => by
    simpa [hne] using hwv e (by simpa using he)
  have hoff : ∀ h, h ∉ C → w h = 0 := by
    intro h hhC
    by_cases hhB : h ∈ Ba
    · exact hwB h hhB fun heq => hhC (heq ▸ ha)
    have hIind : M.Indep (Ba \ {a}) := hBa.indep.subset Set.sdiff_subset
    have hhI : h ∉ Ba \ {a} := fun hi => hhB hi.1
    refine hA.eq_zero_of_not_indep_insert hIind hhI (fun hins => ?_) hw
      fun i hi => hwB i hi.1 hi.2
    obtain ⟨B', hB', hsubB'⟩ := hins.exists_isBase_superset
    have hcard : B'.encard ≤ (insert h (Ba \ {a})).encard := by
      have h1 : (insert a (Ba \ {a})).encard = (Ba \ {a}).encard + 1 :=
        Set.encard_insert_of_notMem fun hi => hi.2 rfl
      rw [Set.insert_sdiff_self_of_mem haB] at h1
      rw [hB'.encard_eq_encard_of_isBase hBa, h1, Set.encard_insert_of_notMem hhI]
    have heq := (Set.toFinite _).eq_of_subset_of_encard_le hsubB' hcard
    refine hmeet B' hB' (Set.disjoint_left.2 fun u hu huB => ?_)
    rw [← heq] at huB
    rcases huB with rfl | hui
    · exact hhC hu
    · exact hBaC u hu hui.2 hui.1
  refine ⟨w, hw, fun d hd hw0 => ?_, hoff⟩
  by_cases hda : d = a
  · rw [hda, hwa] at hw0
    exact one_ne_zero hw0
  obtain ⟨Bd, hBd, hdB, hBdC⟩ := hbase d hd
  have hzero : ∀ e ∈ Bd, w e = 0 := fun e he => by
    by_cases hed : e = d
    · exact hed ▸ hw0
    · exact hoff e fun heC => hBdC e heC hed he
  have haBd : a ∉ Bd := hBdC a ha (Ne.symm hda)
  have hza := hA.eq_zero_of_not_indep_insert hBd.indep haBd
    (hBd.insert_dep ⟨hsub _ (Set.mem_singleton a), haBd⟩).not_indep hw hzero
  rw [hwa] at hza
  exact one_ne_zero hza

/-- A binary representation of `U_{2,3}`. -/
def twoOfThreeMatrix : Matrix (Fin 2) (Fin 3) (ZMod 2) := !![1, 0, 1; 0, 1, 1]

set_option maxRecDepth 100000 in
theorem twoOfThreeMatrix_represents : IsMatrixRepresentation twoOfThree twoOfThreeMatrix := by
  refine ⟨rfl, fun I => ?_⟩
  obtain ⟨I, rfl⟩ := I.toFinite.exists_finset_coe
  refine (matroidOfFinset_indep I).trans (Iff.trans ?_ linearIndepOn_finset_iff.symm)
  revert I
  decide +kernel

/-- The three elements of `U_{2,3}` form a triangle. -/
theorem twoOfThree_triangle : twoOfThree.IsCircuit {1, 2, 0} :=
  (twoOfThree_isCircuit_iff _).2 triangle_isLabelledTriangle.isCycle

example := (ProjEquiv.refl twoOfThreeMatrix).linearIndepOn_iff {0}

example := twoOfThreeMatrix_represents.of_projEquiv (ProjEquiv.refl twoOfThreeMatrix)

example := exists_mem_rowSpace_eq_on (A := twoOfThreeMatrix) (s := {0, 1})
  ((twoOfThreeMatrix_represents.2 _).1 ((matroidOfFinset_indep _).2 (by decide))) 1

example := exists_col_eq_of_isCircuit twoOfThreeMatrix_represents (by decide) (by decide)
  (by decide) twoOfThree_triangle

example := exists_isTriangleForm twoOfThreeMatrix_represents (by decide) (by decide) (by decide)
  twoOfThree_triangle

/-- The binary matrix of `clawGraph`, whose columns are `tail + head`. -/
def clawMatrix : Matrix (Fin 4) (Fin 5) (ZMod 2) :=
  !![1, 1, 1, 0, 0; 1, 0, 0, 1, 1; 0, 1, 0, 1, 0; 0, 0, 1, 0, 1]

set_option maxRecDepth 100000 in
theorem clawMatrix_represents : IsMatrixRepresentation clawMatroid clawMatrix := by
  refine ⟨rfl, fun I => ?_⟩
  obtain ⟨I, rfl⟩ := I.toFinite.exists_finset_coe
  refine (matroidOfFinset_indep I).trans (Iff.trans ?_ linearIndepOn_finset_iff.symm)
  revert I
  decide +kernel

/-- The star of `clawGraph` is independent. -/
theorem clawMatroid_star_indep : clawMatroid.Indep {0, 1, 2} := by
  have h : clawMatroid.Indep (({0, 1, 2} : Finset (Fin 5)) : Set (Fin 5)) :=
    (matroidOfFinset_indep _).2 (by decide)
  simpa using h

example := exists_linearMap_col_eq (row_mem_rowSpace twoOfThreeMatrix 0)

/-- The row `3` of `clawMatrix` vanishes on `0` and `1`, which span `3`. -/
example : clawMatrix 3 3 = 0 := by
  have hI : clawMatroid.Indep (({0, 1} : Finset (Fin 5)) : Set (Fin 5)) :=
    (matroidOfFinset_indep _).2 (by decide)
  have hdep : ¬clawMatroid.Indep (insert 3 (({0, 1} : Finset (Fin 5)) : Set (Fin 5))) := by
    rw [← Finset.coe_insert]
    exact fun h => absurd ((matroidOfFinset_indep _).1 h) (by decide)
  have h0 : ∀ i ∈ ({0, 1} : Finset (Fin 5)), clawMatrix 3 i = 0 := by decide
  exact clawMatrix_represents.eq_zero_of_not_indep_insert hI (by simp) hdep
    (row_mem_rowSpace clawMatrix 3) fun i hi => h0 i (Finset.mem_coe.1 hi)

example := clawMatrix_represents.exists_mem_rowSpace_of_cocircuit clawMatroid_triad

end Transformer.GainMatroid

module

public import Transformer.GainMatroid.Section2_DeltaProjEquiv
public import Transformer.GainMatroid.Section2_ExchangedMatrices
public import Transformer.GainMatroid.Section2_K4Forms

/-!
# Coindependent sets, and the row space of `Δ_Y ∇_Y B`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, Proposition 2.9. When `I` is coindependent
in a matroid that `A` represents, the columns outside `I` span every column, so a vector of the row
space of `A` vanishing outside `I` vanishes (`IsMatrixRepresentation.eq_zero_of_dual_indep`). For
a coindependent triangle this is the hypothesis of `IsTriangleForm.projEquiv_of_deltaMatrix`.

Let `B` be in the form `IsTriadForm` on the triad `Y = {a, b, c}`, with rows `x, y, z, v`. Then
`∇_Y B` is in triangle form on the rows `x, y, z` (`IsTriadForm.isTriangleForm_nablaMatrix`), and
`Δ_Y ∇_Y B` has the row space of `B` (`IsTriadForm.rowSpace_deltaMatrix_nablaMatrix`): its row
`none` is the row `v` of `B`, its row `some v` is zero since the row `v` of `B` vanishes off `Y`,
and its other rows are those of `B`.

`iK4` over `𝔽₂`, with the labels of `k4Matroid`, has its triangle `{a, b, c}` in triangle form
(`k4Matrix_isTriangleForm`), and this triangle is coindependent (`k4_triangle_coindependent`).
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {E R F : Type*} [Field F]

/-- **A vector of the row space vanishing off a coindependent set vanishes.** -/
theorem IsMatrixRepresentation.eq_zero_of_dual_indep [Finite E] {M : Matroid E}
    {A : Matrix R E F} (hA : IsMatrixRepresentation M A) {I : Set E} (hI : M✶.Indep I)
    {w : E → F} (hw : w ∈ rowSpace A) (h₀ : ∀ e, e ∉ I → w e = 0) : w = 0 := by
  obtain ⟨ℓ, hℓ⟩ := exists_linearMap_col_eq hw
  have hle : Submodule.span F ((fun f r => A r f) '' Iᶜ) ≤ LinearMap.ker ℓ := by
    rw [Submodule.span_le]
    rintro _ ⟨f, hf, rfl⟩
    rw [SetLike.mem_coe, LinearMap.mem_ker, hℓ]
    exact h₀ f hf
  funext e
  rw [← hℓ, Pi.zero_apply]
  exact LinearMap.mem_ker.1 (hle ((hA.dual_indep_iff I).1 hI e))

namespace IsTriadForm

variable [DecidableEq E] [DecidableEq R] {B : Matrix R E F} {v : R} {a b c : E} {x y z : R}

/-- `∇_Y B` is in triangle form. -/
theorem isTriangleForm_nablaMatrix (hform : IsTriadForm B v a b c x y z) :
    IsTriangleForm (nablaMatrix B a b c x y z) a b c x y z := by
  have hba := hform.a_ne_b.symm
  have hca := hform.a_ne_c.symm
  have hcb := hform.b_ne_c.symm
  refine ⟨hform.ne_xy, hform.ne_xz, hform.ne_yz, fun r => ?_, fun r => ?_, fun r => ?_⟩ <;>
    simp [nablaMatrix, hba, hca, hcb]

/-- The row `none` of `Δ_Y ∇_Y B` is the row `v` of `B`. -/
theorem deltaMatrix_nablaMatrix_none (hform : IsTriadForm B v a b c x y z) :
    (deltaMatrix (nablaMatrix B a b c x y z) a b c x y z).row none = B.row v := by
  have hba := hform.a_ne_b.symm
  have hca := hform.a_ne_c.symm
  have hcb := hform.b_ne_c.symm
  ext e
  simp only [Matrix.row_apply]
  by_cases h₁ : e = a
  · rw [h₁, hform.col_a]
    simp [deltaMatrix, hform.ne_vx]
  by_cases h₂ : e = b
  · rw [h₂, hform.col_b]
    simp [deltaMatrix, hba, hform.ne_vy]
  by_cases h₃ : e = c
  · rw [h₃, hform.col_c]
    simp [deltaMatrix, hca, hcb, hform.ne_vz]
  rw [hform.row_v e h₁ h₂ h₃]
  simp [deltaMatrix, h₁, h₂, h₃]

/-- A row `some r` of `Δ_Y ∇_Y B` with `r ≠ v` is the row `r` of `B`. -/
theorem deltaMatrix_nablaMatrix_some (hform : IsTriadForm B v a b c x y z) {r : R} (hr : r ≠ v) :
    (deltaMatrix (nablaMatrix B a b c x y z) a b c x y z).row (some r) = B.row r := by
  have hba := hform.a_ne_b.symm
  have hca := hform.a_ne_c.symm
  have hcb := hform.b_ne_c.symm
  ext e
  simp only [Matrix.row_apply]
  by_cases h₁ : e = a
  · rw [h₁, hform.col_a]
    simp [deltaMatrix, hr]
  by_cases h₂ : e = b
  · rw [h₂, hform.col_b]
    simp [deltaMatrix, hba, hr]
  by_cases h₃ : e = c
  · rw [h₃, hform.col_c]
    simp [deltaMatrix, hca, hcb, hr]
  simp [deltaMatrix, nablaMatrix, h₁, h₂, h₃]

/-- The row `some v` of `Δ_Y ∇_Y B` is zero. -/
theorem deltaMatrix_nablaMatrix_some_v (hform : IsTriadForm B v a b c x y z) :
    (deltaMatrix (nablaMatrix B a b c x y z) a b c x y z).row (some v) = 0 := by
  have hba := hform.a_ne_b.symm
  have hca := hform.a_ne_c.symm
  have hcb := hform.b_ne_c.symm
  ext e
  simp only [Matrix.row_apply, Pi.zero_apply]
  by_cases h₁ : e = a
  · simp [h₁, deltaMatrix, hform.ne_vx]
  by_cases h₂ : e = b
  · simp [h₂, deltaMatrix, hba, hform.ne_vy]
  by_cases h₃ : e = c
  · simp [h₃, deltaMatrix, hca, hcb, hform.ne_vz]
  simp [deltaMatrix, nablaMatrix, h₁, h₂, h₃, hform.row_v e h₁ h₂ h₃]

/-- **`Δ_Y ∇_Y B` has the row space of `B`.** -/
theorem rowSpace_deltaMatrix_nablaMatrix (hform : IsTriadForm B v a b c x y z) :
    rowSpace (deltaMatrix (nablaMatrix B a b c x y z) a b c x y z) = rowSpace B := by
  rw [rowSpace, rowSpace]
  refine le_antisymm (Submodule.span_le.2 ?_) (Submodule.span_le.2 ?_)
  · rintro _ ⟨r, rfl⟩
    cases r with
    | none =>
      rw [hform.deltaMatrix_nablaMatrix_none]
      exact Submodule.subset_span ⟨v, rfl⟩
    | some r =>
      by_cases hr : r = v
      · rw [hr, hform.deltaMatrix_nablaMatrix_some_v]
        exact Submodule.zero_mem _
      · rw [hform.deltaMatrix_nablaMatrix_some hr]
        exact Submodule.subset_span ⟨r, rfl⟩
  · rintro _ ⟨r, rfl⟩
    by_cases hr : r = v
    · rw [hr, ← hform.deltaMatrix_nablaMatrix_none]
      exact Submodule.subset_span ⟨none, rfl⟩
    · rw [← hform.deltaMatrix_nablaMatrix_some hr]
      exact Submodule.subset_span ⟨some r, rfl⟩

end IsTriadForm

set_option maxRecDepth 100000 in
/-- The triangle `{a, b, c}` of `iK4` over `𝔽₂`, with the labels of `k4Matroid`, is in triangle
form. -/
theorem k4Matrix_isTriangleForm :
    IsTriangleForm ((iK4 : Matrix (Fin 4) (Fin 6) (ZMod 2)).submatrix id k4Label) (.inl 0)
      (.inl 1) (.inl 2) 0 1 2 := by
  refine ⟨by decide, by decide, by decide, ?_, ?_, ?_⟩ <;> decide +kernel

/-- The triangle `{a, b, c}` of `M(K₄)` is coindependent: the triad `{a', b', c'}` is a base. -/
theorem k4_triangle_coindependent :
    k4Matroid✶.Indep ({.inl 0, .inl 1, .inl 2} : Set (Fin 3 ⊕ Fin 3)) := by
  have h : ({.inl 0, .inl 1, .inl 2} : Set (Fin 3 ⊕ Fin 3)) =
      (({.inl 0, .inl 1, .inl 2} : Finset (Fin 3 ⊕ Fin 3)) : Set (Fin 3 ⊕ Fin 3)) := by simp
  rw [h]
  exact (matroidOfFinset_dual_indep_iff _).2 ⟨{.inr 0, .inr 1, .inr 2}, by decide +kernel,
    by decide⟩

example := iK4_represents_binary.eq_zero_of_dual_indep k4_triangle_coindependent
  (Submodule.zero_mem _) fun _ _ => rfl

example := k4Matrix_isTriangleForm.projEquiv_of_deltaMatrix k4Matrix_isTriangleForm
  (fun _ hw h₀ => iK4_represents_binary.eq_zero_of_dual_indep k4_triangle_coindependent hw h₀)
  (ProjEquiv.refl _)

example := (iK4_isTriadForm (F := ℚ)).isTriangleForm_nablaMatrix

example := (iK4_isTriadForm (F := ℚ)).deltaMatrix_nablaMatrix_none

example := (iK4_isTriadForm (F := ℚ)).deltaMatrix_nablaMatrix_some (r := 0) (by decide)

example := (iK4_isTriadForm (F := ℚ)).deltaMatrix_nablaMatrix_some_v

example := (iK4_isTriadForm (F := ℚ)).rowSpace_deltaMatrix_nablaMatrix

end Transformer.GainMatroid

module

public import Transformer.GainMatroid.Section2_DeltaRowSpace

/-!
# Projective equivalence across `Δ_X A`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, Proposition 2.9. Let `A` and `A'` be in the
form `IsTriangleForm` on the triangle `X = {a, b, c}`, possibly on different rows. The vectors of
the row space of such a matrix satisfy `w b = w a + w c` (`IsTriangleForm.eq_add_of_mem_rowSpace`).
So a scaling of the columns carrying the row space of `A` onto that of `A'` is constant on `X`
(`IsTriangleForm.scale_eq`), and then commutes with `triangleFold` (`triangleFold_scaleEquiv`).
Hence projectively equivalent matrices in this form have projectively equivalent exchanges
(`IsTriangleForm.projEquiv_deltaMatrix`).

Conversely, the row `none` of `Δ_X A` is supported on `X`. A scaling carrying the row space of
`Δ_X A` onto that of `Δ_X A'` sends it to a vector whose fold lies in the row space of `A'` and
vanishes off `X`. When no nonzero vector of the row space of `A'` vanishes off `X`, this fold is
zero, so the scaling is constant on `X`, and `A` and `A'` are projectively equivalent
(`IsTriangleForm.projEquiv_of_deltaMatrix`).
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {E R R' F : Type*} [DecidableEq E] [Field F] {a b c : E}

/-- A scaling constant on `{a, b, c}` commutes with `triangleFold`. -/
theorem triangleFold_scaleEquiv {s : E → Fˣ} (ha : s a = s c) (hb : s b = s c) (v : E → F) :
    triangleFold a b c (scaleEquiv s v) = scaleEquiv s (triangleFold a b c v) := by
  ext e
  change triangleFold a b c (fun j => v j * s j) e = triangleFold a b c v e * s e
  simp only [triangleFold_apply]
  split_ifs with h₁ h₂ h₃
  · rw [h₁, ha, hb]
    ring
  · rw [h₂, ha, hb]
    ring
  · rw [h₃, ha, hb]
    ring
  · rfl

/-- The inverse of a scaling constant on `{a, b, c}` commutes with `triangleFold`. -/
theorem triangleFold_scaleEquiv_symm {s : E → Fˣ} (ha : s a = s c) (hb : s b = s c)
    (v : E → F) :
    triangleFold a b c ((scaleEquiv s).symm v) = (scaleEquiv s).symm (triangleFold a b c v) := by
  apply (scaleEquiv s).injective
  rw [LinearEquiv.apply_symm_apply, ← triangleFold_scaleEquiv ha hb, LinearEquiv.apply_symm_apply]

namespace IsTriangleForm

variable [DecidableEq R] [DecidableEq R'] {A : Matrix R E F} {A' : Matrix R' E F} {x y z : R}
  {x' y' z' : R'}

omit [DecidableEq E] in
/-- The vectors of the row space of a matrix in triangle form satisfy `w b = w a + w c`. -/
theorem eq_add_of_mem_rowSpace (hform : IsTriangleForm A a b c x y z) {w : E → F}
    (hw : w ∈ rowSpace A) : w b = w a + w c := by
  induction hw using Submodule.span_induction with
  | mem v hv =>
    obtain ⟨r, rfl⟩ := hv
    simp only [Matrix.row_apply]
    rw [hform.col_a, hform.col_b, hform.col_c]
    ring
  | zero => simp
  | add v u _ _ hv hu =>
    simp only [Pi.add_apply]
    rw [hv, hu]
    ring
  | smul t v _ hv =>
    simp only [Pi.smul_apply, smul_eq_mul]
    rw [hv, mul_add]

omit [DecidableEq E] in
/-- A scaling of the columns carrying the row space of one matrix in triangle form onto that of
another is constant on the triangle. -/
theorem scale_eq (hform : IsTriangleForm A a b c x y z) (hform' : IsTriangleForm A' a b c x' y' z')
    {s : E → Fˣ} (hs : rowSpace (scaleCols s A) = rowSpace A') : s a = s c ∧ s b = s c := by
  have key (r : R) : A r b * s b = A r a * s a + A r c * s c :=
    hform'.eq_add_of_mem_rowSpace (w := (scaleCols s A).row r) (by
      rw [← hs]
      exact Submodule.subset_span ⟨r, rfl⟩)
  have hx := key x
  have hy := key y
  rw [hform.col_a, hform.col_b, hform.col_c] at hx hy
  simp [hform.ne_xy, hform.ne_xz, hform.ne_yz, hform.ne_xy.symm] at hx hy
  exact ⟨Units.ext (by linear_combination -hy), Units.ext hx⟩

/-- **Projectively equivalent matrices in triangle form have projectively equivalent
exchanges.** -/
theorem projEquiv_deltaMatrix (hform : IsTriangleForm A a b c x y z)
    (hform' : IsTriangleForm A' a b c x' y' z') (h : ProjEquiv A A') :
    ProjEquiv (deltaMatrix A a b c x y z) (deltaMatrix A' a b c x' y' z') := by
  obtain ⟨s, hs⟩ := h
  obtain ⟨ha, hb⟩ := hform.scale_eq hform' hs
  refine ⟨s, ?_⟩
  rw [rowSpace_scaleCols, hform.rowSpace_deltaMatrix, hform'.rowSpace_deltaMatrix, ← hs,
    rowSpace_scaleCols]
  ext v
  rw [Submodule.mem_map_equiv, Submodule.mem_comap, Submodule.mem_comap, Submodule.mem_map_equiv,
    triangleFold_scaleEquiv_symm ha hb]

/-- **Projectively equivalent exchanges come from projectively equivalent matrices**, when no
nonzero vector of the row space of `A'` vanishes off the triangle. -/
theorem projEquiv_of_deltaMatrix (hform : IsTriangleForm A a b c x y z)
    (hform' : IsTriangleForm A' a b c x' y' z')
    (hco : ∀ w ∈ rowSpace A', (∀ e, e ∉ ({a, b, c} : Set E) → w e = 0) → w = 0)
    (h : ProjEquiv (deltaMatrix A a b c x y z) (deltaMatrix A' a b c x' y' z')) :
    ProjEquiv A A' := by
  have hba := hform.a_ne_b.symm
  have hca := hform.a_ne_c.symm
  have hcb := hform.b_ne_c.symm
  obtain ⟨s, hs⟩ := h
  set u := -(deltaMatrix A a b c x y z).row none with hu
  have hSu (t : E) : scaleEquiv s u t = -deltaMatrix A a b c x y z none t * s t := rfl
  have hmem : triangleFold a b c (scaleEquiv s u) ∈ rowSpace A' := by
    have : scaleEquiv s u ∈ rowSpace (deltaMatrix A' a b c x' y' z') := by
      rw [← hs, rowSpace_scaleCols]
      exact Submodule.mem_map_of_mem (Submodule.neg_mem _ (Submodule.subset_span ⟨none, rfl⟩))
    rwa [hform'.rowSpace_deltaMatrix, Submodule.mem_comap] at this
  have h₀ := hco _ hmem fun e he => by
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff, not_or] at he
    obtain ⟨h₁, h₂, h₃⟩ := he
    rw [triangleFold_apply, ite_eq_right h₁, ite_eq_right h₂, ite_eq_right h₃, hSu]
    simp [deltaMatrix, h₁, h₂, h₃]
  have hA := congrFun h₀ a
  have hB := congrFun h₀ b
  rw [triangleFold_apply, ite_eq_left rfl, hSu, hSu] at hA
  rw [triangleFold_apply, ite_eq_right hba, ite_eq_left rfl, hSu, hSu] at hB
  simp [deltaMatrix, hba, hca, hcb, sub_eq_zero] at hA hB
  have hcomm : (scaleEquiv s).toLinearMap ∘ₗ triangleFold a b c =
      triangleFold a b c ∘ₗ (scaleEquiv s).toLinearMap :=
    LinearMap.ext fun v => (triangleFold_scaleEquiv (Units.ext hB) (Units.ext hA) v).symm
  refine ⟨s, ?_⟩
  rw [rowSpace_scaleCols, ← hform.map_rowSpace_deltaMatrix, ← hform'.map_rowSpace_deltaMatrix,
    ← hs, rowSpace_scaleCols, ← Submodule.map_comp, ← Submodule.map_comp, hcomm]

end IsTriangleForm

example := triangleFold_scaleEquiv (F := ℚ) (a := (0 : Fin 3)) (b := 1) (c := 2) (s := 1) rfl rfl 0

example := triangleFold_scaleEquiv_symm (F := ℚ) (a := (0 : Fin 3)) (b := 1) (c := 2) (s := 1) rfl
  rfl 0

example := (iK4_isTriangleForm (F := ℚ)).eq_add_of_mem_rowSpace (Submodule.zero_mem _)

example := (iK4_isTriangleForm (F := ℚ)).scale_eq iK4_isTriangleForm (s := 1)
  (by rw [scaleCols_one])

example := (iK4_isTriangleForm (F := ℚ)).projEquiv_deltaMatrix iK4_isTriangleForm (ProjEquiv.refl _)

end Transformer.GainMatroid

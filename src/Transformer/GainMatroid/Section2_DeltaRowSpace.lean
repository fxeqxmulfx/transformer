module

public import Transformer.GainMatroid.Section2_ConnectionMatroid

/-!
# The row space of `Δ_X A`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, Proposition 2.9: the projective classes of
representations of `M` and of `Δ_X M` correspond. Let `A` be in the form `IsTriangleForm` on the
triangle `X = {a, b, c}`, with rows `x, y, z`: its columns there are `y − z`, `x − z`, `x − y`,
and those of `Δ_X A` are `x − none`, `y − none`, `z − none`. So a vector of the row space of
`Δ_X A` determines one of `A` by differences on `X`. `triangleFold` is this linear map: it replaces
`v a`, `v b`, `v c` by `v b − v c`, `v a − v c`, `v a − v b` and keeps the other coordinates.

* It sends the row `some w` of `Δ_X A` to the row `w` of `A` and the row `none` to zero
  (`IsTriangleForm.triangleFold_row_some`, `IsTriangleForm.triangleFold_row_none`), so it maps the
  row space of `Δ_X A` onto that of `A` (`IsTriangleForm.map_rowSpace_deltaMatrix`).
* Its kernel is spanned by the row `none`, so the row space of `Δ_X A` is the preimage of that of
  `A` (`IsTriangleForm.rowSpace_deltaMatrix`).
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {E R F : Type*} [DecidableEq E] [Field F] {a b c : E}

/-- Reads a vector of the row space of `Δ_X A` as one of `A` (§2.4): the values `v a`, `v b`,
`v c` on the columns `x − none`, `y − none`, `z − none` become `v b − v c`, `v a − v c`,
`v a − v b` on the columns `y − z`, `x − z`, `x − y`. -/
def triangleFold (a b c : E) : (E → F) →ₗ[F] (E → F) where
  toFun v e :=
    if e = a then v b - v c else if e = b then v a - v c else if e = c then v a - v b else v e
  map_add' v w := by
    ext e
    simp only [Pi.add_apply]
    split_ifs <;> ring
  map_smul' t v := by
    ext e
    simp only [Pi.smul_apply, smul_eq_mul, RingHom.id_apply]
    split_ifs <;> ring

theorem triangleFold_apply (v : E → F) (e : E) :
    triangleFold a b c v e =
      if e = a then v b - v c else if e = b then v a - v c else if e = c then v a - v b
      else v e :=
  rfl

namespace IsTriangleForm

variable [DecidableEq R] {A : Matrix R E F} {x y z : R}

/-- The row `some w` of `Δ_X A` folds to the row `w` of `A`. -/
theorem triangleFold_row_some (hform : IsTriangleForm A a b c x y z) (w : R) :
    triangleFold a b c ((deltaMatrix A a b c x y z).row (some w)) = A.row w := by
  have hba := hform.a_ne_b.symm
  have hca := hform.a_ne_c.symm
  have hcb := hform.b_ne_c.symm
  ext e
  simp only [triangleFold_apply, Matrix.row_apply]
  by_cases h₁ : e = a
  · rw [ite_eq_left h₁, h₁, hform.col_a]
    simp [deltaMatrix, hba, hca, hcb]
  by_cases h₂ : e = b
  · rw [ite_eq_right h₁, ite_eq_left h₂, h₂, hform.col_b]
    simp [deltaMatrix, hca, hcb]
  by_cases h₃ : e = c
  · rw [ite_eq_right h₁, ite_eq_right h₂, ite_eq_left h₃, h₃, hform.col_c]
    simp [deltaMatrix, hba]
  rw [ite_eq_right h₁, ite_eq_right h₂, ite_eq_right h₃]
  simp [deltaMatrix, h₁, h₂, h₃]

/-- The row `none` of `Δ_X A` folds to zero. -/
theorem triangleFold_row_none (hform : IsTriangleForm A a b c x y z) :
    triangleFold a b c ((deltaMatrix A a b c x y z).row none) = 0 := by
  have hba := hform.a_ne_b.symm
  have hca := hform.a_ne_c.symm
  have hcb := hform.b_ne_c.symm
  ext e
  rw [triangleFold_apply, Pi.zero_apply]
  by_cases h₁ : e = a
  · simp [h₁, deltaMatrix, hba, hca, hcb]
  by_cases h₂ : e = b
  · simp [h₂, deltaMatrix, hba, hca, hcb]
  by_cases h₃ : e = c
  · simp [h₃, deltaMatrix, hba, hca, hcb]
  simp [h₁, h₂, h₃, deltaMatrix]

/-- `triangleFold` maps the row space of `Δ_X A` onto that of `A`. -/
theorem map_rowSpace_deltaMatrix (hform : IsTriangleForm A a b c x y z) :
    (rowSpace (deltaMatrix A a b c x y z)).map (triangleFold a b c) = rowSpace A := by
  rw [rowSpace, Submodule.map_span, rowSpace]
  refine le_antisymm (Submodule.span_le.2 ?_) (Submodule.span_le.2 ?_)
  · rintro _ ⟨_, ⟨r, rfl⟩, rfl⟩
    cases r with
    | none =>
      rw [hform.triangleFold_row_none]
      exact Submodule.zero_mem _
    | some w =>
      rw [hform.triangleFold_row_some]
      exact Submodule.subset_span ⟨w, rfl⟩
  · rintro _ ⟨w, rfl⟩
    exact Submodule.subset_span ⟨_, ⟨some w, rfl⟩, hform.triangleFold_row_some w⟩

/-- **The row space of `Δ_X A`** is the preimage of that of `A` under `triangleFold`. -/
theorem rowSpace_deltaMatrix (hform : IsTriangleForm A a b c x y z) :
    rowSpace (deltaMatrix A a b c x y z) = (rowSpace A).comap (triangleFold a b c) := by
  have hba := hform.a_ne_b.symm
  have hca := hform.a_ne_c.symm
  have hcb := hform.b_ne_c.symm
  rw [← hform.map_rowSpace_deltaMatrix, Submodule.comap_map_eq, left_eq_sup]
  intro d hd
  have h₀ (e : E) : triangleFold a b c d e = 0 := by
    rw [LinearMap.mem_ker.1 hd, Pi.zero_apply]
  have hab : d b = d c := by
    have := h₀ a
    rw [triangleFold_apply, ite_eq_left rfl] at this
    exact sub_eq_zero.1 this
  have hac : d a = d c := by
    have := h₀ b
    rw [triangleFold_apply, ite_eq_right hba, ite_eq_left rfl] at this
    exact sub_eq_zero.1 this
  have hd' : d = (-d c) • (deltaMatrix A a b c x y z).row none := by
    ext e
    rw [Pi.smul_apply, smul_eq_mul, Matrix.row_apply]
    by_cases h₁ : e = a
    · simp [h₁, hac, deltaMatrix]
    by_cases h₂ : e = b
    · simp [h₂, hab, deltaMatrix]
    by_cases h₃ : e = c
    · simp [h₃, deltaMatrix]
    have := h₀ e
    rw [triangleFold_apply, ite_eq_right h₁, ite_eq_right h₂, ite_eq_right h₃] at this
    simp [this, h₁, h₂, h₃, deltaMatrix]
  rw [hd']
  exact Submodule.smul_mem _ _ (Submodule.subset_span ⟨none, rfl⟩)

end IsTriangleForm

example := (iK4_isTriangleForm (F := ℚ)).triangleFold_row_some 0

example := (iK4_isTriangleForm (F := ℚ)).triangleFold_row_none

example := (iK4_isTriangleForm (F := ℚ)).map_rowSpace_deltaMatrix

example := (iK4_isTriangleForm (F := ℚ)).rowSpace_deltaMatrix

end Transformer.GainMatroid

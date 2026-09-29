/-
# Expanded kaleidoscope operators as genuine matrices

Arora et al., arXiv:2312.04927v1, Appendix `def: kaleidoscope` and
`def: W-kmat`. Expansion is zero padding, application of a product of
butterfly factors, and cropping. We prove this remains linear and identify
its matrix through its action on coordinate basis vectors.
-/

import Transformer.Zoology.Appendix_KaleidoscopeLinear

open scoped BigOperators

namespace Transformer.Zoology

/-- Zero padding preserves addition. Source: Appendix `def: kaleidoscope`,
matrix `Sᵀ`. -/
theorem padButterfly_add {k e : ℕ}
    (u v : Fin (butterflyWidth k) → ℝ) :
    padButterfly (e := e) (u + v) = padButterfly u + padButterfly v := by
  funext i
  by_cases h : i.val < butterflyWidth k <;>
    simp [padButterfly, Pi.add_apply, h]

/-- Zero padding preserves scalar multiplication. Source: Appendix
`def: kaleidoscope`, matrix `Sᵀ`. -/
theorem padButterfly_smul {k e : ℕ} (r : ℝ)
    (u : Fin (butterflyWidth k) → ℝ) :
    padButterfly (e := e) (r • u) = r • padButterfly u := by
  funext i
  by_cases h : i.val < butterflyWidth k <;>
    simp [padButterfly, Pi.smul_apply, h]

/-- Cropping preserves addition. Source: Appendix `def: kaleidoscope`,
matrix `S`. -/
theorem cropButterfly_add {k e : ℕ}
    (u v : Fin (butterflyWidth (k + e)) → ℝ) :
    cropButterfly (u + v) = cropButterfly u + cropButterfly v := by
  funext i
  simp [cropButterfly, Pi.add_apply]

/-- Cropping preserves scalar multiplication. Source: Appendix
`def: kaleidoscope`, matrix `S`. -/
theorem cropButterfly_smul {k e : ℕ} (r : ℝ)
    (u : Fin (butterflyWidth (k + e)) → ℝ) :
    cropButterfly (r • u) = r • cropButterfly u := by
  funext i
  simp [cropButterfly, Pi.smul_apply]

/-- An expanded kaleidoscope operator preserves addition. Source:
Appendix `def: kaleidoscope`, class `(BB*)^w_e`. -/
theorem ExpandedKaleidoscope.apply_add {k e : ℕ}
    (K : ExpandedKaleidoscope k e)
    (u v : Fin (butterflyWidth k) → ℝ) :
    K.apply (u + v) = K.apply u + K.apply v := by
  unfold ExpandedKaleidoscope.apply
  rw [padButterfly_add, K.inner.apply_add, cropButterfly_add]

/-- An expanded kaleidoscope operator preserves scalar multiplication.
Source: Appendix `def: kaleidoscope`, class `(BB*)^w_e`. -/
theorem ExpandedKaleidoscope.apply_smul {k e : ℕ}
    (K : ExpandedKaleidoscope k e) (r : ℝ)
    (u : Fin (butterflyWidth k) → ℝ) :
    K.apply (r • u) = r • K.apply u := by
  unfold ExpandedKaleidoscope.apply
  rw [padButterfly_smul, K.inner.apply_smul, cropButterfly_smul]

/-- An expanded kaleidoscope element as a real-linear operator.
Source: Appendix `def: kaleidoscope` and `def: W-kmat`. -/
def ExpandedKaleidoscope.toLinearMap {k e : ℕ}
    (K : ExpandedKaleidoscope k e) :
    (Fin (butterflyWidth k) → ℝ) →ₗ[ℝ]
      (Fin (butterflyWidth k) → ℝ) := {
  toFun := K.apply
  map_add' := K.apply_add
  map_smul' := K.apply_smul
}

/-- The standard coordinate basis vector. Source: Appendix
`def: W-kmat`, matrix-vector multiplication convention. -/
def coordinateBasis {n : ℕ} (j : Fin n) : Fin n → ℝ :=
  fun i => if i = j then 1 else 0

/-- Every real finite vector is the sum of its scaled coordinate vectors.
Source: Appendix `def: W-kmat`, matrix-vector semantics. -/
theorem vector_eq_sum_coordinateBasis {n : ℕ} (u : Fin n → ℝ) :
    u = ∑ j : Fin n, u j • coordinateBasis j := by
  funext i
  simp [coordinateBasis, Finset.sum_apply, Pi.smul_apply,
    smul_eq_mul]

/-- Matrix coefficients of an expanded kaleidoscope operator, obtained by
applying it to basis vectors. Source: Appendix `def: kaleidoscope`,
`M = S E Sᵀ`. -/
def ExpandedKaleidoscope.matrix {k e : ℕ}
    (K : ExpandedKaleidoscope k e)
    (i j : Fin (butterflyWidth k)) : ℝ :=
  K.apply (coordinateBasis j) i

/-- The stored `BB*` product acts by the matrix defined above. This gives
a direct semantic bridge to the paper's K-matrix weight `W`.
Source: Appendix `def: W-kmat`, linear projection. -/
theorem ExpandedKaleidoscope.apply_eq_matrix {k e : ℕ}
    (K : ExpandedKaleidoscope k e)
    (u : Fin (butterflyWidth k) → ℝ)
    (i : Fin (butterflyWidth k)) :
    K.apply u i = ∑ j : Fin (butterflyWidth k),
      K.matrix i j * u j := by
  let L := K.toLinearMap
  calc
    K.apply u i = L u i := rfl
    _ = L (∑ j : Fin (butterflyWidth k), u j • coordinateBasis j) i := by
      rw [← vector_eq_sum_coordinateBasis u]
    _ = ∑ j : Fin (butterflyWidth k), K.matrix i j * u j := by
      simp [L, ExpandedKaleidoscope.toLinearMap,
        map_sum, Finset.sum_apply, ExpandedKaleidoscope.matrix,
        mul_comm]

end Transformer.Zoology

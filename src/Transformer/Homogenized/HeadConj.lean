/-
# Homogenized Transformers — rotating a configuration and a head

The two deterministic halves of the proof of `lem:drift_on_simplex_selfcontained`
of arXiv:2604.01978v1:

* two configurations with the same Gram matrix differ by an orthogonal map `O`
  (`exists_linearIsometryEquiv_of_inner_eq`);
* `⟨A O x, O y⟩ = ⟨Oᵀ A O x, y⟩`, so rotating the tokens by `O` is the same as
  conjugating the query-key matrix by `O` (`softBary_conjMat`); and for
  `A = W W'ᵀ` the conjugate is `(Oᵀ W)(Oᵀ W')ᵀ` (`conjMat_mul_transpose`), the
  form in which the Gaussian law's invariance can be seen.

Source: arXiv:2604.01978v1, proof of `lem:drift_on_simplex_selfcontained`.
-/

import Transformer.Homogenized.Barycenter

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### Equal Gram matrices -/

/-- The norm of `Σ_k c_k x_k` is determined by the Gram matrix of the `x_k`. -/
theorem norm_linearCombination_eq {d n : ℕ} {X Y : Idx n → EucSpace d}
    (hgram : ∀ k l : Idx n, inner (𝕜 := ℝ) (X k) (X l) = inner (𝕜 := ℝ) (Y k) (Y l))
    (c : Idx n → ℝ) :
    ‖Fintype.linearCombination ℝ X c‖ = ‖Fintype.linearCombination ℝ Y c‖ := by
  have h : ∀ Z : Idx n → EucSpace d, ‖Fintype.linearCombination ℝ Z c‖ ^ 2 =
      ∑ k, ∑ l, c k * c l * inner (𝕜 := ℝ) (Z k) (Z l) := fun Z => by
    rw [← real_inner_self_eq_norm_sq, Fintype.linearCombination_apply, sum_inner]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [inner_sum]
    refine Finset.sum_congr rfl fun l _ => ?_
    rw [inner_smul_left, inner_smul_right]
    simp only [conj_trivial]; ring
  rw [← sq_eq_sq₀ (norm_nonneg _) (norm_nonneg _), h, h]
  simp only [hgram]

/-- **Two configurations with the same Gram matrix differ by an orthogonal
map**: `x̃_k = O x_k` for a linear isometry `O` of `ℝ^d`.  The source's "there
exists an orthogonal `d × d` matrix `O` such that `x_i = O x̃_i`".

Source: arXiv:2604.01978v1, proof of `lem:drift_on_simplex_selfcontained`. -/
theorem exists_linearIsometryEquiv_of_inner_eq {d n : ℕ} {X Y : Idx n → EucSpace d}
    (hgram : ∀ k l : Idx n, inner (𝕜 := ℝ) (X k) (X l) = inner (𝕜 := ℝ) (Y k) (Y l)) :
    ∃ O : EucSpace d ≃ₗᵢ[ℝ] EucSpace d, ∀ k, O (X k) = Y k := by
  set fX := Fintype.linearCombination ℝ X
  set fY := Fintype.linearCombination ℝ Y
  have hker : LinearMap.ker fX ≤ LinearMap.ker fY := fun c hc => by
    rw [LinearMap.mem_ker, ← norm_eq_zero] at hc ⊢
    rw [← norm_linearCombination_eq hgram c]; exact hc
  set L₀ : LinearMap.range fX →ₗ[ℝ] EucSpace d :=
    (LinearMap.ker fX).liftQ fY hker ∘ₗ fX.quotKerEquivRange.symm.toLinearMap
  have hL₀ : ∀ c, L₀ ⟨fX c, LinearMap.mem_range_self fX c⟩ = fY c := fun c => by
    simp [L₀]
  let L : LinearMap.range fX →ₗᵢ[ℝ] EucSpace d :=
    { toLinearMap := L₀
      norm_map' := by
        rintro ⟨z, c, rfl⟩
        exact (hL₀ c).symm ▸ (norm_linearCombination_eq hgram c).symm }
  refine ⟨L.extend.toLinearIsometryEquiv rfl, fun k => ?_⟩
  have hk : fX (Pi.single k 1) = X k := by simp [fX, Fintype.linearCombination_apply_single]
  have := LinearIsometry.extend_apply L ⟨fX (Pi.single k 1), LinearMap.mem_range_self fX _⟩
  simp only [L, LinearIsometry.coe_mk, hL₀] at this
  simp only [LinearIsometry.coe_toLinearIsometryEquiv]
  rw [← hk, this]
  simp [fY, Fintype.linearCombination_apply_single]

/-! ### Conjugating the query-key matrix -/

/-- `A ↦ Oᵀ A O`, the query-key matrix seen from the rotated frame. -/
noncomputable def conjMat {d : ℕ} (O : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    Matrix (Fin d) (Fin d) ℝ ≃ₗ[ℝ] Matrix (Fin d) (Fin d) ℝ :=
  Matrix.toEuclideanLin.trans
    ((O.toLinearEquiv.arrowCongr O.toLinearEquiv).symm.trans Matrix.toEuclideanLin.symm)

theorem toEuclideanLin_conjMat {d : ℕ} (O : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (A : Matrix (Fin d) (Fin d) ℝ) (x : EucSpace d) :
    Matrix.toEuclideanLin (conjMat O A) x = O.symm (Matrix.toEuclideanLin A (O x)) := by
  simp [conjMat]

theorem softWeight_conjMat {d : ℕ} (β : ℝ) (O : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (A : Matrix (Fin d) (Fin d) ℝ) (x y : EucSpace d) :
    softWeight β (conjMat O A) x y = softWeight β A (O x) (O y) := by
  rw [softWeight, softWeight, toEuclideanLin_conjMat, ← O.inner_map_map, O.apply_symm_apply]

/-- Rotating the tokens by `O` is conjugating the query-key matrix by `O`:
`m_{β,A}[μ_{OX}](O x) = O m_{β,OᵀAO}[μ_X](x)`. -/
theorem softBary_conjMat {d n : ℕ} (β : ℝ) (O : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (A : Matrix (Fin d) (Fin d) ℝ) (X : Idx n → EucSpace d) (x : EucSpace d) :
    O (softBary β (conjMat O A) (empMeasure X) x) =
      softBary β A (empMeasure fun k => O (X k)) (O x) := by
  simp only [softBary, integral_empMeasure, softWeight_conjMat, map_smul, map_sum]

theorem toEuclideanLin_symm_apply {d : ℕ} (f : EucSpace d →ₗ[ℝ] EucSpace d) (i j : Fin d) :
    Matrix.toEuclideanLin.symm f i j = f (EuclideanSpace.single j 1) i := by
  obtain ⟨M, rfl⟩ := Matrix.toEuclideanLin.surjective f
  simp [Matrix.toLpLin_apply, Matrix.mulVec_single]

theorem inner_toEuclideanLin_transpose {d : ℕ} (P : Matrix (Fin d) (Fin d) ℝ)
    (u y : EucSpace d) :
    inner (𝕜 := ℝ) (Matrix.toEuclideanLin P u) y =
      inner (𝕜 := ℝ) u (Matrix.toEuclideanLin P.transpose y) := by
  rw [← Matrix.conjTranspose_eq_transpose_of_trivial,
    Matrix.toEuclideanLin_conjTranspose_eq_adjoint, LinearMap.adjoint_inner_right]

theorem inner_toEuclideanLin_mul_transpose {d : ℕ} (P Q : Matrix (Fin d) (Fin d) ℝ)
    (x y : EucSpace d) :
    inner (𝕜 := ℝ) (Matrix.toEuclideanLin (P * Q.transpose) x) y =
      inner (𝕜 := ℝ) (Matrix.toEuclideanLin Q.transpose x)
        (Matrix.toEuclideanLin P.transpose y) := by
  rw [Matrix.toLpLin_mul_same, LinearMap.comp_apply, inner_toEuclideanLin_transpose]

/-- `W ↦ Oᵀ W`: the columns of `W` rotated by `O⁻¹`. -/
noncomputable def colMap {d : ℕ} (O : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (W : Matrix (Fin d) (Fin d) ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  Matrix.toEuclideanLin.symm (O.symm.toLinearMap ∘ₗ Matrix.toEuclideanLin W)

theorem colMap_apply {d : ℕ} (O : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (W : Matrix (Fin d) (Fin d) ℝ) (i j : Fin d) :
    colMap O W i j = O.symm (WithLp.toLp 2 fun k => W k j) i := by
  rw [colMap, toEuclideanLin_symm_apply]
  simp [Matrix.toLpLin_apply, Matrix.mulVec_single]
  rfl

theorem toEuclideanLin_colMap_transpose {d : ℕ} (O : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (W : Matrix (Fin d) (Fin d) ℝ) (y : EucSpace d) :
    Matrix.toEuclideanLin (colMap O W).transpose y =
      Matrix.toEuclideanLin W.transpose (O y) := by
  refine ext_inner_left ℝ fun u => ?_
  rw [← inner_toEuclideanLin_transpose, ← inner_toEuclideanLin_transpose, colMap,
    LinearEquiv.apply_symm_apply, LinearMap.comp_apply]
  change inner ℝ (O.symm _) y = _
  rw [← O.inner_map_map, O.apply_symm_apply]

/-- `Oᵀ (W W'ᵀ) O = (Oᵀ W)(Oᵀ W')ᵀ`. -/
theorem conjMat_mul_transpose {d : ℕ} (O : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (W W' : Matrix (Fin d) (Fin d) ℝ) :
    conjMat O (W * W'.transpose) = colMap O W * (colMap O W').transpose := by
  refine Matrix.toEuclideanLin.injective
    (LinearMap.ext fun x => ext_inner_right ℝ fun y => ?_)
  rw [toEuclideanLin_conjMat, ← O.inner_map_map, O.apply_symm_apply,
    inner_toEuclideanLin_mul_transpose, inner_toEuclideanLin_mul_transpose,
    toEuclideanLin_colMap_transpose, toEuclideanLin_colMap_transpose]

end Homogenized
end Transformer

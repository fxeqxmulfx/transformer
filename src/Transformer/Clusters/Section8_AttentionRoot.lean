/-
# The positive attention root

ArXiv:2305.05465v6, §8.1, `eq: A`: positive definiteness of `QᵀK`
supplies a symmetric positive square root with
`⟨Qu, Kv⟩ = ⟨Au, Av⟩`. The finite-dimensional spectral theorem
provides an orthonormal eigenbasis. Strict positivity of the original
quadratic form makes every eigenvalue positive; replacing each by its
positive square root constructs the required continuous linear map.

The diagonal-map formulas below verify symmetry, positivity, and the
bilinear identity explicitly. Thus the root is derived from the paper's
original hypothesis rather than added as a convergence assumption.
-/

import Transformer.Clusters.Section1_Dynamics
import Mathlib.Analysis.InnerProductSpace.Spectrum
import Mathlib.Analysis.InnerProductSpace.Adjoint

open scoped BigOperators
open Real

namespace Transformer.Clusters

variable {d : ℕ}

/-- A map diagonal in an orthonormal basis, used to construct `eq: A`.
Source: arXiv:2305.05465v6, §8.1, `eq: A`. -/
noncomputable def orthonormalDiagonal (b : OrthonormalBasis (Idx d) ℝ (EucSpace d))
    (a : Idx d → ℝ) : ParamMatrix d :=
  ∑ i : Idx d, a i • (innerSL ℝ (b i)).smulRight (b i)

/-- The diagonal map has its usual coordinate expansion.
Source: arXiv:2305.05465v6, §8.1, the spectral construction in `eq: A`. -/
theorem orthonormalDiagonal_apply (b : OrthonormalBasis (Idx d) ℝ (EucSpace d))
    (a : Idx d → ℝ) (u : EucSpace d) :
    orthonormalDiagonal b a u = ∑ i : Idx d, (a i * inner (𝕜 := ℝ) (b i) u) • b i := by
  simp only [orthonormalDiagonal, sum_apply, smul_apply,
    ContinuousLinearMap.smulRight_apply, innerSL_apply_apply, smul_smul]

/-- Each coordinate is multiplied by its diagonal coefficient.
Source: arXiv:2305.05465v6, §8.1, `eq: A`. -/
theorem inner_orthonormalDiagonal (b : OrthonormalBasis (Idx d) ℝ (EucSpace d))
    (a : Idx d → ℝ) (u : EucSpace d) (i : Idx d) :
    inner (𝕜 := ℝ) (b i) (orthonormalDiagonal b a u) = a i * inner (𝕜 := ℝ) (b i) u := by
  classical
  rw [orthonormalDiagonal_apply, inner_sum]
  simp [real_inner_smul_right, b.inner_eq_ite]

/-- The quadratic form is the weighted sum of products of coordinates.
Source: arXiv:2305.05465v6, §8.1, `eq: A`. -/
theorem orthonormalDiagonal_inner (b : OrthonormalBasis (Idx d) ℝ (EucSpace d))
    (a : Idx d → ℝ) (u v : EucSpace d) :
    inner (𝕜 := ℝ) (orthonormalDiagonal b a u) v =
      ∑ i : Idx d, a i * inner (𝕜 := ℝ) (b i) u * inner (𝕜 := ℝ) (b i) v := by
  rw [orthonormalDiagonal_apply, sum_inner]
  simp only [real_inner_smul_left]

/-- Real diagonal coefficients give a symmetric map.
Source: arXiv:2305.05465v6, §8.1, the positive root in `eq: A`. -/
theorem orthonormalDiagonal_symmetric (b : OrthonormalBasis (Idx d) ℝ (EucSpace d))
    (a : Idx d → ℝ) (u v : EucSpace d) :
    inner (𝕜 := ℝ) (orthonormalDiagonal b a u) v =
      inner (𝕜 := ℝ) u (orthonormalDiagonal b a v) := by
  rw [← real_inner_comm u (orthonormalDiagonal b a v),
    orthonormalDiagonal_inner, orthonormalDiagonal_inner]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- Strictly positive diagonal coefficients give a strictly positive
quadratic form. Source: arXiv:2305.05465v6, §8.1, `eq: A`.
A nonzero vector has a nonzero coordinate in the orthonormal basis. -/
theorem orthonormalDiagonal_pos (b : OrthonormalBasis (Idx d) ℝ (EucSpace d))
    (a : Idx d → ℝ) (ha : ∀ i, 0 < a i) (u : EucSpace d) (hu : u ≠ 0) :
    0 < inner (𝕜 := ℝ) (orthonormalDiagonal b a u) u := by
  classical
  have hcoord : ∃ i : Idx d, inner (𝕜 := ℝ) (b i) u ≠ 0 := by
    by_contra h
    push Not at h
    apply hu
    simpa only [h, zero_smul, Finset.sum_const_zero] using (b.sum_repr' u).symm
  rw [orthonormalDiagonal_inner]
  apply Finset.sum_pos'
  · intro i _
    nlinarith [ha i, sq_nonneg (inner (𝕜 := ℝ) (b i) u)]
  · obtain ⟨i, hi⟩ := hcoord
    refine ⟨i, Finset.mem_univ i, ?_⟩
    have hp := mul_pos (ha i) (sq_pos_of_ne_zero hi)
    nlinarith

/-- Positive coefficients and a nonzero vector coexist in dimension one.
Source: arXiv:2305.05465v6, §8.1, `eq: A`. -/
example : (∀ i : Idx 1, 0 < (fun _ : Idx 1 => (1 : ℝ)) i) ∧
    (basePoint 0 : EucSpace 1) ≠ 0 := by
  exact ⟨fun _ => zero_lt_one, by simp [basePoint]⟩

/-- **The positive root of `QᵀK` exists.** Source: arXiv:2305.05465v6,
§8.1, `eq: A`. Diagonalizing the symmetric operator associated with the
positive bilinear form and taking positive scalar square roots verifies
all three parts of `IsAttentionRoot`, including the exact score identity. -/
theorem IsPosDefQK.exists_attentionRoot {Q K : ParamMatrix d}
    (hQK : IsPosDefQK Q K) : ∃ A : ParamMatrix d, IsAttentionRoot Q K A := by
  classical
  let T : EucSpace d →ₗ[ℝ] EucSpace d := (K.adjoint.comp Q).toLinearMap
  have hform (u v : EucSpace d) : inner (𝕜 := ℝ) (T u) v = inner (𝕜 := ℝ) (Q u) (K v) :=
    K.adjoint_inner_left v (Q u)
  have hT : T.IsSymmetric := by
    intro u v
    rw [hform, ← real_inner_comm u (T v), hform]
    exact hQK.1 u v
  have hn : Module.finrank ℝ (EucSpace d) = d := finrank_euclideanSpace_fin
  let b := hT.eigenvectorBasis hn
  let eigenvalue := hT.eigenvalues hn
  have heigen (i : Idx d) : T (b i) = eigenvalue i • b i := hT.apply_eigenvectorBasis hn i
  have hposEigenvalue (i : Idx d) : 0 < eigenvalue i := by
    have hbi : b i ≠ 0 := by
      intro hz
      have hone := b.inner_eq_one i
      rw [hz, inner_zero_left] at hone
      norm_num at hone
    have hpos := hQK.2 (b i) hbi
    rw [← hform, heigen, real_inner_smul_left, b.inner_eq_one, mul_one] at hpos
    exact hpos
  have hcoord (u : EucSpace d) (i : Idx d) :
      inner (𝕜 := ℝ) (b i) (T u) = eigenvalue i * inner (𝕜 := ℝ) (b i) u := by
    rw [← hT (b i) u, heigen, real_inner_smul_left]
  let a : Idx d → ℝ := fun i => Real.sqrt (eigenvalue i)
  let A := orthonormalDiagonal b a
  refine ⟨A, orthonormalDiagonal_symmetric b a,
    fun u hu => orthonormalDiagonal_pos b a
      (fun i => Real.sqrt_pos.mpr (hposEigenvalue i)) u hu, ?_⟩
  intro u v
  calc
    inner (𝕜 := ℝ) (Q u) (K v) = inner (𝕜 := ℝ) (T u) v := (hform u v).symm
    _ = ∑ i : Idx d, eigenvalue i * inner (𝕜 := ℝ) (b i) u * inner (𝕜 := ℝ) (b i) v := by
      rw [← b.sum_inner_mul_inner (T u) v]
      apply Finset.sum_congr rfl
      intro i _
      rw [← real_inner_comm (T u) (b i), hcoord]
    _ = inner (𝕜 := ℝ) (A u) (A v) := by
      rw [orthonormalDiagonal_inner]
      apply Finset.sum_congr rfl
      intro i _
      rw [inner_orthonormalDiagonal]
      change eigenvalue i * inner (𝕜 := ℝ) (b i) u * inner (𝕜 := ℝ) (b i) v =
        Real.sqrt (eigenvalue i) * inner (𝕜 := ℝ) (b i) u *
          (Real.sqrt (eigenvalue i) * inner (𝕜 := ℝ) (b i) v)
      nth_rw 1 [← Real.sq_sqrt (hposEigenvalue i).le]
      ring

/-- The original positivity hypothesis holds for identity query and key
maps. Source: arXiv:2305.05465v6, §8.1, `eq: A`. -/
example : IsPosDefQK (ContinuousLinearMap.id ℝ (EucSpace d))
    (ContinuousLinearMap.id ℝ (EucSpace d)) :=
  isPosDefQK_of_isAttentionRoot (isAttentionRoot_id d)

end Transformer.Clusters

/-
# Exact moments of the sparse attention kernel

For the law in `Section3_SparseHeadLaw`, the actual attention field is the
signed second coordinate of the uniform token average, along the first axis.
Its mean is zero. On two unit tokens the projected field has norm at most
one, and that bound is attained when both tokens are the second basis vector.
Thus the variance proxy is exactly `√p`, whereas the third moment norm at
that configuration is `p^(1/3)`.

Source: arXiv:2604.01978v1, §3.1, `eq:G_def`, `eq: defining.alpha`,
counterexample to `lem:Kernel_regularity`(2).
-/

import Transformer.Homogenized.Section3_SparseHeadLaw
import Mathlib.Analysis.SpecialFunctions.Pow.Real

open scoped BigOperators
open Real MeasureTheory

namespace Transformer.Homogenized

/-- The actual rank-one value map. Source: arXiv:2604.01978v1, §2.1,
`EQ:VELOCITY_FIELD_SELF_ATTENTION`, for the sparse head. -/
theorem valueMap_sparseMatrix (b : Fin 3) (y : EucSpace 2) :
    valueMap (sparseMatrix b, 0) y =
      (sparseSign b * y 1) • EuclideanSpace.single 0 (1 : ℝ) := by
  ext i
  fin_cases i <;>
    simp [valueMap, sparseMatrix, Matrix.toEuclideanLin, Matrix.mulVec,
      dotProduct, Fin.sum_univ_two]

/-- With `A = 0` the two-token softmax is uniform, at every temperature.
Source: arXiv:2604.01978v1, §2.1, `EQ:VELOCITY_FIELD_SELF_ATTENTION`. -/
theorem attnField_sparseMatrix (β : ℝ) (b : Fin 3)
    (x : Idx 2 → EucSpace 2) (z : EucSpace 2) :
    attnField β (sparseMatrix b, 0) x z =
      (sparseSign b * ((x 0 1 + x 1 1) / 2)) • EuclideanSpace.single 0 (1 : ℝ) := by
  have hw (y : EucSpace 2) : attnWeight β (sparseMatrix b, 0) z y = 1 := by
    simp [attnWeight, qkMap]
  simp only [attnField, hw, Fin.sum_univ_two, one_smul, valueMap_sparseMatrix]
  rw [← add_smul, smul_smul]
  congr 1
  ring

/-- The genuine mean field, including the zero atom, vanishes. Source:
arXiv:2604.01978v1, §2.2.1, the drift/fluctuation decomposition. -/
theorem meanField_sparseHeadLaw {p : ℝ} (hp : p ∈ Set.Icc 0 1) (β : ℝ)
    (x : Idx 2 → EucSpace 2) (z : EucSpace 2) :
    meanField β (sparseHeadLaw p) x z = 0 := by
  rw [meanField, sparseHeadLaw, integral_sparseMixture hp]
  simp only [attnField_sparseMatrix]
  simp [sparseSign]

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- The actual centered and projected kernel at each source atom. Source:
arXiv:2604.01978v1, §3.1, `eq:G_def`. -/
theorem Gfield_sparseHeadLaw {p : ℝ} (hp : p ∈ Set.Icc 0 1) (β : ℝ)
    (x : Idx 2 → EucSpace 2) (b : Fin 3) (i : Idx 2) :
    Gfield β (sparseHeadLaw p) x (sparseMatrix b, 0) i =
      sparseSign b • proj 2 (x i)
        (((x 0 1 + x 1 1) / 2) • EuclideanSpace.single 0 (1 : ℝ)) := by
  rw [Gfield, fluct, meanField_sparseHeadLaw hp, sub_zero, attnField_sparseMatrix]
  simp only [proj, inner_smul_right, ← smul_smul, smul_sub]

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- Every positive integer moment is the rare mass times the moment of the
positive atom. Source: arXiv:2604.01978v1, §3.1, auxiliary to
`lem:Kernel_regularity`(2). -/
theorem integral_norm_Gfield_sparse {p : ℝ} (hp : p ∈ Set.Icc 0 1)
    (β : ℝ) (x : Idx 2 → EucSpace 2) (i : Idx 2) (k : ℕ) :
    ∫ θ, ‖Gfield β (sparseHeadLaw p) x θ i‖ ^ (k + 1) ∂sparseHeadLaw p =
      p * ‖proj 2 (x i)
        (((x 0 1 + x 1 1) / 2) • EuclideanSpace.single 0 (1 : ℝ))‖ ^ (k + 1) := by
  change (∫ θ, ‖Gfield β (sparseHeadLaw p) x θ i‖ ^ (k + 1)
    ∂sparseMixture p (sparseMatrix 0, 0) (sparseMatrix 1, 0) (sparseMatrix 2, 0)) = _
  rw [integral_sparseMixture hp]
  simp [Gfield_sparseHeadLaw hp, sparseSign]
  ring

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- Unit tokens give a kernel of norm at most one, before multiplying by
the sparse sign. Source: arXiv:2604.01978v1, §3.1, `eq: defining.alpha`. -/
theorem norm_sparse_projection_le_one (x : Idx 2 → EucSpace 2)
    (hx : ∀ i, ‖x i‖ = 1) (i : Idx 2) :
    ‖proj 2 (x i)
      (((x 0 1 + x 1 1) / 2) • EuclideanSpace.single 0 (1 : ℝ))‖ ≤ 1 := by
  apply (norm_proj_le (hx i) _).trans
  simp only [norm_smul, PiLp.norm_single, Real.norm_eq_abs, abs_one, mul_one]
  have h₀ : |x 0 1| ≤ 1 := by simpa [hx 0] using PiLp.norm_apply_le (x 0) 1
  have h₁ : |x 1 1| ≤ 1 := by simpa [hx 1] using PiLp.norm_apply_le (x 1) 1
  rcases abs_le.mp h₀ with ⟨hl₀, hu₀⟩
  rcases abs_le.mp h₁ with ⟨hl₁, hu₁⟩
  rw [abs_le]
  constructor <;> linarith

example : ∀ _i : Idx 2, ‖(EuclideanSpace.single (1 : Fin 2) (1 : ℝ))‖ = 1 := by
  intro i; simp

/-- At two coincident second-axis tokens the projected positive atom has
norm exactly one. Source: arXiv:2604.01978v1, §3.1, `eq:G_def`. -/
theorem sparse_projection_unit :
    proj 2 (EuclideanSpace.single 1 (1 : ℝ))
      (EuclideanSpace.single 0 (1 : ℝ)) = EuclideanSpace.single 0 (1 : ℝ) := by
  norm_num [proj, EuclideanSpace.inner_single_left]

/-- The exact variance proxy, with its maximum attained by two unit tokens.
Source: arXiv:2604.01978v1, §3.1, `eq: defining.alpha`. -/
theorem isVarianceProxy_sparse {q : ℝ} (hq : q ∈ Set.Icc 0 1) (β : ℝ) :
    IsVarianceProxy 2 2 β (sparseHeadLaw (q ^ 6)) (q ^ 3) := by
  have hp : q ^ 6 ∈ Set.Icc 0 1 := ⟨by positivity, pow_le_one₀ hq.1 hq.2⟩
  refine ⟨pow_nonneg hq.1 _, ⟨?_, ?_⟩⟩
  · refine ⟨fun _ => EuclideanSpace.single 1 1, fun _ => by simp, 0, ?_⟩
    rw [integral_norm_Gfield_sparse hp β _ _ 1]
    norm_num [proj, EuclideanSpace.inner_single_left]
    ring
  · rintro v ⟨x, hx, i, rfl⟩
    rw [integral_norm_Gfield_sparse hp β x i 1]
    have hnorm := norm_sparse_projection_le_one x hx i
    have hs : ‖proj 2 (x i)
        (((x 0 1 + x 1 1) / 2) • EuclideanSpace.single 0 (1 : ℝ))‖ ^ 2 ≤ 1 := by
      nlinarith [norm_nonneg (proj 2 (x i)
        (((x 0 1 + x 1 1) / 2) • EuclideanSpace.single 0 (1 : ℝ)))]
    calc
      _ ≤ q ^ 6 * 1 := mul_le_mul_of_nonneg_left hs (by positivity)
      _ = (q ^ 3) ^ 2 := by ring

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- The exact third moment norm at two unit tokens is `q²`, while the
variance proxy is `q³`. Source: arXiv:2604.01978v1, §3.1,
counterexample to the uniform bound in `lem:Kernel_regularity`(2). -/
theorem sparse_third_moment {q : ℝ} (hq : q ∈ Set.Icc 0 1) (β : ℝ) :
    (∫ θ, ‖Gfield β (sparseHeadLaw (q ^ 6))
      (fun _ : Idx 2 => EuclideanSpace.single 1 1) θ 0‖ ^ 3 ∂sparseHeadLaw (q ^ 6)) ^
        ((1 : ℝ) / 3) = q ^ 2 := by
  have hp : q ^ 6 ∈ Set.Icc 0 1 := ⟨by positivity, pow_le_one₀ hq.1 hq.2⟩
  have hm : (∫ θ, ‖Gfield β (sparseHeadLaw (q ^ 6))
      (fun _ : Idx 2 => EuclideanSpace.single 1 1) θ 0‖ ^ 3 ∂sparseHeadLaw (q ^ 6)) =
      q ^ 6 := by
    rw [integral_norm_Gfield_sparse hp β _ _ 2]
    norm_num [proj, EuclideanSpace.inner_single_left]
  rw [hm]
  calc
    (q ^ 6) ^ ((1 : ℝ) / 3) = q ^ ((6 : ℝ) * (1 / 3)) := by
      simpa only [Real.rpow_ofNat] using
        (Real.rpow_mul hq.1 (6 : ℝ) (1 / 3)).symm
    _ = q ^ 2 := by norm_num

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

end Transformer.Homogenized

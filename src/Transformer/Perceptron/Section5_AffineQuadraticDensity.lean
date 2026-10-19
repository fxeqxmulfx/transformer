module

public import Transformer.Perceptron.Section5_PolynomialSupport

/-!
# Original density inversion with linear and quadratic terms

ArXiv:2601.21366v2, §5.1, `lem: quadpol` (i), degrees at most two,
and §3.1, `rem: ext` (i), the biased perceptron. An original biased
quadratic branch has constant, linear and quadratic terms. The original
uniform kernel coefficients at all three degrees have been proved
nonzero at positive temperature.

The quadratic coefficients are divided by their original degree-two
coefficient and their actual constant contribution is corrected. The
original linear vector is divided by the original degree-one coefficient.
The resulting actual polynomial candidate has the original transform.
The proved finite-measure identification then gives its pointwise
nonnegativity and the original probability as its actual density law.

The actual degree bound is proved with actual polynomial witnesses.
The proved analytic-density support theorem gives full original support
for an original affine-quadratic transform. No inverse, density,
coefficient identity, positivity or support conclusion is an input.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped BigOperators ENNReal

namespace Transformer.Perceptron

/-- A constant, an actual original linear function and actual weighted
neuron-coordinate squares have original degree at most two.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i), and §3.1,
`rem: ext` (i). The actual original coordinate data determine the witness. -/
theorem isPolyOfDegreeLE_affine_add_weighted_squares {d : ℕ} {ι : Type*}
    (S : Finset ι) (A : ℝ) (v : EucSpace d) (q : ι → ℝ) (a : ι → EucSpace d) :
    IsPolyOfDegreeLE 2 (fun y : SSphere d =>
      (A + ∑ i ∈ S, q i * inner (𝕜 := ℝ) (a i) (y : EucSpace d) ^ 2) +
        inner (𝕜 := ℝ) v (y : EucSpace d)) := by
  apply isPolyOfDegreeLE_add (isPolyOfDegreeLE_const_add_weighted_squares S A q a)
  exact isPolyOfDegreeLE_mono (isPolyOfDegreeLE_inner v) (by omega)

/-- An original affine-quadratic transform gives an actual nonnegative
quadratic density against every actual original uniform law.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i), degree two,
and §3.1, `rem: ext` (i). The original three kernel coefficients are
proved nonzero, and the actual candidate's transform is computed below. -/
theorem exists_quadratic_density_of_affine_add_squares_transform {d : ℕ} {ι : Type*}
    (hd : 2 ≤ d) (S : Finset ι) (β : ℝ) (hβ : 0 < β) (μ : Perspective.ProbSphere d)
    (C : ℝ) (v : EucSpace d) (q : ι → ℝ) (a : ι → EucSpace d)
    (hquad : ∀ x : SSphere d, attentionTransform β μ x =
      C + inner (𝕜 := ℝ) v (x : EucSpace d) +
        ∑ i ∈ S, q i * inner (𝕜 := ℝ) (a i) (x : EucSpace d) ^ 2)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    ∃ g : SSphere d → ℝ, IsPolyOfDegreeLE 2 g ∧ (∀ y, 0 ≤ g y) ∧
      (μ : Measure (SSphere d)) = ν.withDensity (fun y => ENNReal.ofReal (g y)) := by
  let _ : IsProbabilityMeasure ν := hν.1
  obtain ⟨c₀, p, c₂, hc₀, hc₂, hconst, hq⟩ := exists_uniform_constant_quadratic_coefficients hd β hβ ν hν
  obtain ⟨c₁, hc₁, hlin⟩ := exists_uniform_linear_kernel_coefficient β hβ ν hν
  let r := fun i => q i / c₂
  let A := (C - p * ∑ i ∈ S, r i * ‖a i‖ ^ 2) / c₀
  let Q := fun y : SSphere d => A + ∑ i ∈ S, r i * inner (𝕜 := ℝ) (a i) (y : EucSpace d) ^ 2
  let L := fun y : SSphere d => inner (𝕜 := ℝ) (c₁⁻¹ • v) (y : EucSpace d)
  let g := fun y => Q y + L y
  have hg : IsPolyOfDegreeLE 2 g := isPolyOfDegreeLE_affine_add_weighted_squares S A (c₁⁻¹ • v) r a
  have hQ : Continuous Q := continuous_of_isPolyOfDegreeLE (isPolyOfDegreeLE_const_add_weighted_squares S A r a)
  have hL : Continuous L := continuous_const.inner continuous_subtype_val
  have hT (x : SSphere d) : attentionTransform β μ x =
      ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) * g y ∂ν := by
    have hK : Continuous (fun y : SSphere d =>
        exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))) :=
      Real.continuous_exp.comp (continuous_const.mul (continuous_const.inner continuous_subtype_val))
    change attentionTransform β μ x =
      ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) * (Q y + L y) ∂ν
    simp_rw [mul_add]
    rw [integral_add
      (f := fun y : SSphere d => exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) * Q y)
      (g := fun y : SSphere d => exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) * L y)
      (Perspective.integrable_of_continuous_compact (hK.mul hQ) ν)
      (Perspective.integrable_of_continuous_compact (hK.mul hL) ν)]
    rw [integral_exp_inner_mul_const_add_squares S β ν A r a x, hlin, hconst x, hquad x]
    simp_rw [hq]
    have hl : c₁ * inner (𝕜 := ℝ) (c₁⁻¹ • v) (x : EucSpace d) = inner (𝕜 := ℝ) v (x : EucSpace d) := by
      rw [real_inner_smul_left, ← mul_assoc, mul_inv_cancel₀ hc₁, one_mul]
    rw [hl]
    have hsum : (∑ i ∈ S, r i * (p * ‖a i‖ ^ 2 + c₂ * inner (𝕜 := ℝ) (x : EucSpace d) (a i) ^ 2)) =
        p * ∑ i ∈ S, r i * ‖a i‖ ^ 2 +
        ∑ i ∈ S, q i * inner (𝕜 := ℝ) (a i) (x : EucSpace d) ^ 2 := by
      rw [Finset.mul_sum, ← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro i _
      dsimp [r]
      rw [real_inner_comm (a i) (x : EucSpace d)]
      field_simp
    rw [hsum]
    dsimp [A]
    field_simp
    ring
  obtain ⟨hnn, heq⟩ := nonneg_and_eq_withDensity_of_uniform_transform_eq β hβ
    (μ : Measure (SSphere d)) ν hν g (continuous_of_isPolyOfDegreeLE hg) hT
  exact ⟨g, hg, hnn, heq⟩

/-- The original probability with an actual affine-quadratic transform
has full original support at positive temperature.
Source: arXiv:2601.21366v2, §5.3, the full-support step after `eq:global-Id`,
with §3.1, `rem: ext` (i). Its actual density and nonnegativity are proved,
then the actual polynomial zero-set argument gives full support. -/
theorem support_eq_univ_of_affine_add_squares_transform {d : ℕ} {ι : Type*}
    (hd : 2 ≤ d) (S : Finset ι) (β : ℝ) (hβ : 0 < β) (μ : Perspective.ProbSphere d)
    (C : ℝ) (v : EucSpace d) (q : ι → ℝ) (a : ι → EucSpace d)
    (hquad : ∀ x : SSphere d, attentionTransform β μ x =
      C + inner (𝕜 := ℝ) v (x : EucSpace d) +
        ∑ i ∈ S, q i * inner (𝕜 := ℝ) (a i) (x : EucSpace d) ^ 2) :
    (μ : Measure (SSphere d)).support = univ := by
  have hν := Metastability.isUniformOn_uniformLaw (d := d) (by omega)
  obtain ⟨g, hg, hnn, heq⟩ := exists_quadratic_density_of_affine_add_squares_transform
    hd S β hβ μ C v q a hquad (MeanField.uniformLaw d) hν
  exact support_eq_univ_of_nonneg_polynomial_density hd μ (MeanField.uniformLaw d) hν g hg hnn heq

/-- The original affine-quadratic-transform probability is not finitely
atomic in dimension at least two. Source: arXiv:2601.21366v2, §5.1,
`lem: quadpol` (i), and §3.1, `rem: ext` (i).
The proved actual polynomial density gives the actual uniform absolute
continuity that excludes any original finite Dirac presentation. -/
theorem not_isFinitelyAtomic_of_affine_add_squares_transform {d : ℕ} {ι : Type*}
    (hd : 2 ≤ d) (S : Finset ι) (β : ℝ) (hβ : 0 < β) (μ : Perspective.ProbSphere d)
    (C : ℝ) (v : EucSpace d) (q : ι → ℝ) (a : ι → EucSpace d)
    (hquad : ∀ x : SSphere d, attentionTransform β μ x =
      C + inner (𝕜 := ℝ) v (x : EucSpace d) +
        ∑ i ∈ S, q i * inner (𝕜 := ℝ) (a i) (x : EucSpace d) ^ 2) :
    ¬ IsFinitelyAtomic μ := by
  have hν := Metastability.isUniformOn_uniformLaw (d := d) (by omega)
  obtain ⟨g, -, -, heq⟩ := exists_quadratic_density_of_affine_add_squares_transform
    hd S β hβ μ C v q a hquad (MeanField.uniformLaw d) hν
  exact not_isFinitelyAtomic_of_uniform_density hd μ (MeanField.uniformLaw d) hν g heq

/-- All inverse and full-support inputs hold jointly for the original
uniform circle probability, its actual constant transform, and actual
zero linear and quadratic coefficients.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i), degree two. -/
example : ∃ μ : Perspective.ProbSphere 2, (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    Metastability.IsUniformOn 2 (μ : Measure (SSphere 2)) ∧
    ∀ x : SSphere 2, attentionTransform 1 μ x = attentionTransform 1 μ (basePoint 1) +
      inner (𝕜 := ℝ) (0 : EucSpace 2) (x : EucSpace 2) +
      ∑ i ∈ (Finset.univ : Finset (Fin 2)), (0 : ℝ) * inner (𝕜 := ℝ)
        (EuclideanSpace.single i (1 : ℝ)) (x : EucSpace 2) ^ 2 := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  exact ⟨μ, le_rfl, one_pos, hν, fun x => by
    simpa using attentionTransform_eq_of_uniform 1 μ hν x (basePoint 1)⟩

end Transformer.Perceptron

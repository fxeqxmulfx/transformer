module

public import Transformer.Perceptron.Section5_QuadraticCoefficients
public import Transformer.Perceptron.Section5_QuadraticPolynomials

/-!
# Actual quadratic density from the original quadratic transform

ArXiv:2601.21366v2, §5.1, `lem: quadpol` (i), degree two,
and §5.3, `rem:genericity-vacuity`. The actual original kernel
coefficients at degrees zero and two were proved nonzero. Dividing the
original quadratic coefficients by the latter and correcting the
constant contribution gives an actual finite quadratic candidate.

Every original kernel integral of that candidate is computed here.
Its equality with the original probability's transform then proves
that the candidate is nonnegative at every actual sphere point and
that the original probability is its actual `withDensity` law.
Its degree bound comes from an actual multivariate polynomial witness.

No density, positivity, coefficient identity or inverse is assumed.
This proves the quadratic consequence used by the source's linear
stationary-measure example, with the original probability and every
original quadratic test vector and coefficient retained.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped BigOperators ENNReal

namespace Transformer.Perceptron

/-- Every original probability with an actual density against the actual
uniform law is not finitely atomic in dimension at least two.
Source: arXiv:2601.21366v2, §5.3, `rem:genericity-vacuity`.
Absolute continuity transfers the proved uniform nullness of its actual
finite support, contradicting its original total probability mass. -/
theorem not_isFinitelyAtomic_of_uniform_density {d : ℕ} (hd : 2 ≤ d)
    (μ : Perspective.ProbSphere d) (ν : Measure (SSphere d))
    (hν : Metastability.IsUniformOn d ν) (g : SSphere d → ℝ)
    (heq : (μ : Measure (SSphere d)) = ν.withDensity (fun y => ENNReal.ofReal (g y))) :
    ¬ IsFinitelyAtomic μ := by
  intro hfinite
  have hz := measure_finite_support_eq_zero_of_uniform hd (μ : Measure (SSphere d))
    (finite_support_of_finitely_atomic μ hfinite) ν hν
  have habs : (μ : Measure (SSphere d)) ≪ ν := by
    rw [heq]
    exact withDensity_absolutelyContinuous ν _
  have hs := habs hz
  have hmass := measure_union_le (μ := (μ : Measure (SSphere d)))
    (μ : Measure (SSphere d)).support (μ : Measure (SSphere d)).supportᶜ
  simp only [union_compl_self, measure_univ, hs, Measure.measure_compl_support, add_zero] at hmass
  exact one_ne_zero (le_antisymm hmass bot_le)

/-- The actual finite quadratic candidate has its original kernel
integral term by term. Source: arXiv:2601.21366v2, §5.1,
`eq:FH-basic`, degrees zero and two, and `lem: quadpol` (i).
Every actual term is continuous and integrable under the finite law. -/
theorem integral_exp_inner_mul_const_add_squares {d : ℕ} {ι : Type*}
    (S : Finset ι) (β : ℝ) (ν : Measure (SSphere d)) [IsFiniteMeasure ν]
    (A : ℝ) (q : ι → ℝ) (a : ι → EucSpace d) (x : SSphere d) :
    (∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) *
      (A + ∑ i ∈ S, q i * inner (𝕜 := ℝ) (a i) (y : EucSpace d) ^ 2) ∂ν) =
      (∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) ∂ν) * A +
        ∑ i ∈ S, q i * secondMomentKernel β ν (x : EucSpace d) (a i) (a i) := by
  let K := fun y : SSphere d => exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))
  have hK : Continuous K := Real.continuous_exp.comp
    (continuous_const.mul (continuous_const.inner continuous_subtype_val))
  have hterm (i : ι) : Continuous (fun y : SSphere d =>
      K y * (q i * inner (𝕜 := ℝ) (a i) (y : EucSpace d) ^ 2)) :=
    hK.mul (continuous_const.mul ((continuous_const.inner continuous_subtype_val).pow 2))
  simp_rw [mul_add, Finset.mul_sum]
  change (∫ y, K y * A + ∑ i ∈ S, K y * (q i * inner (𝕜 := ℝ) (a i) (y : EucSpace d) ^ 2) ∂ν) = _
  rw [integral_add (f := fun y => K y * A)
    (g := fun y => ∑ i ∈ S, K y * (q i * inner (𝕜 := ℝ) (a i) (y : EucSpace d) ^ 2))
    (Perspective.integrable_of_continuous_compact (hK.mul continuous_const) ν)
    (integrable_finsetSum S fun i _ => Perspective.integrable_of_continuous_compact (hterm i) ν),
    integral_mul_const, integral_finsetSum S (fun i _ => Perspective.integrable_of_continuous_compact (hterm i) ν)]
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  rw [secondMomentKernel_eq_integral_sq, ← integral_const_mul]
  apply integral_congr_ae
  exact ae_of_all ν fun y => by dsimp; ring

/-- An original probability with a constant-plus-quadratic transform has
an actual nonnegative quadratic density against every actual uniform law.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i), degree two,
and §5.3, `rem:genericity-vacuity`. The candidate is constructed from
the proved original coefficients, then positivity and measure equality
are proved by finite-measure injectivity rather than supplied as inputs. -/
theorem exists_quadratic_density_of_const_add_squares_transform {d : ℕ} {ι : Type*}
    (hd : 2 ≤ d) (S : Finset ι) (β : ℝ) (hβ : 0 < β) (μ : Perspective.ProbSphere d)
    (C : ℝ) (q : ι → ℝ) (a : ι → EucSpace d)
    (hquad : ∀ x : SSphere d, attentionTransform β μ x =
      C + ∑ i ∈ S, q i * inner (𝕜 := ℝ) (a i) (x : EucSpace d) ^ 2)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    ∃ g : SSphere d → ℝ, IsPolyOfDegreeLE 2 g ∧ (∀ y, 0 ≤ g y) ∧
      (μ : Measure (SSphere d)) = ν.withDensity (fun y => ENNReal.ofReal (g y)) := by
  let _ : IsProbabilityMeasure ν := hν.1
  obtain ⟨c₀, p, c, hc₀, hc, hconst, hq⟩ := exists_uniform_constant_quadratic_coefficients hd β hβ ν hν
  let r := fun i => q i / c
  let A := (C - p * ∑ i ∈ S, r i * ‖a i‖ ^ 2) / c₀
  let g := fun y : SSphere d => A + ∑ i ∈ S, r i * inner (𝕜 := ℝ) (a i) (y : EucSpace d) ^ 2
  have hg : IsPolyOfDegreeLE 2 g := isPolyOfDegreeLE_const_add_weighted_squares S A r a
  have hT (x : SSphere d) : attentionTransform β μ x =
      ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) * g y ∂ν := by
    rw [hquad x, integral_exp_inner_mul_const_add_squares S β ν A r a x, hconst x]
    simp_rw [hq]
    have hsum : (∑ i ∈ S, r i * (p * ‖a i‖ ^ 2 + c * inner (𝕜 := ℝ) (x : EucSpace d) (a i) ^ 2)) =
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

/-- An original probability with a constant-plus-quadratic original
transform is not finitely atomic. Source: arXiv:2601.21366v2, §5.3,
`rem:genericity-vacuity`. The actual uniform law is constructed, the
actual quadratic density is proved, and absolute continuity excludes
the actual finite Dirac predicate without assuming full support. -/
theorem not_isFinitelyAtomic_of_const_add_squares_transform {d : ℕ} {ι : Type*}
    (hd : 2 ≤ d) (S : Finset ι) (β : ℝ) (hβ : 0 < β) (μ : Perspective.ProbSphere d)
    (C : ℝ) (q : ι → ℝ) (a : ι → EucSpace d)
    (hquad : ∀ x : SSphere d, attentionTransform β μ x =
      C + ∑ i ∈ S, q i * inner (𝕜 := ℝ) (a i) (x : EucSpace d) ^ 2) :
    ¬ IsFinitelyAtomic μ := by
  have hν := Metastability.isUniformOn_uniformLaw (d := d) (by omega)
  obtain ⟨g, -, -, heq⟩ := exists_quadratic_density_of_const_add_squares_transform
    hd S β hβ μ C q a hquad (MeanField.uniformLaw d) hν
  exact not_isFinitelyAtomic_of_uniform_density hd μ (MeanField.uniformLaw d) hν g heq

/-- All quadratic-transform inputs hold jointly for the genuine original
uniform circle probability and its actual constant transform.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i), degree two.
The zero original quadratic weights retain their actual coordinate vectors. -/
example : ∃ μ : Perspective.ProbSphere 2, (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    Metastability.IsUniformOn 2 (μ : Measure (SSphere 2)) ∧
    (∀ x : SSphere 2, attentionTransform 1 μ x = attentionTransform 1 μ (basePoint 1) +
      ∑ i ∈ (Finset.univ : Finset (Fin 2)), (0 : ℝ) *
        inner (𝕜 := ℝ) (EuclideanSpace.single i (1 : ℝ)) (x : EucSpace 2) ^ 2) ∧
    (μ : Measure (SSphere 2)) = (μ : Measure (SSphere 2)).withDensity (fun _ => ENNReal.ofReal 1) := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  exact ⟨μ, le_rfl, one_pos, hν, (fun x => by
    simpa using attentionTransform_eq_of_uniform 1 μ hν x (basePoint 1)), by simp⟩

end Transformer.Perceptron

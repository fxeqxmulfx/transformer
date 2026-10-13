/-
# The original uniform exponential kernel on linear functions

ArXiv:2601.21366v2, §5.1, `eq:FH-basic` and `lem: quadpol` (i).
Orthogonal equivariance of the actual ambient integral makes it radial
on the actual unit sphere. The actual orthogonal map exchanging two unit
points shows that its radial coefficient is constant there.

Passing an actual linear functional through that vector integral proves
that every original linear coordinate function has the same original
kernel coefficient. At positive temperature this coefficient is nonzero:
otherwise the proved signed-density injectivity would make a nonzero
unit coordinate vanish at its own actual unit point.

The same computation also keeps the degree-zero coefficient and proves
the original transform identity for every constant plus linear function.
No eigenvalue formula, harmonic expansion or inverse is assumed. The
coefficient is produced from the original integral and original uniform
law, rather than placed into the density definition as an unproved input.
-/

import Transformer.Perceptron.Section5_DensityIdentification

open Real MeasureTheory Set
open scoped ENNReal

namespace Transformer.Perceptron

/-- The actual uniform ambient exponential integral has one constant
radial coefficient on the actual sphere. Source: arXiv:2601.21366v2,
§5.1, `eq:FH-basic`, degree one. Its radiality and constancy are proved
from the original projected field and orthogonal equivariance. -/
theorem exists_uniform_ambient_radial_coefficient {d : ℕ} (β : ℝ)
    (μ : Perspective.ProbSphere d) (hμ : Metastability.IsUniformOn d (μ : Measure (SSphere d))) :
    ∃ c : ℝ, ∀ x : SSphere d,
      ambientAttentionGradMap (β • LinearMap.id) μ (x : EucSpace d) = c • (x : EucSpace d) := by
  let G := ambientAttentionGradMap (β • LinearMap.id) μ
  let x₀ : SSphere d := Classical.choice (nonempty_of_isProbabilityMeasure (μ : Measure (SSphere d)))
  have hr (x : SSphere d) : G (x : EucSpace d) =
      inner (𝕜 := ℝ) (x : EucSpace d) (G (x : EucSpace d)) • (x : EucSpace d) := by
    have hp := attentionGradMap_zero_of_uniform β μ hμ x
    rw [attentionGradMap_eq_proj_ambientGradMap] at hp
    exact sub_eq_zero.mp hp
  refine ⟨inner (𝕜 := ℝ) (x₀ : EucSpace d) (G (x₀ : EucSpace d)), fun x => ?_⟩
  let U := (ℝ ∙ ((x₀ : EucSpace d) - (x : EucSpace d)))ᗮ.reflection
  have hU : U (x₀ : EucSpace d) = (x : EucSpace d) :=
    Submodule.reflection_sub ((norm_eq_of_mem_sphere x₀).trans (norm_eq_of_mem_sphere x).symm)
  have hG : G (x : EucSpace d) = U (G (x₀ : EucSpace d)) := by
    simpa only [hU] using ambientAttentionGradMap_isometry_of_uniform β μ hμ (x₀ : EucSpace d) U
  change G (x : EucSpace d) = _
  rw [hr]
  congr 1
  rw [hG, ← hU]
  exact U.inner_map_map _ _

/-- The actual original kernel acts by a common nonzero coefficient on
all original linear functions at positive temperature.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`, degree one.
Nonvanishing is proved by signed-density injectivity, using an actual
unit coordinate that equals one at its actual unit base point. -/
theorem exists_uniform_linear_kernel_coefficient {d : ℕ} (β : ℝ) (hβ : 0 < β)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    ∃ c : ℝ, c ≠ 0 ∧ ∀ (b : EucSpace d) (x : SSphere d),
      (∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) *
        inner (𝕜 := ℝ) b (y : EucSpace d) ∂ν) = c * inner (𝕜 := ℝ) b (x : EucSpace d) := by
  let _ : IsProbabilityMeasure ν := hν.1
  let μ : Perspective.ProbSphere d := ⟨ν, hν.1⟩
  obtain ⟨a, ha⟩ := exists_uniform_ambient_radial_coefficient β μ hν
  have heval (b : EucSpace d) (x : SSphere d) :
      (∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) *
        inner (𝕜 := ℝ) b (y : EucSpace d) ∂ν) = (a / β) * inner (𝕜 := ℝ) b (x : EucSpace d) := by
    have hi := (innerSL ℝ b).integral_comp_comm
      (integrable_ambientAttentionKernel (β • LinearMap.id) μ (x : EucSpace d))
    change (∫ y, inner (𝕜 := ℝ) b
      (exp (inner (𝕜 := ℝ) (x : EucSpace d) (β • (y : EucSpace d))) •
        (β • (y : EucSpace d))) ∂ν) =
      inner (𝕜 := ℝ) b (ambientAttentionGradMap (β • LinearMap.id) μ (x : EucSpace d)) at hi
    have hmul : β * (∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) *
        inner (𝕜 := ℝ) b (y : EucSpace d) ∂ν) = a * inner (𝕜 := ℝ) b (x : EucSpace d) := by
      rw [← integral_const_mul, ← real_inner_smul_right, ← ha]
      rw [← hi]
      apply integral_congr_ae
      exact ae_of_all ν fun y => by
        change β * (exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) *
          inner (𝕜 := ℝ) b (y : EucSpace d)) = inner (𝕜 := ℝ) b
            (exp (inner (𝕜 := ℝ) (x : EucSpace d) (β • (y : EucSpace d))) •
              (β • (y : EucSpace d)))
        rw [real_inner_smul_right, real_inner_smul_right, real_inner_smul_right]
        ring
    apply mul_left_cancel₀ hβ.ne'
    calc
      _ = a * inner (𝕜 := ℝ) b (x : EucSpace d) := hmul
      _ = β * ((a / β) * inner (𝕜 := ℝ) b (x : EucSpace d)) := by field_simp
  refine ⟨a / β, ?_, heval⟩
  intro hc
  let x₀ : SSphere d := Classical.choice (nonempty_of_isProbabilityMeasure ν)
  have hz := eq_zero_of_integral_exp_inner_mul_eq_zero_uniform β hβ ν hν
    (fun y => inner (𝕜 := ℝ) (x₀ : EucSpace d) (y : EucSpace d))
    (continuous_const.inner continuous_subtype_val) fun x => by rw [heval, hc, zero_mul]
  have h := congrFun hz x₀
  change inner (𝕜 := ℝ) (x₀ : EucSpace d) (x₀ : EucSpace d) = 0 at h
  rw [real_inner_self_eq_norm_sq, norm_eq_of_mem_sphere] at h
  norm_num at h

/-- The original kernel preserves the original constant and linear terms.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`, degrees zero and one.
The actual two coefficients are simultaneously nonzero at positive
temperature, and the identity holds for every original affine function. -/
theorem exists_uniform_affine_kernel_coefficients {d : ℕ} (β : ℝ) (hβ : 0 < β)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    ∃ c₀ c₁ : ℝ, c₀ ≠ 0 ∧ c₁ ≠ 0 ∧ ∀ (A : ℝ) (b : EucSpace d) (x : SSphere d),
      (∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) *
        (A + inner (𝕜 := ℝ) b (y : EucSpace d)) ∂ν) =
        c₀ * A + c₁ * inner (𝕜 := ℝ) b (x : EucSpace d) := by
  let _ : IsProbabilityMeasure ν := hν.1
  let μ : Perspective.ProbSphere d := ⟨ν, hν.1⟩
  let x₀ : SSphere d := Classical.choice (nonempty_of_isProbabilityMeasure ν)
  obtain ⟨c₁, hc₁, hlin⟩ := exists_uniform_linear_kernel_coefficient β hβ ν hν
  have hmass : ν ≠ 0 := by
    intro hzero
    have h := congrArg (fun ρ : Measure (SSphere d) => ρ univ) hzero
    simp at h
  refine ⟨attentionTransform β μ x₀, c₁,
    (integral_exp_inner_pos_of_finite_nonzero β ν hmass x₀).ne', hc₁, fun A b x => ?_⟩
  have hK : Continuous (fun y : SSphere d =>
      exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))) :=
    Real.continuous_exp.comp (continuous_const.mul
      (continuous_const.inner continuous_subtype_val))
  simp_rw [mul_add]
  rw [integral_add
    (f := fun y : SSphere d => exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) * A)
    (g := fun y : SSphere d => exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) *
      inner (𝕜 := ℝ) b (y : EucSpace d))
    (Perspective.integrable_of_continuous_compact (hK.mul continuous_const) ν)
    (Perspective.integrable_of_continuous_compact
      (hK.mul (continuous_const.inner continuous_subtype_val)) ν), integral_mul_const, hlin]
  change attentionTransform β μ x * A + _ = _
  rw [attentionTransform_eq_of_uniform β μ hν x x₀]

/-- The actual original circle law satisfies every coefficient input
jointly at positive temperature. Source: arXiv:2601.21366v2, §5.1,
`eq:FH-basic`. Both the original probability and unit coordinate exist. -/
example : ∃ μ : Perspective.ProbSphere 2, (0 : ℝ) < 1 ∧
    Metastability.IsUniformOn 2 (μ : Measure (SSphere 2)) ∧
    Continuous (fun y : SSphere 2 => inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2) (y : EucSpace 2)) := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  exact ⟨⟨MeanField.uniformLaw 2, hν.1⟩, one_pos, hν,
    continuous_const.inner continuous_subtype_val⟩

end Transformer.Perceptron

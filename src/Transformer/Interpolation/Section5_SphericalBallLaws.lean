/-
# Diffuse small-cap probabilities with null ball boundaries

ArXiv:2411.04551v3, §5, `cl: balls`. The source's actual measures
are absolutely continuous and have small support. Here such inputs
are constructed in every ambient dimension at least two, by restricting
the genuine uniform probability to a positive-radius closed cap and
normalizing its positive mass.

Every ambient metric sphere on the unit sphere is an affine
inner-product level set. Its uniform measure is zero by the proved
analytic spherical uniqueness theorem: that affine function cannot
vanish both at the center and its antipode. Restriction and rescaling
preserve these null boundaries. The probability, its full-measure
small carrier, and the absence of atoms are all actual conclusions.
The radius can be arbitrarily small and is chosen before the law.
Thus these examples also apply after the geometric scale in the
equal-mass construction has been fixed.
-/

import Transformer.Interpolation.AtomlessSphere
import Transformer.Perceptron.Section6_SphereUniqueness

open Real MeasureTheory Metric Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Every actual metric sphere has zero original uniform mass in
dimension at least two. Source: arXiv:2411.04551v3, §5, `cl: balls`,
the null boundaries used for continuity. The analytic uniqueness input
is the proved `Perceptron.measure_sphere_zeros_eq_zero_of_analytic_nonzero`. -/
theorem uniform_sphere_boundary_null (hd : 2 ≤ d)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (y : SSphere d) (r : ℝ) : ν (sphere y r) = 0 := by
  let f : EucSpace d → ℝ := fun z =>
    inner (𝕜 := ℝ) (y : EucSpace d) z - (1 - r ^ 2 / 2)
  have hf : AnalyticOnNhd ℝ f univ := fun z _ =>
    ((innerSL ℝ (y : EucSpace d)).analyticAt z).sub analyticAt_const
  have hyy : inner (𝕜 := ℝ) (y : EucSpace d) (y : EucSpace d) = 1 := by simp
  have hne : ∃ x : SSphere d, f (x : EucSpace d) ≠ 0 := by
    by_cases hr : r = 0
    · refine ⟨antipode d y, ?_⟩
      change inner (𝕜 := ℝ) (y : EucSpace d) (-(y : EucSpace d)) - (1 - r ^ 2 / 2) ≠ 0
      rw [inner_neg_right, hyy, hr]
      norm_num
    · refine ⟨y, ?_⟩
      dsimp [f]
      rw [hyy]
      nlinarith [sq_pos_of_ne_zero hr]
  have hz := Perceptron.measure_sphere_zeros_eq_zero_of_analytic_nonzero hd f hf ν hν hne
  apply measure_mono_null _ hz
  intro x hx
  have hdist : dist x y = r := hx
  have hs := norm_sub_sq_real (x : EucSpace d) (y : EucSpace d)
  rw [norm_eq_of_mem_sphere, norm_eq_of_mem_sphere] at hs
  have heq : ‖(x : EucSpace d) - (y : EucSpace d)‖ = r := hdist
  rw [heq] at hs
  change f (x : EucSpace d) = 0
  dsimp [f]
  rw [real_inner_comm]
  nlinarith

/-- The null-boundary inputs hold jointly for the actual uniform
circle law and every real radius and center. Source:
arXiv:2411.04551v3, §5, the diffuse ball-mass function in `cl: balls`. -/
example : (2 : ℕ) ≤ 2 ∧ Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) ∧
    ∀ y : SSphere 2, ∀ r : ℝ, MeanField.uniformLaw 2 (sphere y r) = 0 := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  exact ⟨le_rfl, hν, uniform_sphere_boundary_null le_rfl _ hν⟩

/-- There is an actual diffuse probability in every positive-radius
closed cap, with zero mass on every actual ball boundary.
Source: arXiv:2411.04551v3, §5, `cl: balls`, the smallness and
absolute-continuity inputs. The constructed law is the normalized
restriction of the original uniform law; it is not an assumed measure. -/
theorem exists_small_cap_ball_diffuse (hd : 2 ≤ d) (p : SSphere d)
    (δ : ℝ) (hδ : 0 < δ) :
    ∃ μ : Perspective.ProbSphere d,
      (∀ y : SSphere d, ∀ r : ℝ, (μ : Measure (SSphere d)) (sphere y r) = 0) ∧
      (μ : Measure (SSphere d)) (closedBall p δ)ᶜ = 0 := by
  let σ := MeanField.uniformLaw d
  have hσ : Metastability.IsUniformOn d σ := Metastability.isUniformOn_uniformLaw (by omega)
  have hprob : IsProbabilityMeasure σ := hσ.1
  let S := closedBall p δ
  have hpos : 0 < σ S := by
    have hb := Causal.uniformSphere_pos_of_isOpen d (Metric.isOpen_ball (x := p))
      (Metric.nonempty_ball.mpr hδ)
    rw [← MeanField.uniformLaw_apply] at hb
    exact hb.trans_le (measure_mono ball_subset_closedBall)
  have htop : σ S ≠ ⊤ := measure_ne_top _ _
  let μ : Measure (SSphere d) := (σ S)⁻¹ • σ.restrict S
  have hμ : IsProbabilityMeasure μ := ⟨by
    change ((σ S)⁻¹ • σ.restrict S) univ = 1
    rw [Measure.smul_apply, Measure.restrict_apply_univ, smul_eq_mul,
      ENNReal.inv_mul_cancel hpos.ne' htop]⟩
  refine ⟨⟨μ, hμ⟩, fun y r => ?_, ?_⟩
  · have hle : σ.restrict S (sphere y r) ≤ σ (sphere y r) := Measure.restrict_le_self _
    have hnull : σ.restrict S (sphere y r) = 0 :=
      le_zero_iff.mp (hle.trans_eq (uniform_sphere_boundary_null hd σ hσ y r))
    change ((σ S)⁻¹ • σ.restrict S) (sphere y r) = 0
    rw [Measure.smul_apply, hnull, smul_zero]
  · change ((σ S)⁻¹ • σ.restrict S) Sᶜ = 0
    rw [Measure.smul_apply, Measure.restrict_apply isClosed_closedBall.measurableSet.compl]
    simp [S]

/-- Null actual spheres imply absence of actual point masses, since
a zero-radius sphere is its center. Source: arXiv:2411.04551v3,
§5, `cl: balls`, the absence of atoms behind the ball quantiles. -/
theorem singleton_null_of_sphere_null (μ : Measure (SSphere d))
    (hμ : ∀ y : SSphere d, ∀ r : ℝ, μ (sphere y r) = 0) (y : SSphere d) :
    μ {y} = 0 := by simpa using hμ y 0

/-- The construction also puts the actual topological support inside
the prescribed closed cap. Source: arXiv:2411.04551v3, §5,
`eq: boundary.ball` and the absolute-continuity input of `cl: balls`.
The original support is obtained from the proved full-measure carrier,
using closedness of that actual cap. -/
theorem exists_small_support_ball_diffuse (hd : 2 ≤ d) (p : SSphere d)
    (δ : ℝ) (hδ : 0 < δ) :
    ∃ μ : Perspective.ProbSphere d,
      (∀ y : SSphere d, ∀ r : ℝ, (μ : Measure (SSphere d)) (sphere y r) = 0) ∧
      (μ : Measure (SSphere d)).support ⊆ closedBall p δ := by
  obtain ⟨μ, hμ, hcap⟩ := exists_small_cap_ball_diffuse hd p δ hδ
  refine ⟨μ, hμ, ?_⟩
  exact Measure.support_subset_of_isClosed isClosed_closedBall hcap

/-- Every geometric and measure hypothesis holds on a nonempty
family with a genuinely diffuse law. The count `M = 2` admits two
distinct cumulative masses. Source: arXiv:2411.04551v3, §5, `cl: balls`. -/
example : ∃ μ : Perspective.ProbSphere 2,
    (2 : ℕ) ≤ 2 ∧ (0 : ℕ) < 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧
      (∀ y : SSphere 2, ∀ r : ℝ, (μ : Measure (SSphere 2)) (sphere y r) = 0) ∧
      (μ : Measure (SSphere 2)) (closedBall (basePoint 1) (1 / 1000))ᶜ = 0 ∧
      basePoint 1 ∈ closedBall (basePoint 1 : SSphere 2) (1 / 1000) ∧
      ∀ x ∈ closedBall (basePoint 1 : SSphere 2) (1 / 1000),
        dist x (basePoint 1) ≤ (1 / 1000 : ℝ) := by
  obtain ⟨μ, hμ, hcap⟩ := exists_small_cap_ball_diffuse le_rfl (basePoint 1) (1 / 1000) (by norm_num)
  exact ⟨μ, le_rfl, by decide, one_pos, by norm_num, hμ, hcap, by simp,
    fun x hx => hx⟩

/-- The same actual small-cap probability realizes the absence of
atoms and every null-boundary input simultaneously. The carrying set
is nonempty and has actual full mass. Source: arXiv:2411.04551v3,
§5, `cl: balls`, the original absolutely continuous small-support regime. -/
example : ∃ μ : Perspective.ProbSphere 2,
    (∀ y : SSphere 2, (μ : Measure (SSphere 2)) {y} = 0) ∧
      ∀ y : SSphere 2, ∀ r : ℝ, (μ : Measure (SSphere 2)) (sphere y r) = 0 := by
  obtain ⟨μ, hμ, _⟩ := exists_small_cap_ball_diffuse le_rfl (basePoint 1) 1 one_pos
  exact ⟨μ, singleton_null_of_sphere_null _ hμ, hμ⟩

end Transformer.Interpolation

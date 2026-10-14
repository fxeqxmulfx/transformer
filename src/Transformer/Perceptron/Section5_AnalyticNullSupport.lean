/-
# Strict analytic stationary supports are uniform-null in every dimension

ArXiv:2601.21366v2, §3.1, `thm: any.d` (i), and its proof in
§5.3, the real-analytic alternative. Positive original uniform
measure of the original stationary support makes the actual
analytic first variation constant on the entire sphere.

The original full energy has the proved mixed second derivative
along every actual smooth-gradient Monge transport. Its kernel
is bounded by a fixed constant times the product of the two
field norms. A nonzero localized smooth gradient has support of
arbitrarily small original mass, so Cauchy–Schwarz contradicts
the original positive L² coercivity constant.

All original primitive, activation, neurons, temperature and measure
are retained. The conditional derivative input of the original
strict-SOPD predicate is proved before that inequality is used.
The support mass scale is chosen from that positive coercivity
constant and the explicit original mixed-kernel bound. This makes
the resulting contradiction quantitative for the original L² norm.
-/

import Transformer.Perceptron.Section5_SphericalHessian
import Transformer.Perceptron.Section5_SphericalNorms
import Transformer.Perceptron.Section5_NonatomicGradients

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- The analytic strict-SOPD alternative of `thm: any.d` (i), with the
original hypotheses and actual original support.
Source: arXiv:2601.21366v2, §3.1 and §5.3. Analytic continuation,
existence of the actual energy Hessian, the localized smooth-gradient
construction and the small-mass norm estimate are all proved inputs. -/
theorem analytic_strict_measure_support_eq_zero {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : 0 < β) (φ σ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * σ s) s)
    (hσ : AnalyticOnNhd ℝ σ univ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (hμ : IsStrictSOPD β φ σ ω a μ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    ν (μ : Measure (SSphere d)).support = 0 := by
  by_contra hz
  have hpos : 0 < ν (μ : Measure (SSphere d)).support := bot_lt_iff_ne_bot.mpr hz
  obtain ⟨C, hC⟩ := exists_constant_firstVariation_of_positive_uniform_support hd β hβ.ne'
    φ σ hφ hσ ω a μ hμ.1 ν hν hpos
  obtain ⟨κ, hκ, hstrict⟩ := hμ.2
  let D : ℝ := Real.exp β * (β ^ 2 + β)
  let A : ℝ := β⁻¹ * D
  have hA : 0 < A := by dsimp only [A, D]; positivity
  obtain ⟨U, -, hUpos, hUsmall, ξ, hgrad, hcξ, hsupp, hQ⟩ :=
    exists_small_mass_gradient_of_positive_uniform_support hd μ ν hν hpos (κ / A) (div_pos hκ hA)
  let K : SSphere d × SSphere d → ℝ := fun p => spherePairMixedCurvature β
    (p.1 : EucSpace d) (ξ p.1) (p.2 : EucSpace d) (ξ p.2)
  have hK : Continuous K := continuous_spherePairMixedCurvature_field β ξ hcξ
  have hbound (p : SSphere d × SSphere d) : |K p| ≤ D * ‖ξ p.1‖ * ‖ξ p.2‖ := by
    simpa only [K, D, mul_assoc] using abs_spherePairMixedCurvature_le β hβ.le
      (v := ξ p.1) (w := ξ p.2) (norm_eq_of_mem_sphere p.1) (norm_eq_of_mem_sphere p.2)
  have hmix := abs_integral_pair_le_sq_integral_norm μ ξ hcξ K hK D hbound
  have hnorm := sq_integral_norm_le_local_mass μ ξ hcξ U hsupp hUpos
  have hderiv := hasDerivAt_deriv_energy_gradientTransport_of_constant_firstVariation
    β hβ φ (continuous_of_hasDerivAt hφ) ω a μ C hC ξ hgrad
  have hcoercive := hstrict ξ hgrad
    (fun t => μ.map (sphereTransport ξ (inner_isGradientField hgrad) t))
    (isGeodesicFrom_gradientTransport ξ hgrad μ) _ hderiv
  have hupper : β⁻¹ * (∫ p, K p ∂(μ : Measure (SSphere d)).prod μ) ≤
      A * (∫ x, ‖ξ x‖ ∂(μ : Measure (SSphere d))) ^ 2 := by
    calc
      _ ≤ β⁻¹ * |∫ p, K p ∂(μ : Measure (SSphere d)).prod μ| :=
        mul_le_mul_of_nonneg_left (le_abs_self _) (inv_nonneg.mpr hβ.le)
      _ ≤ β⁻¹ * (D * (∫ x, ‖ξ x‖ ∂(μ : Measure (SSphere d))) ^ 2) :=
        mul_le_mul_of_nonneg_left hmix (inv_nonneg.mpr hβ.le)
      _ = _ := by dsimp only [A]; ring
  have hlocal : A * (∫ x, ‖ξ x‖ ∂(μ : Measure (SSphere d))) ^ 2 ≤
      (A * (μ : Measure (SSphere d)).real U) * (∫ x, ‖ξ x‖ ^ 2 ∂(μ : Measure (SSphere d))) := by
    calc
      _ ≤ A * ((μ : Measure (SSphere d)).real U *
          (∫ x, ‖ξ x‖ ^ 2 ∂(μ : Measure (SSphere d)))) := mul_le_mul_of_nonneg_left hnorm hA.le
      _ = _ := by ring
  have hsmall : A * (μ : Measure (SSphere d)).real U < κ := by
    simpa only [mul_comm] using (lt_div_iff₀ hA).mp hUsmall
  exact (not_lt_of_ge (hcoercive.trans (hupper.trans hlocal)))
    (mul_lt_mul_of_pos_right hsmall hQ)

/-- The original strict analytic stationary probability is singular
with respect to every original uniform law.
Source: arXiv:2601.21366v2, §3.1, the consequence of `thm: any.d` (i). -/
theorem analytic_strict_mutuallySingular_uniform {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : 0 < β) (φ σ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * σ s) s)
    (hσ : AnalyticOnNhd ℝ σ univ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (hμ : IsStrictSOPD β φ σ ω a μ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    (μ : Measure (SSphere d)) ⟂ₘ ν :=
  mutuallySingular_of_measure_support_eq_zero
    (analytic_strict_measure_support_eq_zero hd β hβ φ σ hφ hσ ω a μ hμ ν hν)

/-- An original strict analytic stationary support is a proper subset of
the sphere. Source: arXiv:2601.21366v2, §3.1, `thm: any.d` (i),
since the original uniform law has total mass one. -/
theorem support_ne_univ_of_analytic_strict {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : 0 < β) (φ σ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * σ s) s)
    (hσ : AnalyticOnNhd ℝ σ univ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (hμ : IsStrictSOPD β φ σ ω a μ) :
    (μ : Measure (SSphere d)).support ≠ univ := by
  have hν := Metastability.isUniformOn_uniformLaw (d := d) (by omega)
  have hprob : IsProbabilityMeasure (MeanField.uniformLaw d) := hν.1
  have hnull := analytic_strict_measure_support_eq_zero hd β hβ φ σ hφ hσ ω a μ hμ _ hν
  intro heq
  rw [heq] at hnull
  simp at hnull

/-- No original uniform probability is strict SOPD with an analytic
activation. Source: arXiv:2601.21366v2, §3.1 and §5.3,
the uniform-support consequence of the analytic alternative of `thm: any.d` (i). -/
theorem not_isStrictSOPD_of_uniform_analytic {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : 0 < β) (φ σ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * σ s) s)
    (hσ : AnalyticOnNhd ℝ σ univ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (hν : Metastability.IsUniformOn d (μ : Measure (SSphere d))) :
    ¬ IsStrictSOPD β φ σ ω a μ := by
  intro hμ
  exact support_ne_univ_of_analytic_strict hd β hβ φ σ hφ hσ ω a μ hμ
    (support_eq_univ_of_uniform μ hν)

/-- The source's actual pinned Dirac simultaneously realizes dimension,
positive temperature, analytic activation, actual primitive, and strict
SOPD. It can be tested against the actual uniform law.
Source: arXiv:2601.21366v2, §2.2 and §3.1, the nonvacuous strict alternative. -/
example : (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    (∀ s : ℝ, HasDerivAt (fun r : ℝ => r) (2 * (2 : ℝ)⁻¹) s) ∧
    AnalyticOnNhd ℝ (fun _ : ℝ => (2 : ℝ)⁻¹) univ ∧
    IsStrictSOPD 1 (fun s => s) (fun _ => (2 : ℝ)⁻¹) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) ∧
    Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) :=
  ⟨le_rfl, one_pos, fun s => by simpa using hasDerivAt_id' (𝕜 := ℝ) (x := s),
    fun _ _ => analyticAt_const, isStrictSOPD_pin 1 0 _,
    Metastability.isUniformOn_uniformLaw (by norm_num)⟩

/-- The actual uniform probability with zero weights supplies all
hypotheses of the uniform exclusion, including the actual analytic
activation and its actual primitive. -/
example : ∃ μ : Perspective.ProbSphere 2, (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    (∀ s : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (2 * (0 : ℝ → ℝ) s) s) ∧
    AnalyticOnNhd ℝ (0 : ℝ → ℝ) univ ∧
    Metastability.IsUniformOn 2 (μ : Measure (SSphere 2)) := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  exact ⟨⟨MeanField.uniformLaw 2, hν.1⟩, le_rfl, one_pos,
    fun s => by simpa using hasDerivAt_const s (0 : ℝ), fun _ _ => analyticAt_const, hν⟩

end Transformer.Perceptron

module

public import Transformer.Perceptron.Section5_QuadraticDensity
public import Transformer.Perceptron.Section6_SphereUniqueness

/-!
# Full original support of a nonnegative polynomial density

ArXiv:2601.21366v2, §5.3, proof of `thm: any.d` (i), following
`eq:global-Id`. The original quadratic density supplied by `lem: quadpol`
is analytic and nontrivial. Its actual uniform-null zero set forces the
original stationary probability to have the full original sphere as support.

Here an actual multivariate polynomial gives an actual ambient-analytic
extension of its original spherical restriction. Original probability
normalization prevents its density from vanishing identically. The proved
spherical analytic zero-set theorem then makes that original density
strictly positive almost everywhere, using its pointwise nonnegativity.

Consequently the actual uniform law is absolutely continuous with respect
to the original `withDensity` law. Actual support inclusion and the proved
full uniform support give the original full-support conclusion. The
quadratic transform case uses the proved original inverse, including its
proved nonnegativity; no full support or density is assumed there.
Pointwise nonnegativity is essential because `ENNReal.ofReal` clips
negative values; the original inverse proves it before this support step.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped BigOperators ENNReal

namespace Transformer.Perceptron

/-- Evaluation of an actual multivariate polynomial at the actual
ambient coordinates is analytic. Source: arXiv:2601.21366v2, §5.3,
the polynomial-density analyticity step following `eq:global-Id`.
The actual polynomial is built from constants, addition and variables. -/
theorem analyticOnNhd_mvPolynomial_eval {d : ℕ} (P : MvPolynomial (Idx d) ℝ) :
    AnalyticOnNhd ℝ (fun z : EucSpace d => MvPolynomial.eval (fun i => z i) P) univ := by
  intro z _
  induction P using MvPolynomial.induction_on with
  | C c => simpa only [MvPolynomial.eval_C] using (analyticAt_const (x := z) (v := c))
  | add P Q hP hQ => simpa only [MvPolynomial.eval_add] using hP.fun_add hQ
  | mul_X P i hP =>
    have hi : AnalyticAt ℝ (fun x : EucSpace d => x i) z := by
      simpa [EuclideanSpace.inner_single_left] using (innerSL ℝ (EuclideanSpace.single i (1 : ℝ))).analyticAt z
    simpa only [MvPolynomial.eval_mul, MvPolynomial.eval_X] using hP.fun_mul hi

/-- An actual spherical polynomial has an actual ambient-analytic
extension, agreeing at every original sphere point.
Source: arXiv:2601.21366v2, §5.3, following `eq:global-Id`.
The extension is the actual evaluation of its original polynomial witness. -/
theorem exists_analytic_extension_of_isPolyOfDegreeLE {d k : ℕ} (g : SSphere d → ℝ)
    (hg : IsPolyOfDegreeLE k g) : ∃ F : EucSpace d → ℝ,
      AnalyticOnNhd ℝ F univ ∧ ∀ y : SSphere d, g y = F (y : EucSpace d) := by
  obtain ⟨P, -, hval⟩ := hg
  exact ⟨fun z => MvPolynomial.eval (fun i => z i) P, analyticOnNhd_mvPolynomial_eval P, hval⟩

/-- An actual density of an original probability cannot vanish at every
original sphere point. Source: arXiv:2601.21366v2, §5.3,
the nontrivial-density step following `eq:global-Id`.
The actual original total probability mass is retained. -/
theorem exists_nonzero_of_probability_density {d : ℕ} (μ : Perspective.ProbSphere d)
    (ν : Measure (SSphere d)) (g : SSphere d → ℝ)
    (heq : (μ : Measure (SSphere d)) = ν.withDensity (fun y => ENNReal.ofReal (g y))) :
    ∃ y, g y ≠ 0 := by
  by_contra h
  have hg : ∀ y, g y = 0 := by simpa using h
  have hzero : (μ : Measure (SSphere d)) = 0 := by
    rw [heq]
    have hf : (fun y => ENNReal.ofReal (g y)) = (0 : SSphere d → ℝ≥0∞) := by
      funext y
      simp [hg y]
    rw [hf, withDensity_zero]
  exact IsProbabilityMeasure.ne_zero (μ : Measure (SSphere d)) hzero

/-- The actual uniform law is absolutely continuous with respect to
an original probability having an actual nonnegative polynomial density.
Source: arXiv:2601.21366v2, §5.3, following `eq:global-Id`.
Its actual zero set is proved null using analytic nontriviality, rather
than assuming strict positivity or full support of the original density. -/
theorem uniform_absolutelyContinuous_of_nonneg_polynomial_density {d k : ℕ}
    (hd : 2 ≤ d) (μ : Perspective.ProbSphere d) (ν : Measure (SSphere d))
    (hν : Metastability.IsUniformOn d ν) (g : SSphere d → ℝ) (hg : IsPolyOfDegreeLE k g)
    (hnn : ∀ y, 0 ≤ g y)
    (heq : (μ : Measure (SSphere d)) = ν.withDensity (fun y => ENNReal.ofReal (g y))) :
    ν ≪ (μ : Measure (SSphere d)) := by
  obtain ⟨F, hF, hval⟩ := exists_analytic_extension_of_isPolyOfDegreeLE g hg
  obtain ⟨y, hy⟩ := exists_nonzero_of_probability_density μ ν g heq
  have hz := measure_sphere_zeros_eq_zero_of_analytic_nonzero hd F hF ν hν ⟨y, by rwa [← hval y]⟩
  have hset : {y : SSphere d | F (y : EucSpace d) = 0} = {y | g y = 0} := by
    ext y
    change F (y : EucSpace d) = 0 ↔ g y = 0
    rw [hval y]
  rw [hset] at hz
  have hae : ∀ᵐ y ∂ν, g y ≠ 0 := by simpa only [ae_iff, not_not] using hz
  rw [heq]
  apply withDensity_absolutelyContinuous' (ENNReal.continuous_ofReal.comp
    (continuous_of_isPolyOfDegreeLE hg)).aemeasurable
  filter_upwards [hae] with y hy
  intro hzero
  have hn := ENNReal.ofReal_eq_zero.mp hzero
  exact hy (le_antisymm hn (hnn y))

/-- The actual original probability with an actual nonnegative polynomial
density has full original topological support in dimension at least two.
Source: arXiv:2601.21366v2, §5.3, following `eq:global-Id`.
The actual support conclusion follows from the proved absolute continuity
of the actual uniform law in this direction. -/
theorem support_eq_univ_of_nonneg_polynomial_density {d k : ℕ} (hd : 2 ≤ d)
    (μ : Perspective.ProbSphere d) (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (g : SSphere d → ℝ) (hg : IsPolyOfDegreeLE k g) (hnn : ∀ y, 0 ≤ g y)
    (heq : (μ : Measure (SSphere d)) = ν.withDensity (fun y => ENNReal.ofReal (g y))) :
    (μ : Measure (SSphere d)).support = univ := by
  have hs := (uniform_absolutelyContinuous_of_nonneg_polynomial_density hd μ ν hν g hg hnn heq).support_mono
  have hunif := support_eq_univ_of_uniform (⟨ν, hν.1⟩ : Perspective.ProbSphere d) hν
  change ν.support = univ at hunif
  rw [hunif] at hs
  exact eq_univ_of_univ_subset hs

/-- The original quadratic-transform probability has full original
support at positive temperature in every dimension at least two.
Source: arXiv:2601.21366v2, §5.3, `eq:global-Id` and its full-support
consequence. Its actual nonnegative density is proved by the original
kernel inverse, using the constructed genuine uniform law. -/
theorem support_eq_univ_of_const_add_squares_transform {d : ℕ} {ι : Type*}
    (hd : 2 ≤ d) (S : Finset ι) (β : ℝ) (hβ : 0 < β) (μ : Perspective.ProbSphere d)
    (C : ℝ) (q : ι → ℝ) (a : ι → EucSpace d)
    (hquad : ∀ x : SSphere d, attentionTransform β μ x =
      C + ∑ i ∈ S, q i * inner (𝕜 := ℝ) (a i) (x : EucSpace d) ^ 2) :
    (μ : Measure (SSphere d)).support = univ := by
  have hν := Metastability.isUniformOn_uniformLaw (d := d) (by omega)
  obtain ⟨g, hg, hnn, heq⟩ := exists_quadratic_density_of_const_add_squares_transform
    hd S β hβ μ C q a hquad (MeanField.uniformLaw d) hν
  exact support_eq_univ_of_nonneg_polynomial_density hd μ (MeanField.uniformLaw d) hν g hg hnn heq

/-- All density, nonnegativity, polynomial, temperature and dimensional
inputs hold jointly for the original uniform circle probability and actual
constant density one. Source: arXiv:2601.21366v2, §5.3,
the polynomial full-support step following `eq:global-Id`. -/
example : ∃ μ : Perspective.ProbSphere 2, (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    Metastability.IsUniformOn 2 (μ : Measure (SSphere 2)) ∧
    IsPolyOfDegreeLE 0 (fun _ : SSphere 2 => (1 : ℝ)) ∧ (∀ _y : SSphere 2, (0 : ℝ) ≤ 1) ∧
    (μ : Measure (SSphere 2)) = (μ : Measure (SSphere 2)).withDensity (fun _ => ENNReal.ofReal 1) ∧
    ∀ x : SSphere 2, attentionTransform 1 μ x = attentionTransform 1 μ (basePoint 1) +
      ∑ i ∈ (Finset.univ : Finset (Fin 2)), (0 : ℝ) *
        inner (𝕜 := ℝ) (EuclideanSpace.single i (1 : ℝ)) (x : EucSpace 2) ^ 2 := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  exact ⟨μ, le_rfl, one_pos, hν, isPolyOfDegreeLE_const 1, fun _ => zero_le_one, by simp,
    fun x => by simpa using attentionTransform_eq_of_uniform 1 μ hν x (basePoint 1)⟩

end Transformer.Perceptron

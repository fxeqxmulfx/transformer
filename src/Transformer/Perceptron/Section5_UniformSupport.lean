module

public import Transformer.Perceptron.Section5_ConstantTransform
public import Transformer.Perceptron.Section6_AntipodalFields
public import Transformer.Perceptron.Section6_CircleNullSupport

/-!
# The original uniform support and stationary linear-density example

ArXiv:2601.21366v2, §5.1, `lem: quadpol` (i), and §5.3,
proof of `rem:genericity-vacuity`. The constant density one gives the
original uniform measure, whose support is the entire original sphere.
In dimension at least two it is not finitely atomic.

An actual support point exists by probability normalization. The actual
reflection exchanging it with any other unit point transports support
under the source's orthogonal invariance. Thus full support is proved
without a density or support conclusion as an input. The previously
proved uniform atomlessness then excludes a finite Dirac presentation.

With every original perceptron weight zero, the actual perceptron drift
vanishes. The actual original uniform attention field was proved zero,
so both original stationary predicates hold at nonzero temperature.
This produces the full stationary/interior-support witness missing from
the earlier linear-density statement, retaining its genuine probability
measure and the primitive `φ(s)=s²` of the original linear activation.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped BigOperators ENNReal

namespace Transformer.Perceptron

/-- The original uniform law has full actual topological support.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i),
and §5.3, the full-support consequence for analytic densities.
Probability gives one support point; actual orthogonal transitivity
and the actual measure invariance give every other sphere point. -/
theorem support_eq_univ_of_uniform {d : ℕ} (μ : Perspective.ProbSphere d)
    (hμ : Metastability.IsUniformOn d (μ : Measure (SSphere d))) :
    (μ : Measure (SSphere d)).support = univ := by
  have hne : (μ : Measure (SSphere d)) ≠ 0 := by
    intro heq
    have h := congrArg (fun m : Measure (SSphere d) => m univ) heq
    simp at h
  obtain ⟨y, hy⟩ := Measure.nonempty_support hne
  apply eq_univ_iff_forall.mpr
  intro x
  let U := (ℝ ∙ ((y : EucSpace d) - (x : EucSpace d)))ᗮ.reflection
  have heq : Perspective.sphereMap d U y = x := Subtype.ext
    (Submodule.reflection_sub ((norm_eq_of_mem_sphere y).trans (norm_eq_of_mem_sphere x).symm))
  have hx := mem_support_map_of_continuous_sphere (μ : Measure (SSphere d))
    (Perspective.sphereMap d U) (Perspective.continuous_sphereMap d U)
    (Perspective.continuous_sphereMap d U).measurable y hy
  rwa [hμ.2 U, heq] at hx

/-- The original support-interior hypothesis is realized by the actual
uniform probability. Source: arXiv:2601.21366v2, §5.3,
`rem:genericity-vacuity`. Full support supplies an actual interior point
on the original sphere, whose nonemptiness follows from probability. -/
theorem interior_support_nonempty_of_uniform {d : ℕ} (μ : Perspective.ProbSphere d)
    (hμ : Metastability.IsUniformOn d (μ : Measure (SSphere d))) :
    (interior (μ : Measure (SSphere d)).support).Nonempty := by
  rw [support_eq_univ_of_uniform μ hμ, interior_univ]
  obtain ⟨x⟩ := nonempty_of_isProbabilityMeasure (μ : Measure (SSphere d))
  exact ⟨x, mem_univ _⟩

/-- The original uniform probability is not finitely atomic in every
dimension at least two. Source: arXiv:2601.21366v2, §5.3,
`rem:genericity-vacuity`, the non-atomic polynomial-density example.
Full support and the proved uniform nullness of finite sets contradict
the original probability mass, using the actual finite Dirac predicate. -/
theorem not_isFinitelyAtomic_of_uniform {d : ℕ} (hd : 2 ≤ d)
    (μ : Perspective.ProbSphere d)
    (hμ : Metastability.IsUniformOn d (μ : Measure (SSphere d))) : ¬ IsFinitelyAtomic μ := by
  intro hfinite
  have hz := measure_finite_support_eq_zero_of_uniform hd (μ : Measure (SSphere d))
    (finite_support_of_finitely_atomic μ hfinite) (μ : Measure (SSphere d)) hμ
  rw [support_eq_univ_of_uniform μ hμ] at hz
  simp at hz

/-- Every original neuron has zero contribution when its original weight
is zero. Source: arXiv:2601.21366v2, §2.3, the original perceptron drift.
The actual activation, neuron vectors and tangent projection are retained. -/
theorem drift_zero_weights {d : ℕ} (σ : ℝ → ℝ) (a : Idx d → EucSpace d) (x : EucSpace d) :
    drift σ (0 : Idx d → ℝ) a x = 0 := by
  simp [drift, proj]

/-- The original uniform probability is stationary for zero original
weights at every nonzero temperature and every actual activation.
Source: arXiv:2601.21366v2, §5.3, `rem:genericity-vacuity`.
The source's linear positive-temperature example is included without
assuming stationarity or replacing the probability by a Dirac mass. -/
theorem isStationary_zero_weights_of_uniform {d : ℕ} (β : ℝ) (hβ : β ≠ 0)
    (σ : ℝ → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (hμ : Metastability.IsUniformOn d (μ : Measure (SSphere d))) :
    IsStationary β σ (0 : Idx d → ℝ) a μ := by
  intro x _
  change attentionGrad β μ (x : EucSpace d) + drift σ 0 a (x : EucSpace d) = 0
  rw [attentionGrad_zero_of_uniform β hβ μ hμ x, drift_zero_weights, add_zero]

/-- The same original uniform probability is stationary for the original
normalized field at zero original weights. Source: arXiv:2601.21366v2,
§6, `eq: fulltrans.stat`. The original positive weight multiplies the
actual zero drift; no normalized or unnormalized field is redefined. -/
theorem isNormalizedStationary_zero_weights_of_uniform {d : ℕ} (β : ℝ) (hβ : β ≠ 0)
    (σ : ℝ → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (hμ : Metastability.IsUniformOn d (μ : Measure (SSphere d))) :
    IsNormalizedStationary β σ (0 : Idx d → ℝ) a μ := by
  intro x _
  rw [attentionGrad_zero_of_uniform β hβ μ hμ x, drift_zero_weights, smul_zero, add_zero]

/-- The original uniform probability has its actual constant polynomial
density against every original uniform law, at every degree bound.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i),
and §5.3, `rem:genericity-vacuity`. The proved transform criterion
identifies the two actual laws before the constant density is supplied. -/
theorem exists_polynomial_density_of_uniform {d : ℕ} (k : ℕ)
    (μ : Perspective.ProbSphere d)
    (hμ : Metastability.IsUniformOn d (μ : Measure (SSphere d)))
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    ∃ g : SSphere d → ℝ, IsPolyOfDegreeLE k g ∧
      (μ : Measure (SSphere d)) = ν.withDensity fun y => ENNReal.ofReal (g y) := by
  let x : SSphere d := Classical.choice (nonempty_of_isProbabilityMeasure (μ : Measure (SSphere d)))
  have heq := eq_uniform_of_constant_attentionTransform 1 one_ne_zero μ ν hν
    ⟨attentionTransform 1 μ x, fun y => attentionTransform_eq_of_uniform 1 μ hμ y x⟩
  refine ⟨fun _ => 1, ⟨MvPolynomial.C 1, by simp, fun _ => by simp⟩, ?_⟩
  simpa using heq

/-- Every hypothesis of the source's linear stationary-density statement
holds jointly, including nonempty interior of the actual support.
Source: arXiv:2601.21366v2, §5.3, `rem:genericity-vacuity`.
The original uniform circle law realizes the claimed non-atomic example. -/
example : ∃ μ : Perspective.ProbSphere 2, (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    Metastability.IsUniformOn 2 (μ : Measure (SSphere 2)) ∧
    (∀ s : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * s) s) ∧
    IsStationary 1 (fun s => s) (0 : Idx 2 → ℝ) (0 : Idx 2 → EucSpace 2) μ ∧
    IsNormalizedStationary 1 (fun s => s) (0 : Idx 2 → ℝ) (0 : Idx 2 → EucSpace 2) μ ∧
    (interior (μ : Measure (SSphere 2)).support).Nonempty ∧
    (∀ ν : Measure (SSphere 2), Metastability.IsUniformOn 2 ν →
      ∃ g : SSphere 2 → ℝ, IsPolyOfDegreeLE 2 g ∧
        (μ : Measure (SSphere 2)) = ν.withDensity fun y => ENNReal.ofReal (g y)) ∧
    ¬ IsFinitelyAtomic μ := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  refine ⟨μ, le_rfl, one_pos, hν, fun s => ?_,
    isStationary_zero_weights_of_uniform 1 one_ne_zero (fun s => s) 0 μ hν,
    isNormalizedStationary_zero_weights_of_uniform 1 one_ne_zero (fun s => s) 0 μ hν,
    interior_support_nonempty_of_uniform μ hν, fun ν hν' => exists_polynomial_density_of_uniform 2 μ hν ν hν',
    not_isFinitelyAtomic_of_uniform le_rfl μ hν⟩
  · simpa using hasDerivAt_pow 2 s

end Transformer.Perceptron

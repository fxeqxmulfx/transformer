module

public import Transformer.Perceptron.Section5_BiasedObstruction
public import Transformer.Perceptron.Section5_ReLUAtomicity

/-!
# Original biased ReLU probabilities have finite or null support

ArXiv:2601.21366v2, §3.1, `rem: ext` (i), extending `thm: circle`
and the ReLU alternative of `thm: any.d` (i). The proofs follow §5.2
and §5.3 with every original bias retained in the quadratic branch.

On the circle, infinite original support gives infinitely many actual
angular lifts in the compact circle interval. One actual biased pattern
contains infinitely many of these lifts. Its original analytic vector
branch must then vanish on the whole actual circle.

In higher dimensions, positive uniform measure of the original support
gives a biased pattern with positive original measure. Analytic vector
uniqueness again forces that original branch to vanish globally.

The proved affine-quadratic inverse and original first-variation identity
exclude both zero branches under the source's non-analyticity hypothesis.
The actual supplied sphere point fixes their constants. The source's
transverse section supplies that point in `BiasedAtomicity.lean`.
The original arbitrary primitive, probability, neuron data and biases
are retained throughout. Finite support gives actual finite Dirac masses;
uniform-null support gives actual mutual singularity.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- The original biased stationary ReLU probability has finite actual
support on the original circle. Source: arXiv:2601.21366v2, §3.1,
`rem: ext` (i), carrying `thm: circle`, with its proof in §5.2.
The supplied actual sphere point fixes the first-variation constant;
the infinite-pattern contradiction retains every original bias. -/
theorem finite_support_of_biased_stationary_relu_circle (β : ℝ) (hβ : 0 < β) (φ : ℝ → ℝ)
    (hφ : ∀ s, HasDerivAt φ (2 * max s 0) s) (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (b : Idx 2 → ℝ) (hana : ¬ IsBiasedAnalyticOnSphere φ ω a b)
    (μ : Perspective.ProbSphere 2) (hμ : IsBiasedStationary β (fun s => max s 0) ω a b μ)
    (o : SSphere 2) : (μ : Measure (SSphere 2)).support.Finite := by
  by_contra hInf
  let S : Set ℝ := Icc (-π) π ∩ circlePoint ⁻¹' (μ : Measure (SSphere 2)).support
  have hS : S.Infinite := by
    intro hfinite
    apply hInf
    apply (hfinite.image circlePoint).subset
    intro x hx
    obtain ⟨θ, hθ, heq⟩ := exists_circlePoint_eq x
    refine ⟨θ, ⟨hθ, ?_⟩, heq⟩
    change circlePoint θ ∈ (μ : Measure (SSphere 2)).support
    rwa [heq]
  obtain ⟨J, hT⟩ := exists_infinite_biased_activation_angles a b S hS
  let T : Set ℝ := S ∩ {θ | IsBiasedActivationPattern a b J (circlePoint θ : EucSpace 2)}
  let F := fun x : EucSpace 2 => proj 2 x (biasedAttentionVector β ω a b μ J x)
  have hzero : ∀ θ ∈ T, F (circlePoint θ : EucSpace 2) = 0 := by
    intro θ hθ
    change proj 2 (circlePoint θ : EucSpace 2)
      (biasedAttentionVector β ω a b μ J (circlePoint θ : EucSpace 2)) = 0
    rw [projected_biasedAttentionVector_eq_relu_at_pattern β ω a b μ J _ hθ.2,
      hμ (circlePoint θ) hθ.1.2, smul_zero]
  have hglobal := vector_eq_zero_on_circle_of_infinite_zeros F
    (analyticOnNhd_projected_biasedAttentionVector β ω a b μ J)
    T hT (fun _ hθ => hθ.1.1) hzero
  exact not_zero_projected_biased_branch_of_nonanalytic le_rfl β hβ φ hφ ω a b hana μ hμ J o hglobal

/-- Finite actual support supplies the source's actual finite Dirac
presentation of the same original biased stationary probability.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (i), carrying `thm: circle`.
The weights in the presentation are the original singleton masses. -/
theorem isFinitelyAtomic_of_biased_stationary_relu_circle (β : ℝ) (hβ : 0 < β) (φ : ℝ → ℝ)
    (hφ : ∀ s, HasDerivAt φ (2 * max s 0) s) (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (b : Idx 2 → ℝ) (hana : ¬ IsBiasedAnalyticOnSphere φ ω a b)
    (μ : Perspective.ProbSphere 2) (hμ : IsBiasedStationary β (fun s => max s 0) ω a b μ)
    (o : SSphere 2) : IsFinitelyAtomic μ :=
  isFinitelyAtomic_of_finite_topological_support μ
    (finite_support_of_biased_stationary_relu_circle β hβ φ hφ ω a b hana μ hμ o)

/-- The original support has zero measure for every actual uniform law
in dimension at least two. Source: arXiv:2601.21366v2, §3.1,
`rem: ext` (i), carrying the ReLU alternative of `thm: any.d` (i),
with §5.3 after `eq:global-Id`. The original weak biased patterns cover
every original support point, including all activation boundaries. -/
theorem biased_relu_measure_support_eq_zero {d : ℕ} (hd : 2 ≤ d) (β : ℝ) (hβ : 0 < β)
    (φ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * max s 0) s) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (b : Idx d → ℝ) (hana : ¬ IsBiasedAnalyticOnSphere φ ω a b)
    (μ : Perspective.ProbSphere d) (hμ : IsBiasedStationary β (fun s => max s 0) ω a b μ)
    (o : SSphere d) : ∀ ν : Measure (SSphere d), Metastability.IsUniformOn d ν →
      ν (μ : Measure (SSphere d)).support = 0 := by
  intro ν hν
  by_contra hne
  obtain ⟨J, hpos⟩ := exists_positive_measure_biased_activationPattern ν
    (μ : Measure (SSphere d)).support (pos_iff_ne_zero.mpr hne) a b
  let F := fun x : EucSpace d => proj d x (biasedAttentionVector β ω a b μ J x)
  have hposF : 0 < ν {x : SSphere d | F (x : EucSpace d) = 0} :=
    hpos.trans_le (measure_mono (fun x hx => by
      change proj d (x : EucSpace d) (biasedAttentionVector β ω a b μ J (x : EucSpace d)) = 0
      rw [projected_biasedAttentionVector_eq_relu_at_pattern β ω a b μ J _ hx.2,
        hμ x hx.1, smul_zero]))
  have hglobal := vector_eq_zero_on_sphere_of_positive_zero_measure hd F
    (analyticOnNhd_projected_biasedAttentionVector β ω a b μ J) ν hν hposF
  exact not_zero_projected_biased_branch_of_nonanalytic hd β hβ φ hφ ω a b hana μ hμ J o hglobal

/-- The original biased stationary probability is mutually singular
with every original uniform law. Source: arXiv:2601.21366v2, §3.1,
`rem: ext` (i), carrying the final assertion of `thm: any.d` (i).
Its actual topological support is the proved uniform-null carrier. -/
theorem biased_relu_mutuallySingular_uniform {d : ℕ} (hd : 2 ≤ d) (β : ℝ) (hβ : 0 < β)
    (φ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * max s 0) s) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (b : Idx d → ℝ) (hana : ¬ IsBiasedAnalyticOnSphere φ ω a b)
    (μ : Perspective.ProbSphere d) (hμ : IsBiasedStationary β (fun s => max s 0) ω a b μ)
    (o : SSphere d) (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    (μ : Measure (SSphere d)) ⟂ₘ ν :=
  mutuallySingular_of_measure_support_eq_zero
    (biased_relu_measure_support_eq_zero hd β hβ φ hφ ω a b hana μ hμ o ν hν)

/-- The repository's actual original uniform probability also gives
zero mass to the original biased stationary support.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (i), carrying `thm: any.d` (i).
Its original rotation-invariant probability property is proved separately;
the supplied sphere point still fixes the original branch constant. -/
theorem biased_relu_support_uniformLaw_eq_zero {d : ℕ} (hd : 2 ≤ d) (β : ℝ) (hβ : 0 < β)
    (φ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * max s 0) s) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (b : Idx d → ℝ) (hana : ¬ IsBiasedAnalyticOnSphere φ ω a b)
    (μ : Perspective.ProbSphere d) (hμ : IsBiasedStationary β (fun s => max s 0) ω a b μ)
    (o : SSphere d) : MeanField.uniformLaw d (μ : Measure (SSphere d)).support = 0 :=
  biased_relu_measure_support_eq_zero hd β hβ φ hφ ω a b hana μ hμ o
    (MeanField.uniformLaw d) (Metastability.isUniformOn_uniformLaw (by omega))

/-- Every original finite-support, support-null and singularity input
holds jointly for the source's original pinned Dirac with actual zero
biases. Source: arXiv:2601.21366v2, §3.1, `rem: ext` (i), carrying
`thm: circle` and `thm: any.d` (i). The actual base point fixes the
first-variation constant, and the actual original uniform law supplies
the higher-dimensional conclusion's measure input. -/
example : (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    (∀ s, HasDerivAt (fun t : ℝ => max t 0 ^ 2) (2 * max s 0) s) ∧
    ¬ IsBiasedAnalyticOnSphere (fun s => max s 0 ^ 2) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-(basePoint 1 : EucSpace 2))) 0 ∧
    IsBiasedStationary 1 (fun s => max s 0) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-(basePoint 1 : EucSpace 2))) 0 (Perspective.diracProb 2 (basePoint 1)) ∧
    Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) := by
  refine ⟨le_rfl, one_pos, hasDerivAt_reluSq, ?_, ?_, Metastability.isUniformOn_uniformLaw (by norm_num)⟩
  · rw [isBiasedAnalyticOnSphere_zero_iff]
    exact not_isAnalyticOnSphere_relu 0 norm_basePoint_one norm_secondAxis inner_basePoint_secondAxis
  · rw [isBiasedStationary_zero_iff]
    exact isStationary_relu_pin 1 0 (basePoint 1)

end Transformer.Perceptron

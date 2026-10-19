module

public import Transformer.Perceptron.Section5_ReLUObstruction

/-!
# Finite original support for stationary ReLU probabilities on the circle

ArXiv:2601.21366v2, §5.2, proof of `thm: circle`.
An infinite original support has infinitely many actual angular lifts
in the compact original circle interval. The finite original activation
patterns select one with infinitely many original support points.

The actual projected quadratic branch field is analytic in the ambient
space and hence along the original circle. Testing it against every
actual vector gives an analytic scalar function with infinitely many
actual zeros in the compact interval. The proved analytic identity
theorem makes the original vector branch zero at every sphere point.

The original quadratic inverse then forces full support, and original
stationarity makes the original perceptron potential analytic. This
contradicts the source's actual non-analyticity hypothesis, as proved in
the preceding obstruction module. The resulting finite actual support
gives a finite original Dirac presentation with actual singleton masses.
The original primitive, neuron data and probability are never replaced.
In every dimension at least two, the same branch obstruction proves the
ReLU alternative of §5.3, `thm: any.d` (i): the original support is null
for every actual original uniform law. A positive-measure pattern would
make its original analytic branch zero globally, giving the contradiction.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- Infinite actual angular zeros of an ambient-analytic vector field
in the original compact circle interval force zero on the actual circle.
Source: arXiv:2601.21366v2, §5.2, the identity step of `thm: circle`.
Scalar inner-product tests retain the actual vector field and actual circle. -/
theorem vector_eq_zero_on_circle_of_infinite_zeros {m : ℕ}
    (F : EucSpace 2 → EucSpace m) (hF : AnalyticOnNhd ℝ F univ)
    (S : Set ℝ) (hS : S.Infinite) (hbound : S ⊆ Icc (-π) π)
    (hzero : ∀ θ ∈ S, F (circlePoint θ : EucSpace 2) = 0) :
    ∀ x : SSphere 2, F (x : EucSpace 2) = 0 := by
  intro x
  obtain ⟨θ, -, hθ⟩ := exists_circlePoint_eq x
  let v := F (x : EucSpace 2)
  let f := fun r : ℝ => inner (𝕜 := ℝ) v (F (circlePoint r : EucSpace 2))
  have hc : AnalyticOnNhd ℝ (fun r => (circlePoint r : EucSpace 2)) univ :=
    analyticOnNhd_greatCircle (basePoint 1 : EucSpace 2) secondAxis
  have hf : AnalyticOnNhd ℝ f univ := fun r hr =>
    ((innerSL ℝ v).analyticAt _).comp ((hF _ (mem_univ _)).comp (hc r hr))
  have hz := eq_zero_of_analytic_infinite_zeros f hf (Icc (-π) π) isCompact_Icc S hS hbound
    (fun r hr => by change inner (𝕜 := ℝ) v (F (circlePoint r : EucSpace 2)) = 0; rw [hzero r hr, inner_zero_right]) θ
  have hx : inner (𝕜 := ℝ) (F (x : EucSpace 2)) (F (x : EucSpace 2)) = 0 := by
    simpa only [f, v, hθ] using hz
  exact inner_self_eq_zero.mp hx

/-- The original stationary ReLU probability has finite actual
topological support on the actual circle under the original non-analytic
potential hypothesis. Source: arXiv:2601.21366v2, §5.2, `thm: circle`.
Original support points lift into the actual compact angular interval;
an infinite pattern would give the forbidden globally zero branch. -/
theorem finite_support_of_stationary_relu_circle (β : ℝ) (hβ : 0 < β) (φ : ℝ → ℝ)
    (hφ : ∀ s, HasDerivAt φ (2 * max s 0) s) (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (hana : ¬ IsAnalyticOnSphere φ ω a) (μ : Perspective.ProbSphere 2)
    (hμ : IsStationary β (fun s => max s 0) ω a μ) :
    (μ : Measure (SSphere 2)).support.Finite := by
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
  obtain ⟨J, hT⟩ := exists_infinite_activation_angles a S hS
  let T : Set ℝ := S ∩ {θ | IsActivationPattern a J (circlePoint θ : EucSpace 2)}
  let F := fun x : EucSpace 2 => proj 2 x (linearAttentionVector β (branchMaskedWeights ω J) a μ x)
  have hzero : ∀ θ ∈ T, F (circlePoint θ : EucSpace 2) = 0 := by
    intro θ hθ
    change proj 2 (circlePoint θ : EucSpace 2)
      (linearAttentionVector β (branchMaskedWeights ω J) a μ (circlePoint θ : EucSpace 2)) = 0
    rw [projected_linearAttentionVector_eq_relu_at_pattern β ω a μ J _ hθ.2,
      hμ (circlePoint θ) hθ.1.2, smul_zero]
  have hglobal := vector_eq_zero_on_circle_of_infinite_zeros F
    (analyticOnNhd_projected_linearAttentionVector β (branchMaskedWeights ω J) a μ)
    T hT (fun _ hθ => hθ.1.1) hzero
  exact not_zero_projected_relu_branch_of_nonanalytic le_rfl β hβ φ hφ ω a hana μ hμ J hglobal

/-- **Theorem `thm: circle`:** the original stationary ReLU probability
is a finite combination of original Dirac masses when its original
perceptron potential is not analytic on the circle.
Source: arXiv:2601.21366v2, §3.1, `thm: circle`, proved in §5.2.
The original arbitrary primitive of twice ReLU, positive temperature,
neuron weights and vectors, probability, stationarity and non-analyticity
are retained. The original measure's actual singleton masses give the
Dirac coefficients, by the proved finite-support representation theorem. -/
theorem circle_isFinitelyAtomic (β : ℝ) (hβ : 0 < β) (φ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * max s 0) s) (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (hana : ¬ IsAnalyticOnSphere φ ω a)
    (μ : Perspective.ProbSphere 2) (hμ : IsStationary β (fun s => max s 0) ω a μ) :
    IsFinitelyAtomic μ :=
  isFinitelyAtomic_of_finite_topological_support μ
    (finite_support_of_stationary_relu_circle β hβ φ hφ ω a hana μ hμ)

/-- The original stationary ReLU probability has actual uniform-null
support in every dimension at least two when its original potential
is non-analytic. Source: arXiv:2601.21366v2, §3.1, `thm: any.d` (i),
the ReLU alternative, proved in §5.3 after `eq:global-Id`.
The actual pattern intersection has positive original uniform measure
under a contrary assumption; the proved vector uniqueness gives the
forbidden globally zero original branch. Every source hypothesis remains. -/
theorem relu_measure_support_eq_zero {d : ℕ} (hd : 2 ≤ d) (β : ℝ) (hβ : 0 < β)
    (φ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * max s 0) s) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (hana : ¬ IsAnalyticOnSphere φ ω a)
    (μ : Perspective.ProbSphere d) (hμ : IsStationary β (fun s => max s 0) ω a μ) :
    ∀ ν : Measure (SSphere d), Metastability.IsUniformOn d ν →
      ν (μ : Measure (SSphere d)).support = 0 := by
  intro ν hν
  by_contra hne
  obtain ⟨J, hpos⟩ := exists_positive_measure_activationPattern ν (μ : Measure (SSphere d)).support
    (pos_iff_ne_zero.mpr hne) a
  let F := fun x : EucSpace d => proj d x (linearAttentionVector β (branchMaskedWeights ω J) a μ x)
  have hposF : 0 < ν {x : SSphere d | F (x : EucSpace d) = 0} :=
    hpos.trans_le (measure_mono (fun x hx => by
      change proj d (x : EucSpace d)
        (linearAttentionVector β (branchMaskedWeights ω J) a μ (x : EucSpace d)) = 0
      rw [projected_linearAttentionVector_eq_relu_at_pattern β ω a μ J _ hx.2,
        hμ x hx.1, smul_zero]))
  have hglobal := vector_eq_zero_on_sphere_of_positive_zero_measure hd F
    (analyticOnNhd_projected_linearAttentionVector β (branchMaskedWeights ω J) a μ) ν hν hposF
  exact not_zero_projected_relu_branch_of_nonanalytic hd β hβ φ hφ ω a hana μ hμ J hglobal

/-- Every analytic, infinite-zero-set and compact angular hypothesis
holds jointly for the actual zero vector field on the original circle
and the actual entire original angle interval.
Source: arXiv:2601.21366v2, §5.2, the identity step of `thm: circle`.
The interval is infinite because its actual length is strictly positive. -/
example : AnalyticOnNhd ℝ (fun _ : EucSpace 2 => (0 : EucSpace 2)) univ ∧
    (Icc (-π) π).Infinite ∧ (Icc (-π) π ⊆ Icc (-π) π) ∧
    ∀ θ ∈ Icc (-π) π, (fun _ : EucSpace 2 => (0 : EucSpace 2)) (circlePoint θ : EucSpace 2) = 0 :=
  ⟨fun _ _ => analyticAt_const, Icc_infinite (by linarith [Real.pi_pos]), subset_rfl, fun _ _ => rfl⟩

/-- Every original atomicity hypothesis holds jointly for the original
pinned Dirac, actual one-neuron ReLU weights and original primitive.
Source: arXiv:2601.21366v2, §3.1, `thm: circle`, and its proof in §5.2.
The actual probability, all original support points and the zero second
neuron are retained in the witness. -/
example : (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    (∀ s, HasDerivAt (fun t : ℝ => max t 0 ^ 2) (2 * max s 0) s) ∧
    ¬ IsAnalyticOnSphere (fun s => max s 0 ^ 2) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-(basePoint 1 : EucSpace 2))) ∧
    IsStationary 1 (fun s => max s 0) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-(basePoint 1 : EucSpace 2))) (Perspective.diracProb 2 (basePoint 1)) ∧
    Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) :=
  ⟨le_rfl, one_pos, hasDerivAt_reluSq,
    not_isAnalyticOnSphere_relu 0 norm_basePoint_one norm_secondAxis inner_basePoint_secondAxis,
    isStationary_relu_pin 1 0 (basePoint 1), Metastability.isUniformOn_uniformLaw (by norm_num)⟩

end Transformer.Perceptron

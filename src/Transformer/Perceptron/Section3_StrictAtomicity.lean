module

public import Transformer.Perceptron.Section5_LocalHessianBound
public import Transformer.Perceptron.Section5_Accumulation

/-!
# Finite strict SOPD support on the circle with a continuous derivative

ArXiv:2601.21366v2, §3.1, `thm: circle.gelu`, and its proof in
§5.2. The source assumes a real-analytic activation and obtains
global cancellation of the first-variation curvature if support
is infinite. The following stronger support result only assumes
that the actual activation derivative is continuous.

At a stationary support accumulation point the actual angular
force derivative is zero. Continuity makes that curvature and
the pair-kernel oscillation small nearby. Two localized smooth
gradients give a nonzero direction with zero measure mean. The
proved actual Hessian along it is smaller than half its strict
SOPD lower bound, a contradiction. Every quantity is from the
original energy, first variation, measure and smooth gradient field.

The analytic source theorem is retained below as a corollary.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology

namespace Transformer.Perceptron

/-- Strict SOPD and a continuous actual activation derivative force finite
topological support in dimension two. Source: arXiv:2601.21366v2, §3.1,
`thm: circle.gelu`, and §5.2. This strengthens the source's analytic
activation hypothesis. The proof uses local cancellation at one actual
support accumulation point, retaining the original full Hessian formula. -/
theorem finite_support_of_strictSOPD_continuous_derivative (β : ℝ) (hβ : 0 < β)
    (φ σ σ' : ℝ → ℝ) (hφ : ∀ r, HasDerivAt φ (2 * σ r) r)
    (hσ : ∀ r, HasDerivAt σ (σ' r) r) (hσ' : Continuous σ')
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (hμ : IsStrictSOPD β φ σ ω a μ) : (μ : Measure (SSphere 2)).support.Finite := by
  by_contra hInf
  let S : Set ℝ := Icc (-π) π ∩ circlePoint ⁻¹' (μ : Measure (SSphere 2)).support
  have hcompact : IsCompact S := isCompact_Icc.inter_right
    (Measure.isClosed_support.preimage continuous_circlePoint)
  have hS : S.Infinite := by
    intro hfinite
    apply hInf
    apply (hfinite.image circlePoint).subset
    intro x hx
    obtain ⟨θ, hθ, heq⟩ := exists_circlePoint_eq x
    refine ⟨θ, ⟨hθ, ?_⟩, heq⟩
    change circlePoint θ ∈ (μ : Measure (SSphere 2)).support
    rwa [heq]
  obtain ⟨p, -, hacc⟩ := hS.exists_accPt_of_subset_isCompact hcompact (subset_refl S)
  let A : SSphere 2 → ℝ := circleFirstVariationCurvature β σ σ' ω a μ
  let K : SSphere 2 × SSphere 2 → ℝ := fun z => circlePairCurvature β z.1 z.2
  have hσc : Continuous σ := continuous_iff_continuousAt.mpr fun r => (hσ r).continuousAt
  have hA : Continuous A := continuous_circleFirstVariationCurvature β hβ σ σ' hσc hσ' ω a μ
  have hK : Continuous K := continuous_circlePairCurvature β
  have hzero : ∀ q ∈ S, deriv (angularFirstVariation β φ ω a μ) q = 0 :=
    fun q hq => deriv_angularFirstVariation_zero_on_support β hβ.ne' φ σ hφ ω a μ hμ.1 q hq.2
  have hAp : A (circlePoint p) = 0 := derivative_zero_at_accPt_of_zeros
    (deriv (angularFirstVariation β φ ω a μ)) S p (A (circlePoint p)) hacc hzero
    (hasDerivAt_angular_force_circleCurvature β hβ φ σ σ' hφ hσ ω a μ p)
  obtain ⟨κ, hκ, hH⟩ := strict_circle_hessian_coercive_expanded β hβ φ σ σ' hφ hσ hσ' ω a μ hμ
  obtain ⟨U, hU, hpU, hlocal, hosc⟩ := exists_open_local_curvature_bounds
    A hA K hK (circlePoint p) hAp (κ / 4) (by positivity)
  obtain ⟨θ, ψ, hθS, hψS, hne, hθU, hψU⟩ := exists_two_circlePoints_of_accPt S p hacc U hU hpU
  obtain ⟨s, hs, hgrad, hsupp, hmean, hnorm⟩ := exists_zero_mean_gradient_in_open μ U hU
    (circlePoint θ) (circlePoint ψ) hne hθS.2 hψS.2 hθU hψU
  have hlower := hH s hs hgrad
  change κ * (∫ x, s x ^ 2 ∂(μ : Measure (SSphere 2))) ≤
    (∫ x, A x * s x ^ 2 ∂(μ : Measure (SSphere 2))) -
      (∫ z, K z * s z.1 * s z.2
        ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) at hlower
  have hupper := local_expanded_hessian_le μ A hA K hK s hs U hsupp hmean
    (K (circlePoint p, circlePoint p)) (κ / 4) (by positivity) hlocal hosc
  nlinarith [mul_pos hκ hnorm]

/-- The stronger strict SOPD result gives the source's actual finite
Dirac presentation, with the original probability measure's singleton
masses. Source: arXiv:2601.21366v2, §3.1, `thm: circle.gelu`, and §5.2.
The analytic activation assumption is strengthened to a continuous
actual derivative; finite atomicity is unchanged. -/
theorem isFinitelyAtomic_of_strictSOPD_continuous_derivative (β : ℝ) (hβ : 0 < β)
    (φ σ σ' : ℝ → ℝ) (hφ : ∀ r, HasDerivAt φ (2 * σ r) r)
    (hσ : ∀ r, HasDerivAt σ (σ' r) r) (hσ' : Continuous σ')
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (hμ : IsStrictSOPD β φ σ ω a μ) : IsFinitelyAtomic μ :=
  isFinitelyAtomic_of_finite_topological_support μ
    (finite_support_of_strictSOPD_continuous_derivative β hβ φ σ σ' hφ hσ hσ' ω a μ hμ)

/-- All primitive, derivative, continuity, positive-temperature and strict
SOPD hypotheses are jointly realized by the actual pinned Dirac witness:
`φ=id`, `σ=1/2`, `σ'=0`, one neuron opposite the atom, and `β=1`. -/
example : (0 : ℝ) < 1 ∧
    (∀ r : ℝ, HasDerivAt (fun t : ℝ => t) (2 * (fun _ : ℝ => (2 : ℝ)⁻¹) r) r) ∧
    (∀ r : ℝ, HasDerivAt (fun _ : ℝ => (2 : ℝ)⁻¹) ((0 : ℝ → ℝ) r) r) ∧
    Continuous (0 : ℝ → ℝ) ∧
    IsStrictSOPD 1 (fun t => t) (fun _ => (2 : ℝ)⁻¹) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) := by
  exact ⟨one_pos, fun r => by simpa using hasDerivAt_id' (𝕜 := ℝ) (x := r),
    fun r => by simpa using hasDerivAt_const r ((2 : ℝ)⁻¹), continuous_const,
    isStrictSOPD_pin 1 0 (basePoint 1)⟩

/-- The source's analytic activation has a continuous actual derivative,
so the stronger local result gives finite topological support.
Source: arXiv:2601.21366v2, §3.1, `thm: circle.gelu`, and §5.2.
No global cancellation or finite-support hypothesis is needed. -/
theorem finite_support_of_strictSOPD_analytic (β : ℝ) (hβ : 0 < β) (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s)
    (hσ : AnalyticOnNhd ℝ σ Set.univ) (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (μ : Perspective.ProbSphere 2) (hμ : IsStrictSOPD β φ σ ω a μ) :
    (μ : Measure (SSphere 2)).support.Finite := by
  have hs1 : ContDiff ℝ 1 σ := hσ.contDiff
  exact finite_support_of_strictSOPD_continuous_derivative β hβ φ σ (deriv σ) hφ
    (fun r => (hs1.differentiable (by norm_num)).differentiableAt.hasDerivAt)
    hs1.continuous_deriv_one ω a μ hμ

/-- **Theorem (thm: circle.gelu).** Let `d = 2`, `β > 0`, fix weights `ϑ` and
a real-analytic `σ : ℝ → ℝ`. If `μ ∈ P(𝕊¹)` is a strict SOPD Wasserstein
critical point of `E_{β,ϑ}` in the sense of `eq:strict2order`, then `μ` is
purely atomic and has finite support.

Equivalently: a SOPD critical point with infinite support is degenerate.
The source statement is unchanged. The proof uses the stronger local
curvature argument above, whose actual derivative and Hessian estimates
already apply to continuously differentiable activations.

Source: arXiv:2601.21366v2, §3.1, `thm: circle.gelu`, and §5.2. -/
theorem circle_gelu_isFinitelyAtomic (β : ℝ) (hβ : 0 < β) (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s)
    (hσ : AnalyticOnNhd ℝ σ Set.univ) (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (μ : Perspective.ProbSphere 2) (hμ : IsStrictSOPD β φ σ ω a μ) :
    IsFinitelyAtomic μ :=
  isFinitelyAtomic_of_finite_topological_support μ
    (finite_support_of_strictSOPD_analytic β hβ φ σ hφ hσ ω a μ hμ)

/-- The hypotheses of both analytic conclusions are jointly satisfiable:
`β = 1`, `φ = id`, the real-analytic constant activation `σ ≡ 1/2`, and the
pinned Dirac mass of `isStrictSOPD_pin`, with strict SOPD constant `κ = 1/2`.
The same primitive and actual measure realize every analytic assumption. -/
example :
    (0 : ℝ) < 1 ∧ (∀ s : ℝ, HasDerivAt (fun t : ℝ => t) (2 * (fun _ : ℝ => (2 : ℝ)⁻¹) s) s) ∧
      AnalyticOnNhd ℝ (fun _ : ℝ => (2 : ℝ)⁻¹) Set.univ ∧
      IsStrictSOPD 1 (fun s => s) (fun _ => (2 : ℝ)⁻¹) (Pi.single 0 (1 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
        (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨one_pos, fun s => by simpa using hasDerivAt_id' (𝕜 := ℝ) (x := s), fun _ _ => analyticAt_const,
    isStrictSOPD_pin 1 0 (basePoint 1)⟩

end Transformer.Perceptron

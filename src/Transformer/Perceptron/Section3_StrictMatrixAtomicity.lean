module

public import Transformer.Perceptron.Section5_MapExpandedHessian
public import Transformer.Perceptron.Section5_LocalHessianBound
public import Transformer.Perceptron.Section5_Accumulation

/-!
# Finite strict critical support for symmetric matrix attention on the circle

ArXiv:2601.21366v2, §3.1, `rem: ext` (ii), extending
`thm: circle.gelu`, and the localization proof of §5.2.
The source assumes an invertible symmetric matrix and an analytic
activation. The proved local argument requires only symmetry of
the matrix and a continuous actual activation derivative.

Stationarity makes the actual scalar force zero at all support angles.
Infinite support gives an accumulation point where its actual derivative
vanishes. The proved expanded Hessian and two localized smooth gradients
then contradict the strict positive lower bound on the same energy.
Neither a finite measure presentation nor a Hessian identity is assumed.

The mixed coefficient need not depend only on the angular difference.
Local continuity supplies its estimate without rotational invariance.
The analytic corollaries use the activation's actual derivative.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology

namespace Transformer.Perceptron

/-- Strict SOPD for the actual symmetric matrix energy forces finite
topological support when the actual activation derivative is continuous.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), extending
`thm: circle.gelu`, and §5.2. This strengthens the source by removing
matrix invertibility and replacing activation analyticity by `C¹`.
The proof retains the actual diagonal and mixed matrix coefficients. -/
theorem finite_support_of_strictSOPDMap_continuous_derivative
    (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (hsymm : B.IsSymmetric)
    (φ σ σ' : ℝ → ℝ) (hφ : ∀ r, HasDerivAt φ (2 * σ r) r)
    (hσ : ∀ r, HasDerivAt σ (σ' r) r) (hσ' : Continuous σ')
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (hμ : IsStrictSOPDMap B φ σ ω a μ) : (μ : Measure (SSphere 2)).support.Finite := by
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
  let A : SSphere 2 → ℝ := circleCurvatureMap B σ σ' ω a μ
  let K : SSphere 2 × SSphere 2 → ℝ := fun z => -circleMapPairMixedCurvature B z.1 z.2
  let g : ℝ → ℝ := fun θ => circleForceMap B σ ω a μ (circlePoint θ)
  have hσc : Continuous σ := continuous_iff_continuousAt.mpr fun r => (hσ r).continuousAt
  have hA : Continuous A := continuous_circleCurvatureMap B σ σ' hσc hσ' ω a μ
  have hK : Continuous K := (continuous_circleMapPairMixedCurvature B).neg
  have hzero : ∀ q ∈ S, g q = 0 :=
    fun q hq => circleForceMap_zero_on_support B σ ω a μ hμ.1 (circlePoint q) hq.2
  have hAp : A (circlePoint p) = 0 := derivative_zero_at_accPt_of_zeros
    g S p (A (circlePoint p)) hacc hzero (hasDerivAt_circleForceMap B σ σ' hσ ω a μ p)
  obtain ⟨κ, hκ, hH⟩ := strict_circle_map_hessian_coercive_expanded
    B hsymm φ σ σ' hφ hσ hσ' ω a μ hμ
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

/-- The stronger matrix support theorem supplies the original measure's
actual finite Dirac presentation. Source: arXiv:2601.21366v2, §3.1,
`rem: ext` (ii), extending `thm: circle.gelu`, and §5.2.
Symmetry and a continuous actual activation derivative suffice. -/
theorem isFinitelyAtomic_of_strictSOPDMap_continuous_derivative
    (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (hsymm : B.IsSymmetric)
    (φ σ σ' : ℝ → ℝ) (hφ : ∀ r, HasDerivAt φ (2 * σ r) r)
    (hσ : ∀ r, HasDerivAt σ (σ' r) r) (hσ' : Continuous σ')
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (hμ : IsStrictSOPDMap B φ σ ω a μ) : IsFinitelyAtomic μ :=
  isFinitelyAtomic_of_finite_topological_support μ
    (finite_support_of_strictSOPDMap_continuous_derivative B hsymm φ σ σ' hφ hσ hσ' ω a μ hμ)

/-- Every primitive, actual derivative, continuity, symmetry and strict
SOPD input is jointly realized by the actual pinned Dirac at `B=id`,
with `φ=id`, constant activation one half and derivative zero. -/
example : (LinearMap.id (R := ℝ) (M := EucSpace 2)).IsSymmetric ∧
    (∀ r : ℝ, HasDerivAt (fun t : ℝ => t) (2 * (fun _ : ℝ => (2 : ℝ)⁻¹) r) r) ∧
    (∀ r : ℝ, HasDerivAt (fun _ : ℝ => (2 : ℝ)⁻¹) ((0 : ℝ → ℝ) r) r) ∧
    Continuous (0 : ℝ → ℝ) ∧
    IsStrictSOPDMap (LinearMap.id (R := ℝ) (M := EucSpace 2)) (fun t => t)
      (fun _ => (2 : ℝ)⁻¹) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨fun _ _ => rfl, fun r => by simpa using hasDerivAt_id' (𝕜 := ℝ) (x := r),
    fun r => by simpa using hasDerivAt_const r ((2 : ℝ)⁻¹), continuous_const,
    (isStrictSOPDMap_id_iff _ _ _ _ _).mpr (isStrictSOPD_pin 1 0 _)⟩

/-- Analytic activation supplies the actual continuous derivative needed
by the stronger matrix support result. Source: arXiv:2601.21366v2,
§3.1, `rem: ext` (ii), extending `thm: circle.gelu`, and §5.2.
Matrix invertibility is not needed for this strengthened conclusion. -/
theorem finite_support_of_strictSOPDMap_analytic (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (hsymm : B.IsSymmetric) (φ σ : ℝ → ℝ)
    (hφ : ∀ r, HasDerivAt φ (2 * σ r) r) (hσ : AnalyticOnNhd ℝ σ Set.univ)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (hμ : IsStrictSOPDMap B φ σ ω a μ) : (μ : Measure (SSphere 2)).support.Finite := by
  have hs1 : ContDiff ℝ 1 σ := hσ.contDiff
  exact finite_support_of_strictSOPDMap_continuous_derivative B hsymm φ σ (deriv σ) hφ
    (fun r => (hs1.differentiable (by norm_num)).differentiableAt.hasDerivAt)
    hs1.continuous_deriv_one ω a μ hμ

/-- The source's analytic matrix activation yields its actual finite
Dirac conclusion. Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii),
extending `thm: circle.gelu`, and §5.2. Symmetry alone suffices for `B`. -/
theorem isFinitelyAtomic_of_strictSOPDMap_analytic (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (hsymm : B.IsSymmetric) (φ σ : ℝ → ℝ)
    (hφ : ∀ r, HasDerivAt φ (2 * σ r) r) (hσ : AnalyticOnNhd ℝ σ Set.univ)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (hμ : IsStrictSOPDMap B φ σ ω a μ) : IsFinitelyAtomic μ :=
  isFinitelyAtomic_of_finite_topological_support μ
    (finite_support_of_strictSOPDMap_analytic B hsymm φ σ hφ hσ ω a μ hμ)

/-- The source's original analytic and invertible-matrix inputs hold
jointly at `B=id` for the same actual pinned Dirac and primitive.
This realizes the analytic corollaries without weakening strictness. -/
example : (LinearMap.id (R := ℝ) (M := EucSpace 2)).IsSymmetric ∧
    Function.Bijective (LinearMap.id (R := ℝ) (M := EucSpace 2)) ∧
    (∀ r : ℝ, HasDerivAt (fun t : ℝ => t) (2 * (fun _ : ℝ => (2 : ℝ)⁻¹) r) r) ∧
    AnalyticOnNhd ℝ (fun _ : ℝ => (2 : ℝ)⁻¹) Set.univ ∧
    IsStrictSOPDMap (LinearMap.id (R := ℝ) (M := EucSpace 2)) (fun t => t)
      (fun _ => (2 : ℝ)⁻¹) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨fun _ _ => rfl, Function.bijective_id,
    fun r => by simpa using hasDerivAt_id' (𝕜 := ℝ) (x := r), fun _ _ => analyticAt_const,
    (isStrictSOPDMap_id_iff _ _ _ _ _).mpr (isStrictSOPD_pin 1 0 _)⟩

end Transformer.Perceptron

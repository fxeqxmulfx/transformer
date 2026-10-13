/-
# Actual normalized stationarity in circle coordinates

ArXiv:2601.21366v2, §2.3, `eq: fulltrans.stat`, and §6,
proof of `prop: unified.log` and `rem:unifiedlog-B`.
The angular derivative of the original weight is the tangent
component of the actual vector integral `attentionGradMap`.
The perceptron branch is the tangent component of the actual drift.

Consequently stationarity makes the analytic multiplied branch force
zero at every actual support angle with that pattern. No identity
between force and stationarity is assumed as a hypothesis.
Activation-boundary terms retain their actual zero contributions.
The equation uses the original support, before any atomicity conclusion.
The final support lemma transports support points through a continuous
measurable map. It will carry an actual support point to its antipode
once evenness of the invertible transform supplies measure invariance.
-/

import Transformer.Perceptron.Section6_BranchCalculus
import Transformer.Perceptron.Section5_MapForce

open Real MeasureTheory Set
open scoped BigOperators Topology

namespace Transformer.Perceptron

/-- The angular derivative of the original weight is its actual
tangent vector-gradient component. Source: arXiv:2601.21366v2,
§2.3, `eq: fulltrans.stat`, and §6, `rem:unifiedlog-B`.
Compact continuity justifies passing the inner product through the integral. -/
theorem deriv_circleAngularWeightMap_eq_inner_attentionGradMap
    (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (μ : Perspective.ProbSphere 2) (θ : ℝ) :
    deriv (circleAngularWeightMap B μ) θ =
      inner (𝕜 := ℝ) (circleVel θ) (attentionGradMap B μ (circlePoint θ : EucSpace 2)) := by
  let F : SSphere 2 → EucSpace 2 := fun y => circleMapPairKernel B (circlePoint θ) y •
    proj 2 (circlePoint θ : EucSpace 2) (B (y : EucSpace 2))
  have hFC : Continuous F := by
    have hB := B.continuous_of_finiteDimensional
    have hK : Continuous (fun y : SSphere 2 => circleMapPairKernel B (circlePoint θ) y) :=
      (continuous_circleMapPairKernel B).comp (continuous_const.prodMk continuous_id)
    dsimp only [F, proj]
    fun_prop
  have hFI : Integrable F (μ : Measure (SSphere 2)) :=
    hFC.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  rw [(hasDerivAt_circleAngularWeightMap B μ θ).deriv]
  change (∫ y, circleMapPairForce B (circlePoint θ) y ∂(μ : Measure (SSphere 2))) =
    inner (𝕜 := ℝ) (circleVel θ) (∫ y, F y ∂(μ : Measure (SSphere 2)))
  rw [← integral_inner hFI]
  apply integral_congr_ae
  exact ae_of_all _ fun y => by
    rw [← circleTangent_circlePoint]
    simp only [F, real_inner_smul_right, inner_circleTangent_proj,
      circleMapPairForce, circleMapPairKernel]

/-- The original perceptron force is the actual tangent drift component.
Source: arXiv:2601.21366v2, §2.3, `eq: primitive.field`,
and §6, the gradient of the branch potential in `H_log`. -/
theorem circlePotentialForce_eq_inner_drift (σ : ℝ → ℝ) (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (x : SSphere 2) :
    circlePotentialForce σ ω a x = inner (𝕜 := ℝ) (circleTangent (x : EucSpace 2))
      (drift σ ω a (x : EucSpace 2)) := by
  rw [drift, inner_circleTangent_proj, inner_sum]
  unfold circlePotentialForce
  apply Finset.sum_congr rfl
  intro j _
  rw [real_inner_smul_right, real_inner_comm (a j) (circleTangent (x : EucSpace 2))]
  ring

/-- At an actual angle with the supplied pattern, the globally defined
branch force is the tangent component of the original normalized field.
Source: arXiv:2601.21366v2, §6, the support inclusion `{g_log=0}`,
with its positive weight multiplied through as in `eq: fulltrans.stat`. -/
theorem circleNormalizedBranchForce_eq_inner_field (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (J : Finset (Idx 2)) (θ : ℝ)
    (hθ : IsActivationPattern a J (circlePoint θ : EucSpace 2)) :
    circleNormalizedBranchForce B ω a μ J θ = inner (𝕜 := ℝ) (circleVel θ)
      (attentionGradMap B μ (circlePoint θ : EucSpace 2) +
        attentionWeightMap B μ (circlePoint θ : EucSpace 2) •
          drift (fun s => max s 0) ω a (circlePoint θ : EucSpace 2)) := by
  rw [circleNormalizedBranchForce, deriv_circleAngularWeightMap_eq_inner_attentionGradMap,
    ← circlePotentialForce_relu_eq_branch ω a J θ hθ,
    circlePotentialForce_eq_inner_drift, circleTangent_circlePoint,
    inner_add_right, real_inner_smul_right]
  rfl

/-- Normalized stationarity makes the actual branch force zero at every
actual support angle with that pattern. Source: arXiv:2601.21366v2,
§6, the support inclusion used in part (b) of `prop: unified.log`. -/
theorem circleNormalizedBranchForce_zero_on_support (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (hμ : IsNormalizedStationaryMap B (fun s => max s 0) ω a μ)
    (J : Finset (Idx 2)) (θ : ℝ)
    (hx : circlePoint θ ∈ (μ : Measure (SSphere 2)).support)
    (hθ : IsActivationPattern a J (circlePoint θ : EucSpace 2)) :
    circleNormalizedBranchForce B ω a μ J θ = 0 := by
  rw [circleNormalizedBranchForce_eq_inner_field B ω a μ J θ hθ,
    hμ (circlePoint θ) hx, inner_zero_right]

/-- A continuous measurable sphere map carries support points into
the support of the actual pushforward. Source: arXiv:2601.21366v2,
§6, passing from antipodal invariance of the measure to its support.
The proof uses positive measure of each actual open neighborhood. -/
theorem mem_support_map_of_continuous_circle (μ : Measure (SSphere 2))
    (f : SSphere 2 → SSphere 2) (hc : Continuous f) (hm : Measurable f)
    (x : SSphere 2) (hx : x ∈ μ.support) : f x ∈ (μ.map f).support := by
  rw [Measure.support_eq_forall_isOpen]
  intro U hxU hU
  rw [Measure.map_apply hm hU.measurableSet]
  exact (Measure.mem_support_iff_forall x).mp hx _ ((hU.preimage hc).mem_nhds hxU)

/-- The actual opposite neuron is inactive at the point it pins, and
all zero neurons lie on their activation boundaries.
Source: arXiv:2601.21366v2, §6, the active branch in `H_log`.
This realizes the pattern input of the stationary Dirac witness. -/
theorem isActivationPattern_single_neg_empty {d : ℕ} (j₀ : Idx d) (x : EucSpace d) :
    IsActivationPattern (Pi.single j₀ (-x)) ∅ x := by
  classical
  constructor
  · intro j hj
    simp at hj
  · intro j _
    by_cases h : j = j₀
    · subst j
      rw [Pi.single_eq_same, inner_neg_left]
      exact neg_nonpos.mpr real_inner_self_nonneg
    · rw [Pi.single_eq_of_ne h, inner_zero_left]

/-- All pattern, normalized stationarity and support hypotheses hold
jointly for the actual pinned Dirac at angle zero, with empty active set.
The continuous measurable support-map inputs hold for the identity at
that same actual point. Source: arXiv:2601.21366v2, §6,
`eq: fulltrans.stat` and the support argument of `prop: unified.log`. -/
example : IsNormalizedStationaryMap (LinearMap.id (R := ℝ) (M := EucSpace 2))
      (fun s => max s 0) (Pi.single 0 1) (Pi.single 0 (-(circlePoint 0 : EucSpace 2)))
      (Perspective.diracProb 2 (circlePoint 0)) ∧
    circlePoint 0 ∈ (Perspective.diracProb 2 (circlePoint 0) : Measure (SSphere 2)).support ∧
    IsActivationPattern (Pi.single 0 (-(circlePoint 0 : EucSpace 2))) ∅
      (circlePoint 0 : EucSpace 2) ∧
    Continuous (id : SSphere 2 → SSphere 2) ∧ Measurable (id : SSphere 2 → SSphere 2) := by
  refine ⟨isNormalizedStationaryMap_relu_pin 0 (circlePoint 0), ?_, ?_, continuous_id, measurable_id⟩
  · apply (Measure.mem_support_iff_forall _).mpr
    intro U hU
    have hx : circlePoint 0 ∈ U := mem_of_mem_nhds hU
    change 0 < Measure.dirac (circlePoint 0) U
    simp [hx]
  · exact isActivationPattern_single_neg_empty 0 (circlePoint 0 : EucSpace 2)

end Transformer.Perceptron

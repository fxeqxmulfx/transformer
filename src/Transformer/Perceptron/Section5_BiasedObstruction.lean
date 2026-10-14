/-
# Non-analytic original biased potentials obstruct zero branches

ArXiv:2601.21366v2, §3.1, `rem: ext` (i), with the contradiction
after §5.3, `eq:global-Id`. The original first variation contains the
original exponential partition integral and the original biased potential.
Its actual derivative and projected field are proved below.

If the original biased stationary probability has full support, this
actual first variation is constant on the original sphere. Its actual
expression then supplies an ambient-analytic extension of the same
original biased perceptron potential. The source's non-analyticity
hypothesis therefore excludes full support.

A globally zero original biased branch would give full support by the
proved affine-quadratic inverse. Thus it is excluded for the original
non-analytic biased potential. The original arbitrary primitive of twice
ReLU, every original bias, and the original stationary probability remain.
The actual supplied sphere point fixes the source's constant value.
-/

import Transformer.Perceptron.Section5_BiasedTransform
import Transformer.Perceptron.Section5_ReLUObstruction

open Real MeasureTheory Set
open scoped BigOperators

namespace Transformer.Perceptron

/-- The actual original biased first variation on the original ambient
space. Source: arXiv:2601.21366v2, §2.3 and §3.1, `rem: ext` (i).
The original probability, primitive, coefficients and biases are retained. -/
noncomputable def biasedFirstVariation {d : ℕ} (β : ℝ) (φ : ℝ → ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (b : Idx d → ℝ) (μ : Perspective.ProbSphere d) (x : EucSpace d) : ℝ :=
  β⁻¹ * Perspective.partitionMu d β μ x + (2 : ℝ)⁻¹ * biasedPotential φ ω a b x

/-- The actual original ambient gradient of the original biased first
variation. Source: arXiv:2601.21366v2, §2.3 and §3.1, `rem: ext` (i).
The proof copies `hasGradientAt_firstVariation`, using the proved original
biased-potential gradient with the actual unchanged biases. -/
theorem hasFDerivAt_biasedFirstVariation {d : ℕ} (β : ℝ) (hβ : β ≠ 0) (φ σ : ℝ → ℝ)
    (hφ : ∀ s, HasDerivAt φ (2 * σ s) s) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (b : Idx d → ℝ) (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    HasFDerivAt (biasedFirstVariation β φ ω a b μ)
      (innerSL ℝ ((∫ y, exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • (y : EucSpace d)
        ∂(μ : Measure (SSphere d))) + ∑ j, (ω j * σ (inner (𝕜 := ℝ) (a j) x + b j)) • a j)) x := by
  have hZ := hasGradientAt_iff_hasFDerivAt.mp (Perspective.hasGradientAt_partitionMu d β μ x)
  have hv := hasGradientAt_iff_hasFDerivAt.mp (hasGradientAt_biasedPotential φ σ hφ ω a b x)
  refine ((hZ.const_mul β⁻¹).fun_add (hv.const_mul (2 : ℝ)⁻¹)).congr_fderiv ?_
  ext v
  simp only [add_apply, smul_apply, InnerProductSpace.toDual_apply_apply,
    innerSL_apply_apply, smul_eq_mul, inner_add_left, real_inner_smul_left]
  field_simp

/-- The actual original biased field is the original tangent projection
of its actual ambient vector. Source: arXiv:2601.21366v2, §2.3,
`eq: steady.state`, and §3.1, `rem: ext` (i).
The proof copies `energyGrad_eq_proj` for the actual biased drift. -/
theorem biasedEnergyGrad_eq_proj {d : ℕ} (β : ℝ) (σ : ℝ → ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (b : Idx d → ℝ) (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    biasedEnergyGrad β σ ω a b μ x = proj d x
      ((∫ y, exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • (y : EucSpace d)
        ∂(μ : Measure (SSphere d))) + ∑ j, (ω j * σ (inner (𝕜 := ℝ) (a j) x + b j)) • a j) := by
  let L : EucSpace d →L[ℝ] EucSpace d := ContinuousLinearMap.id ℝ (EucSpace d) - (innerSL ℝ x).smulRight x
  have hL (v : EucSpace d) : L v = proj d x v := rfl
  have hint : (∫ y, exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • proj d x (y : EucSpace d)
      ∂(μ : Measure (SSphere d))) =
      L (∫ y, exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • (y : EucSpace d) ∂(μ : Measure (SSphere d))) := by
    rw [← L.integral_comp_comm (Perspective.integrable_expInner_smul d β μ x)]
    apply integral_congr_ae
    exact ae_of_all _ fun y => by change _ = L (_ • (y : EucSpace d)); rw [L.map_smul, hL]
  rw [biasedEnergyGrad, biasedDrift, hint, ← hL, ← hL, L.map_add]

/-- Original biased stationarity at full original support makes the
same original biased potential analytic on the actual sphere.
Source: arXiv:2601.21366v2, §5.3, the contradiction after `eq:global-Id`,
with §3.1, `rem: ext` (i). The original first variation is made constant
using its proved actual derivative and original stationarity. -/
theorem isBiasedAnalyticOnSphere_of_stationary_full_support {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : β ≠ 0) (φ σ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * σ s) s)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (b : Idx d → ℝ) (μ : Perspective.ProbSphere d)
    (hμ : IsBiasedStationary β σ ω a b μ) (hfull : (μ : Measure (SSphere d)).support = univ)
    (o : SSphere d) : IsBiasedAnalyticOnSphere φ ω a b := by
  let D := fun x : EucSpace d =>
    (∫ y, exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • (y : EucSpace d) ∂(μ : Measure (SSphere d))) +
      ∑ j, (ω j * σ (inner (𝕜 := ℝ) (a j) x + b j)) • a j
  have hp (z : EucSpace d) (hz : ‖z‖ = 1) : proj d z (D z) = 0 := by
    let x : SSphere d := ⟨z, mem_sphere_zero_iff_norm.mpr hz⟩
    have hx : x ∈ (μ : Measure (SSphere d)).support := by rw [hfull]; exact mem_univ _
    rw [← biasedEnergyGrad_eq_proj]
    exact hμ x hx
  have hconst := eq_on_sphere_of_projected_derivative_zero hd (biasedFirstVariation β φ ω a b μ) D
    (hasFDerivAt_biasedFirstVariation β hβ φ σ hφ ω a b μ) hp
  let C := biasedFirstVariation β φ ω a b μ (o : EucSpace d)
  refine ⟨fun z => 2 * (C - β⁻¹ * Perspective.partitionMu d β μ z), ?_, fun z hz => ?_⟩
  · intro z _
    exact analyticAt_const.fun_mul (analyticAt_const.fun_sub
      (analyticAt_const.fun_mul (analyticOnNhd_partitionMu_original β μ z (mem_univ z))))
  · have h := hconst (⟨z, mem_sphere_zero_iff_norm.mpr hz⟩ : SSphere d) o
    change β⁻¹ * Perspective.partitionMu d β μ z + (2 : ℝ)⁻¹ * biasedPotential φ ω a b z = C at h
    linarith

/-- The original non-analytic biased potential excludes full original
support for the original stationary probability.
Source: arXiv:2601.21366v2, §5.3, after `eq:global-Id`, with §3.1,
`rem: ext` (i). The original activation and primitive need only their
actual derivative relation, and every original bias is retained. -/
theorem not_full_support_of_nonanalytic_biased_stationary {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : β ≠ 0) (φ σ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * σ s) s)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (b : Idx d → ℝ)
    (hana : ¬ IsBiasedAnalyticOnSphere φ ω a b) (μ : Perspective.ProbSphere d)
    (hμ : IsBiasedStationary β σ ω a b μ) (o : SSphere d) : (μ : Measure (SSphere d)).support ≠ univ := by
  intro hfull
  exact hana (isBiasedAnalyticOnSphere_of_stationary_full_support hd β hβ φ σ hφ ω a b μ hμ hfull o)

/-- A globally zero original biased branch is excluded by the original
non-analytic biased potential and original biased stationarity.
Source: arXiv:2601.21366v2, §5.3, after `eq:global-Id`, with §3.1,
`rem: ext` (i). The proved affine-quadratic inverse first forces full
support; the original arbitrary primitive then gives the contradiction. -/
theorem not_zero_projected_biased_branch_of_nonanalytic {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : 0 < β) (φ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * max s 0) s)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (b : Idx d → ℝ)
    (hana : ¬ IsBiasedAnalyticOnSphere φ ω a b) (μ : Perspective.ProbSphere d)
    (hμ : IsBiasedStationary β (fun s => max s 0) ω a b μ) (J : Finset (Idx d)) (o : SSphere d) :
    ¬ ∀ x : SSphere d, proj d (x : EucSpace d) (biasedAttentionVector β ω a b μ J (x : EucSpace d)) = 0 := by
  intro hzero
  have hfull := support_eq_univ_of_zero_projected_biased_branch hd β hβ ω a b μ J o hzero
  exact hana (isBiasedAnalyticOnSphere_of_stationary_full_support hd β hβ.ne' φ
    (fun s => max s 0) hφ ω a b μ hμ hfull o)

/-- All full-support, stationary and arbitrary-primitive inputs hold
jointly for the original uniform circle probability and actual zero
weights. Source: arXiv:2601.21366v2, §3.1, `rem: ext` (i), and §5.3. -/
example : ∃ μ : Perspective.ProbSphere 2, (2 : ℕ) ≤ 2 ∧ (1 : ℝ) ≠ 0 ∧
    (∀ s, HasDerivAt (fun t : ℝ => max t 0 ^ 2) (2 * max s 0) s) ∧
    IsBiasedStationary 1 (fun s => max s 0) (0 : Idx 2 → ℝ) 0 0 μ ∧
    (μ : Measure (SSphere 2)).support = univ := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  exact ⟨μ, le_rfl, one_ne_zero, hasDerivAt_reluSq,
    (isBiasedStationary_zero_iff 1 _ 0 0 μ).mpr (isStationary_zero_weights_of_uniform 1 one_ne_zero _ 0 μ hν),
    support_eq_univ_of_uniform μ hν⟩

/-- The source's pinned original Dirac gives every non-analytic biased
obstruction input jointly, with the actual original zero-bias primitive.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (i), and §5.3. -/
example : (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≠ 0 ∧
    (∀ s, HasDerivAt (fun t : ℝ => max t 0 ^ 2) (2 * max s 0) s) ∧
    ¬ IsBiasedAnalyticOnSphere (fun s => max s 0 ^ 2) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-(basePoint 1 : EucSpace 2))) 0 ∧
    IsBiasedStationary 1 (fun s => max s 0) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-(basePoint 1 : EucSpace 2))) 0 (Perspective.diracProb 2 (basePoint 1)) := by
  refine ⟨le_rfl, one_pos, one_ne_zero, hasDerivAt_reluSq, ?_, ?_⟩
  · rw [isBiasedAnalyticOnSphere_zero_iff]
    exact not_isAnalyticOnSphere_relu 0 norm_basePoint_one norm_secondAxis inner_basePoint_secondAxis
  · rw [isBiasedStationary_zero_iff]
    exact isStationary_relu_pin 1 0 (basePoint 1)

end Transformer.Perceptron

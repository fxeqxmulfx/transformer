/-
# Perceptrons and attention's mean-field landscape — normalized attention with
  a general matrix

The vocabulary of `rem:unifiedlog-B` of arXiv:2601.21366v2: normalized
attention for the energy `E_B[μ] = (1/2)∬ e^{xᵀBy} dμ dμ` of
`GeneralAttention.lean`, whose first variation is
`δE_B/δμ[μ](x) = ∫ e^{xᵀBy} dμ(y)` — the weight `w_B[μ]`, with no prefactor,
because the `1/2` and the symmetry of the kernel cancel.

**Consistency with the isotropic case.**  `E_{β•id} = β E_β`
(`interactionEnergyMap_smul_id`), so the weight of `E_{β•id}` is `β` times the
weight `β⁻¹Z_{β,μ}` of `E_β`, which is `Z_{β,μ}` itself:
`attentionWeightMap_smul_id`.  The stationarity conditions then agree with no
reweighting of `ω` — `isNormalizedStationaryMap_smul_id_iff` — because the
factor `β` multiplies the attention field and its weight alike.  That is
unlike the unnormalized case, where the same factor multiplies only the
interaction half of the coupled energy and `isStationaryMap_smul_id_iff`
carries `β⁻¹ω`.

Source: arXiv:2601.21366v2, `rem:unifiedlog-B`.
-/

import Transformer.Perceptron.Normalized
import Transformer.Perceptron.GeneralAttention

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### The weight and the attention field of `E_B` -/

/-- **The weight `w_B[μ] = δE_B/δμ[μ]`**: `∫ e^{xᵀBy} dμ(y)`.

Source: arXiv:2601.21366v2, `rem:unifiedlog-B`. -/
noncomputable def attentionWeightMap (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (x : EucSpace d) : ℝ :=
  ∫ y, Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) ∂(μ : Measure (SSphere d))

/-- **The attention half of `energyGradMap`**: `∫ e^{xᵀBy} Proj_x(By) dμ(y)`,
the spherical gradient of `w_B[μ]`.

Source: arXiv:2601.21366v2, `rem:unifiedlog-B`. -/
noncomputable def attentionGradMap (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (x : EucSpace d) : EucSpace d :=
  ∫ y, Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) • proj d x (B (y : EucSpace d))
    ∂(μ : Measure (SSphere d))

/-- `∇δE_{B,ϑ}/δμ[μ] = ∇δE_B/δμ[μ] + u_ϑ`. -/
theorem energyGradMap_eq_attentionGradMap_add_drift (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (σ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (x : EucSpace d) :
    energyGradMap B σ ω a μ x = attentionGradMap B μ x + drift σ ω a x := rfl

/-- **`w_{β•id}[μ] = Z_{β,μ} = β w[μ]`.** -/
theorem attentionWeightMap_smul_id (β : ℝ) (hβ : β ≠ 0) (μ : Perspective.ProbSphere d)
    (x : EucSpace d) :
    attentionWeightMap (β • LinearMap.id) μ x = β * attentionWeight β μ x := by
  rw [attentionWeight, ← mul_assoc, mul_inv_cancel₀ hβ, one_mul, attentionWeightMap,
    Perspective.partitionMu]
  simp only [LinearMap.smul_apply, LinearMap.id_coe, id_eq, real_inner_smul_right]

/-- The hypothesis of `attentionWeightMap_smul_id` is satisfiable. -/
example : (1 : ℝ) ≠ 0 := one_ne_zero

/-- **`∇δE_{β•id}/δμ[μ] = β ∇δE_β/δμ[μ]`.** -/
theorem attentionGradMap_smul_id (β : ℝ) (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    attentionGradMap (β • LinearMap.id) μ x = β • attentionGrad β μ x := by
  rw [attentionGradMap, attentionGrad, ← integral_smul]
  refine integral_congr_ae (Filter.Eventually.of_forall fun y => ?_)
  simp only [LinearMap.smul_apply, LinearMap.id_coe, id_eq, real_inner_smul_right,
    proj_smul_right]
  rw [smul_comm]

/-! ### Stationarity -/

/-- **`eq: fulltrans.stat` for `E_B`.**  `μ` is stationary for normalized
attention with the matrix `B`:

  `∇δE_B/δμ[μ] + w_B[μ] u_ϑ = 0`  on `supp μ`.

Source: arXiv:2601.21366v2, `rem:unifiedlog-B`. -/
def IsNormalizedStationaryMap (B : EucSpace d →ₗ[ℝ] EucSpace d) (σ : ℝ → ℝ)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) : Prop :=
  ∀ x ∈ (μ : Measure (SSphere d)).support,
    attentionGradMap B μ (x : EucSpace d)
      + attentionWeightMap B μ (x : EucSpace d) • drift σ ω a (x : EucSpace d) = 0

/-- The normalized stationarity field of `E_{β•id}` is `β` times that of
`E_β`, with no reweighting of `ω`. -/
theorem normalizedField_smul_id (β : ℝ) (hβ : β ≠ 0) (σ : ℝ → ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    attentionGradMap (β • LinearMap.id) μ x
        + attentionWeightMap (β • LinearMap.id) μ x • drift σ ω a x
      = β • (attentionGrad β μ x + attentionWeight β μ x • drift σ ω a x) := by
  rw [attentionGradMap_smul_id, attentionWeightMap_smul_id β hβ, smul_add, mul_smul]

/-- The hypothesis of `normalizedField_smul_id` is satisfiable. -/
example : (1 : ℝ) ≠ 0 := one_ne_zero

/-- **Normalized stationarity for `E_{β•id}` is normalized stationarity for
`E_β`.** -/
theorem isNormalizedStationaryMap_smul_id_iff (β : ℝ) (hβ : β ≠ 0) (σ : ℝ → ℝ)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) :
    IsNormalizedStationaryMap (β • LinearMap.id) σ ω a μ ↔
      IsNormalizedStationary β σ ω a μ := by
  simp only [IsNormalizedStationaryMap, IsNormalizedStationary,
    normalizedField_smul_id β hβ, smul_eq_zero, hβ, false_or]

/-- The hypothesis of `isNormalizedStationaryMap_smul_id_iff` is
satisfiable. -/
example : (1 : ℝ) ≠ 0 := one_ne_zero

/-- At `B = id` normalized stationarity for `E_B` is the case `β = 1`. -/
theorem isNormalizedStationaryMap_id_iff (σ : ℝ → ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) :
    IsNormalizedStationaryMap (LinearMap.id (R := ℝ) (M := EucSpace d)) σ ω a μ ↔
      IsNormalizedStationary 1 σ ω a μ := by
  have h := isNormalizedStationaryMap_smul_id_iff (d := d) 1 one_ne_zero σ ω a μ
  simpa using h

/-- **`δ_x` is stationary for normalized attention with `B = id`** and the
ReLU perceptron pinned at `x`: the witness for the stationarity hypothesis of
`rem:unifiedlog-B`. -/
theorem isNormalizedStationaryMap_relu_pin (j₀ : Idx d) (x : SSphere d) :
    IsNormalizedStationaryMap (LinearMap.id (R := ℝ) (M := EucSpace d))
      (fun s => max s 0) (Pi.single j₀ (1 : ℝ)) (Pi.single j₀ (-(x : EucSpace d)))
      (Perspective.diracProb d x) :=
  (isNormalizedStationaryMap_id_iff _ _ _ _).mpr (isNormalizedStationary_relu_pin 1 j₀ x)

end Perceptron
end Transformer

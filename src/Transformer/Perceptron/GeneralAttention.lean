/-
# Perceptrons and attention's mean-field landscape — a general attention matrix

Formalization of the energy `E_B` of `rem:unifiedlog-B` of arXiv:2601.21366v2,
the vocabulary `rem: ext` (ii) is written in: the interaction energy carries a
symmetric invertible matrix `B` in place of `β I_d`.

**What the source says and what is carried here.**

* `E_B[μ] = (1/2) ∬ e^{xᵀ B y} dμ(x) dμ(y)` is the source's own definition in
  `rem:unifiedlog-B`; `B` is a linear map of `ℝ^d`, as in
  `rem: general-attention` and `Transform.lean`, and the isotropic case is
  `B = β • id`.

* The source's `E_β` carries the prefactor `(2β)⁻¹` and `E_B` the prefactor
  `1/2`, so the two agree only up to the factor `β`:
  `interactionEnergyMap_smul_id`.  In the *coupled* energy that factor is not
  a harmless rescaling — it multiplies the interaction half and not the
  potential half — and the honest statement of what it does is
  `energyMap_smul_id`: the coupled `E_B` at `B = β • id` with weights `ω` is
  `β` times the source's `E_{β,ϑ}` with weights `β⁻¹ω`.  Stationarity is
  therefore the same relation up to that reweighting,
  `isStationaryMap_smul_id_iff`, and at `B = id` the two energies are equal on
  the nose.

* `energyGradMap` is the Wasserstein gradient of the first variation of
  `energyMap` *for symmetric `B`*: it is there that
  `δE_B/δμ[μ](x) = ∫ e^{xᵀBy} dμ(y)`, with ambient gradient
  `∫ e^{xᵀBy} By dμ(y)`.  Every statement of `rem: ext` (ii) carries
  `B.IsSymmetric`, so nothing below uses it outside that case.

Source: arXiv:2601.21366v2, `rem: ext` (ii), `rem:unifiedlog-B`,
`rem: general-attention`.
-/

import Transformer.Perceptron.Geodesic

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### The energy `E_B` -/

/-- **`E_B[μ] = (1/2) ∬ e^{xᵀ B y} dμ(x) dμ(y)`**, the interaction energy of a
general symmetric attention matrix.

Source: arXiv:2601.21366v2, `rem:unifiedlog-B`. -/
noncomputable def interactionEnergyMap (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) : ℝ :=
  (2 : ℝ)⁻¹ *
    ∫ x, ∫ y, Real.exp (inner (𝕜 := ℝ) (x : EucSpace d) (B (y : EucSpace d)))
      ∂(μ : Measure (SSphere d)) ∂(μ : Measure (SSphere d))

/-- **The coupled energy with a general attention matrix**,

  `E_{B,ϑ}[μ] = E_B[μ] + (1/2) ∫ v_ϑ dμ`.

Source: arXiv:2601.21366v2, `rem: ext` (ii), `rem:unifiedlog-B`. -/
noncomputable def energyMap (B : EucSpace d →ₗ[ℝ] EucSpace d) (φ : ℝ → ℝ)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) : ℝ :=
  interactionEnergyMap B μ +
    (2 : ℝ)⁻¹ * ∫ x, potential φ ω a (x : EucSpace d) ∂(μ : Measure (SSphere d))

/-- **The Wasserstein gradient of the first variation of `E_{B,ϑ}`**, for
symmetric `B`:

  `∇ δE_{B,ϑ}/δμ[μ](x) = ∫ e^{xᵀBy} Proj_x(By) dμ(y) + u_ϑ(x)`.

Source: arXiv:2601.21366v2, `rem: ext` (ii), `rem:unifiedlog-B`. -/
noncomputable def energyGradMap (B : EucSpace d →ₗ[ℝ] EucSpace d) (σ : ℝ → ℝ)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (x : EucSpace d) : EucSpace d :=
  (∫ y, Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) • proj d x (B (y : EucSpace d))
      ∂(μ : Measure (SSphere d))) + drift σ ω a x

/-- **`eq: steady.state` for `E_{B,ϑ}`.**

Source: arXiv:2601.21366v2, `rem: ext` (ii). -/
def IsStationaryMap (B : EucSpace d →ₗ[ℝ] EucSpace d) (σ : ℝ → ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) : Prop :=
  ∀ x ∈ (μ : Measure (SSphere d)).support,
    energyGradMap B σ ω a μ (x : EucSpace d) = 0

/-- **`eq:2order` for `E_{B,ϑ}`.**

Source: arXiv:2601.21366v2, `rem: ext` (ii), `eq:2order`. -/
def IsSOPDMap (B : EucSpace d →ₗ[ℝ] EucSpace d) (φ σ : ℝ → ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) : Prop :=
  IsStationaryMap B σ ω a μ ∧
    ∀ ξ : SSphere d → EucSpace d, IsGradientField ξ →
      ∀ ν : ℝ → Perspective.ProbSphere d, IsGeodesicFrom ξ μ ν →
        ∀ H : ℝ, HasDerivAt (deriv fun t => energyMap B φ ω a (ν t)) H 0 → 0 ≤ H

/-- **`eq:strict2order` for `E_{B,ϑ}`.**

Source: arXiv:2601.21366v2, `rem: ext` (ii), `eq:strict2order`. -/
def IsStrictSOPDMap (B : EucSpace d →ₗ[ℝ] EucSpace d) (φ σ : ℝ → ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) : Prop :=
  IsStationaryMap B σ ω a μ ∧
    ∃ κ : ℝ, 0 < κ ∧
      ∀ ξ : SSphere d → EucSpace d, IsGradientField ξ →
        ∀ ν : ℝ → Perspective.ProbSphere d, IsGeodesicFrom ξ μ ν →
          ∀ H : ℝ, HasDerivAt (deriv fun t => energyMap B φ ω a (ν t)) H 0 →
            κ * ∫ x, ‖ξ x‖ ^ 2 ∂(μ : Measure (SSphere d)) ≤ H

/-! ### Homogeneity in the weights -/

/-- The potential is linear in the output weights. -/
theorem potential_smul (c : ℝ) (φ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (x : EucSpace d) : potential φ (c • ω) a x = c * potential φ ω a x := by
  simp only [potential, Pi.smul_apply, smul_eq_mul, Finset.mul_sum, mul_assoc]

/-- `Proj_x` is homogeneous in the vector projected. -/
theorem proj_smul_right (c : ℝ) (x y : EucSpace d) :
    proj d x (c • y) = c • proj d x y := by
  simp only [proj, real_inner_smul_right, smul_sub, smul_smul]

/-- The drift is linear in the output weights. -/
theorem drift_smul (c : ℝ) (σ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (x : EucSpace d) : drift σ (c • ω) a x = c • drift σ ω a x := by
  have hsum : ∑ j : Idx d, ((c • ω) j * σ (inner (𝕜 := ℝ) (a j) x)) • a j
      = c • ∑ j : Idx d, (ω j * σ (inner (𝕜 := ℝ) (a j) x)) • a j := by
    rw [Finset.smul_sum]
    exact Finset.sum_congr rfl fun j _ => by
      simp only [Pi.smul_apply, smul_eq_mul, smul_smul, mul_assoc]
  rw [drift, drift, hsum, proj_smul_right]

/-! ### The isotropic case -/

/-- `E_B` at `B = β • id` is `β E_β`: the source's two prefactors, `1/2` and
`(2β)⁻¹`, differ by exactly that factor. -/
theorem interactionEnergyMap_smul_id (β : ℝ) (hβ : β ≠ 0) (μ : Perspective.ProbSphere d) :
    interactionEnergyMap (β • LinearMap.id) μ = β * Perspective.interactionEnergy d β μ := by
  rw [interactionEnergyMap, Perspective.interactionEnergy, ← mul_assoc]
  congr 1
  · field_simp
  · simp only [LinearMap.smul_apply, LinearMap.id_coe, id_eq, real_inner_smul_right]

/-- The hypothesis of `interactionEnergyMap_smul_id` is satisfiable: `β = 1`. -/
example : (1 : ℝ) ≠ 0 := one_ne_zero

/-- **The coupled `E_B` at `B = β • id`** is `β` times the source's coupled
energy at the reweighted perceptron `β⁻¹ϑ`. -/
theorem energyMap_smul_id (β : ℝ) (hβ : β ≠ 0) (φ : ℝ → ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) :
    energyMap (β • LinearMap.id) φ ω a μ = β * energy β φ (β⁻¹ • ω) a μ := by
  rw [energyMap, energy, mul_add, interactionEnergyMap_smul_id β hβ]
  congr 1
  simp only [potential_smul, integral_const_mul]
  field_simp

/-- The hypothesis of `energyMap_smul_id` is satisfiable: `β = 1`. -/
example : (1 : ℝ) ≠ 0 := one_ne_zero

/-- The first variation's gradient at `B = β • id`, again at the reweighted
perceptron. -/
theorem energyGradMap_smul_id (β : ℝ) (hβ : β ≠ 0) (σ : ℝ → ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    energyGradMap (β • LinearMap.id) σ ω a μ x = β • energyGrad β σ (β⁻¹ • ω) a μ x := by
  rw [energyGradMap, energyGrad, smul_add, drift_smul, smul_smul, mul_inv_cancel₀ hβ, one_smul]
  congr 1
  rw [← integral_smul]
  refine integral_congr_ae (Filter.Eventually.of_forall fun y => ?_)
  simp only [LinearMap.smul_apply, LinearMap.id_coe, id_eq, real_inner_smul_right,
    proj_smul_right]
  rw [smul_comm]

/-- The hypothesis of `energyGradMap_smul_id` is satisfiable: `β = 1`. -/
example : (1 : ℝ) ≠ 0 := one_ne_zero

/-- **Stationarity for `E_B` at `B = β • id`** is the source's stationarity at
the reweighted perceptron `β⁻¹ϑ`. -/
theorem isStationaryMap_smul_id_iff (β : ℝ) (hβ : β ≠ 0) (σ : ℝ → ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) :
    IsStationaryMap (β • LinearMap.id) σ ω a μ ↔ IsStationary β σ (β⁻¹ • ω) a μ := by
  simp only [IsStationaryMap, IsStationary, energyGradMap_smul_id β hβ, smul_eq_zero, hβ,
    false_or]

/-- The hypothesis of `isStationaryMap_smul_id_iff` is satisfiable: `β = 1`. -/
example : (1 : ℝ) ≠ 0 := one_ne_zero

/-- At `B = id` the coupled `E_B` is the source's coupled energy at `β = 1`. -/
theorem energyMap_id_eq :
    energyMap (LinearMap.id (R := ℝ) (M := EucSpace d)) = energy (d := d) 1 := by
  funext φ ω a μ
  have h := energyMap_smul_id (d := d) 1 one_ne_zero φ ω a μ
  simpa using h

/-- At `B = id` stationarity for `E_B` is stationarity at `β = 1`. -/
theorem isStationaryMap_id_iff (σ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) :
    IsStationaryMap (LinearMap.id (R := ℝ) (M := EucSpace d)) σ ω a μ ↔
      IsStationary 1 σ ω a μ := by
  have h := isStationaryMap_smul_id_iff (d := d) 1 one_ne_zero σ ω a μ
  simpa using h

/-- At `B = id` the strict SOPD critical points of `E_B` are those of `E_1`. -/
theorem isStrictSOPDMap_id_iff (φ σ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) :
    IsStrictSOPDMap (LinearMap.id (R := ℝ) (M := EucSpace d)) φ σ ω a μ ↔
      IsStrictSOPD 1 φ σ ω a μ := by
  simp only [IsStrictSOPDMap, IsStrictSOPD, isStationaryMap_id_iff, energyMap_id_eq]

/-- At `B = id` the SOPD critical points of `E_B` are those of `E_1`. -/
theorem isSOPDMap_id_iff (φ σ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) :
    IsSOPDMap (LinearMap.id (R := ℝ) (M := EucSpace d)) φ σ ω a μ ↔ IsSOPD 1 φ σ ω a μ := by
  simp only [IsSOPDMap, IsSOPD, isStationaryMap_id_iff, energyMap_id_eq]

end Perceptron
end Transformer

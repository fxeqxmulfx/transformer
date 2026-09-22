/-
# Perceptrons and attention's mean-field landscape — the first variation

The first variation of the coupled energy,

  `δE_{β,ϑ}/δμ[μ](x) = β⁻¹ ∫ e^{β x·y} dμ(y) + ½ v_ϑ(x)`,

as a function on all of `ℝ^d` (`firstVariation`), and its gradient.

* Its ambient gradient projects onto the field of `eq: steady.state`:
  `∇ δE/δμ[μ](x) = Proj_x ∇_{ℝ^d}(δE/δμ[μ])(x)` (`hasGradientAt_firstVariation`,
  `energyGrad_eq_proj`).

* It is the slope of the energy towards a Dirac mass (`MixDirac`).

Source: arXiv:2601.21366v2, §2.1 (the first variation of `𝖤_β`) and §2.3 (that
of `E_{β,ϑ}`).
-/

import Transformer.Perceptron.Minimizer
import Transformer.Perspective.PartitionGradient

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### The first variation and its gradient -/

/-- **The first variation of the coupled energy**, extended to `ℝ^d`:

  `δE_{β,ϑ}/δμ[μ](x) = β⁻¹ ∫ e^{β x·y} dμ(y) + ½ v_ϑ(x)`.

Source: arXiv:2601.21366v2, §2.1 (`δ𝖤_β/δμ`) and §2.3. -/
noncomputable def firstVariation (β : ℝ) (φ : ℝ → ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (x : EucSpace d) : ℝ :=
  β⁻¹ * Perspective.partitionMu d β μ x + (2 : ℝ)⁻¹ * potential φ ω a x

/-- **The ambient gradient of the first variation**, for `β ≠ 0`:

  `∇_{ℝ^d}(δE/δμ[μ])(x) = ∫ e^{β x·y} y dμ(y) + Σ_j ω_j σ(a_j·x) a_j`. -/
theorem hasGradientAt_firstVariation {β : ℝ} (hβ : β ≠ 0) (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    HasGradientAt (firstVariation β φ ω a μ)
      ((∫ y, Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • (y : EucSpace d)
          ∂(μ : Measure (SSphere d))) +
        ∑ j : Idx d, (ω j * σ (inner (𝕜 := ℝ) (a j) x)) • a j) x := by
  have hZ := hasGradientAt_iff_hasFDerivAt.mp (Perspective.hasGradientAt_partitionMu d β μ x)
  have hv := hasGradientAt_iff_hasFDerivAt.mp (hasGradientAt_potential φ σ hφ ω a x)
  rw [hasGradientAt_iff_hasFDerivAt]
  refine ((hZ.const_mul β⁻¹).fun_add (hv.const_mul (2 : ℝ)⁻¹)).congr_fderiv ?_
  ext y
  simp only [add_apply, smul_apply,
    InnerProductSpace.toDual_apply_apply, smul_eq_mul, inner_add_left, real_inner_smul_left]
  field_simp

/-- The hypothesis of `hasGradientAt_firstVariation` on `φ` is satisfiable:
`σ = 0` has the primitive `φ = 0`. -/
example : (1 : ℝ) ≠ 0 ∧ ∀ s : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (2 * (0 : ℝ → ℝ) s) s :=
  ⟨one_ne_zero, fun s => by simpa using hasDerivAt_const s (0 : ℝ)⟩

/-- The first variation is continuous, for `β ≠ 0`: it is differentiable. -/
theorem continuous_firstVariation {β : ℝ} (hβ : β ≠ 0) (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) : Continuous (firstVariation β φ ω a μ) :=
  continuous_iff_continuousAt.mpr fun x =>
    (hasGradientAt_firstVariation hβ φ σ hφ ω a μ x).continuousAt

/-- The hypotheses of `continuous_firstVariation` are satisfiable: `β = 1`, and
`σ = 0` with the primitive `φ = 0`. -/
example : (1 : ℝ) ≠ 0 ∧ ∀ s : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (2 * (0 : ℝ → ℝ) s) s :=
  ⟨one_ne_zero, fun s => by simpa using hasDerivAt_const s (0 : ℝ)⟩

/-- **The field of `eq: steady.state` is the tangential gradient of the first
variation**: `∇ δE/δμ[μ](x) = Proj_x (∫ e^{β x·y} y dμ(y) + Σ_j ω_j σ(a_j·x) a_j)`,
since `Proj_x` is linear and commutes with the integral.

Source: arXiv:2601.21366v2, §2.3, the display above `eq: steady.state`. -/
theorem energyGrad_eq_proj (β : ℝ) (σ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    energyGrad β σ ω a μ x = proj d x
      ((∫ y, Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • (y : EucSpace d)
          ∂(μ : Measure (SSphere d))) +
        ∑ j : Idx d, (ω j * σ (inner (𝕜 := ℝ) (a j) x)) • a j) := by
  set L : EucSpace d →L[ℝ] EucSpace d :=
    ContinuousLinearMap.id ℝ (EucSpace d) - (innerSL ℝ x).smulRight x
  have hL : ∀ y, L y = proj d x y := fun y => by simp [L, proj]
  have hint : ∫ y, Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • proj d x (y : EucSpace d)
        ∂(μ : Measure (SSphere d)) =
      L (∫ y, Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • (y : EucSpace d)
        ∂(μ : Measure (SSphere d))) := by
    rw [← L.integral_comp_comm (Perspective.integrable_expInner_smul d β μ x)]
    exact integral_congr_ae (ae_of_all _ fun y => by dsimp only; rw [map_smul, hL])
  rw [energyGrad, drift, hint, ← hL, ← hL, map_add]

end Perceptron
end Transformer

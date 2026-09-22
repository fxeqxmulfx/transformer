/-
# Perceptrons and attention's mean-field landscape — moving mass to a point

The segment `ν_ε = (1 - ε) μ + ε δ_z` from a measure `μ` to a Dirac mass at a
point `z` of the sphere (`mixDirac`), and the energy along it.

The energy is exactly quadratic in `ε`: the interaction term is a quadratic
form in the measure, the potential term linear.  Its slope at `ε = 0` is the
first variation at `z`, centred (`energy_mixDirac`):

  `E[ν_ε] = E[μ] + ε (δE/δμ[μ](z) - ∫ δE/δμ[μ] dμ) + ε² M`.

This is the variation the source's "it is clear" rests on: at a minimizer the
slope is nonnegative for every `z`, which is the first-order condition
(`MinStationary`).

Source: arXiv:2601.21366v2, §2.1 and §2.3.
-/

import Transformer.Perceptron.FirstVariation

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### Moving mass to a point -/

/-- **The mixture `(1 - ε) μ + ε δ_z`**, for `0 ≤ ε ≤ 1`: `μ` with the fraction
`ε` of its mass moved to the point `z`. -/
noncomputable def mixDirac (μ : Perspective.ProbSphere d) (z : SSphere d) (ε : ℝ)
    (h0 : 0 ≤ ε) (h1 : ε ≤ 1) : Perspective.ProbSphere d :=
  ⟨ENNReal.ofReal (1 - ε) • (μ : Measure (SSphere d)) + ENNReal.ofReal ε • Measure.dirac z,
    ⟨by
      simp only [Measure.add_apply, Measure.smul_apply, measure_univ, smul_eq_mul, mul_one]
      rw [← ENNReal.ofReal_add (by linarith) h0, sub_add_cancel, ENNReal.ofReal_one]⟩⟩

@[simp] theorem coe_mixDirac (μ : Perspective.ProbSphere d) (z : SSphere d) {ε : ℝ}
    (h0 : 0 ≤ ε) (h1 : ε ≤ 1) :
    (mixDirac μ z ε h0 h1 : Measure (SSphere d)) =
      ENNReal.ofReal (1 - ε) • (μ : Measure (SSphere d)) + ENNReal.ofReal ε • Measure.dirac z :=
  rfl

/-- `∫ f d((1 - ε) μ + ε δ_z) = (1 - ε) ∫ f dμ + ε f(z)`, for continuous `f`. -/
theorem integral_mixDirac (μ : Perspective.ProbSphere d) (z : SSphere d) {ε : ℝ}
    (h0 : 0 ≤ ε) (h1 : ε ≤ 1) {f : SSphere d → ℝ} (hf : Continuous f) :
    ∫ x, f x ∂(mixDirac μ z ε h0 h1 : Measure (SSphere d)) =
      (1 - ε) * ∫ x, f x ∂(μ : Measure (SSphere d)) + ε * f z := by
  rw [coe_mixDirac, integral_add_measure
    ((Perspective.integrable_of_continuous_compact hf _).smul_measure ENNReal.ofReal_ne_top)
    ((Perspective.integrable_of_continuous_compact hf _).smul_measure ENNReal.ofReal_ne_top),
    integral_smul_measure, integral_smul_measure, integral_dirac,
    ENNReal.toReal_ofReal (by linarith), ENNReal.toReal_ofReal h0, smul_eq_mul, smul_eq_mul]

/-- The hypotheses of `integral_mixDirac` are satisfiable: `ε = 0` and `f = 0`. -/
example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ Continuous (fun _ : SSphere 1 => (0 : ℝ)) :=
  ⟨le_rfl, zero_le_one, continuous_const⟩

/-- `Z_{β,μ}` is continuous on the sphere, for every `β`: it is differentiable
on `ℝ^d`. -/
theorem continuous_partitionMu_sphere (β : ℝ) (μ : Perspective.ProbSphere d) :
    Continuous fun x : SSphere d => Perspective.partitionMu d β μ (x : EucSpace d) :=
  (continuous_iff_continuousAt.mpr fun x =>
    (Perspective.hasGradientAt_partitionMu d β μ x).continuousAt).comp continuous_subtype_val

/-- **The interaction energy of `(1 - ε) μ + ε δ_z`**:

  `𝖤_β[ν] = (2β)⁻¹ ((1 - ε)² ∫ Z_μ dμ + 2ε(1 - ε) Z_μ(z) + ε² e^β)`.

The cross terms are both `Z_μ(z)`, the kernel being symmetric, and the
`δ_z ⊗ δ_z` term is `e^{β|z|²} = e^β`. -/
theorem interactionEnergy_mixDirac (β : ℝ) (μ : Perspective.ProbSphere d) (z : SSphere d)
    {ε : ℝ} (h0 : 0 ≤ ε) (h1 : ε ≤ 1) :
    Perspective.interactionEnergy d β (mixDirac μ z ε h0 h1) = (2 * β)⁻¹ *
      ((1 - ε) ^ 2 * (∫ x, Perspective.partitionMu d β μ (x : EucSpace d)
          ∂(μ : Measure (SSphere d))) +
        2 * ε * (1 - ε) * Perspective.partitionMu d β μ (z : EucSpace d) +
        ε ^ 2 * Real.exp β) := by
  have hZ := continuous_partitionMu_sphere β μ
  have hK := Perspective.continuous_expInner_left d β z
  have hin : ∀ x : SSphere d,
      ∫ x', Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (x' : EucSpace d))
          ∂(mixDirac μ z ε h0 h1 : Measure (SSphere d)) =
        (1 - ε) * Perspective.partitionMu d β μ (x : EucSpace d) +
          ε * Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (z : EucSpace d)) := fun x =>
    integral_mixDirac μ z h0 h1 (Perspective.continuous_expInner_right d β x)
  have hsym : ∫ x, Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (z : EucSpace d))
      ∂(μ : Measure (SSphere d)) = Perspective.partitionMu d β μ (z : EucSpace d) :=
    integral_congr_ae (ae_of_all _ fun x => by simp only [real_inner_comm])
  have hzz : Real.exp (β * inner (𝕜 := ℝ) (z : EucSpace d) (z : EucSpace d)) = Real.exp β := by
    rw [real_inner_self_eq_norm_sq, norm_eq_of_mem_sphere, one_pow, mul_one]
  have hZε : Continuous fun x : SSphere d =>
      (1 - ε) * Perspective.partitionMu d β μ (x : EucSpace d) := continuous_const.mul hZ
  have hKε : Continuous fun x : SSphere d =>
      ε * Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (z : EucSpace d)) :=
    continuous_const.mul hK
  have hsum : Continuous fun x : SSphere d =>
      (1 - ε) * Perspective.partitionMu d β μ (x : EucSpace d) +
        ε * Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (z : EucSpace d)) := hZε.add hKε
  rw [Perspective.interactionEnergy]
  simp only [hin]
  rw [integral_mixDirac μ z h0 h1 hsum,
    integral_add (Perspective.integrable_of_continuous_compact hZε _)
      (Perspective.integrable_of_continuous_compact hKε _),
    integral_const_mul, integral_const_mul, hsym, hzz]
  ring

/-- The hypotheses of `interactionEnergy_mixDirac` are satisfiable: `ε = 0`. -/
example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := ⟨le_rfl, zero_le_one⟩

/-- **Moving mass to a point**: for `ν = (1 - ε) μ + ε δ_z`,

  `E[ν] = E[μ] + ε (δE/δμ[μ](z) - ∫ δE/δμ[μ] dμ)
      + ε² (2β)⁻¹ (∫ Z_μ dμ - 2 Z_μ(z) + e^β)`,

exactly: the energy is quadratic along the segment from `μ` to `δ_z`, and its
slope at `μ` is the first variation at `z`, centred. -/
theorem energy_mixDirac (β : ℝ) {φ : ℝ → ℝ} (hφ : Continuous φ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (z : SSphere d) {ε : ℝ}
    (h0 : 0 ≤ ε) (h1 : ε ≤ 1) :
    energy β φ ω a (mixDirac μ z ε h0 h1) = energy β φ ω a μ +
      ε * (firstVariation β φ ω a μ (z : EucSpace d) -
        ∫ x, firstVariation β φ ω a μ (x : EucSpace d) ∂(μ : Measure (SSphere d))) +
      ε ^ 2 * ((2 * β)⁻¹ * ((∫ x, Perspective.partitionMu d β μ (x : EucSpace d)
          ∂(μ : Measure (SSphere d))) -
        2 * Perspective.partitionMu d β μ (z : EucSpace d) + Real.exp β)) := by
  have hZ := continuous_partitionMu_sphere β μ
  have hv : Continuous fun x : SSphere d => potential φ ω a (x : EucSpace d) :=
    (continuous_potential hφ ω a).comp continuous_subtype_val
  have hE : Perspective.interactionEnergy d β μ = (2 * β)⁻¹ *
      ∫ x, Perspective.partitionMu d β μ (x : EucSpace d) ∂(μ : Measure (SSphere d)) := rfl
  have hF : ∫ x, firstVariation β φ ω a μ (x : EucSpace d) ∂(μ : Measure (SSphere d)) =
      β⁻¹ * (∫ x, Perspective.partitionMu d β μ (x : EucSpace d) ∂(μ : Measure (SSphere d))) +
        (2 : ℝ)⁻¹ * ∫ x, potential φ ω a (x : EucSpace d) ∂(μ : Measure (SSphere d)) := by
    have hZβ : Continuous fun x : SSphere d =>
        β⁻¹ * Perspective.partitionMu d β μ (x : EucSpace d) := continuous_const.mul hZ
    have hv2 : Continuous fun x : SSphere d => (2 : ℝ)⁻¹ * potential φ ω a (x : EucSpace d) :=
      continuous_const.mul hv
    rw [← integral_const_mul, ← integral_const_mul, ← integral_add
      (Perspective.integrable_of_continuous_compact hZβ _)
      (Perspective.integrable_of_continuous_compact hv2 _)]
    rfl
  rw [energy, energy, interactionEnergy_mixDirac, integral_mixDirac μ z h0 h1 hv, hE, hF,
    firstVariation]
  ring

/-- The hypotheses of `energy_mixDirac` are satisfiable: `φ = 0` and `ε = 0`. -/
example : Continuous (fun _ : ℝ => (0 : ℝ)) ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 :=
  ⟨continuous_const, le_rfl, zero_le_one⟩

end Perceptron
end Transformer

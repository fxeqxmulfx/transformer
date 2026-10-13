/-
# Continuous signed densities in the original exponential transform

ArXiv:2601.21366v2, §5.1, proof of `lem: quadpol` (i).
The source first constructs a finite signed density and then identifies
it with the original nonnegative measure. Its nonnegativity must follow
from that identification, rather than being assumed during inversion.

Here the actual positive and negative parts are finite nonnegative
measures. Their original transforms differ by the integral of the
original signed density. The proved finite-measure injectivity theorem
therefore identifies the parts when this integral vanishes everywhere.
Equality of their actual densities gives zero almost everywhere.

The actual uniform law has full support, as proved from its original
orthogonal invariance. Continuous densities equal almost everywhere
against that law are thus equal everywhere. This proves injectivity for
continuous signed densities without an inverse or harmonic expansion
as an input. All integrals use the original exponential kernel.
-/

import Transformer.Perceptron.Section5_FiniteTransform
import Transformer.Perceptron.Section5_UniformSupport
import Mathlib.MeasureTheory.Function.AEEqOfLIntegral
import Mathlib.MeasureTheory.Measure.OpenPos

open Real MeasureTheory Set
open scoped ENNReal

namespace Transformer.Perceptron

/-- The actual uniform law charges every actual nonempty open set.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i), and §5.3,
the full-support consequence for analytic densities. This follows from
the proved full support, without supplying positivity as a hypothesis. -/
theorem isOpenPosMeasure_of_uniform {d : ℕ} (ν : Measure (SSphere d))
    (hν : Metastability.IsUniformOn d ν) : ν.IsOpenPosMeasure := by
  refine ⟨fun U hU hne => ?_⟩
  obtain ⟨x, hx⟩ := hne
  have hs : x ∈ ν.support := by
    have heq := support_eq_univ_of_uniform (⟨ν, hν.1⟩ : Perspective.ProbSphere d) hν
    change ν.support = univ at heq
    rw [heq]
    exact mem_univ _
  exact ((Measure.mem_support_iff_forall x).mp hs U (hU.mem_nhds hx)).ne'

/-- Integration against the actual positive part of a continuous signed
density is integration against its actual pointwise positive part.
Source: arXiv:2601.21366v2, §5.1, proof of `lem: quadpol` (i).
This explicitly retains the clipping performed by `ENNReal.ofReal`. -/
theorem integral_withDensity_ofReal_eq_posPart {d : ℕ} (ν : Measure (SSphere d))
    (g : SSphere d → ℝ) (hg : Continuous g) (f : SSphere d → ℝ) :
    (∫ y, f y ∂ν.withDensity (fun y => ENNReal.ofReal (g y))) =
      ∫ y, max (g y) 0 * f y ∂ν := by
  simpa only [ENNReal.toReal_ofReal', smul_eq_mul] using
    integral_withDensity_eq_integral_toReal_smul (f := fun y => ENNReal.ofReal (g y))
    (ENNReal.continuous_ofReal.comp hg).measurable
    (ae_of_all ν fun _ => ENNReal.ofReal_lt_top) f

/-- The original signed-density transform is the difference of the
original transforms of its actual positive and negative parts.
Source: arXiv:2601.21366v2, §5.1, proof of `lem: quadpol` (i).
Continuity on the compact sphere makes both original integrals finite. -/
theorem integral_exp_inner_mul_eq_positive_parts {d : ℕ} (β : ℝ)
    (ν : Measure (SSphere d)) [IsFiniteMeasure ν] (g : SSphere d → ℝ)
    (hg : Continuous g) (x : SSphere d) :
    (∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) * g y ∂ν) =
      (∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))
        ∂ν.withDensity (fun y => ENNReal.ofReal (g y))) -
      ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))
        ∂ν.withDensity (fun y => ENNReal.ofReal (-g y)) := by
  let K := fun y : SSphere d =>
    exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))
  have hK : Continuous K := Real.continuous_exp.comp
    (continuous_const.mul (continuous_const.inner continuous_subtype_val))
  rw [integral_withDensity_ofReal_eq_posPart ν g hg,
    integral_withDensity_ofReal_eq_posPart ν (fun y => -g y) hg.neg]
  change (∫ y, K y * g y ∂ν) =
    (∫ y, max (g y) 0 * K y ∂ν) - ∫ y, max (-g y) 0 * K y ∂ν
  rw [← integral_sub (f := fun y => max (g y) 0 * K y)
    (g := fun y => max (-g y) 0 * K y)
    (Perspective.integrable_of_continuous_compact ((hg.max continuous_const).mul hK) ν)
    (Perspective.integrable_of_continuous_compact ((hg.neg.max continuous_const).mul hK) ν)]
  apply integral_congr_ae
  refine ae_of_all ν fun y => ?_
  change K y * g y = max (g y) 0 * K y - max (-g y) 0 * K y
  by_cases h : 0 ≤ g y
  · rw [max_eq_left h, max_eq_right (neg_nonpos.mpr h)]
    ring
  · rw [max_eq_right (le_of_not_ge h), max_eq_left (neg_nonneg.mpr (le_of_not_ge h))]
    ring

/-- A continuous signed density with zero original transform is zero
almost everywhere against its original finite measure.
Source: arXiv:2601.21366v2, §5.1, injectivity in `lem: quadpol`.
The finite-measure theorem compares its actual positive and negative parts. -/
theorem ae_eq_zero_of_integral_exp_inner_mul_eq_zero {d : ℕ}
    (β : ℝ) (hβ : 0 < β) (ν : Measure (SSphere d)) [IsFiniteMeasure ν]
    (g : SSphere d → ℝ) (hg : Continuous g)
    (hzero : ∀ x : SSphere d,
      ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) * g y ∂ν = 0) :
    g =ᵐ[ν] 0 := by
  let _ := isFiniteMeasure_withDensity_ofReal
    (Perspective.integrable_of_continuous_compact hg ν).hasFiniteIntegral
  let _ := isFiniteMeasure_withDensity_ofReal
    (Perspective.integrable_of_continuous_compact hg.neg ν).hasFiniteIntegral
  have heq := eq_of_integral_exp_inner_eq_finite β hβ
    (ν.withDensity fun y => ENNReal.ofReal (g y))
    (ν.withDensity fun y => ENNReal.ofReal (-g y)) fun x => by
      have h := integral_exp_inner_mul_eq_positive_parts β ν g hg x
      rw [hzero x] at h
      exact sub_eq_zero.mp h.symm
  have hae := (withDensity_eq_iff_of_sigmaFinite
    (ENNReal.continuous_ofReal.comp hg).aemeasurable
    (ENNReal.continuous_ofReal.comp hg.neg).aemeasurable).mp heq
  filter_upwards [hae] with y hy
  have h := congrArg ENNReal.toReal hy
  change max (g y) 0 = max (-g y) 0 at h
  change g y = 0
  by_cases hgy : 0 ≤ g y
  · rw [max_eq_left hgy, max_eq_right (neg_nonpos.mpr hgy)] at h
    exact h
  · rw [max_eq_right (le_of_not_ge hgy), max_eq_left (neg_nonneg.mpr (le_of_not_ge hgy))] at h
    linarith

/-- Zero original transform forces a continuous signed density to vanish
at every original sphere point for the actual uniform law.
Source: arXiv:2601.21366v2, §5.1, proof of `lem: quadpol` (i).
The pointwise conclusion uses the proved full support of that same law. -/
theorem eq_zero_of_integral_exp_inner_mul_eq_zero_uniform {d : ℕ}
    (β : ℝ) (hβ : 0 < β) (ν : Measure (SSphere d))
    (hν : Metastability.IsUniformOn d ν) (g : SSphere d → ℝ) (hg : Continuous g)
    (hzero : ∀ x : SSphere d,
      ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) * g y ∂ν = 0) :
    g = 0 := by
  let _ : IsProbabilityMeasure ν := hν.1
  let _ : ν.IsOpenPosMeasure := isOpenPosMeasure_of_uniform ν hν
  exact Measure.eq_of_ae_eq
    (ae_eq_zero_of_integral_exp_inner_mul_eq_zero β hβ ν g hg hzero) hg continuous_const

/-- The original kernel identifies two arbitrary continuous signed
densities against the actual uniform law. Source: arXiv:2601.21366v2,
§5.1, proof of `lem: quadpol` (i). No nonnegativity or probability
normalization is imposed on either actual density candidate. -/
theorem eq_of_integral_exp_inner_mul_eq_uniform {d : ℕ} (β : ℝ) (hβ : 0 < β)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (f g : SSphere d → ℝ) (hf : Continuous f) (hg : Continuous g)
    (h : ∀ x : SSphere d,
      (∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) * f y ∂ν) =
      ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) * g y ∂ν) :
    f = g := by
  let _ : IsProbabilityMeasure ν := hν.1
  have hz := eq_zero_of_integral_exp_inner_mul_eq_zero_uniform β hβ ν hν
    (fun y => f y - g y) (hf.sub hg) fun x => by
      have hK : Continuous (fun y : SSphere d =>
          exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))) :=
        Real.continuous_exp.comp (continuous_const.mul
          (continuous_const.inner continuous_subtype_val))
      simp_rw [mul_sub]
      rw [integral_sub
        (f := fun y : SSphere d => exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) * f y)
        (g := fun y : SSphere d => exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) * g y)
        (Perspective.integrable_of_continuous_compact (hK.mul hf) ν)
        (Perspective.integrable_of_continuous_compact (hK.mul hg) ν), h x, sub_self]
  funext y
  have hy := congrFun hz y
  exact sub_eq_zero.mp hy

/-- All signed-density, positive-temperature and actual uniform inputs
hold jointly. Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i).
The zero density supplies the actual zero transform on the genuine circle;
the constant one also supplies the positive-part integral inputs. -/
example : ∃ ν : Measure (SSphere 2), (0 : ℝ) < 1 ∧
    Metastability.IsUniformOn 2 ν ∧ IsFiniteMeasure ν ∧
    Continuous (0 : SSphere 2 → ℝ) ∧ Continuous (1 : SSphere 2 → ℝ) ∧
    (∀ x : SSphere 2,
      ∫ y, exp (1 * inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2)) * (0 : ℝ) ∂ν = 0) ∧
    ∀ x : SSphere 2,
      (∫ y, exp (1 * inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2)) * (1 : ℝ) ∂ν) =
      ∫ y, exp (1 * inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2)) * (1 : ℝ) ∂ν := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let _ : IsProbabilityMeasure (MeanField.uniformLaw 2) := hν.1
  exact ⟨MeanField.uniformLaw 2, one_pos, hν, inferInstance,
    continuous_const, continuous_const, (fun _ => by simp), fun _ => rfl⟩

end Transformer.Perceptron

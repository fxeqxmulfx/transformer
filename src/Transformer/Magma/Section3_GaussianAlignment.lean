/-
# Gaussian negative-alignment probabilities

Formalization of arXiv:2602.15322v1, Section 3, the inline Phi formula.
The Gaussian projection model, nonzero momentum, and positive noise scale
are explicit. The formula is not a distribution-free property of EMA
momentum. A Chernoff bound specifies its stated exponential decay.
-/

import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.CDF
import Mathlib.Analysis.InnerProductSpace.Continuous
import Mathlib.Tactic

open MeasureTheory ProbabilityTheory
open scoped InnerProductSpace

noncomputable section

namespace Transformer.Magma

/-- The actual standard normal cumulative distribution function.
Source: arXiv:2602.15322v1, Section 3, Phi in the negative-alignment model. -/
def normalCDF (x : ℝ) : ℝ := cdf (gaussianReal 0 1) x

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace E] [BorelSpace E]

/-- Standardized centered projection along momentum. Source:
arXiv:2602.15322v1, Section 3, Gaussian negative-alignment model. -/
def alignmentProjection (momentum : E) (σ : ℝ) (g : E) : ℝ :=
  ⟪momentum, g - momentum⟫_ℝ / (σ * ‖momentum‖)

/-- Correctly scoped Gaussian formula for the probability of opposing
momentum. The standardized projection is assumed standard normal, as in
the isotropic Gaussian model motivating the source formula. Arbitrary
stochastic gradients need not have this law. Source:
arXiv:2602.15322v1, Section 3, inline P(mu^T g<0)=Phi(-norm(mu)/sigma). -/
theorem gaussian_negative_alignment_probability (momentum : E) (σ : ℝ)
    (hμ : momentum ≠ 0) (hσ : 0 < σ) (law : Measure E)
    (hgaussian : law.map (alignmentProjection momentum σ) = gaussianReal 0 1) :
    law.real {g | ⟪momentum, g⟫_ℝ < 0} = normalCDF (-‖momentum‖ / σ) := by
  have hn : 0 < ‖momentum‖ := norm_pos_iff.mpr hμ
  have hd : 0 < σ * ‖momentum‖ := mul_pos hσ hn
  have heq : (-‖momentum‖ / σ) * (σ * ‖momentum‖) = -‖momentum‖ ^ 2 := by
    field_simp
  have hset : {g | ⟪momentum, g⟫_ℝ < 0} =
      alignmentProjection momentum σ ⁻¹' Set.Iio (-‖momentum‖ / σ) := by
    ext g
    change ⟪momentum, g⟫_ℝ < 0 ↔
      ⟪momentum, g - momentum⟫_ℝ / (σ * ‖momentum‖) < -‖momentum‖ / σ
    rw [inner_sub_right, real_inner_self_eq_norm_sq, div_lt_iff₀ hd, heq]
    constructor <;> intro h <;> linarith
  have hm : Measurable (alignmentProjection momentum σ) :=
    ((continuous_const.inner (continuous_id.sub continuous_const)).div_const _).measurable
  have hlaw : law {g | ⟪momentum, g⟫_ℝ < 0} =
      (gaussianReal 0 1) (Set.Iio (-‖momentum‖ / σ)) := by
    rw [hset, ← Measure.map_apply hm measurableSet_Iio, hgaussian]
  change (law {g | ⟪momentum, g⟫_ℝ < 0}).toReal = _
  rw [hlaw, normalCDF, cdf_eq_real]
  let : NullSingletonClass (gaussianReal 0 1) := nullSingletonClass_gaussianReal (by norm_num)
  exact congrArg ENNReal.toReal (measure_congr Iio_ae_eq_Iic)

/-- The Gaussian model and both strict-domain conditions are jointly
satisfiable for scalar momentum=1, noise scale=1, and N(1,1) gradients.
Source: arXiv:2602.15322v1, Section 3, Gaussian alignment formula. -/
example : (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧
    (gaussianReal 1 1).map (alignmentProjection (1 : ℝ) 1) = gaussianReal 0 1 := by
  refine ⟨by norm_num, by norm_num, ?_⟩
  have hproj : alignmentProjection (1 : ℝ) 1 = fun x => x - 1 := by
    funext x
    norm_num [alignmentProjection, RCLike.inner_apply, Real.norm_eq_abs]
  rw [hproj]
  simpa using (gaussianReal_map_sub_const (μ := 1) (v := 1) 1)

omit [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E] [BorelSpace E] in
/-- The motivating Gaussian negative-alignment probability decays at
least as exp(-r^2/2) with signal-to-noise ratio r>=0. Source:
arXiv:2602.15322v1, Section 3, exponential-decay statement. -/
theorem normal_lower_tail_exponential (r : ℝ) (hr : 0 ≤ r) :
    normalCDF (-r) ≤ Real.exp (-r ^ 2 / 2) := by
  rw [normalCDF, cdf_eq_real]
  have h := measure_le_le_exp_mul_mgf (X := id) (μ := gaussianReal 0 1)
    (t := -r) (-r) (by linarith) (integrable_exp_mul_gaussianReal (-r))
  rw [mgf_id_gaussianReal] at h
  calc
    _ ≤ Real.exp (-(-r) * (-r)) * Real.exp (0 * (-r) + (1 : ℝ) * (-r) ^ 2 / 2) := h
    _ = _ := by
      rw [← Real.exp_add]
      congr 1
      ring

/-- A nonzero signal-to-noise ratio lies in the tail-bound domain.
Source: arXiv:2602.15322v1, Section 3. -/
example : (0 : ℝ) ≤ 1 := by norm_num

end Transformer.Magma

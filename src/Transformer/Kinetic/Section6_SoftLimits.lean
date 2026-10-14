/-
# Kinetic theory for Transformers — primacy and the interior recency limit

We transfer the actual aggregate `ℋ` to the original soft correction
`𝒮_t(σ₀)=ℋ(Y(σ₀))` on `(0,1)`, as in arXiv:2605.09213v1, §6,
proof of `thm:U-shape`. Its actual interior derivative has the usual
chain-rule formula. The coercive aggregate gives primacy, and its
negative initial slope times the negative output derivative of `Y`
gives a strictly positive left limit of the interior derivative.

The original totalized Bessel formula takes value zero on the diagonal;
the continuous series extension has a positive left limit there. The
recency claim is therefore stated as the limit of actual interior
derivatives, as explained in `Hardy` and `Section1_RecencyEndpoint`.
`Section6_OutputAggregation` keeps the original weights and eigenvalues.
-/

import Transformer.Kinetic.Section6_OutputAggregation
import Transformer.Kinetic.Section6_SourceLimits
import Transformer.Kinetic.Section1_RecencyEndpoint

open Set Filter Real
open scoped Topology

namespace Transformer
namespace Kinetic

/-- The original soft correction has its actual chain-rule derivative
throughout its interior domain. Source: arXiv:2605.09213v1, §6, `thm:U-shape`. -/
theorem hasDerivAt_softCorrection {β lam M t C s : ℝ} (hβ : 0 < β) (hlam : 0 < lam)
    (hM : 0 < M) (ht : 0 < t) (hC : ∀ n, outputModeParameter β t n ≤ C)
    (hs0 : 0 < s) (hs1 : s < 1) :
    HasDerivAt (softCorrection β lam M t)
      (seriesDeriv lam (weightedPsiCoeffs (gaussianModeWeight M) (outputModeParameter β t))
        (Yfun lam 1 s) * (-(lam * Real.exp (lam * s) / (Real.exp (lam * s) - 1)))) s := by
  have hc := (hasDerivAt_softProfile hβ hM ht hC lam (Yfun lam 1 s)).comp s
    (hasDerivAt_Yfun_one hlam hs0)
  apply hc.congr_of_eventuallyEq
  filter_upwards [Ioo_mem_nhds hs0 hs1] with z hz
  exact softCorrection_eq_softProfile hβ hlam hM ht hC hz.1 hz.2

/-- The actual derivative function of the original soft correction on `(0,1)`.
Source: arXiv:2605.09213v1, §6, the recency derivative in `thm:U-shape`. -/
theorem deriv_softCorrection {β lam M t C s : ℝ} (hβ : 0 < β) (hlam : 0 < lam)
    (hM : 0 < M) (ht : 0 < t) (hC : ∀ n, outputModeParameter β t n ≤ C)
    (hs0 : 0 < s) (hs1 : s < 1) :
    deriv (softCorrection β lam M t) s =
      seriesDeriv lam (weightedPsiCoeffs (gaussianModeWeight M) (outputModeParameter β t))
        (Yfun lam 1 s) * (-(lam * Real.exp (lam * s) / (Real.exp (lam * s) - 1))) :=
  (hasDerivAt_softCorrection hβ hlam hM ht hC hs0 hs1).deriv

/-- Primacy for the original Fourier sum follows from its actual positive
linear coefficient. Source: arXiv:2605.09213v1, §6, `thm:U-shape` (i). -/
theorem softCorrection_primacy {β lam M t C : ℝ} (hβ : 0 < β) (hlam : 0 < lam)
    (hM : 0 < M) (ht : 0 < t) (hC : ∀ n, outputModeParameter β t n ≤ C) :
    Tendsto (softCorrection β lam M t) (𝓝[>] 0) atTop := by
  have hp := (softProfile_tendsto_atTop hβ hlam hM ht hC).comp (Yfun_one_tendsto_atTop hlam)
  apply hp.congr'
  filter_upwards [self_mem_nhdsWithin (a := (0 : ℝ)) (s := Ioi 0),
    (eventually_lt_nhds (show (0 : ℝ) < 1 from one_pos)).filter_mono nhdsWithin_le_nhds] with s hs0 hs1
  exact (softCorrection_eq_softProfile hβ hlam hM ht hC hs0 hs1).symm

/-- Recency: the actual derivative on the original open interval has a
strictly positive left limit, under the non-strict source smallness bound.
Source: arXiv:2605.09213v1, §6, `thm:U-shape` (ii), with its endpoint interpretation corrected. -/
theorem softCorrection_recency {β lam M t C : ℝ} (hβ : 0 < β) (hlam : 0 < lam)
    (hM : 0 < M) (ht : 0 < t) (hC : ∀ n, outputModeParameter β t n ≤ C)
    (hsmall : ∀ n, outputModeParameter β t n ≤ 2 * (1 - Real.exp (-lam))) :
    ∃ D : ℝ, 0 < D ∧ Tendsto (deriv (softCorrection β lam M t)) (𝓝[<] 1) (𝓝 D) := by
  let b := weightedPsiCoeffs (gaussianModeWeight M) (outputModeParameter β t)
  have hb : IsEntire b := isEntire_softProfileCoeffs hβ hM ht hC
  have hslope : seriesDeriv lam b 0 < 0 := softProfile_initialSlope_neg hβ hlam hM ht hC hsmall
  have hYneg : -(lam * Real.exp lam / (Real.exp lam - 1)) < 0 := by
    simpa only [mul_one] using Yfun_one_derivative_neg hlam (show (0 : ℝ) < 1 from one_pos)
  let D := seriesDeriv lam b 0 * (-(lam * Real.exp lam / (Real.exp lam - 1)))
  refine ⟨D, mul_pos_of_neg_of_neg hslope hYneg, ?_⟩
  have hH := (continuous_seriesDeriv hb lam).tendsto 0
  have hlim := (hH.comp (Yfun_one_tendsto_zero hlam)).mul (Yfun_one_derivative_tendsto hlam)
  apply hlim.congr'
  filter_upwards [self_mem_nhdsWithin (a := (1 : ℝ)) (s := Iio 1),
    (eventually_gt_nhds (show (0 : ℝ) < 1 from one_pos)).filter_mono nhdsWithin_le_nhds] with s hs1 hs0
  exact (deriv_softCorrection hβ hlam hM ht hC hs0 hs1).symm

/-- The actual correction has the source's continuous-series left limit
at the output endpoint. Source: arXiv:2605.09213v1, §6, `ψ_c(0)=c` in `thm:U-shape`. -/
theorem softCorrection_output_limit {β lam M t C : ℝ} (hβ : 0 < β) (hlam : 0 < lam)
    (hM : 0 < M) (ht : 0 < t) (hC : ∀ n, outputModeParameter β t n ≤ C) :
    Tendsto (softCorrection β lam M t) (𝓝[<] 1) (𝓝 (softProfile β lam M t 0)) := by
  have hp := ((continuous_softProfile hβ hM ht hC lam).tendsto 0).comp
    (Yfun_one_tendsto_zero hlam)
  apply hp.congr'
  filter_upwards [self_mem_nhdsWithin (a := (1 : ℝ)) (s := Iio 1),
    (eventually_gt_nhds (show (0 : ℝ) < 1 from one_pos)).filter_mono nhdsWithin_le_nhds] with s hs1 hs0
  exact (softCorrection_eq_softProfile hβ hlam hM ht hC hs0 hs1).symm

/-- The original series extension has a positive output value: its
literal totalized diagonal value is not its interior left limit.
Source: arXiv:2605.09213v1, §6, the continuous extension in `thm:U-shape`. -/
theorem softProfile_output_pos {β lam M t C : ℝ} (hβ : 0 < β) (hlam : 0 < lam)
    (hM : 0 < M) (ht : 0 < t) (hC : ∀ n, outputModeParameter β t n ≤ C) :
    0 < softProfile β lam M t 0 :=
  seriesProfile_weighted_zero_pos (summable_gaussianModeWeight hM)
    (fun n => (gaussianModeWeight_pos M n).le) (fun n => (outputModeParameter_pos hβ ht n).le)
    hC hlam 0 (gaussianModeWeight_pos M 0) (outputModeParameter_pos hβ ht 0)

/-- The original correction is positive at every interior source position.
Source: arXiv:2605.09213v1, §6, the positive Fourier sum in `thm:U-shape`. -/
theorem softCorrection_pos {β lam M t C s : ℝ} (hβ : 0 < β) (hlam : 0 < lam)
    (hM : 0 < M) (ht : 0 < t) (hC : ∀ n, outputModeParameter β t n ≤ C)
    (hs0 : 0 < s) (hs1 : s < 1) : 0 < softCorrection β lam M t s := by
  rw [softCorrection_eq_softProfile hβ hlam hM ht hC hs0 hs1]
  exact seriesProfile_weighted_pos (summable_gaussianModeWeight hM)
    (fun n => (gaussianModeWeight_pos M n).le) (fun n => (outputModeParameter_pos hβ ht n).le)
    hC hlam 0 (gaussianModeWeight_pos M 0) (outputModeParameter_pos hβ ht 0)
    (Yfun_one_pos hlam hs0 hs1).le

/-- The literal totalized correction is discontinuous at its output
endpoint; its positive interior limit differs from its zero diagonal value.
Source: arXiv:2605.09213v1, §6, the series extension in `thm:U-shape`; the printed
Bessel formula `eq:Bessel-full` is defined only for `σ₀ < σ`. -/
theorem not_continuousAt_softCorrection_one {β lam M t C : ℝ} (hβ : 0 < β) (hlam : 0 < lam)
    (hM : 0 < M) (ht : 0 < t) (hC : ∀ n, outputModeParameter β t n ≤ C) :
    ¬ ContinuousAt (softCorrection β lam M t) 1 := by
  intro hc
  have hz := hc.tendsto.mono_left (nhdsWithin_le_nhds (s := Iio 1))
  rw [softCorrection_one] at hz
  have heq := tendsto_nhds_unique (softCorrection_output_limit hβ hlam hM ht hC) hz
  exact (softProfile_output_pos hβ hlam hM ht hC).ne' heq

/-- At the original positive unit parameters, some actual positive time
meets all bounds above. The position `1/2` meets both interior derivative
hypotheses. Source: arXiv:2605.09213v1, `eq:affine-smallness`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 2 ∧ (1 : ℝ) / 2 < 1 ∧
    ∃ t : ℝ, 0 < t ∧ (∀ n, outputModeParameter 1 t n ≤ 3 - Real.sqrt 3) ∧
      (∀ n, outputModeParameter 1 t n ≤ 2 * (1 - Real.exp (-1))) := by
  refine ⟨one_pos, one_half_pos, one_half_lt_one, ?_⟩
  obtain ⟨t, ht, hb⟩ := exists_affine_smallness 1 1 one_pos
  exact ⟨t, ht, fun n => (hb (n + 1) (by omega)).trans (min_le_left _ _),
    fun n => (hb (n + 1) (by omega)).trans (min_le_right _ _)⟩

/-- The endpoint discontinuity occurs at actual admissible source
parameters, with a genuinely positive time satisfying both printed bounds.
Source: arXiv:2605.09213v1, §6, `eq:affine-smallness` and the recency extension. -/
example : ∃ t : ℝ, 0 < t ∧ ¬ ContinuousAt (softCorrection 1 1 1 t) 1 := by
  obtain ⟨t, ht, hb⟩ := exists_affine_smallness 1 1 one_pos
  exact ⟨t, ht, not_continuousAt_softCorrection_one one_pos one_pos one_pos ht
    (fun n => (hb (n + 1) (by omega)).trans (min_le_left _ _))⟩

end Kinetic
end Transformer

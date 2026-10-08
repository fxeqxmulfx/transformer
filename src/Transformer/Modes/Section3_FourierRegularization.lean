import Transformer.Modes.Section3_GaussianRegularization
import Mathlib.Analysis.Fourier.Inversion
import Mathlib.MeasureTheory.Measure.CharacteristicFunction.TaylorExpansion

/-!
# The exact Fourier integral of a Gaussian regularization

Section 5.4 of arXiv:2412.09080v3 invokes Fourier inversion of the
normalized-sum density. The input law need not yet have a density.
This module proves the regularized inversion identity directly for
an arbitrary probability law on the two-dimensional real space `ℂ`.

The Gaussian multiplier is `exp(-|ξ|²/c)`, whose Fourier transform is
`π c exp(-π² c |x|²)`. The positive-sign characteristic function must
be evaluated at `2π ξ` to match Mathlib's negative-sign Fourier transform.
This accounts explicitly for the source's differing Fourier convention.

The Fourier integrand is integrable on the product of frequency volume
and the actual law: its norm is the integrable Gaussian, since the
oscillatory factor has norm one. Fubini then identifies the Fourier
integral with the real nonnegative mass-one smoothing already constructed.
The identity uses no absolute-continuity assumption and no moment bound.

The multiplier tends pointwise to one as the regularization scale tends
to infinity. When the characteristic function is integrable, this identity
therefore supplies dominated-convergence approximations to its inverse
Fourier integral. Positivity and total-mass control can be passed to that
limit before invoking the unregularized inversion theorem.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4, Fourier inversion.
-/

open Real MeasureTheory Filter
open scoped FourierTransform Topology ENNReal

namespace Transformer.Modes

/-- The Fourier Gaussian multiplier corresponding to the mass-one
regularizer. Source: arXiv:2412.09080v3, §5.4, Fourier inversion. -/
noncomputable def gaussianDamping2 (c : ℝ) (z : ℂ) : ℂ :=
  Complex.exp (-(c : ℂ)⁻¹ * ‖z‖ ^ 2)

/-- The positive-scale Gaussian multiplier has norm at most one.
Source: arXiv:2412.09080v3, §5.4, Fourier inversion. -/
theorem norm_gaussianDamping2_le_one {c : ℝ} (hc : 0 < c) (z : ℂ) :
    ‖gaussianDamping2 c z‖ ≤ 1 := by
  unfold gaussianDamping2
  rw [← Complex.ofReal_inv, ← Complex.ofReal_neg, ← Complex.ofReal_pow,
    ← Complex.ofReal_mul, Complex.norm_exp_ofReal]
  apply Real.exp_le_one_iff.mpr
  exact mul_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr (inv_pos.mpr hc).le) (sq_nonneg _)

example : (0 : ℝ) < 1 := by norm_num

/-- A positive-scale Fourier Gaussian is integrable.
Source: arXiv:2412.09080v3, §5.4, Fourier inversion. -/
theorem integrable_gaussianDamping2 {c : ℝ} (hc : 0 < c) :
    Integrable (gaussianDamping2 c) := by
  unfold gaussianDamping2
  simpa only [zero_mul, add_zero] using
    GaussianFourier.integrable_cexp_neg_mul_sq_norm_add
      (V := ℂ) (b := (c : ℂ)⁻¹) (by simpa using inv_pos.mpr hc) 0 0

example : (0 : ℝ) < 1 := by norm_num

/-- The Gaussian Fourier transform has the exact mass-one
normalization used here. Source: arXiv:2412.09080v3, §5.4.
The source uses `exp(-i⟨ξ,x⟩)`; Mathlib uses `exp(-2π i⟨ξ,x⟩)`. -/
theorem fourier_dampedGaussian {c : ℝ} (hc : 0 < c) (w : ℂ) :
    𝓕 (fun z : ℂ => Complex.exp (-(c : ℂ)⁻¹ * ‖z‖ ^ 2)) w =
      (gaussianHeatKernel2 c w : ℂ) := by
  rw [fourier_gaussian_innerProductSpace (by simpa using inv_pos.mpr hc)]
  simp only [Complex.finrank_real_complex, Nat.cast_ofNat,
    div_self (by norm_num : (2 : ℂ) ≠ 0), Complex.cpow_one]
  rw [div_inv_eq_mul, div_inv_eq_mul]
  simp only [gaussianHeatKernel2, Complex.ofReal_mul, Complex.ofReal_pow,
    Complex.ofReal_neg, Complex.ofReal_exp]
  ring_nf

example : (0 : ℝ) < 1 := by norm_num

/-- The smoothed law is the inverse characteristic-function
integral with a Gaussian multiplier. Fubini is justified by the
integrable Gaussian and the unit-modulus phase under the actual law.
Source: arXiv:2412.09080v3, §3 and §5.4, Fourier inversion. -/
theorem gaussianRegularization2_fourier (μ : Measure ℂ) [IsProbabilityMeasure μ]
    {c : ℝ} (hc : 0 < c) (w : ℂ) :
    (gaussianRegularization2 μ c w : ℂ) =
      𝓕 (fun ξ => gaussianDamping2 c ξ * charFun μ ((2 * Real.pi) • ξ)) w := by
  let F : ℂ × ℂ → ℂ := fun p =>
    gaussianDamping2 c p.1 *
      Complex.exp (((-2 * Real.pi * inner ℝ p.1 (w - p.2) : ℝ) : ℂ) * Complex.I)
  have hFcont : Continuous F := by
    unfold F gaussianDamping2
    fun_prop
  have hF : Integrable F (volume.prod μ) := by
    have hb := (integrable_gaussianDamping2 hc).norm.mul_prod
      (integrable_const (1 : ℝ) : Integrable (fun _ : ℂ => (1 : ℝ)) μ)
    apply hb.mono' hFcont.aestronglyMeasurable
    exact ae_of_all _ fun p => by
      change ‖gaussianDamping2 c p.1 *
        Complex.exp (((-2 * Real.pi * inner ℝ p.1 (w - p.2) : ℝ) : ℂ) * Complex.I)‖ ≤
          ‖gaussianDamping2 c p.1‖ * 1
      rw [norm_mul, Complex.norm_exp_ofReal_mul_I]
  have hswap := integral_integral_swap (f := fun ξ z => F (ξ, z)) hF
  have hleft (ξ : ℂ) : (∫ z, F (ξ, z) ∂μ) =
      Complex.exp (((-2 * Real.pi * inner ℝ ξ w : ℝ) : ℂ) * Complex.I) *
        (gaussianDamping2 c ξ * charFun μ ((2 * Real.pi) • ξ)) := by
    rw [charFun_apply, ← integral_const_mul, ← integral_const_mul]
    apply integral_congr_ae
    exact ae_of_all _ fun z => by
      dsimp only [F]
      rw [inner_sub_right, inner_smul_right, real_inner_comm z ξ]
      have he : (((-2 * Real.pi * (inner ℝ ξ w - inner ℝ z ξ) : ℝ) : ℂ) * Complex.I) =
          (((-2 * Real.pi * inner ℝ ξ w : ℝ) : ℂ) * Complex.I) +
            (((2 * Real.pi * inner ℝ z ξ : ℝ) : ℂ) * Complex.I) := by
        simp only [Complex.ofReal_sub, Complex.ofReal_mul, Complex.ofReal_neg]
        ring
      rw [he, Complex.exp_add]
      ring
  have hright (z : ℂ) : (∫ ξ, F (ξ, z)) = (gaussianHeatKernel2 c (w - z) : ℂ) := by
    rw [← fourier_dampedGaussian hc (w - z), Real.fourier_eq']
    apply integral_congr_ae
    exact ae_of_all _ fun ξ => by simp [F, gaussianDamping2, smul_eq_mul, mul_comm]
  rw [Real.fourier_eq']
  simp only [smul_eq_mul]
  rw [← integral_congr_ae (ae_of_all _ hleft), hswap,
    integral_congr_ae (ae_of_all _ hright), integral_complex_ofReal]
  rfl


example : IsProbabilityMeasure (Measure.dirac (0 : ℂ)) ∧ (0 : ℝ) < 1 :=
  ⟨inferInstance, by norm_num⟩

/-- The Gaussian Fourier multiplier is continuous in frequency.
Source: arXiv:2412.09080v3, §5.4, Fourier inversion. -/
theorem continuous_gaussianDamping2 (c : ℝ) : Continuous (gaussianDamping2 c) := by
  unfold gaussianDamping2
  fun_prop

/-- Removing the Gaussian multiplier recovers one at each frequency.
Source: arXiv:2412.09080v3, §5.4, Fourier inversion. -/
theorem tendsto_gaussianDamping2 (z : ℂ) :
    Tendsto (fun c : ℝ => gaussianDamping2 c z) atTop (𝓝 1) := by
  have hi : Tendsto (fun c : ℝ => (c : ℂ)⁻¹) atTop (𝓝 0) := by
    simpa only [Function.comp_def, Complex.ofReal_inv, Complex.ofReal_zero] using
      Complex.continuous_ofReal.continuousAt.tendsto.comp tendsto_inv_atTop_zero
  have he := Complex.continuous_exp.continuousAt.tendsto.comp
    (hi.neg.mul_const ((‖z‖ : ℂ) ^ 2))
  simpa only [Function.comp_def, gaussianDamping2, neg_zero, zero_mul, Complex.exp_zero] using he

end Transformer.Modes

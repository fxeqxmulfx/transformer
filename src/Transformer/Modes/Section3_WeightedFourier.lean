import Transformer.Modes.Section3_DerivativeIntegralComparison
import Mathlib.Analysis.Fourier.FourierTransformDeriv
/-!
# Second-order spatial weights through Fourier derivatives

The inversion argument of arXiv:2412.09080v3, §5.4
`eq:higher-error-goal`, converts characteristic derivatives into spatial
weights. The complex coordinate model has the Euclidean norm and
preserves volume, whereas the project's product norm is the sup norm.
The operator norm of the coordinate map is therefore retained explicitly.
The coordinate estimates hold at every finite derivative order, including
the third order needed for the cubic spatial weight.

Positive frequency scaling has the exact Jacobian `c⁻²`. Transporting a
second derivative contributes `c²`, so these factors cancel in its norm
integral. Fourier differentiation then bounds `|w|²` times the Fourier
integral by `(2π)⁻²` times the second-derivative norm integral. All required
derivatives are genuinely integrable; no boundary term is assumed.
-/

open Real MeasureTheory ProbabilityTheory Filter VectorFourier
open scoped ENNReal Topology FourierTransform
namespace Transformer.Modes

/-- Coordinate transport and frequency scaling bound the actual iterated derivative.
Source: arXiv:2412.09080v3, §5.4 `eq:higher-error-goal`, its Fourier
coordinate change. The product-to-Euclidean norm comparison is explicit. -/
theorem norm_iteratedFDeriv_realProd_smul_le {N : ℕ} (f : ℝ × ℝ → ℂ)
    (hf : ContDiff ℝ N f) {k : ℕ} (hk : k ≤ N) (c : ℝ) (hc : 0 < c) (ξ : ℂ) :
    ‖iteratedFDeriv ℝ k (fun ζ : ℂ => f (c • Complex.equivRealProdCLM ζ)) ξ‖ ≤
      (c * ‖Complex.equivRealProdCLM.toContinuousLinearMap‖) ^ k *
        ‖iteratedFDeriv ℝ k f (c • Complex.equivRealProdCLM ξ)‖ := by
  let L : ℂ →L[ℝ] ℝ × ℝ := c • Complex.equivRealProdCLM.toContinuousLinearMap
  change ‖iteratedFDeriv ℝ k (f ∘ L) ξ‖ ≤ _
  rw [L.iteratedFDeriv_comp_right hf ξ (by exact_mod_cast hk)]
  have h := (iteratedFDeriv ℝ k f (L ξ)).norm_compContinuousLinearMap_le (fun _ : Fin k => L)
  simpa [L, norm_smul, Real.norm_eq_abs, abs_of_pos hc, mul_comm] using h

example : ContDiff ℝ 2 (characteristic2 stdGauss2) ∧ 2 ≤ (2 : ℕ) ∧ (0 : ℝ) < 2 * Real.pi :=
  ⟨contDiff_characteristic2 stdGauss2 (k := 2) (IsGaussian.memLp_id _ _ (by simp)), le_rfl, by positivity⟩

/-- The frequency coordinate change preserves volume and scaling gives `c⁻²`.
Source: arXiv:2412.09080v3, §5.4 `eq:higher-error-goal`, with the
normalization factor in the manuscript's `exp(-i dot)` convention. -/
theorem integral_realProd_smul (f : ℝ × ℝ → ℝ) (c : ℝ) (hc : 0 < c) :
    (∫ ξ : ℂ, f (c • Complex.equivRealProdCLM ξ)) = (c ^ 2)⁻¹ * ∫ ξ : ℝ × ℝ, f ξ := by
  have h := Complex.volume_preserving_equiv_real_prod.integral_comp
    Complex.measurableEquivRealProd.measurableEmbedding (fun ξ => f (c • ξ))
  calc
    _ = ∫ ξ : ℝ × ℝ, f (c • ξ) := h
    _ = _ := by
      rw [Measure.integral_comp_smul_of_nonneg volume f c (hR := hc.le)]
      simp [Module.finrank_prod, smul_eq_mul]

example : (0 : ℝ) < 2 * Real.pi := by positivity

/-- Coordinate transport preserves derivative integrability and bounds its norm integral.
Source: arXiv:2412.09080v3, §5.4 `eq:higher-error-goal`, the normalized
Fourier derivative step. The derivative scale and volume scale are both retained. -/
theorem integrable_iteratedFDeriv_realProd_smul_and_bound {N : ℕ} (f : ℝ × ℝ → ℂ)
    (hf : ContDiff ℝ N f) {k : ℕ} (hk : k ≤ N)
    (hI : Integrable (fun ξ => ‖iteratedFDeriv ℝ k f ξ‖)) (c : ℝ) (hc : 0 < c) :
    Integrable (iteratedFDeriv ℝ k (fun ζ : ℂ => f (c • Complex.equivRealProdCLM ζ))) ∧
      (∫ ξ : ℂ, ‖iteratedFDeriv ℝ k (fun ζ : ℂ => f (c • Complex.equivRealProdCLM ζ)) ξ‖) ≤
        (c * ‖Complex.equivRealProdCLM.toContinuousLinearMap‖) ^ k * (c ^ 2)⁻¹ *
          ∫ ξ : ℝ × ℝ, ‖iteratedFDeriv ℝ k f ξ‖ := by
  let L : ℂ →L[ℝ] ℝ × ℝ := c • Complex.equivRealProdCLM.toContinuousLinearMap
  have hIc := hI.comp_smul hc.ne'
  have hIp := (Complex.volume_preserving_equiv_real_prod.integrable_comp hIc.aestronglyMeasurable).mpr hIc
  have hM : Integrable (fun ξ : ℂ =>
      (c * ‖Complex.equivRealProdCLM.toContinuousLinearMap‖) ^ k *
        ‖iteratedFDeriv ℝ k f (c • Complex.equivRealProdCLM ξ)‖) := hIp.const_mul _
  have hC : Continuous (iteratedFDeriv ℝ k (fun ζ : ℂ => f (c • Complex.equivRealProdCLM ζ))) :=
    (hf.comp L.contDiff).continuous_iteratedFDeriv (by exact_mod_cast hk)
  have hD : Integrable (iteratedFDeriv ℝ k (fun ζ : ℂ => f (c • Complex.equivRealProdCLM ζ))) :=
    hM.mono' hC.aestronglyMeasurable
      (ae_of_all _ (norm_iteratedFDeriv_realProd_smul_le f hf hk c hc))
  refine ⟨hD, ?_⟩
  calc
    _ ≤ ∫ ξ : ℂ, (c * ‖Complex.equivRealProdCLM.toContinuousLinearMap‖) ^ k *
        ‖iteratedFDeriv ℝ k f (c • Complex.equivRealProdCLM ξ)‖ :=
      integral_mono_ae hD.norm hM (ae_of_all _ (norm_iteratedFDeriv_realProd_smul_le f hf hk c hc))
    _ = _ := by
      rw [integral_const_mul, integral_realProd_smul (fun ξ => ‖iteratedFDeriv ℝ k f ξ‖) c hc]
      ring

example : ContDiff ℝ 2 (characteristic2 stdGauss2) ∧ 2 ≤ (2 : ℕ) ∧
    Integrable (fun ξ => ‖iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ‖) ∧ (0 : ℝ) < 2 * Real.pi :=
  ⟨contDiff_characteristic2 stdGauss2 (k := 2) (IsGaussian.memLp_id _ _ (by simp)), le_rfl,
    integrable_characteristic2_stdGauss2_derivatives.2.norm, by positivity⟩

/-- Second Fourier differentiation controls the quadratic spatial weight.
Source: arXiv:2412.09080v3, §5.4 `eq:higher-error-goal`, derivative order
two. The exact factor `(2π)⁻²` comes from Mathlib's Fourier convention. -/
theorem norm_sq_mul_norm_fourier_le_second_derivative (f : ℂ → ℂ) (hf : ContDiff ℝ 2 f)
    (hI : ∀ k : ℕ, k ≤ 2 → Integrable (iteratedFDeriv ℝ k f)) (w : ℂ) :
    ‖w‖ ^ 2 * ‖𝓕 f w‖ ≤ ((2 * Real.pi) ^ 2)⁻¹ * ∫ ξ : ℂ, ‖iteratedFDeriv ℝ 2 f ξ‖ := by
  have hJ : 0 ≤ ∫ ξ : ℂ, ‖iteratedFDeriv ℝ 2 f ξ‖ := integral_nonneg fun ξ => norm_nonneg _
  by_cases hw : w = 0
  · subst w
    simp only [norm_zero, zero_pow (by omega : 2 ≠ 0), zero_mul]
    positivity
  have hT : 𝓕 (iteratedFDeriv ℝ 2 f) w = fourierPowSMulRight (-innerSL ℝ) (𝓕 f) w 2 := by
    exact congrFun (Real.fourier_iteratedFDeriv (N := 2) hf
      (fun k hk => hI k (by exact_mod_cast hk)) (n := 2) (by norm_num)) w
  have hDiag : (𝓕 (iteratedFDeriv ℝ 2 f) w) (fun _ : Fin 2 => w) =
      (-(2 * (Real.pi : ℂ) * Complex.I)) ^ 2 •
        ((-(‖w‖ ^ 2)) * (-(‖w‖ ^ 2))) • 𝓕 f w := by
    rw [hT, fourierPowSMulRight_apply, Fin.prod_univ_two]
    change (-(2 * (Real.pi : ℂ) * Complex.I)) ^ 2 •
      ((-inner ℝ w w) * (-inner ℝ w w)) • 𝓕 f w = _
    rw [inner_self_eq_norm_sq_to_K]
    norm_cast
  have hDiagNorm : ‖(𝓕 (iteratedFDeriv ℝ 2 f) w) (fun _ : Fin 2 => w)‖ =
      (2 * Real.pi) ^ 2 * ‖w‖ ^ 4 * ‖𝓕 f w‖ := by
    rw [hDiag]
    simp only [norm_smul, norm_pow, norm_neg, norm_mul, Complex.norm_ofNat, Complex.norm_real,
      Real.norm_eq_abs, abs_of_pos Real.pi_pos, Complex.norm_I, mul_one,
      abs_of_nonneg (norm_nonneg w)]
    ring
  have hFT : ‖𝓕 (iteratedFDeriv ℝ 2 f) w‖ ≤ ∫ ξ : ℂ, ‖iteratedFDeriv ℝ 2 f ξ‖ := by
    rw [Real.fourier_eq']
    apply (norm_integral_le_integral_norm _).trans_eq
    apply integral_congr_ae
    exact ae_of_all _ fun ξ => by
      dsimp only
      rw [norm_smul, Complex.norm_exp_ofReal_mul_I, one_mul]
  have hL := (𝓕 (iteratedFDeriv ℝ 2 f) w).le_opNorm (fun _ : Fin 2 => w)
  rw [hDiagNorm, Fin.prod_univ_two] at hL
  have h : (2 * Real.pi) ^ 2 * ‖w‖ ^ 4 * ‖𝓕 f w‖ ≤
      (∫ ξ : ℂ, ‖iteratedFDeriv ℝ 2 f ξ‖) * (‖w‖ * ‖w‖) :=
    hL.trans (mul_le_mul_of_nonneg_right hFT (by positivity))
  have hNorm : 0 < ‖w‖ := norm_pos_iff.mpr hw
  have hsmall : (2 * Real.pi) ^ 2 * (‖w‖ ^ 2 * ‖𝓕 f w‖) ≤ ∫ ξ : ℂ, ‖iteratedFDeriv ℝ 2 f ξ‖ := by
    apply le_of_mul_le_mul_right (a := ‖w‖ ^ 2) _ (by positivity)
    convert h using 1 <;> ring
  exact (le_inv_mul_iff₀ (by positivity : 0 < (2 * Real.pi) ^ 2)).mpr hsmall

example : ∃ f : ℂ → ℂ, ContDiff ℝ 2 f ∧ ∀ k : ℕ, k ≤ 2 → Integrable (iteratedFDeriv ℝ k f) := by
  have hf := contDiff_characteristic2 stdGauss2 (k := 2) (IsGaussian.memLp_id _ _ (by simp))
  refine ⟨fun ξ => characteristic2 stdGauss2 ((1 : ℝ) • Complex.equivRealProdCLM ξ),
    hf.comp ((1 : ℝ) • Complex.equivRealProdCLM.toContinuousLinearMap).contDiff, ?_⟩
  intro k hk
  have hI : Integrable (fun ξ => ‖iteratedFDeriv ℝ k (characteristic2 stdGauss2) ξ‖) := by
    interval_cases k
    · simpa only [norm_iteratedFDeriv_zero] using integrable_characteristic2_stdGauss2.norm
    · simpa only [norm_iteratedFDeriv_one] using integrable_characteristic2_stdGauss2_derivatives.1.norm
    · exact integrable_characteristic2_stdGauss2_derivatives.2.norm
  exact (integrable_iteratedFDeriv_realProd_smul_and_bound (characteristic2 stdGauss2)
    hf hk hI 1 (by norm_num)).1

end Transformer.Modes

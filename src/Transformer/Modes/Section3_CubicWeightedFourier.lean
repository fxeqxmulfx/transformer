import Transformer.Modes.Section3_WeightedFourier
import Mathlib.Analysis.Distribution.SchwartzSpace.Basic

/-!
# Cubic spatial weights through the actual third Fourier derivative

The Fourier inversion step of arXiv:2412.09080v3, §5.4
`eq:higher-error-goal`, uses all derivatives through order three when
`s = 3`. Evaluating the transformed third derivative three times along
the spatial vector gives the cubic weight. The operator norm bound and
the integral of the derivative norm then control that weighted transform.

Frequency scaling by `2π` contributes three powers to the derivative and
two inverse powers to the volume. Fourier differentiation contributes
three inverse powers, leaving precisely the source's factor `(2π)⁻²`.
The product-to-Euclidean coordinate map retains its operator norm cubed.
Combining orders zero and three gives the full weight `1 + |x|³`.

These are analytic inversion estimates for actual smooth functions with
integrable derivatives. The inverse-n rate for the normalized-sum
derivative error remains a separate spectral comparison in proving §3
`thm:br`, `s = 3`. Schwartz functions provide examples satisfying every
smoothness and integrability hypothesis.
-/

open Real MeasureTheory ProbabilityTheory Filter VectorFourier
open scoped ENNReal Topology FourierTransform
namespace Transformer.Modes

/-- Third Fourier differentiation controls the cubic spatial weight.
Source: arXiv:2412.09080v3, §5.4 `eq:higher-error-goal`, derivative order
three. The precise factor `(2π)⁻³` uses Mathlib's Fourier convention. -/
theorem norm_cube_mul_norm_fourier_le_third_derivative (f : ℂ → ℂ)
    (hf : ContDiff ℝ 3 f) (hI : ∀ k : ℕ, k ≤ 3 → Integrable (iteratedFDeriv ℝ k f))
    (w : ℂ) :
    ‖w‖ ^ 3 * ‖𝓕 f w‖ ≤ ((2 * Real.pi) ^ 3)⁻¹ * ∫ ξ : ℂ, ‖iteratedFDeriv ℝ 3 f ξ‖ := by
  have hJ : 0 ≤ ∫ ξ : ℂ, ‖iteratedFDeriv ℝ 3 f ξ‖ := integral_nonneg fun ξ => norm_nonneg _
  by_cases hw : w = 0
  · subst w
    simp only [norm_zero, zero_pow (by omega : 3 ≠ 0), zero_mul]
    positivity
  have hT : 𝓕 (iteratedFDeriv ℝ 3 f) w = fourierPowSMulRight (-innerSL ℝ) (𝓕 f) w 3 :=
    congrFun (Real.fourier_iteratedFDeriv (N := 3) hf
      (fun k hk => hI k (by exact_mod_cast hk)) (n := 3) (by norm_num)) w
  have hDiag : (𝓕 (iteratedFDeriv ℝ 3 f) w) (fun _ : Fin 3 => w) =
      (-(2 * (Real.pi : ℂ) * Complex.I)) ^ 3 •
        ((-(‖w‖ ^ 2)) * (-(‖w‖ ^ 2)) * (-(‖w‖ ^ 2))) • 𝓕 f w := by
    rw [hT, fourierPowSMulRight_apply, Fin.prod_univ_three]
    change (-(2 * (Real.pi : ℂ) * Complex.I)) ^ 3 •
      ((-inner ℝ w w) * (-inner ℝ w w) * (-inner ℝ w w)) • 𝓕 f w = _
    rw [inner_self_eq_norm_sq_to_K]
    norm_cast
  have hDiagNorm : ‖(𝓕 (iteratedFDeriv ℝ 3 f) w) (fun _ : Fin 3 => w)‖ =
      (2 * Real.pi) ^ 3 * ‖w‖ ^ 6 * ‖𝓕 f w‖ := by
    rw [hDiag]
    simp only [norm_smul, norm_pow, norm_neg, norm_mul, Complex.norm_ofNat, Complex.norm_real,
      Real.norm_eq_abs, abs_of_pos Real.pi_pos, Complex.norm_I, mul_one,
      abs_of_nonneg (norm_nonneg w)]
    ring
  have hFT : ‖𝓕 (iteratedFDeriv ℝ 3 f) w‖ ≤ ∫ ξ : ℂ, ‖iteratedFDeriv ℝ 3 f ξ‖ := by
    rw [Real.fourier_eq']
    apply (norm_integral_le_integral_norm _).trans_eq
    apply integral_congr_ae
    exact ae_of_all _ fun ξ => by
      dsimp only
      rw [norm_smul, Complex.norm_exp_ofReal_mul_I, one_mul]
  have hL := (𝓕 (iteratedFDeriv ℝ 3 f) w).le_opNorm (fun _ : Fin 3 => w)
  rw [hDiagNorm, Fin.prod_univ_three] at hL
  have h : (2 * Real.pi) ^ 3 * ‖w‖ ^ 6 * ‖𝓕 f w‖ ≤
      (∫ ξ : ℂ, ‖iteratedFDeriv ℝ 3 f ξ‖) * (‖w‖ * ‖w‖ * ‖w‖) :=
    hL.trans (mul_le_mul_of_nonneg_right hFT (by positivity))
  have hNorm : 0 < ‖w‖ := norm_pos_iff.mpr hw
  have hsmall : (2 * Real.pi) ^ 3 * (‖w‖ ^ 3 * ‖𝓕 f w‖) ≤ ∫ ξ : ℂ, ‖iteratedFDeriv ℝ 3 f ξ‖ := by
    apply le_of_mul_le_mul_right (a := ‖w‖ ^ 3) _ (by positivity)
    convert h using 1 <;> ring
  exact (le_inv_mul_iff₀ (by positivity : 0 < (2 * Real.pi) ^ 3)).mpr hsmall

example (f : SchwartzMap ℂ ℂ) : ContDiff ℝ 3 f ∧
    ∀ k : ℕ, k ≤ 3 → Integrable (iteratedFDeriv ℝ k f) := by
  refine ⟨f.smooth 3, fun k hk => ?_⟩
  apply (integrable_norm_iff ((f.smooth 3).continuous_iteratedFDeriv
    (by exact_mod_cast hk)).aestronglyMeasurable).mp
  simpa only [pow_zero, one_mul] using f.integrable_pow_mul_iteratedFDeriv (μ := volume) 0 k

/-- The source's real-pair coordinates retain the cubic spatial weight and exact Jacobian.
Source: arXiv:2412.09080v3, §5.4 `eq:higher-error-goal`, derivative order
three, with explicit control of the product-to-Euclidean operator norm. -/
theorem eucl_cube_mul_norm_fourier_realProd_le (f : ℝ × ℝ → ℂ)
    (hf : ContDiff ℝ 3 f) (hI : ∀ k : ℕ, k ≤ 3 → Integrable (fun ξ => ‖iteratedFDeriv ℝ k f ξ‖))
    (x : ℝ × ℝ) :
    eucl x ^ 3 * ‖𝓕 (fun ξ : ℂ => f ((2 * Real.pi) • Complex.equivRealProdCLM ξ))
      (Complex.measurableEquivRealProd.symm x)‖ ≤ ((2 * Real.pi) ^ 2)⁻¹ *
        ‖Complex.equivRealProdCLM.toContinuousLinearMap‖ ^ 3 * ∫ ξ, ‖iteratedFDeriv ℝ 3 f ξ‖ := by
  let L : ℂ →L[ℝ] ℝ × ℝ := (2 * Real.pi) • Complex.equivRealProdCLM.toContinuousLinearMap
  let g : ℂ → ℂ := fun ξ => f ((2 * Real.pi) • Complex.equivRealProdCLM ξ)
  have hg : ContDiff ℝ 3 g := hf.comp L.contDiff
  have hIg (k : ℕ) (hk : k ≤ 3) : Integrable (iteratedFDeriv ℝ k g) :=
    (integrable_iteratedFDeriv_realProd_smul_and_bound f hf hk (hI k hk)
      (2 * Real.pi) (by positivity)).1
  have h3 := norm_cube_mul_norm_fourier_le_third_derivative g hg hIg
    (Complex.measurableEquivRealProd.symm x)
  have hEucl : ‖Complex.measurableEquivRealProd.symm x‖ = eucl x := by
    rw [Complex.norm_eq_sqrt_sq_add_sq]
    rfl
  rw [hEucl] at h3
  have hD := (integrable_iteratedFDeriv_realProd_smul_and_bound f hf le_rfl (hI 3 le_rfl)
    (2 * Real.pi) (by positivity)).2
  calc
    _ ≤ ((2 * Real.pi) ^ 3)⁻¹ * ∫ ξ : ℂ, ‖iteratedFDeriv ℝ 3 g ξ‖ := h3
    _ ≤ ((2 * Real.pi) ^ 3)⁻¹ *
        ((2 * Real.pi * ‖Complex.equivRealProdCLM.toContinuousLinearMap‖) ^ 3 *
          ((2 * Real.pi) ^ 2)⁻¹ * ∫ ξ, ‖iteratedFDeriv ℝ 3 f ξ‖) :=
      mul_le_mul_of_nonneg_left hD (by positivity)
    _ = _ := by field_simp

example (f : SchwartzMap (ℝ × ℝ) ℂ) : ContDiff ℝ 3 f ∧
    ∀ k : ℕ, k ≤ 3 → Integrable (fun ξ => ‖iteratedFDeriv ℝ k f ξ‖) := by
  refine ⟨f.smooth 3, fun k _ => ?_⟩
  simpa only [pow_zero, one_mul] using f.integrable_pow_mul_iteratedFDeriv (μ := volume) 0 k

/-- Orders zero and three give the full cubic spatial weight in Fourier inversion.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, and §5.4
`eq:higher-error-goal`. This is the analytic bound before estimating the spectral error. -/
theorem weighted_norm_fourier_realProd_le (f : ℝ × ℝ → ℂ)
    (hf : ContDiff ℝ 3 f) (hI : ∀ k : ℕ, k ≤ 3 → Integrable (fun ξ => ‖iteratedFDeriv ℝ k f ξ‖))
    (x : ℝ × ℝ) :
    (1 + eucl x ^ 3) * ‖𝓕 (fun ξ : ℂ => f ((2 * Real.pi) • Complex.equivRealProdCLM ξ))
      (Complex.measurableEquivRealProd.symm x)‖ ≤ ((2 * Real.pi) ^ 2)⁻¹ *
        ((∫ ξ, ‖f ξ‖) + ‖Complex.equivRealProdCLM.toContinuousLinearMap‖ ^ 3 *
          ∫ ξ, ‖iteratedFDeriv ℝ 3 f ξ‖) := by
  let g : ℂ → ℂ := fun ξ => f ((2 * Real.pi) • Complex.equivRealProdCLM ξ)
  let w : ℂ := Complex.measurableEquivRealProd.symm x
  have h0 : ‖𝓕 g w‖ ≤ ((2 * Real.pi) ^ 2)⁻¹ * ∫ ξ, ‖f ξ‖ := by
    calc
      _ ≤ ∫ ξ : ℂ, ‖g ξ‖ := by
        rw [Real.fourier_eq']
        apply (norm_integral_le_integral_norm _).trans_eq
        apply integral_congr_ae
        exact ae_of_all _ fun ξ => by
          dsimp only
          rw [norm_smul, Complex.norm_exp_ofReal_mul_I, one_mul]
      _ = _ := integral_realProd_smul (fun ξ => ‖f ξ‖) (2 * Real.pi) (by positivity)
  have h3 := eucl_cube_mul_norm_fourier_realProd_le f hf hI x
  calc
    _ = ‖𝓕 g w‖ + eucl x ^ 3 * ‖𝓕 g w‖ := by dsimp only [g, w]; ring
    _ ≤ ((2 * Real.pi) ^ 2)⁻¹ * (∫ ξ, ‖f ξ‖) + ((2 * Real.pi) ^ 2)⁻¹ *
        ‖Complex.equivRealProdCLM.toContinuousLinearMap‖ ^ 3 * ∫ ξ, ‖iteratedFDeriv ℝ 3 f ξ‖ :=
      add_le_add h0 h3
    _ = _ := by ring

example (f : SchwartzMap (ℝ × ℝ) ℂ) : ContDiff ℝ 3 f ∧
    ∀ k : ℕ, k ≤ 3 → Integrable (fun ξ => ‖iteratedFDeriv ℝ k f ξ‖) := by
  refine ⟨f.smooth 3, fun k _ => ?_⟩
  simpa only [pow_zero, one_mul] using f.integrable_pow_mul_iteratedFDeriv (μ := volume) 0 k

end Transformer.Modes

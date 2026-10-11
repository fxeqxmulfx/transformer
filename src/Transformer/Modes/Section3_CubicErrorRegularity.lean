import Transformer.Modes.Section3_ThirdDerivativeTail
import Transformer.Modes.Section3_CubicCorrectionDerivativeBounds

/-!
# Regularity of the actual cubic-corrected characteristic error

Fourier inversion with the cubic weight in arXiv:2412.09080v3,
§5.4 `eq:higher-error-goal`, requires continuous derivatives and
integrable derivative norms at every order from zero through three.
These conditions now follow from a finite third moment and the
original integrable-characteristic-power hypothesis.

The normalized sum, Gaussian function, and Gaussian-damped cubic
coefficient are differentiated separately. Subtraction and constant
scalar multiplication give the literal derivative of their error;
no derivative bound or regularity condition is assumed on that error.

The first and second derivatives use the original integrable power
with the global characteristic modulus bound one. The third uses
the proved third-derivative integrability result. Norm integrability
is sufficient in real-pair coordinates, where the product norm is
not the Euclidean norm used by Fourier inversion.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The actual cubic-corrected normalized-sum error has three continuous derivatives.
Source: arXiv:2412.09080v3, §5.4 `eq:higher-error-goal`,
the regularity required before the cubic Fourier derivative. -/
theorem contDiff_characteristic_scaledSum_cubic_error
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ) (n : ℕ) :
    ContDiff ℝ 3 (fun η =>
      characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) η -
        characteristic2 stdGauss2 η - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
          characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) := by
  have hs := contDiff_characteristic_scaledSum_of_memLp μ 3 hmom n
  have hg := contDiff_characteristic2 stdGauss2 (k := 3) (IsGaussian.memLp_id _ _ (by simp))
  have hb : ContDiff ℝ 3 (cubicCharacteristicCorrection μ) :=
    (contDiff_cubicCharacteristicCorrection_of_memLp μ hmom).of_le (by simp)
  exact (hs.sub hg).sub ((contDiff_const.mul hg).mul hb)

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- Differentiating the actual corrected error retains the full derivative of the Gaussian cubic product.
Source: arXiv:2412.09080v3, §5.4 `eq:higher-error-goal` and
`eq:big-z-poly`, including derivatives of both Gaussian and cubic factors. -/
theorem iteratedFDeriv_characteristic_scaledSum_cubic_error
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (n k : ℕ) (hk : k ≤ 3) (ξ : ℝ × ℝ) :
    iteratedFDeriv ℝ k (fun η =>
      characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) η -
        characteristic2 stdGauss2 η - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
          characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) ξ =
      iteratedFDeriv ℝ k (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ -
        iteratedFDeriv ℝ k (characteristic2 stdGauss2) ξ -
          (((Real.sqrt n)⁻¹ : ℝ) : ℂ) • iteratedFDeriv ℝ k
            (fun η => characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) ξ := by
  have hs := (contDiff_characteristic_scaledSum_of_memLp μ 3 hmom n).of_le
    (show (k : WithTop ℕ∞) ≤ 3 by exact_mod_cast hk)
  have hg := (contDiff_characteristic2 stdGauss2 (k := 3)
    (IsGaussian.memLp_id _ _ (by simp))).of_le (show (k : WithTop ℕ∞) ≤ 3 by exact_mod_cast hk)
  have hb := (contDiff_cubicCharacteristicCorrection_of_memLp μ hmom).of_le
    (show (k : WithTop ℕ∞) ≤ ⊤ by simp)
  have he : (fun η => (((Real.sqrt n)⁻¹ : ℝ) : ℂ) * characteristic2 stdGauss2 η *
      cubicCharacteristicCorrection μ η) =
      fun η => (((Real.sqrt n)⁻¹ : ℝ) : ℂ) •
        (characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) := by
    funext η
    simp only [smul_eq_mul, mul_assoc]
  have hcorr : ContDiff ℝ (k : WithTop ℕ∞) (fun η =>
      (((Real.sqrt n)⁻¹ : ℝ) : ℂ) * characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) :=
    (contDiff_const.mul hg).mul hb
  rw [fun_iteratedFDeriv_sub (hs.sub hg) hcorr, fun_iteratedFDeriv_sub hs hg, he]
  dsimp only
  rw [iteratedFDeriv_const_smul_apply' (hg.mul hb).contDiffAt]

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 ∧ (3 : ℕ) ≤ 3 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp), le_rfl⟩

/-- Every actual normalized-sum derivative norm through order three is eventually integrable.
Source: arXiv:2412.09080v3, §5.4 `eq:higher-error-goal` and
`eq:big-z-exp`, using the original integrable characteristic power. -/
theorem eventually_integrable_norm_iteratedFDeriv_characteristic_scaledSum
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (hcf : HasIntegrableCharFun μ) :
    ∀ᶠ n : ℕ in atTop, ∀ k : ℕ, k ≤ 3 →
      Integrable (fun ξ => ‖iteratedFDeriv ℝ k
        (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ‖) := by
  have hpower := hcf
  obtain ⟨ν, hν, hI⟩ := hpower
  have hmom2 : MemLp id 2 μ := hmom.mono_exponent (by norm_num)
  filter_upwards [eventually_integrable_characteristic_scaledSum μ hcf,
    eventually_integrable_iteratedFDeriv_three_characteristic_scaledSum μ hmom hcf,
    eventually_ge_atTop (ν + 3)] with n hI0 hI3 hn
  have hI1 := (integrable_fderiv_characteristic_scaledSum_and_tail μ hmom2 (n := n) hν (by omega) hI
    zero_le_one (a := 0) (fun ξ _ => norm_characteristic2_le_one μ ξ)).1
  have hI2 := (integrable_iteratedFDeriv_two_characteristic_scaledSum_and_tail μ hmom2 (n := n) hν (by omega) hI
    zero_le_one (a := 0) (fun ξ _ => norm_characteristic2_le_one μ ξ)).1
  intro k hk
  interval_cases k
  · simpa only [norm_iteratedFDeriv_zero] using hI0.norm
  · simpa only [norm_iteratedFDeriv_one] using hI1.norm
  · exact hI2.norm
  · exact hI3.norm

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp), hasIntegrableCharFun_stdGauss2⟩

/-- The actual cubic-corrected error has all derivative norm integrals required by cubic Fourier inversion.
Source: arXiv:2412.09080v3, §5.4 `eq:higher-error-goal`,
with integrability derived at every order zero through three. -/
theorem eventually_integrable_norm_iteratedFDeriv_characteristic_scaledSum_cubic_error
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (hcf : HasIntegrableCharFun μ) :
    ∀ᶠ n : ℕ in atTop, ∀ k : ℕ, k ≤ 3 →
      Integrable (fun ξ => ‖iteratedFDeriv ℝ k (fun η =>
        characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) η -
          characteristic2 stdGauss2 η - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
            characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) ξ‖) := by
  filter_upwards [eventually_integrable_norm_iteratedFDeriv_characteristic_scaledSum μ hmom hcf]
    with n hIS
  intro k hk
  have hs : ContDiff ℝ 3 (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) :=
    contDiff_characteristic_scaledSum_of_memLp μ 3 hmom n
  have hg := contDiff_characteristic2 stdGauss2 (k := 3) (IsGaussian.memLp_id _ _ (by simp))
  have hISk := (integrable_norm_iff (hs.continuous_iteratedFDeriv
    (by exact_mod_cast hk)).aestronglyMeasurable).mp (hIS k hk)
  have hIGnorm : Integrable (fun ξ => ‖iteratedFDeriv ℝ k (characteristic2 stdGauss2) ξ‖) := by
    simpa only [pow_zero, one_mul] using integrable_norm_pow_iteratedFDeriv_characteristic2_stdGauss2 0 k hk
  have hIG := (integrable_norm_iff (hg.continuous_iteratedFDeriv
    (by exact_mod_cast hk)).aestronglyMeasurable).mp hIGnorm
  have hIB := integrable_iteratedFDeriv_gaussian_cubicCorrection μ hmom k hk
  have hI := ((hISk.sub hIG).sub (Integrable.smul (((Real.sqrt n)⁻¹ : ℝ) : ℂ) hIB)).norm
  simpa only [iteratedFDeriv_characteristic_scaledSum_cubic_error μ hmom n k hk,
    Pi.sub_apply, Pi.smul_apply] using hI

/-- The Gaussian law supplies both the third moment
and the integrable characteristic power. -/
example : ∀ᶠ n : ℕ in atTop, ∀ k : ℕ, k ≤ 3 →
    Integrable (fun ξ => ‖iteratedFDeriv ℝ k (fun η =>
      characteristic2 ((Measure.pi fun _ : Fin n => stdGauss2).map (scaledSum n)) η -
        characteristic2 stdGauss2 η - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
          characteristic2 stdGauss2 η * cubicCharacteristicCorrection stdGauss2 η) ξ‖) := by
  exact eventually_integrable_norm_iteratedFDeriv_characteristic_scaledSum_cubic_error stdGauss2
    (IsGaussian.memLp_id _ _ (by simp)) hasIntegrableCharFun_stdGauss2

end Transformer.Modes

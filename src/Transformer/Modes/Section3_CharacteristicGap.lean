import Transformer.Modes.Section3_DensityInversion
import Mathlib.Analysis.Fourier.RiemannLebesgueLemma
import Mathlib.Analysis.Convex.Integral
import Mathlib.Analysis.InnerProductSpace.Convex
import Mathlib.Analysis.Normed.Group.CocompactMap
/-!
# A strict characteristic-function bound for the general Edgeworth law

The suitable condition in §3 `thm:br` of arXiv:2412.09080v3 is an
integrable norm power of the characteristic function. It already gives
actual normalized sums with integrable characteristic functions.
Fourier inversion and the Riemann–Lebesgue lemma then prove decay for
those sums. Strict convexity of the complex norm excludes modulus one
at a nonzero frequency: that modulus would persist along every natural
frequency multiple, including for the normalized sum, contradicting
decay. This supplies the pointwise strict inequality needed in §5.4's
large-frequency argument, without extra moment or density hypotheses.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped FourierTransform Topology ENNReal
namespace Transformer.Modes

/-- Norm one forces a probability law's phase to be almost surely
constant, so natural frequency multiples give powers. Source:
arXiv:2412.09080v3, §5.4, the strict bound defining `ε` in `eq:big-z-exp`. -/
theorem characteristic2_nat_mul_eq_pow_of_norm_eq_one
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (ξ : ℝ × ℝ)
    (hξ : ‖characteristic2 μ ξ‖ = 1) (n : ℕ) :
    characteristic2 μ ((n : ℝ) • ξ) = characteristic2 μ ξ ^ n := by
  let f : ℝ × ℝ → ℂ := fun z =>
    Complex.exp (((ξ.1 * z.1 + ξ.2 * z.2 : ℝ) : ℂ) * Complex.I)
  have hbound : ∀ᵐ z ∂μ, ‖f z‖ ≤ 1 := ae_of_all _ fun z => by
    dsimp [f]; simp [Complex.norm_exp]
  rcases ae_eq_const_or_norm_integral_lt_of_norm_le_const hbound with hc | hs
  · have he : f =ᵐ[μ] fun _ => characteristic2 μ ξ := by
      filter_upwards [hc] with z hz
      change f z = (⨍ y, f y ∂μ) at hz
      rw [average_eq_integral] at hz
      exact hz
    have hphase (z : ℝ × ℝ) :
        Complex.exp (((((n : ℝ) • ξ).1 * z.1 + ((n : ℝ) • ξ).2 * z.2 : ℝ) : ℂ) *
          Complex.I) = f z ^ n := by
      rw [← Complex.exp_nat_mul]
      apply congrArg Complex.exp
      simp only [Prod.smul_fst, Prod.smul_snd, smul_eq_mul, Complex.ofReal_add,
        Complex.ofReal_mul, Complex.ofReal_natCast]
      ring
    unfold characteristic2
    calc
      _ = ∫ z, f z ^ n ∂μ := integral_congr_ae (ae_of_all _ hphase)
      _ = ∫ _ : ℝ × ℝ, characteristic2 μ ξ ^ n ∂μ :=
        integral_congr_ae (he.fun_comp (fun z => z ^ n))
      _ = _ := by simp [characteristic2]
  · have hlt : ‖characteristic2 μ ξ‖ < 1 := by
      simpa only [probReal_univ, one_mul, f, characteristic2] using hs
    exact False.elim (by linarith)

example : IsProbabilityMeasure stdGauss2 ∧ ‖characteristic2 stdGauss2 (0 : ℝ × ℝ)‖ = 1 := by
  exact ⟨inferInstance, by simp [characteristic2]⟩

/-- An integrable characteristic function tends to zero at infinity.
Fourier inversion uses the actual integrable inverse density proved
previously, followed by the Riemann–Lebesgue lemma. Source:
arXiv:2412.09080v3, §5.4, justifying the large-frequency bound defining `ε`. -/
theorem tendsto_charFun_zero_of_integrable (μ : Measure ℂ) [IsProbabilityMeasure μ]
    (hI : Integrable (charFun μ)) : Tendsto (charFun μ) (cocompact ℂ) (𝓝 0) := by
  have hψI := integrable_normalizedCharacteristic μ hI
  have hψc : Continuous (normalizedCharacteristic μ) := by
    unfold normalizedCharacteristic
    fun_prop
  have hQI : Integrable (𝓕 (normalizedCharacteristic μ)) := by
    have he : 𝓕 (normalizedCharacteristic μ) = fun w => (inverseCharacteristic μ w : ℂ) := by
      funext w
      exact (ofReal_inverseCharacteristic μ hI w).symm
    rw [he]
    exact (integrable_inverseCharacteristic μ hI).ofReal
  have hinv := hψc.fourierInv_fourier_eq hψI hQI
  let f : ℂ → ℂ := 𝓕 (normalizedCharacteristic μ)
  have ht : Tendsto (𝓕 f) (cocompact ℂ) (𝓝 0) :=
    tendsto_integral_exp_inner_smul_cocompact f
  have hχ : charFun μ = fun ξ => 𝓕 f ((-(2 * Real.pi)⁻¹ : ℝ) • ξ) := by
    funext ξ
    have h := congrFun hinv ((2 * Real.pi)⁻¹ • ξ)
    have he : normalizedCharacteristic μ ((2 * Real.pi)⁻¹ • ξ) = charFun μ ξ := by
      unfold normalizedCharacteristic
      rw [smul_smul, mul_inv_cancel₀ (by positivity : (2 * Real.pi : ℝ) ≠ 0), one_smul]
    rw [he, Real.fourierInv_eq_fourier_neg] at h
    simpa only [neg_smul, f] using h.symm
  rw [hχ]
  apply ht.comp
  apply Filter.tendsto_cocompact_cocompact_of_norm
  intro ε
  have hc : 0 < ‖(-(2 * Real.pi)⁻¹ : ℝ)‖ := by
    rw [Real.norm_eq_abs, abs_neg, abs_of_pos (by positivity)]
    positivity
  refine ⟨ε / ‖(-(2 * Real.pi)⁻¹ : ℝ)‖, ?_⟩
  intro ξ hξ
  rw [norm_smul]
  have h := (div_lt_iff₀ hc).mp hξ
  simpa only [mul_comm] using h

example := tendsto_charFun_zero_of_integrable _
  (integrable_charFun_complexLaw_scaledSum_lawY (n := 17)
    (by norm_num : (0 : ℝ) < 3) 0 (by norm_num))

/-- The source's integrable-characteristic-power condition forces strict
modulus below one at every nonzero frequency, for an arbitrary probability
law. Norm one would persist along frequency multiples of a normalized
sum with integrable characteristic function, contradicting its decay.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4, definition of `ε`. -/
theorem norm_characteristic2_lt_one
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hcf : HasIntegrableCharFun μ)
    {ξ : ℝ × ℝ} (hξ : ξ ≠ 0) : ‖characteristic2 μ ξ‖ < 1 := by
  by_contra hlt
  have heq : ‖characteristic2 μ ξ‖ = 1 :=
    le_antisymm (norm_characteristic2_le_one μ ξ) (not_lt.mp hlt)
  obtain ⟨n, hIn, hn⟩ :=
    ((eventually_integrable_characteristic_scaledSum μ hcf).and (eventually_ge_atTop 1)).exists
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n from hn)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
  let ν : Measure (ℝ × ℝ) := (Measure.pi fun _ : Fin n => μ).map (scaledSum n)
  have hI : Integrable (charFun (complexLaw ν)) := integrable_charFun_complexLaw ν hIn
  let ζ : ℂ := ⟨ξ.1, ξ.2⟩
  have hζ : ζ ≠ 0 := by
    intro h
    apply hξ
    apply Prod.ext
    · exact congrArg Complex.re h
    · exact congrArg Complex.im h
  let w : ℕ → ℂ := fun k => ((k : ℝ) * Real.sqrt n) • ζ
  have hunit (k : ℕ) : ‖charFun (complexLaw ν) (w k)‖ = 1 := by
    rw [charFun_complexLaw]
    have hp : ((w k).re, (w k).im) = ((k : ℝ) * Real.sqrt n) • ξ := by
      ext <;> simp [w, ζ, Complex.real_smul]
    rw [hp]
    dsimp only [ν]
    rw [characteristic_scaledSum]
    have hfreq : (Real.sqrt (n : ℝ))⁻¹ • (((k : ℝ) * Real.sqrt n) • ξ) = (k : ℝ) • ξ := by
      rw [smul_smul]
      congr 1
      field_simp
    rw [hfreq, characteristic2_nat_mul_eq_pow_of_norm_eq_one μ ξ heq k,
      norm_pow, norm_pow, heq, one_pow, one_pow]
  have hN : Tendsto (fun k : ℕ => ‖w k‖) atTop atTop := by
    have h := Tendsto.atTop_mul_const (mul_pos hs (norm_pos_iff.mpr hζ))
      (tendsto_natCast_atTop_atTop : Tendsto (fun k : ℕ => (k : ℝ)) atTop atTop)
    convert h using 1
    funext k
    change ‖((k : ℝ) * Real.sqrt n) • ζ‖ = _
    rw [norm_smul, Real.norm_of_nonneg (mul_nonneg (Nat.cast_nonneg _) hs.le)]
    ring
  have hw : Tendsto w atTop (cocompact ℂ) :=
    tendsto_cocompact_of_tendsto_dist_comp_atTop (0 : ℂ) (by
      simpa only [dist_zero_right] using hN)
  have ht := (tendsto_charFun_zero_of_integrable (complexLaw ν) hI).comp hw
  have htN : Tendsto (fun k => ‖charFun (complexLaw ν) (w k)‖) atTop (𝓝 (0 : ℝ)) := by
    simpa only [norm_zero, Function.comp_def] using ht.norm
  have hbad : (1 : ℝ) ≤ 0 := ge_of_tendsto' htN fun k => (hunit k).ge
  linarith

example : IsProbabilityMeasure stdGauss2 ∧ HasIntegrableCharFun stdGauss2 ∧
    ((1, 0) : ℝ × ℝ) ≠ 0 :=
  ⟨inferInstance, hasIntegrableCharFun_stdGauss2, by norm_num⟩

end Transformer.Modes

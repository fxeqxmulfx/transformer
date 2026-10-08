import Transformer.Modes.Section3_PhaseTaylor
import Mathlib.MeasureTheory.SpecificCodomains.Pi
/-!
# The quadratic expansion of a standardized characteristic function

In §3 `thm:br` of arXiv:2412.09080v3 the summands are centered with
identity covariance and have a finite third moment when `s = 2`. The
actual phase integral therefore has quadratic term `1 - |ξ|² / 2`.
We integrate the phase Taylor bound, compute the linear and quadratic
moments from these hypotheses, and bound the cubic error uniformly in
frequency by the third sup-norm moment. No exponential moment is used.

These are the single-summand estimates for the small-frequency proof in
§5.4 `eq:br-9.10`. That displayed estimate also involves the normalized
sum and derivatives; those further steps are not asserted here.
-/

open Real MeasureTheory Filter
open scoped ENNReal
namespace Transformer.Modes

/-- Linear forms preserve finite moments. Source: arXiv:2412.09080v3,
§3 `thm:br` and §5.4, the moment input to `eq:br-9.10`. -/
theorem memLp_dot (μ : Measure (ℝ × ℝ)) {p : ℝ≥0∞} (hm : MemLp id p μ)
    (ξ : ℝ × ℝ) : MemLp (fun z => ξ.1 * z.1 + ξ.2 * z.2) p μ := by
  exact (hm.fst.const_mul ξ.1).add (hm.snd.const_mul ξ.2)

example : MemLp id 3 stdGauss2 :=
  ProbabilityTheory.IsGaussian.memLp_id _ _ (by simp)

/-- Every linear form of a standardized law is centered.
Source: arXiv:2412.09080v3, §3 `thm:br`, the mean-zero hypothesis. -/
theorem integral_dot_of_standardized (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ]
    (hμ : IsStandardized μ) (ξ : ℝ × ℝ) :
    ∫ z, ξ.1 * z.1 + ξ.2 * z.2 ∂μ = 0 := by
  have h1 : Integrable (fun z : ℝ × ℝ => z.1) μ := hμ.memLp.fst.integrable (by norm_num)
  have h2 : Integrable (fun z : ℝ × ℝ => z.2) μ := hμ.memLp.snd.integrable (by norm_num)
  rw [integral_add (h1.const_mul ξ.1) (h2.const_mul ξ.2), integral_const_mul,
    integral_const_mul, hμ.mean_fst, hμ.mean_snd]
  ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2⟩

/-- The second moment of a linear form is its frequency's Euclidean
square. Source: arXiv:2412.09080v3, §3 `thm:br`, identity covariance. -/
theorem integral_dot_sq_of_standardized (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ]
    (hμ : IsStandardized μ) (ξ : ℝ × ℝ) :
    ∫ z, (ξ.1 * z.1 + ξ.2 * z.2) ^ 2 ∂μ = ξ.1 ^ 2 + ξ.2 ^ 2 := by
  have h1 : Integrable (fun z : ℝ × ℝ => z.1 ^ 2) μ := hμ.memLp.fst.integrable_sq
  have h2 : Integrable (fun z : ℝ × ℝ => z.2 ^ 2) μ := hμ.memLp.snd.integrable_sq
  have h12 : Integrable (fun z : ℝ × ℝ => z.1 * z.2) μ :=
    hμ.memLp.fst.integrable_mul hμ.memLp.snd
  have he : (fun z : ℝ × ℝ => (ξ.1 * z.1 + ξ.2 * z.2) ^ 2) =
      fun z => ξ.1 ^ 2 * z.1 ^ 2 + 2 * ξ.1 * ξ.2 * (z.1 * z.2) + ξ.2 ^ 2 * z.2 ^ 2 := by
    funext z; ring
  rw [he, integral_add (f := fun z : ℝ × ℝ =>
      ξ.1 ^ 2 * z.1 ^ 2 + 2 * ξ.1 * ξ.2 * (z.1 * z.2))
      ((h1.const_mul (ξ.1 ^ 2)).add (h12.const_mul (2 * ξ.1 * ξ.2)))
      (h2.const_mul (ξ.2 ^ 2)),
    integral_add (h1.const_mul (ξ.1 ^ 2)) (h12.const_mul (2 * ξ.1 * ξ.2)), integral_const_mul,
    integral_const_mul, integral_const_mul, hμ.var_fst, hμ.var_snd, hμ.cov]
  ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2⟩

/-- The actual characteristic function has a quadratic expansion whose
remainder is bounded by the third moment of the linear form. The factor
`1/2` is nonsharp. Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 2`,
and §5.4, the unscaled Taylor step underlying `eq:br-9.10`. -/
theorem norm_characteristic2_sub_quadratic_moment
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (ξ : ℝ × ℝ) :
    ‖characteristic2 μ ξ - (1 - ((ξ.1 ^ 2 + ξ.2 ^ 2 : ℝ) : ℂ) / 2)‖ ≤
      (∫ z, |ξ.1 * z.1 + ξ.2 * z.2| ^ 3 ∂μ) / 2 := by
  let f : ℝ × ℝ → ℝ := fun z => ξ.1 * z.1 + ξ.2 * z.2
  have hf : Continuous f := by fun_prop
  have h3 : Integrable (fun z => |f z| ^ 3) μ := by
    simpa using (memLp_dot μ hmom ξ).integrable_norm_pow (by decide : (3 : ℕ) ≠ 0)
  have h1 : Integrable f μ := (memLp_dot μ hμ.memLp ξ).integrable (by norm_num)
  have h2 : Integrable (fun z => f z ^ 2) μ := (memLp_dot μ hμ.memLp ξ).integrable_sq
  have he (u : ℝ) : 1 + (u : ℂ) * Complex.I + ((u : ℂ) * Complex.I) ^ 2 / 2 =
      (1 - ((u ^ 2 : ℝ) : ℂ) / 2) + (u : ℂ) * Complex.I := by
    rw [mul_pow, Complex.I_sq]
    push_cast
    ring
  have hint : (∫ z, (1 + (f z : ℂ) * Complex.I + ((f z : ℂ) * Complex.I) ^ 2 / 2) ∂μ) =
      1 - ((ξ.1 ^ 2 + ξ.2 ^ 2 : ℝ) : ℂ) / 2 := by
    simp_rw [he]
    have ha := integral_add ((integrable_const (1 : ℂ)).sub (h2.ofReal.div_const 2))
      (h1.ofReal.mul_const Complex.I)
    have hs := integral_sub (integrable_const (1 : ℂ)) (h2.ofReal.div_const 2)
    calc
      _ = (∫ _ : ℝ × ℝ, (1 : ℂ) ∂μ) - (∫ z, ((f z ^ 2 : ℝ) : ℂ) / 2 ∂μ) +
          (∫ z, (f z : ℂ) * Complex.I ∂μ) :=
        ha.trans (congrArg (fun w : ℂ => w + ∫ z, (f z : ℂ) * Complex.I ∂μ) hs)
      _ = _ := by
        rw [integral_mul_const, integral_div, integral_complex_ofReal, integral_complex_ofReal]
        dsimp [f]
        rw [integral_dot_sq_of_standardized μ hμ ξ, integral_dot_of_standardized μ hμ ξ]
        simp
  have h := norm_integral_complexExp_sub_quadratic μ f hf h3
  rw [hint] at h
  exact h

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 3 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, ProbabilityTheory.IsGaussian.memLp_id _ _ (by simp)⟩

/-- The explicit linear form is bounded using the product sup norm.
The factor two accounts for both coordinates. Source: arXiv:2412.09080v3,
§5.4, the moment comparison in the small-frequency argument. -/
theorem abs_dot_le_two_norm_mul (ξ z : ℝ × ℝ) :
    |ξ.1 * z.1 + ξ.2 * z.2| ≤ 2 * ‖ξ‖ * ‖z‖ := by
  have hfξ : |ξ.1| ≤ ‖ξ‖ := by simpa using norm_fst_le ξ
  have hsξ : |ξ.2| ≤ ‖ξ‖ := by simpa using norm_snd_le ξ
  have hfz : |z.1| ≤ ‖z‖ := by simpa using norm_fst_le z
  have hsz : |z.2| ≤ ‖z‖ := by simpa using norm_snd_le z
  calc
    _ ≤ |ξ.1 * z.1| + |ξ.2 * z.2| := abs_add_le _ _
    _ = |ξ.1| * |z.1| + |ξ.2| * |z.2| := by rw [abs_mul, abs_mul]
    _ ≤ ‖ξ‖ * ‖z‖ + ‖ξ‖ * ‖z‖ :=
      add_le_add (mul_le_mul hfξ hfz (abs_nonneg _) (norm_nonneg _))
        (mul_le_mul hsξ hsz (abs_nonneg _) (norm_nonneg _))
    _ = _ := by ring

/-- A global cubic remainder bound from the finite third sup-norm
moment. Its constant is independent of the frequency. Source:
arXiv:2412.09080v3, §3 `thm:br` and §5.4 `eq:br-9.10`; this remains
an estimate for one summand, before taking the normalized-sum power. -/
theorem norm_characteristic2_sub_quadratic
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (ξ : ℝ × ℝ) :
    ‖characteristic2 μ ξ - (1 - ((ξ.1 ^ 2 + ξ.2 ^ 2 : ℝ) : ℂ) / 2)‖ ≤
      4 * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ ^ 3 := by
  have hi : Integrable (fun z => ‖z‖ ^ 3) μ := by
    simpa using hmom.integrable_norm_pow (by decide : (3 : ℕ) ≠ 0)
  have hd : Integrable (fun z => |ξ.1 * z.1 + ξ.2 * z.2| ^ 3) μ := by
    simpa using (memLp_dot μ hmom ξ).integrable_norm_pow (by decide : (3 : ℕ) ≠ 0)
  have hbound : (∫ z, |ξ.1 * z.1 + ξ.2 * z.2| ^ 3 ∂μ) ≤
      8 * ‖ξ‖ ^ 3 * (∫ z, ‖z‖ ^ 3 ∂μ) := by
    calc
      _ ≤ ∫ z, 8 * ‖ξ‖ ^ 3 * ‖z‖ ^ 3 ∂μ := by
        apply integral_mono_ae hd (hi.const_mul _)
        exact ae_of_all _ fun z => by
          have h := pow_le_pow_left₀ (abs_nonneg _) (abs_dot_le_two_norm_mul ξ z) 3
          nlinarith [h]
      _ = _ := integral_const_mul _ _
  have h := norm_characteristic2_sub_quadratic_moment μ hμ hmom ξ
  linarith

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 3 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, ProbabilityTheory.IsGaussian.memLp_id _ _ (by simp)⟩

end Transformer.Modes

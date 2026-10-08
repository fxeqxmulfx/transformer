import Transformer.Modes.Section3_DerivativeMoments
import Transformer.Modes.Section3_WeightedPhase
/-!
# Finite-third-moment Taylor bounds for characteristic derivatives

The first two derivative estimates underlying arXiv:2412.09080v3,
§5.4 `eq:br-9.10`, are obtained by expanding the unit phase before
integrating. The weighted remainders require precisely the third moment.
Standardization computes the first and second polynomial terms as
`-dot ξ v` and `-dot v₀ v₁`, respectively.

The explicit nonsharp factor eight is the product of three coordinate
comparison factors for the product sup norm. The estimates hold for all
frequencies before normalized-sum powers and Gaussian damping are applied.
On the unit frequency ball the first derivative is additionally bounded
by a constant times the frequency norm, as needed by the product rule.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The first actual characteristic derivative has quadratic
Taylor error, controlled by the finite third moment alone. Source:
arXiv:2412.09080v3, §3 `thm:br`, `s = 2`, and §5.4 `eq:br-9.10`,
the unscaled first-derivative expansion. The factor eight is nonsharp. -/
theorem norm_fderiv_characteristic2_add_dot_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (ξ v : ℝ × ℝ) :
    ‖fderiv ℝ (characteristic2 μ) ξ v + ((ξ.1 * v.1 + ξ.2 * v.2 : ℝ) : ℂ)‖ ≤
      8 * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ ^ 2 * ‖v‖ := by
  let w : ℝ × ℝ → ℝ := fun z => v.1 * z.1 + v.2 * z.2
  let f : ℝ × ℝ → ℝ := fun z => ξ.1 * z.1 + ξ.2 * z.2
  have hw : Integrable w μ := (memLp_dot μ hμ.memLp v).integrable (by norm_num)
  have hwf : Integrable (fun z => w z * f z) μ :=
    (memLp_dot μ hμ.memLp v).integrable_mul (memLp_dot μ hμ.memLp ξ)
  have hM := hmom.integrable_norm_pow'
  have hp (z : ℝ × ℝ) : |w z| * |f z| ^ 2 ≤ (8 * ‖ξ‖ ^ 2 * ‖v‖) * ‖z‖ ^ 3 := by
    have hwz := abs_dot_le_two_norm_mul v z
    have hfz := pow_le_pow_left₀ (abs_nonneg _) (abs_dot_le_two_norm_mul ξ z) 2
    calc
      _ ≤ (2 * ‖v‖ * ‖z‖) * (2 * ‖ξ‖ * ‖z‖) ^ 2 :=
        mul_le_mul hwz hfz (by positivity) (by positivity)
      _ = _ := by ring
  have hW : Integrable (fun z => |w z| * |f z| ^ 2) μ := by
    apply (hM.const_mul (8 * ‖ξ‖ ^ 2 * ‖v‖)).mono' (by dsimp [w, f]; fun_prop)
    exact ae_of_all _ fun z => by rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]; exact hp z
  have h := norm_integral_real_weighted_phase_sub_linear μ w f (by dsimp [f]; fun_prop) hw hwf hW
  have hw0 : ∫ z, w z ∂μ = 0 := integral_dot_of_standardized μ hμ v
  have hwf0 : ∫ z, w z * f z ∂μ = v.1 * ξ.1 + v.2 * ξ.2 :=
    integral_dot_mul_of_standardized μ hμ v ξ
  rw [hw0, hwf0, Complex.ofReal_zero, zero_add] at h
  have hd : fderiv ℝ (characteristic2 μ) ξ v =
      Complex.I * ∫ z, (w z : ℂ) * Complex.exp ((f z : ℂ) * Complex.I) ∂μ := by
    rw [fderiv_characteristic2_apply μ (hμ.memLp.mono_exponent (by norm_num)) ξ v]
    congr 1
    apply integral_congr_ae
    exact ae_of_all _ fun z => by dsimp [w, f]; congr 2 <;> congr 1 <;> push_cast <;> ring
  have he : fderiv ℝ (characteristic2 μ) ξ v + ((ξ.1 * v.1 + ξ.2 * v.2 : ℝ) : ℂ) =
      Complex.I * ((∫ z, (w z : ℂ) * Complex.exp ((f z : ℂ) * Complex.I) ∂μ) -
        ((v.1 * ξ.1 + v.2 * ξ.2 : ℝ) : ℂ) * Complex.I) := by
    rw [hd]
    push_cast
    have hI : Complex.I * Complex.I = -1 := Complex.I_mul_I
    calc
      _ = Complex.I * (∫ z, (w z : ℂ) * Complex.exp ((f z : ℂ) * Complex.I) ∂μ) -
          ((v.1 : ℂ) * ξ.1 + (v.2 : ℂ) * ξ.2) * (Complex.I * Complex.I) := by rw [hI]; ring
      _ = _ := by ring
  rw [he, norm_mul, Complex.norm_I, one_mul]
  calc
    _ ≤ ∫ z, |w z| * |f z| ^ 2 ∂μ := h
    _ ≤ ∫ z, (8 * ‖ξ‖ ^ 2 * ‖v‖) * ‖z‖ ^ 3 ∂μ :=
      integral_mono_ae hW (hM.const_mul _) (ae_of_all _ hp)
    _ = _ := by rw [integral_const_mul]; ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The second actual characteristic derivative has linear
Taylor error, controlled by the finite third moment alone. Source:
arXiv:2412.09080v3, §3 `thm:br`, `s = 2`, and §5.4 `eq:br-9.10`,
the unscaled second-derivative expansion. The factor eight is nonsharp. -/
theorem norm_iteratedFDeriv_two_characteristic2_add_dot_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (ξ : ℝ × ℝ) (v : Fin 2 → ℝ × ℝ) :
    ‖iteratedFDeriv ℝ 2 (characteristic2 μ) ξ v +
      (((v 0).1 * (v 1).1 + (v 0).2 * (v 1).2 : ℝ) : ℂ)‖ ≤
        8 * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ * ‖v 0‖ * ‖v 1‖ := by
  let w : ℝ × ℝ → ℝ := fun z => ((v 0).1 * z.1 + (v 0).2 * z.2) *
    ((v 1).1 * z.1 + (v 1).2 * z.2)
  let f : ℝ × ℝ → ℝ := fun z => ξ.1 * z.1 + ξ.2 * z.2
  have hw : Integrable w μ :=
    (memLp_dot μ hμ.memLp (v 0)).integrable_mul (memLp_dot μ hμ.memLp (v 1))
  have hM := hmom.integrable_norm_pow'
  have hp (z : ℝ × ℝ) : |w z| * |f z| ≤ (8 * ‖ξ‖ * ‖v 0‖ * ‖v 1‖) * ‖z‖ ^ 3 := by
    dsimp [w, f]
    rw [abs_mul]
    have h0 := abs_dot_le_two_norm_mul (v 0) z
    have h1 := abs_dot_le_two_norm_mul (v 1) z
    have hf := abs_dot_le_two_norm_mul ξ z
    calc
      _ ≤ ((2 * ‖v 0‖ * ‖z‖) * (2 * ‖v 1‖ * ‖z‖)) * (2 * ‖ξ‖ * ‖z‖) := by
        gcongr
      _ = _ := by ring
  have hW : Integrable (fun z => |w z| * |f z|) μ := by
    apply (hM.const_mul (8 * ‖ξ‖ * ‖v 0‖ * ‖v 1‖)).mono' (by dsimp [w, f]; fun_prop)
    exact ae_of_all _ fun z => by rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]; exact hp z
  have h := norm_integral_real_weighted_phase_sub_one μ w f (by dsimp [f]; fun_prop) hw hW
  have hw0 : ∫ z, w z ∂μ = (v 0).1 * (v 1).1 + (v 0).2 * (v 1).2 :=
    integral_dot_mul_of_standardized μ hμ (v 0) (v 1)
  rw [hw0] at h
  have hd : iteratedFDeriv ℝ 2 (characteristic2 μ) ξ v =
      -(∫ z, (w z : ℂ) * Complex.exp ((f z : ℂ) * Complex.I) ∂μ) := by
    rw [iteratedFDeriv_two_characteristic2_apply μ hμ.memLp ξ v]
    congr 1
    apply integral_congr_ae
    exact ae_of_all _ fun z => by dsimp [w, f]; congr 2 <;> congr 1 <;> push_cast <;> ring
  rw [hd]
  have he : -(∫ z, (w z : ℂ) * Complex.exp ((f z : ℂ) * Complex.I) ∂μ) +
      (((v 0).1 * (v 1).1 + (v 0).2 * (v 1).2 : ℝ) : ℂ) =
      -((∫ z, (w z : ℂ) * Complex.exp ((f z : ℂ) * Complex.I) ∂μ) -
        (((v 0).1 * (v 1).1 + (v 0).2 * (v 1).2 : ℝ) : ℂ)) := by ring
  rw [he, norm_neg]
  calc
    _ ≤ ∫ z, |w z| * |f z| ∂μ := h
    _ ≤ ∫ z, (8 * ‖ξ‖ * ‖v 0‖ * ‖v 1‖) * ‖z‖ ^ 3 ∂μ :=
      integral_mono_ae hW (hM.const_mul _) (ae_of_all _ hp)
    _ = _ := by rw [integral_const_mul]; ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp)⟩

/-- Centering and finite third moment bound the first derivative
by the frequency norm inside the unit ball. Source: arXiv:2412.09080v3,
§5.4 `eq:br-9.10`, the derivative factors in the normalized-sum power
comparison. No exponential moment is needed. -/
theorem norm_fderiv_characteristic2_apply_small
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (ξ v : ℝ × ℝ) (hξ : ‖ξ‖ ≤ 1) :
    ‖fderiv ℝ (characteristic2 μ) ξ v‖ ≤
      (2 + 8 * (∫ z, ‖z‖ ^ 3 ∂μ)) * ‖ξ‖ * ‖v‖ := by
  let D : ℂ := ((ξ.1 * v.1 + ξ.2 * v.2 : ℝ) : ℂ)
  have hM : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  have hd : ‖D‖ ≤ 2 * ‖ξ‖ * ‖v‖ := by
    dsimp [D]
    rw [Complex.norm_real, Real.norm_eq_abs]
    exact abs_dot_le_two_norm_mul ξ v
  have hr : ‖ξ‖ ^ 2 ≤ ‖ξ‖ := by nlinarith [norm_nonneg ξ]
  have hR := norm_fderiv_characteristic2_add_dot_le μ hμ hmom ξ v
  calc
    _ = ‖(fderiv ℝ (characteristic2 μ) ξ v + D) - D‖ := by congr 1; ring
    _ ≤ ‖fderiv ℝ (characteristic2 μ) ξ v + D‖ + ‖D‖ := norm_sub_le _ _
    _ ≤ 8 * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ ^ 2 * ‖v‖ + 2 * ‖ξ‖ * ‖v‖ := add_le_add hR hd
    _ ≤ 8 * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ * ‖v‖ + 2 * ‖ξ‖ * ‖v‖ := by gcongr
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 3 stdGauss2 ∧ ‖((1, 0) : ℝ × ℝ)‖ ≤ 1 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp), by norm_num⟩

/-- The full first-derivative operator norm is bounded by the
frequency norm inside the unit ball. Source: arXiv:2412.09080v3,
§5.4 `eq:br-9.10`, the derivative factors in the normalized-sum power
comparison. This is the operator version of the preceding directional bound. -/
theorem norm_fderiv_characteristic2_small
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (ξ : ℝ × ℝ) (hξ : ‖ξ‖ ≤ 1) :
    ‖fderiv ℝ (characteristic2 μ) ξ‖ ≤ (2 + 8 * (∫ z, ‖z‖ ^ 3 ∂μ)) * ‖ξ‖ := by
  have hM : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  apply ContinuousLinearMap.opNorm_le_bound _ (by positivity)
  exact fun v => norm_fderiv_characteristic2_apply_small μ hμ hmom ξ v hξ

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 3 stdGauss2 ∧ ‖((1, 0) : ℝ × ℝ)‖ ≤ 1 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp), by norm_num⟩

end Transformer.Modes

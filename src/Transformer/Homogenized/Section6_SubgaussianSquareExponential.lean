/-
# Exponential moments of subgaussian squares

Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`.
These are scalar prerequisites for its quadratic-form argument.
The layer-cake formula integrates the proved tail of the original
square, establishing finiteness before using a real-valued integral.
The small positive coefficient bound keeps the original random
variable and its variance proxy; no exponential-square hypothesis
or quadratic-form concentration estimate is assumed.
-/

import Transformer.Homogenized.Section6_SubgaussianTails
import Mathlib.MeasureTheory.Integral.Layercake
import Mathlib.Analysis.SpecialFunctions.ImproperIntegrals

open Real MeasureTheory ProbabilityTheory Set
open scoped NNReal

namespace Transformer.Homogenized

/-- The exponential-square layer-cake primitive, used in the proof
of arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`. -/
theorem integral_exp_square_primitive (r s : ℝ) :
    (∫ u in 0..s, r * Real.exp (r * u)) = Real.exp (r * s) - 1 := by
  have hd : ∀ u ∈ uIcc (0 : ℝ) s,
      HasDerivAt (fun v => Real.exp (r * v)) (r * Real.exp (r * u)) u := by
    intro u _
    simpa only [id_eq, mul_one, mul_comm] using ((hasDerivAt_id u).const_mul r).exp
  have hi : IntervalIntegrable (fun u => r * Real.exp (r * u)) volume 0 s :=
    (continuous_const.mul (Real.continuous_exp.comp (continuous_const.mul continuous_id))).intervalIntegrable 0 s
  simpa only [mul_zero, Real.exp_zero] using
    intervalIntegral.integral_eq_sub_of_hasDerivAt hd hi

/-- Layer cake for the original square, without assuming integrability
of its exponential. Source: arXiv:2604.01978v1, §6.3,
the scalar exponential estimates for `lem:Hanson_wright`. -/
theorem lintegral_exp_square_sub_one {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (X : Ω → ℝ) (hX : AEMeasurable X P)
    (r : ℝ) (hr : 0 ≤ r) :
    (∫⁻ ω, ENNReal.ofReal (Real.exp (r * (X ω) ^ 2) - 1) ∂P) =
      ∫⁻ u in Ioi 0, P {ω | u ≤ (X ω) ^ 2} * ENNReal.ofReal (r * Real.exp (r * u)) := by
  have hs : AEMeasurable (fun ω => (X ω) ^ 2) P := by
    simpa only [pow_two] using hX.fun_mul hX
  have hi : ∀ s > 0, IntervalIntegrable (fun u => r * Real.exp (r * u)) volume 0 s :=
    fun s _ => (continuous_const.mul
      (Real.continuous_exp.comp (continuous_const.mul continuous_id))).intervalIntegrable 0 s
  have hg : ∀ᵐ u ∂volume.restrict (Ioi (0 : ℝ)), 0 ≤ r * Real.exp (r * u) :=
    ae_of_all _ (fun u => mul_nonneg hr (Real.exp_nonneg _))
  simpa only [integral_exp_square_primitive] using
    lintegral_comp_eq_lintegral_meas_le_mul P (ae_of_all _ (fun ω => sq_nonneg (X ω))) hs hi hg

/-- Integrating the actual squared tail bounds the original exponential
increment. Source: arXiv:2604.01978v1, §6.3, the subgaussian
square input to `lem:Hanson_wright`. The positive gap is explicit. -/
theorem subgaussian_square_exp_increment_le {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (X : Ω → ℝ) (c : ℝ≥0)
    (hX : HasSubgaussianMGF X c P) (r : ℝ) (hr : 0 ≤ r)
    (hgap : 0 < 1 / (2 * (c : ℝ)) - r) :
    (∫⁻ ω, ENNReal.ofReal (Real.exp (r * (X ω) ^ 2) - 1) ∂P) ≤
      ENNReal.ofReal (2 * r / (1 / (2 * (c : ℝ)) - r)) := by
  let k : ℝ := 1 / (2 * (c : ℝ)) - r
  have hk : 0 < k := hgap
  rw [lintegral_exp_square_sub_one P X hX.aemeasurable r hr]
  have hbound : (∫⁻ u in Ioi 0,
      P {ω | u ≤ (X ω) ^ 2} * ENNReal.ofReal (r * Real.exp (r * u))) ≤
      ∫⁻ u in Ioi 0, ENNReal.ofReal (2 * r * Real.exp (-k * u)) := by
    apply lintegral_mono_ae
    filter_upwards [ae_restrict_mem (measurableSet_Ioi : MeasurableSet (Ioi (0 : ℝ)))] with u hu
    calc
      P {ω | u ≤ (X ω) ^ 2} * ENNReal.ofReal (r * Real.exp (r * u)) ≤
          ENNReal.ofReal (2 * Real.exp (-u / (2 * (c : ℝ)))) *
            ENNReal.ofReal (r * Real.exp (r * u)) := by
        gcongr
        exact subgaussian_sq_tail_ennreal_le P X c hX u hu.le
      _ = ENNReal.ofReal (2 * r * Real.exp (-k * u)) := by
        rw [← ENNReal.ofReal_mul (by positivity)]
        congr 1
        rw [mul_mul_mul_comm, ← Real.exp_add]
        congr 2
        dsimp [k]
        ring
  have hi : Integrable (fun u : ℝ => 2 * r * Real.exp (-k * u)) (volume.restrict (Ioi 0)) :=
    (integrableOn_exp_mul_Ioi (by linarith : -k < 0) 0).const_mul (2 * r)
  have hn : ∀ᵐ u ∂volume.restrict (Ioi (0 : ℝ)), 0 ≤ 2 * r * Real.exp (-k * u) :=
    ae_of_all _ (fun u => mul_nonneg (by positivity) (Real.exp_nonneg _))
  calc
    _ ≤ ∫⁻ u in Ioi 0, ENNReal.ofReal (2 * r * Real.exp (-k * u)) := hbound
    _ = ENNReal.ofReal (∫ u in Ioi 0, 2 * r * Real.exp (-k * u)) :=
      (ofReal_integral_eq_lintegral_ofReal hi hn).symm
    _ = ENNReal.ofReal (2 * r / k) := by
      rw [integral_const_mul, integral_exp_mul_Ioi (by linarith : -k < 0)]
      simp only [mul_zero, Real.exp_zero, neg_div_neg_eq, mul_one_div]

/-- Actual square exponentials are integrable and bounded by a universal
linear exponential for `r c ≤ 1/4`. Source: arXiv:2604.01978v1,
§6.3, a scalar prerequisite for `lem:Hanson_wright`. -/
theorem subgaussian_square_exp_integral_le {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (X : Ω → ℝ) (c : ℝ≥0)
    (hX : HasSubgaussianMGF X c P) (r : ℝ) (hr : 0 ≤ r) (hsmall : r * (c : ℝ) ≤ 1 / 4) :
    Integrable (fun ω => Real.exp (r * (X ω) ^ 2)) P ∧
      (∫ ω, Real.exp (r * (X ω) ^ 2) ∂P) ≤ Real.exp (8 * r * (c : ℝ)) := by
  by_cases hc : c = 0
  · subst c
    obtain ⟨hi, he⟩ := subgaussian_zero_proxy_square_exp P X hX r
    exact ⟨hi, by simp only [he, NNReal.coe_zero, mul_zero, Real.exp_zero, le_refl]⟩
  have hcpos : 0 < (c : ℝ) := by exact_mod_cast (pos_iff_ne_zero.mpr hc)
  let k : ℝ := 1 / (2 * (c : ℝ)) - r
  have hk : 0 < k := by
    dsimp [k]
    apply sub_pos.mpr
    apply (lt_div_iff₀ (by positivity : 0 < 2 * (c : ℝ))).mpr
    nlinarith
  have hratio : 2 * r / k ≤ 8 * r * (c : ℝ) := by
    apply (div_le_iff₀ hk).mpr
    dsimp [k]
    have hcancel : (c : ℝ) * (1 / (2 * (c : ℝ))) = 1 / 2 := by field_simp
    nlinarith [mul_nonneg hr (show 0 ≤ 1 / 4 - r * (c : ℝ) by linarith)]
  have hinc := subgaussian_square_exp_increment_le P X c hX r hr hk
  have hone : ∀ ω, 1 ≤ Real.exp (r * (X ω) ^ 2) :=
    fun ω => Real.one_le_exp_iff.mpr (mul_nonneg hr (sq_nonneg _))
  have heq : (∫⁻ ω, ENNReal.ofReal (Real.exp (r * (X ω) ^ 2)) ∂P) =
      1 + ∫⁻ ω, ENNReal.ofReal (Real.exp (r * (X ω) ^ 2) - 1) ∂P := by
    have hp : ∀ ω, ENNReal.ofReal (Real.exp (r * (X ω) ^ 2)) =
        1 + ENNReal.ofReal (Real.exp (r * (X ω) ^ 2) - 1) := by
      intro ω
      calc
        _ = ENNReal.ofReal (1 + (Real.exp (r * (X ω) ^ 2) - 1)) := by congr 1; ring
        _ = _ := by
          rw [ENNReal.ofReal_add (by norm_num) (by linarith [hone ω]), ENNReal.ofReal_one]
    rw [lintegral_congr hp, lintegral_add_left measurable_const, lintegral_const]
    simp
  have hle : (∫⁻ ω, ENNReal.ofReal (Real.exp (r * (X ω) ^ 2)) ∂P) ≤
      ENNReal.ofReal (1 + 2 * r / k) := by
    rw [heq, ENNReal.ofReal_add (by norm_num) (by positivity), ENNReal.ofReal_one]
    exact add_le_add le_rfl hinc
  have hn : 0 ≤ᵐ[P] (fun ω => Real.exp (r * (X ω) ^ 2)) :=
    ae_of_all _ (fun ω => Real.exp_nonneg _)
  have hm : AEMeasurable (fun ω => Real.exp (r * (X ω) ^ 2)) P := by
    apply Real.measurable_exp.comp_aemeasurable
    simpa only [pow_two] using aemeasurable_const.fun_mul (hX.aemeasurable.fun_mul hX.aemeasurable)
  have hi : Integrable (fun ω => Real.exp (r * (X ω) ^ 2)) P :=
    ⟨hm.aestronglyMeasurable, (hasFiniteIntegral_iff_ofReal hn).mpr
      (hle.trans_lt ENNReal.ofReal_lt_top)⟩
  refine ⟨hi, ?_⟩
  have hh := ENNReal.toReal_mono ENNReal.ofReal_ne_top hle
  rw [← ofReal_integral_eq_lintegral_ofReal hi hn,
    ENNReal.toReal_ofReal (integral_nonneg (fun ω => Real.exp_nonneg _)),
    ENNReal.toReal_ofReal (by positivity)] at hh
  exact hh.trans ((add_le_add le_rfl hratio).trans (by simpa only [add_comm] using Real.add_one_le_exp (8 * r * (c : ℝ))))

/-- A positive coefficient and the genuine standard normal law jointly
witness every scalar hypothesis, including the positive integration gap. -/
example : HasSubgaussianMGF (id : ℝ → ℝ) 1 (gaussianReal 0 1) ∧
    AEMeasurable (id : ℝ → ℝ) (gaussianReal 0 1) ∧
    IsProbabilityMeasure (gaussianReal 0 1) ∧ (0 : ℝ) ≤ 1 / 8 ∧
    (1 / 8 : ℝ) * (1 : ℝ≥0) ≤ 1 / 4 ∧ 0 < 1 / (2 * (1 : ℝ≥0)) - (1 / 8 : ℝ) :=
  ⟨hasSubgaussianMGF_standardGaussianReal, measurable_id.aemeasurable, inferInstance,
    by norm_num, by norm_num, by norm_num⟩

end Transformer.Homogenized

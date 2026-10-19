module

public import Transformer.Homogenized.Section6_CenteredExponentialBounds

/-!
# MGF of the original centered subgaussian square

Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`.
The actual square is centered by its own integral. Its absolute
exponential moment is proved from the original subgaussian MGF,
then the global centered Taylor estimate yields a quadratic MGF
bound for both signs of a sufficiently small coefficient.
The constants are fixed and independent of the probability space,
coordinate and variance proxy. The zero-proxy case is included.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped NNReal

namespace Transformer.Homogenized

/-- A fixed positive square-exponential rate is integrable with a
universal moment bound. Source: arXiv:2604.01978v1, §6.3,
the scalar square estimate for `lem:Hanson_wright`. -/
theorem subgaussian_square_reference_exp {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (X : Ω → ℝ) (c : ℝ≥0)
    (hX : HasSubgaussianMGF X c P) (hc : 0 < (c : ℝ)) :
    Integrable (fun ω => Real.exp ((1 / (8 * (c : ℝ))) * (X ω) ^ 2)) P ∧
      (∫ ω, Real.exp ((1 / (8 * (c : ℝ))) * (X ω) ^ 2) ∂P) ≤ Real.exp 1 := by
  have he : (1 / (8 * (c : ℝ))) * (c : ℝ) = 1 / 8 := by field_simp
  have h := subgaussian_square_exp_integral_le P X c hX (1 / (8 * (c : ℝ)))
    (by positivity) (by rw [he]; norm_num)
  have he' : 8 * (1 / (8 * (c : ℝ))) * (c : ℝ) = 1 := by nlinarith [he]
  simpa only [he'] using h

/-- The actual mean-centered square is integrable. Source:
arXiv:2604.01978v1, §6.3, the diagonal term of `lem:Hanson_wright`. -/
theorem integrable_centered_subgaussian_square {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (X : Ω → ℝ) (c : ℝ≥0)
    (hX : HasSubgaussianMGF X c P) :
    Integrable (fun ω => (X ω) ^ 2 - ∫ η, (X η) ^ 2 ∂P) P := by
  simpa only [sub_eq_add_neg] using (integrable_subgaussian_square P X c hX).fun_add
    (integrable_const (-(∫ η, (X η) ^ 2 ∂P)))

/-- The original centering cancels the actual square integral.
Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`. -/
theorem integral_centered_subgaussian_square_eq_zero {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (X : Ω → ℝ) (c : ℝ≥0)
    (hX : HasSubgaussianMGF X c P) :
    (∫ ω, (X ω) ^ 2 - ∫ η, (X η) ^ 2 ∂P ∂P) = 0 := by
  rw [integral_sub (integrable_subgaussian_square P X c hX) (integrable_const _)]
  simp

/-- The original second moment is bounded without identifying it with
the variance proxy. Source: arXiv:2604.01978v1, §6.3,
the centered-square exponential step for `lem:Hanson_wright`. -/
theorem subgaussian_square_integral_le {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (X : Ω → ℝ) (c : ℝ≥0)
    (hX : HasSubgaussianMGF X c P) (hc : 0 < (c : ℝ)) :
    (∫ ω, (X ω) ^ 2 ∂P) ≤ 8 * (c : ℝ) * Real.exp 1 := by
  obtain ⟨hi, hb⟩ := subgaussian_square_reference_exp P X c hX hc
  have hp : ∀ ω, (X ω) ^ 2 ≤ 8 * (c : ℝ) * Real.exp ((1 / (8 * (c : ℝ))) * (X ω) ^ 2) := by
    intro ω
    have h := Real.add_one_le_exp ((1 / (8 * (c : ℝ))) * (X ω) ^ 2)
    have hm := mul_le_mul_of_nonneg_left h (show 0 ≤ 8 * (c : ℝ) by positivity)
    have he : 8 * (c : ℝ) * ((1 / (8 * (c : ℝ))) * (X ω) ^ 2) = (X ω) ^ 2 := by field_simp
    nlinarith
  have hh := integral_mono (integrable_subgaussian_square P X c hX) (hi.const_mul (8 * (c : ℝ))) hp
  rw [integral_const_mul] at hh
  exact hh.trans (mul_le_mul_of_nonneg_left hb (by positivity))

/-- The genuine centered square has a finite absolute exponential
moment bounded by a universal constant. Source: arXiv:2604.01978v1,
§6.3, the diagonal MGF prerequisite for `lem:Hanson_wright`. -/
theorem subgaussian_centered_square_abs_exp {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (X : Ω → ℝ) (c : ℝ≥0)
    (hX : HasSubgaussianMGF X c P) (hc : 0 < (c : ℝ)) :
    Integrable (fun ω => Real.exp ((1 / (16 * (c : ℝ) * Real.exp 1)) *
      |(X ω) ^ 2 - ∫ η, (X η) ^ 2 ∂P|)) P ∧
    (∫ ω, Real.exp ((1 / (16 * (c : ℝ) * Real.exp 1)) *
      |(X ω) ^ 2 - ∫ η, (X η) ^ 2 ∂P|) ∂P) ≤ Real.exp 2 := by
  let m := ∫ η, (X η) ^ 2 ∂P
  let a := 1 / (16 * (c : ℝ) * Real.exp 1)
  let r := 1 / (8 * (c : ℝ))
  have hm : 0 ≤ m := integral_nonneg (fun η => sq_nonneg _)
  have hmb : m ≤ 8 * (c : ℝ) * Real.exp 1 := subgaussian_square_integral_le P X c hX hc
  have ha : 0 < a := by dsimp [a]; positivity
  have ham : a * m ≤ 1 / 2 := by
    dsimp [a]
    rw [div_mul_eq_mul_div]
    apply (div_le_iff₀ (by positivity : 0 < 16 * (c : ℝ) * Real.exp 1)).mpr
    nlinarith
  have har : a ≤ r := by
    dsimp [a, r]
    apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
    nlinarith [Real.one_le_exp (by norm_num : (0 : ℝ) ≤ 1)]
  obtain ⟨hi, hb⟩ := subgaussian_square_reference_exp P X c hX hc
  have hp : ∀ ω, Real.exp (a * |(X ω) ^ 2 - m|) ≤
      Real.exp (1 / 2) * Real.exp (r * (X ω) ^ 2) := by
    intro ω
    rw [← Real.exp_add]
    apply Real.exp_le_exp.mpr
    have hs : |(X ω) ^ 2 - m| ≤ (X ω) ^ 2 + m := by
      calc
        _ ≤ |(X ω) ^ 2| + |m| := abs_sub _ _
        _ = _ := by rw [abs_of_nonneg (sq_nonneg (X ω)), abs_of_nonneg hm]
    nlinarith [mul_le_mul_of_nonneg_left hs ha.le,
      mul_le_mul_of_nonneg_right har (sq_nonneg (X ω))]
  have hs : AEMeasurable (fun ω => (X ω) ^ 2 - m) P := by
    have hh : AEMeasurable (fun ω => (X ω) ^ 2) P := by
      simpa only [pow_two] using hX.aemeasurable.fun_mul hX.aemeasurable
    exact hh.fun_sub aemeasurable_const
  have hme : AEMeasurable (fun ω => Real.exp (a * |(X ω) ^ 2 - m|)) P :=
    Real.measurable_exp.comp_aemeasurable (aemeasurable_const.fun_mul hs.abs)
  have hj : Integrable (fun ω => Real.exp (a * |(X ω) ^ 2 - m|)) P := by
    apply (hi.const_mul (Real.exp (1 / 2))).mono' hme.aestronglyMeasurable
    apply ae_of_all
    intro ω
    simpa only [Real.norm_eq_abs, abs_of_nonneg (Real.exp_nonneg _)] using hp ω
  refine ⟨hj, ?_⟩
  have hh := integral_mono hj (hi.const_mul (Real.exp (1 / 2))) hp
  rw [integral_const_mul] at hh
  calc
    _ ≤ Real.exp (1 / 2) * Real.exp 1 := hh.trans (mul_le_mul_of_nonneg_left hb (Real.exp_nonneg _))
    _ ≤ Real.exp 2 := by rw [← Real.exp_add]; apply Real.exp_le_exp.mpr; norm_num

/-- The actual centered-square MGF has a universal quadratic bound for
both coefficient signs. Source: arXiv:2604.01978v1, §6.3,
the diagonal scalar input to `lem:Hanson_wright`. -/
theorem subgaussian_centered_square_exp_integral_le {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (X : Ω → ℝ) (c : ℝ≥0)
    (hX : HasSubgaussianMGF X c P) (t : ℝ) (ht : |t| * (c : ℝ) ≤ 1 / (32 * Real.exp 1)) :
    Integrable (fun ω => Real.exp (t * ((X ω) ^ 2 - ∫ η, (X η) ^ 2 ∂P))) P ∧
    (∫ ω, Real.exp (t * ((X ω) ^ 2 - ∫ η, (X η) ^ 2 ∂P)) ∂P) ≤
      Real.exp (2048 * (Real.exp 1) ^ 4 * t ^ 2 * (c : ℝ) ^ 2) := by
  by_cases hc : c = 0
  · subst c
    have hz := hX.ae_eq_zero_of_hasSubgaussianMGF_zero
    have hs : (fun ω => (X ω) ^ 2) =ᵐ[P] (fun _ => (0 : ℝ)) := by
      filter_upwards [hz] with ω hω
      change X ω = 0 at hω
      simp only [hω, zero_pow (by decide : 2 ≠ 0)]
    have hm : (∫ η, (X η) ^ 2 ∂P) = 0 := by rw [integral_congr_ae hs]; simp
    have he : (fun ω => Real.exp (t * ((X ω) ^ 2 - ∫ η, (X η) ^ 2 ∂P))) =ᵐ[P] (fun _ => (1 : ℝ)) := by
      filter_upwards [hs] with ω hω
      simp only [hω, hm, sub_self, mul_zero, Real.exp_zero]
    refine ⟨(integrable_const (1 : ℝ)).congr he.symm, ?_⟩
    rw [integral_congr_ae he]
    simp
  have hcpos : 0 < (c : ℝ) := by exact_mod_cast (pos_iff_ne_zero.mpr hc)
  let a := 1 / (16 * (c : ℝ) * Real.exp 1)
  have ha : 0 < a := by dsimp [a]; positivity
  have hta : |t| ≤ a / 2 := by
    have hh := (le_div_iff₀ (show 0 < 32 * Real.exp 1 by positivity)).mp ht
    have he : a / 2 = 1 / (32 * (c : ℝ) * Real.exp 1) := by dsimp [a]; field_simp; ring
    rw [he]
    apply (le_div_iff₀ (by positivity)).mpr
    nlinarith
  obtain ⟨hi, hb⟩ := subgaussian_centered_square_abs_exp P X c hX hcpos
  obtain ⟨he, hh⟩ := centered_exp_integral_le P _ (integrable_centered_subgaussian_square P X c hX)
    (integral_centered_subgaussian_square_eq_zero P X c hX) a ha hi t hta
  refine ⟨he, hh.trans (Real.exp_le_exp.mpr ?_)⟩
  have hm := mul_le_mul_of_nonneg_left hb (show 0 ≤ (8 / a ^ 2) * t ^ 2 by positivity)
  have hconst : (8 / a ^ 2) * t ^ 2 * Real.exp 2 = 2048 * (Real.exp 1) ^ 4 * t ^ 2 * (c : ℝ) ^ 2 := by
    have hexp : Real.exp 2 = Real.exp 1 * Real.exp 1 := by
      rw [← Real.exp_add]; norm_num
    rw [hexp]
    dsimp [a]
    field_simp
    ring
  exact hm.trans hconst.le

/-- A genuine nondegenerate scalar law and a nonzero coefficient
witness the subgaussian and small-coefficient assumptions together. -/
example : HasSubgaussianMGF (id : ℝ → ℝ) 1 (gaussianReal 0 1) ∧
    IsProbabilityMeasure (gaussianReal 0 1) ∧ (0 : ℝ) < (1 : ℝ≥0) ∧
    |(1 / (64 * Real.exp 1) : ℝ)| * (1 : ℝ≥0) ≤ 1 / (32 * Real.exp 1) := by
  refine ⟨hasSubgaussianMGF_standardGaussianReal, inferInstance, by norm_num, ?_⟩
  rw [abs_of_nonneg (by positivity), NNReal.coe_one, mul_one]
  apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
  nlinarith [Real.exp_pos (1 : ℝ)]

end Transformer.Homogenized

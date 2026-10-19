module

public import Transformer.Homogenized.Section6_SubgaussianSquareExponential
public import Mathlib.MeasureTheory.Order.Group.Lattice

/-!
# Centered scalar exponential estimates

Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`.
The diagonal part requires an MGF bound quadratic in its coefficient.
These analytic prerequisites apply to an actual centered random variable
with a finite exponential moment of its absolute value. The later
subgaussian-square argument proves that input from the original MGF.
The Taylor remainder is global, so unbounded coordinates are covered.
Integrability is established before cancellation of the actual mean.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- A global real exponential remainder, derived from Mathlib's complex
Taylor estimate. Source: arXiv:2604.01978v1, §6.3, the centered
exponential argument used to prove `lem:Hanson_wright`. -/
theorem abs_exp_sub_one_sub_le (x : ℝ) :
    |Real.exp x - 1 - x| ≤ x ^ 2 * Real.exp |x| := by
  have h := Complex.norm_exp_sub_sum_le_norm_mul_exp (x : ℂ) 2
  have hs : (∑ m ∈ Finset.range 2, (x : ℂ) ^ m / (m.factorial : ℂ)) = 1 + x := by
    norm_num [Finset.sum_range_succ]
  rw [hs, ← Complex.ofReal_exp, ← Complex.ofReal_one, ← Complex.ofReal_add,
    ← Complex.ofReal_sub, Complex.norm_real, Real.norm_eq_abs,
    Complex.norm_real, Real.norm_eq_abs, sq_abs] at h
  simpa only [sub_add_eq_sub_sub] using h

/-- The global upper Taylor bound retains its linear term for actual
mean cancellation. Source: arXiv:2604.01978v1, §6.3,
the scalar centered-square MGF step for `lem:Hanson_wright`. -/
theorem exp_le_one_add_linear_add_remainder (x : ℝ) :
    Real.exp x ≤ 1 + x + x ^ 2 * Real.exp |x| := by
  have h := (le_abs_self (Real.exp x - 1 - x)).trans (abs_exp_sub_one_sub_le x)
  linarith

/-- A quadratic factor is absorbed by half the positive exponential
rate. Source: arXiv:2604.01978v1, §6.3, a scalar integrability
and remainder estimate for `lem:Hanson_wright`. -/
theorem sq_mul_exp_half_abs_le (a : ℝ) (ha : 0 < a) (y : ℝ) :
    y ^ 2 * Real.exp (a * |y| / 2) ≤ (8 / a ^ 2) * Real.exp (a * |y|) := by
  have hv : 0 ≤ a * |y| / 2 := by positivity
  have h := Real.sum_le_exp_of_nonneg hv 3
  norm_num [Finset.sum_range_succ] at h
  have hp : a ^ 2 * y ^ 2 ≤ 8 * Real.exp (a * |y| / 2) := by
    nlinarith [sq_abs y]
  have he : Real.exp (a * |y| / 2) * Real.exp (a * |y| / 2) = Real.exp (a * |y|) := by
    rw [← Real.exp_add]
    congr 1
    ring
  have hm := mul_le_mul_of_nonneg_right hp (Real.exp_nonneg (a * |y| / 2))
  have hm' : a ^ 2 * (y ^ 2 * Real.exp (a * |y| / 2)) ≤ 8 * Real.exp (a * |y|) := by
    calc
      _ ≤ 8 * (Real.exp (a * |y| / 2) * Real.exp (a * |y| / 2)) := by nlinarith [hm]
      _ = _ := by rw [he]
  rw [div_mul_eq_mul_div]
  apply (le_div_iff₀ (sq_pos_of_pos ha)).mpr
  nlinarith [hm']

/-- The same quadratic absorption covers every actual coefficient
with `|t| ≤ a/2`. Source: arXiv:2604.01978v1, §6.3,
the scalar Taylor remainder for `lem:Hanson_wright`. -/
theorem sq_mul_exp_abs_coefficient_le (a : ℝ) (ha : 0 < a)
    (t : ℝ) (ht : |t| ≤ a / 2) (y : ℝ) :
    y ^ 2 * Real.exp |t * y| ≤ (8 / a ^ 2) * Real.exp (a * |y|) := by
  apply le_trans _ (sq_mul_exp_half_abs_le a ha y)
  gcongr
  rw [abs_mul]
  nlinarith [mul_le_mul_of_nonneg_right ht (abs_nonneg y)]

/-- A finite absolute exponential moment gives integrability of the
original scalar MGF throughout its interval. Source: arXiv:2604.01978v1,
§6.3, the scalar centered-square exponential argument. -/
theorem integrable_exp_of_integrable_abs_exp {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (Y : Ω → ℝ) (hY : AEMeasurable Y P) (a : ℝ)
    (hi : Integrable (fun ω => Real.exp (a * |Y ω|)) P)
    (t : ℝ) (ht : |t| ≤ a) : Integrable (fun ω => Real.exp (t * Y ω)) P := by
  have hm : AEMeasurable (fun ω => Real.exp (t * Y ω)) P :=
    Real.measurable_exp.comp_aemeasurable (aemeasurable_const.fun_mul hY)
  apply hi.mono' hm.aestronglyMeasurable
  apply ae_of_all
  intro ω
  rw [Real.norm_eq_abs, abs_of_nonneg (Real.exp_nonneg _)]
  apply Real.exp_le_exp.mpr
  calc
    t * Y ω ≤ |t * Y ω| := le_abs_self _
    _ ≤ a * |Y ω| := by rw [abs_mul]; gcongr

/-- The original centered scalar MGF is quadratic on half the finite
absolute-exponential interval. Source: arXiv:2604.01978v1, §6.3,
a scalar prerequisite for `lem:Hanson_wright`; its absolute exponential
moment remains the genuine original integral. -/
theorem centered_exp_integral_le {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (Y : Ω → ℝ)
    (hY : Integrable Y P) (hmean : (∫ ω, Y ω ∂P) = 0)
    (a : ℝ) (ha : 0 < a) (hi : Integrable (fun ω => Real.exp (a * |Y ω|)) P)
    (t : ℝ) (ht : |t| ≤ a / 2) :
    Integrable (fun ω => Real.exp (t * Y ω)) P ∧
      (∫ ω, Real.exp (t * Y ω) ∂P) ≤
        Real.exp ((8 / a ^ 2) * t ^ 2 * ∫ ω, Real.exp (a * |Y ω|) ∂P) := by
  have ht' : |t| ≤ a := by linarith
  have he := integrable_exp_of_integrable_abs_exp P Y hY.aestronglyMeasurable.aemeasurable a hi t ht'
  have hlinear : Integrable (fun ω => 1 + t * Y ω) P :=
    (integrable_const (1 : ℝ)).fun_add (hY.const_mul t)
  have hbound : Integrable (fun ω => 1 + t * Y ω +
      ((8 / a ^ 2) * t ^ 2) * Real.exp (a * |Y ω|)) P :=
    hlinear.fun_add (hi.const_mul _)
  have hpoint : ∀ ω, Real.exp (t * Y ω) ≤
      1 + t * Y ω + ((8 / a ^ 2) * t ^ 2) * Real.exp (a * |Y ω|) := by
    intro ω
    apply (exp_le_one_add_linear_add_remainder (t * Y ω)).trans
    have hh := mul_le_mul_of_nonneg_left (sq_mul_exp_abs_coefficient_le a ha t ht (Y ω)) (sq_nonneg t)
    nlinarith [hh]
  have hh := integral_mono he hbound hpoint
  rw [integral_add hlinear (hi.const_mul _), integral_add (integrable_const (1 : ℝ)) (hY.const_mul t),
    integral_const_mul, integral_const_mul, hmean] at hh
  simp only [integral_const, probReal_univ, one_smul, mul_zero, add_zero] at hh
  refine ⟨he, hh.trans ?_⟩
  simpa only [add_comm] using
    (Real.add_one_le_exp ((8 / a ^ 2) * t ^ 2 * ∫ ω, Real.exp (a * |Y ω|) ∂P))

/-- The genuine standard Gaussian and a positive absolute-exponential
rate witness the analytic hypotheses, with a nonzero admissible coefficient. -/
example : ∃ Y : ℝ → ℝ, Integrable Y (gaussianReal 0 1) ∧
    (∫ ω, Y ω ∂gaussianReal 0 1) = 0 ∧ AEMeasurable Y (gaussianReal 0 1) ∧
    Integrable (fun ω => Real.exp ((1 / 8 : ℝ) * |Y ω|)) (gaussianReal 0 1) ∧
    (0 : ℝ) < 1 / 8 ∧ |(1 / 32 : ℝ)| ≤ (1 / 8 : ℝ) / 2 := by
  let P := gaussianReal 0 1
  have hx : HasSubgaussianMGF (id : ℝ → ℝ) 1 P := hasSubgaussianMGF_standardGaussianReal
  have hs := (subgaussian_square_exp_integral_le P id 1 hx (1 / 8) (by norm_num) (by norm_num)).1
  have hm : AEMeasurable (fun x : ℝ => Real.exp ((1 / 8 : ℝ) * |x|)) P :=
    Real.measurable_exp.comp_aemeasurable (aemeasurable_const.fun_mul hx.aemeasurable.abs)
  have hi : Integrable (fun x : ℝ => Real.exp ((1 / 8 : ℝ) * |x|)) P := by
    apply (hs.const_mul (Real.exp (1 / 8))).mono' hm.aestronglyMeasurable
    apply ae_of_all
    intro x
    rw [Real.norm_eq_abs, abs_of_nonneg (Real.exp_nonneg _), ← Real.exp_add]
    apply Real.exp_le_exp.mpr
    simp only [id_eq]
    nlinarith [sq_abs x, sq_nonneg (|x| - 1 / 2)]
  exact ⟨id, hx.integrable, subgaussian_integral_eq_zero P id 1 hx, hx.aemeasurable,
    hi, by norm_num, by norm_num⟩

/-- A positive analytic rate and a nonzero coefficient jointly witness
both deterministic quadratic-absorption assumptions. -/
example : (0 : ℝ) < 1 ∧ |(1 / 4 : ℝ)| ≤ (1 : ℝ) / 2 := ⟨by norm_num, by norm_num⟩

end Transformer.Homogenized

import Transformer.Homogenized.Section7_LogisticArcsinTests

/-!
# Domination and limits for the actual logistic angle generators

ArXiv:2604.01978v1, §7, `sec: sde.logistic`.
The preceding smooth tests equal arcsin(a u) near every state
of [-1,1]. Their original generator was computed exactly there.
Here its residual drift and square-test diffusion contribution
are recorded by their explicit arithmetic formulas.

For 0<a<1 the residual drift is bounded by sqrt(1-a²),
and the diffusion contribution lies between 0 and 2(1-u²).
The bounds include both absorbing endpoints. Along a→1
the drift tends to zero and the diffusion term tends to
the source's exact 2(1-u²). The endpoint cases are treated
directly, retaining the zero original coefficients there.

The square-generator identity connects both formulas back to
the actual globally smooth tests and their original generator.
The explicit sequence 1-1/(k+2) stays strictly inside (0,1)
and tends to 1, so the smooth approximants exist at every index.
The examples retain nonzero drift and positive diffusion at an
interior point, together with actual smooth test functions.
-/

open Real MeasureTheory Set Filter
open scoped Topology
namespace Transformer.Homogenized

/-- The computed original angle-test generator; source: arXiv:2604.01978v1,
§7, `sec: sde.logistic`, as calculated in logisticGenerator_arcsin_local. -/
noncomputable def regularizedLogisticDrift (a u : ℝ) : ℝ :=
  -(a * (1 - a ^ 2) * u * (1 - u ^ 2)) / Real.sqrt (1 - (a * u) ^ 2) ^ 3

/-- The computed square-test diffusion term; source: arXiv:2604.01978v1,
§7, `sec: sde.logistic`, the angle's quadratic variation. -/
noncomputable def regularizedLogisticVariance (a u : ℝ) : ℝ :=
  2 * a ^ 2 * (1 - u ^ 2) ^ 2 / (1 - (a * u) ^ 2)

/-- The original state interval keeps every regularized denominator positive.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`. -/
theorem regularizedLogistic_parameter_bounds (a u : ℝ)
    (ha : a ∈ Ioo (0 : ℝ) 1) (hu : u ∈ Icc (-1 : ℝ) 1) :
    0 < 1 - a ^ 2 ∧ 0 ≤ 1 - u ^ 2 ∧ 0 < 1 - (a * u) ^ 2 ∧
      1 - a ^ 2 ≤ 1 - (a * u) ^ 2 ∧ 1 - u ^ 2 ≤ 1 - (a * u) ^ 2 ∧ |a * u| < 1 := by
  have ha2 : a ^ 2 < 1 := (sq_lt_one_iff_abs_lt_one a).mpr (by rw [abs_of_pos ha.1]; exact ha.2)
  have hu2 : u ^ 2 ≤ 1 := (sq_le_one_iff_abs_le_one u).mpr (abs_le.mpr hu)
  have hau : |a * u| < 1 := by
    rw [abs_mul, abs_of_pos ha.1]
    exact (mul_le_mul_of_nonneg_left (abs_le.mpr hu) ha.1.le).trans_lt (by simpa only [mul_one] using ha.2)
  have hau2 : (a * u) ^ 2 < 1 := (sq_lt_one_iff_abs_lt_one (a * u)).mpr hau
  have h1 := mul_le_mul_of_nonneg_left hu2 (sq_nonneg a)
  have h2 := mul_le_mul_of_nonneg_right ha2.le (sq_nonneg u)
  refine ⟨by linarith, by linarith, by linarith, ?_, ?_, hau⟩ <;> nlinarith

/-- Uniform domination of the actual residual drift, with a vanishing bound.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, justifying the angle limit. -/
theorem abs_regularizedLogisticDrift_le (a u : ℝ)
    (ha : a ∈ Ioo (0 : ℝ) 1) (hu : u ∈ Icc (-1 : ℝ) 1) :
    |regularizedLogisticDrift a u| ≤ Real.sqrt (1 - a ^ 2) := by
  obtain ⟨hδ, hv, hD, hδD, hvD, hau⟩ := regularizedLogistic_parameter_bounds a u ha hu
  have hs := Real.sqrt_pos.mpr hD
  have hr := Real.sqrt_nonneg (1 - a ^ 2)
  have hrs := Real.sqrt_le_sqrt hδD
  have hδs : 1 - a ^ 2 ≤ Real.sqrt (1 - a ^ 2) * Real.sqrt (1 - (a * u) ^ 2) := by
    nlinarith [mul_le_mul_of_nonneg_left hrs hr, Real.sq_sqrt hδ.le]
  have hab : a * |u| ≤ 1 := by simpa only [abs_mul, abs_of_pos ha.1] using hau.le
  rw [regularizedLogisticDrift, abs_div, abs_neg, abs_mul, abs_mul, abs_mul,
    abs_of_pos ha.1, abs_of_pos hδ, abs_of_nonneg hv, abs_pow, abs_of_pos hs]
  apply (div_le_iff₀ (pow_pos hs 3)).mpr
  calc
    _ = ((1 - a ^ 2) * (a * |u|)) * (1 - u ^ 2) := by ring
    _ ≤ (1 - a ^ 2) * (1 - (a * u) ^ 2) :=
      mul_le_mul (mul_le_of_le_one_right hδ.le hab) hvD hv hδ.le
    _ ≤ (Real.sqrt (1 - a ^ 2) * Real.sqrt (1 - (a * u) ^ 2)) * (1 - (a * u) ^ 2) :=
      mul_le_mul_of_nonneg_right hδs hD.le
    _ = _ := by rw [pow_succ (Real.sqrt (1 - (a * u) ^ 2)) 2, Real.sq_sqrt hD.le]; ring

/-- The actual regularized diffusion is bounded by the source's limiting one.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, the bracket density. -/
theorem regularizedLogisticVariance_mem_Icc (a u : ℝ)
    (ha : a ∈ Ioo (0 : ℝ) 1) (hu : u ∈ Icc (-1 : ℝ) 1) :
    regularizedLogisticVariance a u ∈ Icc (0 : ℝ) (2 * (1 - u ^ 2)) := by
  obtain ⟨hδ, hv, hD, -, -, -⟩ := regularizedLogistic_parameter_bounds a u ha hu
  constructor
  · unfold regularizedLogisticVariance
    positivity
  · unfold regularizedLogisticVariance
    apply (div_le_iff₀ hD).mpr
    nlinarith [mul_nonneg hδ.le hv]

/-- Both arithmetic terms come from the actual original square generator.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, the square-test identity. -/
theorem logisticGenerator_square_arcsin_local (a u : ℝ)
    (ha : a ∈ Ioo (0 : ℝ) 1) (hu : u ∈ Icc (-1 : ℝ) 1)
    (φ : ℝ → ℝ) (hφ : ContDiff ℝ (⊤ : ℕ∞) φ)
    (heq : φ =ᶠ[𝓝 u] (fun v => Real.arcsin (a * v))) :
    logisticGenerator (fun v => φ v ^ 2) u =
      2 * φ u * regularizedLogisticDrift a u + regularizedLogisticVariance a u := by
  obtain ⟨-, -, hD, -, -, hau⟩ := regularizedLogistic_parameter_bounds a u ha hu
  rw [logisticGenerator_square φ hφ u, logisticGenerator_arcsin_local a u hau φ heq,
    heq.deriv_eq, (hasDerivAt_arcsin_mul a u hau).deriv]
  unfold regularizedLogisticDrift regularizedLogisticVariance
  rw [div_pow, Real.sq_sqrt hD.le]
  ring

/-- The actual residual drift vanishes, including at ±1.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, the drift-free angle. -/
theorem tendsto_regularizedLogisticDrift (a : ℕ → ℝ) (ha : Tendsto a atTop (𝓝 1))
    (u : ℝ) (hu : u ∈ Icc (-1 : ℝ) 1) :
    Tendsto (fun k => regularizedLogisticDrift (a k) u) atTop (𝓝 0) := by
  by_cases hv : 1 - u ^ 2 = 0
  · simpa only [regularizedLogisticDrift, hv, mul_zero, neg_zero, zero_div] using
      (tendsto_const_nhds : Tendsto (fun _ : ℕ => (0 : ℝ)) atTop (𝓝 0))
  have hpos : 0 < 1 - u ^ 2 := lt_of_le_of_ne
    (sub_nonneg.mpr ((sq_le_one_iff_abs_le_one u).mpr (abs_le.mpr hu))) (fun h => hv h.symm)
  have hden : Real.sqrt (1 - (1 * u) ^ 2) ^ 3 ≠ 0 := by
    simpa only [one_mul] using pow_ne_zero 3 (Real.sqrt_pos.mpr hpos).ne'
  have hconst : Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (𝓝 1) := tendsto_const_nhds
  have hnum : Tendsto (fun k => -(a k * (1 - a k ^ 2) * u * (1 - u ^ 2))) atTop (𝓝 0) := by
    simpa only [one_pow, sub_self, mul_zero, zero_mul, neg_zero] using
      (((ha.mul (hconst.sub (ha.pow 2))).mul_const u).mul_const (1 - u ^ 2)).neg
  have hroot : Tendsto (fun k => Real.sqrt (1 - (a k * u) ^ 2) ^ 3) atTop
      (𝓝 (Real.sqrt (1 - (1 * u) ^ 2) ^ 3)) :=
    (hconst.sub ((ha.mul_const u).pow 2)).sqrt.pow 3
  have h := hnum.div hroot hden
  simpa only [Pi.div_def, regularizedLogisticDrift, zero_div] using h

/-- The arithmetic square-test term converges to the exact original bracket density.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`; the limit also holds off the state interval. -/
theorem tendsto_regularizedLogisticVariance (a : ℕ → ℝ) (ha : Tendsto a atTop (𝓝 1))
    (u : ℝ) :
    Tendsto (fun k => regularizedLogisticVariance (a k) u) atTop (𝓝 (2 * (1 - u ^ 2))) := by
  by_cases hv : 1 - u ^ 2 = 0
  · simpa only [regularizedLogisticVariance, hv, zero_pow (by decide : 2 ≠ 0), mul_zero, zero_div] using
      (tendsto_const_nhds : Tendsto (fun _ : ℕ => (0 : ℝ)) atTop (𝓝 0))
  have h := (((ha.pow 2).const_mul 2).mul_const ((1 - u ^ 2) ^ 2)).div
    (tendsto_const_nhds.sub ((ha.mul_const u).pow 2)) (by simpa only [one_mul] using hv)
  have heq : 2 * (1 : ℝ) ^ 2 * (1 - u ^ 2) ^ 2 / (1 - (1 * u) ^ 2) = 2 * (1 - u ^ 2) := by
    simp only [one_pow, mul_one, one_mul]
    field_simp
  simpa only [Pi.div_def, regularizedLogisticVariance, heq] using h

/-- A concrete regularization sequence for the source's angle substitution.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`. -/
noncomputable def logisticArcsinScale (k : ℕ) : ℝ := 1 - 1 / ((k : ℝ) + 2)

/-- Every actual regularization parameter permits the proved global smooth test.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`. -/
theorem logisticArcsinScale_mem (k : ℕ) : logisticArcsinScale k ∈ Ioo (0 : ℝ) 1 := by
  have hk : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  have hp : (0 : ℝ) < (k : ℝ) + 2 := by linarith
  have hl : 1 / ((k : ℝ) + 2) < 1 := (div_lt_one hp).mpr (by linarith)
  have hz : 0 < 1 / ((k : ℝ) + 2) := by positivity
  constructor <;> dsimp [logisticArcsinScale] <;> linarith

/-- The actual regularization parameters tend to the source's angle parameter.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`. -/
theorem tendsto_logisticArcsinScale : Tendsto logisticArcsinScale atTop (𝓝 1) := by
  have h := (tendsto_one_div_add_atTop_nhds_zero_nat (𝕜 := ℝ)).comp (Filter.tendsto_add_atTop_nat 1)
  have hr : Tendsto (fun k : ℕ => 1 / ((k : ℝ) + 2)) atTop (𝓝 0) := by
    simpa only [Function.comp_def, Nat.cast_add, Nat.cast_one, add_assoc, one_add_one_eq_two] using h
  change Tendsto (fun k : ℕ => 1 - 1 / ((k : ℝ) + 2)) atTop (𝓝 1)
  simpa only [sub_zero] using hr.const_sub 1

/-- Every bound and limit premise is realized by actual regularization data
and a genuine smooth test, with nonzero drift and diffusion at u=1/2. -/
example : (1 / 2 : ℝ) ∈ Ioo (0 : ℝ) 1 ∧ (1 / 2 : ℝ) ∈ Icc (-1 : ℝ) 1 ∧
    (∀ k, logisticArcsinScale k ∈ Ioo (0 : ℝ) 1) ∧ Tendsto logisticArcsinScale atTop (𝓝 1) ∧
    regularizedLogisticDrift (1 / 2) (1 / 2) < 0 ∧
    regularizedLogisticVariance (1 / 2) (1 / 2) = 3 / 10 ∧
    ∃ φ : ℝ → ℝ, ContDiff ℝ (⊤ : ℕ∞) φ ∧
      ∀ u ∈ Icc (-1 : ℝ) 1, φ =ᶠ[𝓝 u] (fun v => Real.arcsin ((1 / 2 : ℝ) * v)) := by
  refine ⟨by norm_num, by norm_num, logisticArcsinScale_mem, tendsto_logisticArcsinScale, ?_, ?_,
    exists_contDiff_arcsin_test _ (by norm_num)⟩
  · unfold regularizedLogisticDrift
    exact div_neg_of_neg_of_pos (by norm_num) (by positivity)
  · norm_num [regularizedLogisticVariance]

end Transformer.Homogenized

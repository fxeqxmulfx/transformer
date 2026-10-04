/-
# The variation of the noise scale

arXiv:1812.06162, Appendix D.  The parameter `γ = (∫√𝓑 ds)²/(S_min E_min)` of eq. (D.5)
(`paretoGamma`) is at most 1, by Cauchy–Schwarz (`paretoGamma_le_one`), and "when the noise
scale is constant `γ = 1`" (`paretoGamma_const`).  On as many examples as a constant batch,
the schedule `√(r𝓑)` takes `S_min + γ(S - S_min)` steps, where `S` are the constant batch's
(`totalSteps_adaptiveBatch_sub`): at `γ = 1` "there is no benefit from using an adaptive batch
size", a smaller `γ` is the "predicted improvement in the Pareto front", and the steps saved
vanish as the batch grows: "negligible benefits at large `E_tot/S_tot`"
(`tendsto_totalSteps_sub_adaptiveBatch`).  The footnote's
`γ = E[√𝓑]²/E[𝓑] = 1/(1 + σ²/E[√𝓑]²)`, "where the expectation is over a training run,
weighting each full-batch step equally", holds with `E` the average `⨍` over `ds` and `σ²` the
variance of `√𝓑` under it (`paretoGamma_eq_average`): "more variation in `𝓑` pushes `γ`
closer to 0".

"An SVHN Case Study": for the fit `𝓑_crit(s) ≈ 10√s`,
`γ = (∫ds s^{1/4})²/(s ∫ds √s) = 24/25`, "or around 4%", whatever the length of the run and
the factor 10 (`paretoGamma_sqrt`).
-/

import Transformer.NoiseScale.SectionD_ParetoFront
import Mathlib.MeasureTheory.Integral.Average
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic

open MeasureTheory Filter Topology

namespace Transformer.NoiseScale

variable {T : Type*} [MeasurableSpace T] {μ : Measure T} [IsFiniteMeasure μ] {𝓑 : T → ℝ}

/-- `∫(√𝓑 - m)² ds = E_min - 2m ∫√𝓑 ds + m² S_min`. -/
theorem integral_sqrt_sub_sq (h𝓑 : ∀ s, 0 ≤ 𝓑 s) (hi : Integrable 𝓑 μ) (m : ℝ) :
    ∫ s, (√(𝓑 s) - m) ^ 2 ∂μ =
      (∫ s, 𝓑 s ∂μ) - 2 * m * (∫ s, √(𝓑 s) ∂μ) + m ^ 2 * μ.real Set.univ := by
  have e : (fun s => (√(𝓑 s) - m) ^ 2) = fun s => 𝓑 s - 2 * m * √(𝓑 s) + m ^ 2 :=
    funext fun s => by rw [sub_sq, Real.sq_sqrt (h𝓑 s)]; ring
  have h1 : Integrable (fun s => 𝓑 s - 2 * m * √(𝓑 s)) μ :=
    hi.sub ((integrable_sqrt h𝓑 hi).const_mul _)
  rw [e, integral_add h1 (integrable_const _),
    integral_sub hi ((integrable_sqrt h𝓑 hi).const_mul _), integral_const_mul, integral_const,
    smul_eq_mul, mul_comm (μ.real _)]

/-- The hypotheses of `integral_sqrt_sub_sq` are satisfiable: `𝓑 = 0`. -/
example := integral_sqrt_sub_sq (μ := μ) (𝓑 := 0) (fun _ => le_rfl) (integrable_zero _ _ _) 1

theorem integral_sqrt_pos (h𝓑 : ∀ s, 0 ≤ 𝓑 s) (hi : Integrable 𝓑 μ) (hE : 0 < ∫ s, 𝓑 s ∂μ) :
    0 < ∫ s, √(𝓑 s) ∂μ := by
  have hs : Function.support (fun s => √(𝓑 s)) = Function.support 𝓑 := by
    ext s
    simp only [Function.mem_support, ne_eq, Real.sqrt_eq_zero (h𝓑 s)]
  rw [integral_pos_iff_support_of_nonneg (fun s => Real.sqrt_nonneg _) (integrable_sqrt h𝓑 hi),
    hs]
  exact (integral_pos_iff_support_of_nonneg h𝓑 hi).1 hE

/-- The hypotheses of `integral_sqrt_pos` are satisfiable: `𝓑 = 1` over one step. -/
example := integral_sqrt_pos (μ := Measure.dirac ()) (𝓑 := 1) (fun _ => zero_le_one)
  (integrable_const 1) (by simp)

/-- **Appendix D**: `γ ≤ 1`, by Cauchy–Schwarz `(∫√𝓑 ds)² ≤ S_min E_min`. -/
theorem paretoGamma_le_one (h𝓑 : ∀ s, 0 ≤ 𝓑 s) (hi : Integrable 𝓑 μ) :
    paretoGamma μ 𝓑 ≤ 1 := by
  rcases (measureReal_nonneg (μ := μ) (s := Set.univ)).eq_or_lt with hμ | hμ
  · simp [paretoGamma, ← hμ]
  have h := integral_sqrt_sub_sq h𝓑 hi ((∫ s, √(𝓑 s) ∂μ) / μ.real Set.univ)
  have h0 : 0 ≤ ∫ s, (√(𝓑 s) - (∫ s, √(𝓑 s) ∂μ) / μ.real Set.univ) ^ 2 ∂μ :=
    integral_nonneg fun s => sq_nonneg _
  rw [h] at h0
  refine div_le_one_of_le₀ ?_ (mul_nonneg hμ.le (integral_nonneg h𝓑))
  generalize ∫ s, √(𝓑 s) ∂μ = c at h0
  generalize ∫ s, 𝓑 s ∂μ = E at h0
  generalize μ.real Set.univ = S at hμ h0 ⊢
  have : 0 ≤ (S * E - c ^ 2) / S := by
    convert h0 using 1
    field_simp
    ring
  nlinarith [(div_nonneg_iff.1 this).resolve_right fun h => absurd h.2 (not_le.2 hμ)]

/-- The hypotheses of `paretoGamma_le_one` are satisfiable: `𝓑 = 0`. -/
example := paretoGamma_le_one (μ := μ) (𝓑 := 0) (fun _ => le_rfl) (integrable_zero _ _ _)

omit [IsFiniteMeasure μ] in
/-- **Appendix D**: "when the noise scale is constant `γ = 1`". -/
theorem paretoGamma_const (hμ : 0 < μ.real Set.univ) {b : ℝ} (hb : 0 < b) :
    paretoGamma μ (fun _ => b) = 1 := by
  rw [paretoGamma, integral_const, integral_const, smul_eq_mul, smul_eq_mul, mul_pow,
    Real.sq_sqrt hb.le]
  field_simp

/-- The hypotheses of `paretoGamma_const` are satisfiable: one step, `b = 1`. -/
example := paretoGamma_const (μ := Measure.dirac ()) (by simp) one_pos

/-- **Appendix D**: on as many examples as a constant batch `B`, the schedule `√(r𝓑)` takes
`S_min + γ(S - S_min)` steps, where `S` are the constant batch's: "when the noise scale is
constant `γ = 1` and there is no benefit from using an adaptive batch size; more variation in
`𝓑` pushes `γ` closer to 0, yielding a corresponding predicted improvement in the Pareto
front". -/
theorem totalSteps_adaptiveBatch_sub (hμ : 0 < μ.real Set.univ) (h𝓑 : ∀ s, 0 ≤ 𝓑 s)
    (hi : Integrable 𝓑 μ) (hE : 0 < ∫ s, 𝓑 s ∂μ) {r B : ℝ} (hr : 0 < r) (hB : 0 < B)
    (h : totalExamples μ 𝓑 (adaptiveBatch r 𝓑) = totalExamples μ 𝓑 fun _ => B) :
    totalSteps μ 𝓑 (adaptiveBatch r 𝓑) - μ.real Set.univ =
      paretoGamma μ 𝓑 * (totalSteps μ 𝓑 (fun _ => B) - μ.real Set.univ) := by
  have h1 := totalSteps_adaptiveBatch_tradeoff hμ h𝓑 hi hE hr
  rw [h, ← totalSteps_tradeoff hμ hi hE hB] at h1
  have e : ∀ x, x - μ.real Set.univ = (x / μ.real Set.univ - 1) * μ.real Set.univ := fun x => by
    rw [sub_one_mul, div_mul_cancel₀ _ hμ.ne']
  rw [e, e (totalSteps _ _ _), h1, mul_assoc]

/-- The hypotheses of `totalSteps_adaptiveBatch_sub` are satisfiable: `𝓑 = 1` over one step,
`r = B = 1`. -/
example := totalSteps_adaptiveBatch_sub (μ := Measure.dirac ()) (𝓑 := 1) (by simp)
  (fun _ => zero_le_one) (integrable_const 1) (by simp) one_pos one_pos
  (by simp [totalExamples, adaptiveBatch])

/-- **Appendix D**: "our theoretical analysis would predict negligible benefits at large
`E_tot/S_tot`": the steps saved by the schedule `√(r𝓑)` on as many examples as a constant batch
`B`, at `√r = B S_min/∫√𝓑 ds`, tend to 0 as `B → ∞`. -/
theorem tendsto_totalSteps_sub_adaptiveBatch (hμ : 0 < μ.real Set.univ) (h𝓑 : ∀ s, 0 ≤ 𝓑 s)
    (hi : Integrable 𝓑 μ) (hE : 0 < ∫ s, 𝓑 s ∂μ) :
    Tendsto (fun B => totalSteps μ 𝓑 (fun _ => B) - totalSteps μ 𝓑
      (adaptiveBatch ((B * μ.real Set.univ / ∫ s, √(𝓑 s) ∂μ) ^ 2) 𝓑)) atTop (𝓝 0) := by
  have hc := integral_sqrt_pos h𝓑 hi hE
  have h := ((tendsto_totalSteps hi).sub_const (μ.real Set.univ)).const_mul
    (1 - paretoGamma μ 𝓑)
  rw [sub_self, mul_zero] at h
  refine h.congr' ((eventually_gt_atTop 0).mono fun B hB => ?_)
  have hr : 0 < (B * μ.real Set.univ / ∫ s, √(𝓑 s) ∂μ) ^ 2 := by positivity
  have := totalSteps_adaptiveBatch_sub hμ h𝓑 hi hE hr hB (by
    rw [totalExamples_adaptiveBatch h𝓑 hi hr.le, Real.sqrt_sq (by positivity),
      totalExamples_eq hi (integrable_const B), integral_const, smul_eq_mul,
      div_mul_cancel₀ _ hc.ne', mul_comm])
  linarith

/-- The hypotheses of `tendsto_totalSteps_sub_adaptiveBatch` are satisfiable: `𝓑 = 1` over one
step. -/
example := tendsto_totalSteps_sub_adaptiveBatch (μ := Measure.dirac ()) (𝓑 := 1) (by simp)
  (fun _ => zero_le_one) (integrable_const 1) (by simp)

/-- **Appendix D**, footnote: `γ = E[√𝓑]²/E[𝓑] = 1/(1 + σ²/E[√𝓑]²)`, with `E` the average
over a run, "weighting each full-batch step equally", and `σ²` the variance of `√𝓑` under it. -/
theorem paretoGamma_eq_average (hμ : 0 < μ.real Set.univ) (h𝓑 : ∀ s, 0 ≤ 𝓑 s)
    (hi : Integrable 𝓑 μ) (hE : 0 < ∫ s, 𝓑 s ∂μ) :
    paretoGamma μ 𝓑 = (⨍ s, √(𝓑 s) ∂μ) ^ 2 / ⨍ s, 𝓑 s ∂μ ∧
      paretoGamma μ 𝓑 =
        1 / (1 + (⨍ s, (√(𝓑 s) - ⨍ t, √(𝓑 t) ∂μ) ^ 2 ∂μ) / (⨍ s, √(𝓑 s) ∂μ) ^ 2) := by
  have hc := integral_sqrt_pos h𝓑 hi hE
  simp only [average_eq, smul_eq_mul]
  rw [integral_sqrt_sub_sq h𝓑 hi, paretoGamma]
  generalize ∫ s, √(𝓑 s) ∂μ = c at hc ⊢
  generalize ∫ s, 𝓑 s ∂μ = E at hE ⊢
  generalize μ.real Set.univ = S at hμ ⊢
  have := hμ.ne'
  have := hE.ne'
  constructor
  · field_simp
  · rw [show S⁻¹ * (E - 2 * (S⁻¹ * c) * c + (S⁻¹ * c) ^ 2 * S) / (S⁻¹ * c) ^ 2 =
      (S * E - c ^ 2) / c ^ 2 by field_simp; ring]
    field_simp
    ring

/-- The hypotheses of `paretoGamma_eq_average` are satisfiable: `𝓑 = 1` over one step. -/
example := paretoGamma_eq_average (μ := Measure.dirac ()) (𝓑 := 1) (by simp)
  (fun _ => zero_le_one) (integrable_const 1) (by simp)

/-- **Appendix D**, "An SVHN Case Study": for the fit `𝓑_crit(s) ≈ 10√s`, `γ = 24/25`, "or
around 4%"; here for `𝓑 = c√s` on a run of `S` full-batch steps, any `c, S > 0`. -/
theorem paretoGamma_sqrt {c S : ℝ} (hc : 0 < c) (hS : 0 < S) :
    paretoGamma (volume.restrict (Set.Ioc 0 S)) (fun s => c * √s) = 24 / 25 := by
  have ht := Real.rpow_pos_of_pos hS (1 / 4)
  have h1 : ∫ s in Set.Ioc 0 S, √(c * √s) = √c * (S * S ^ (1 / 4 : ℝ)) / (5 / 4) := by
    rw [setIntegral_congr_fun measurableSet_Ioc (g := fun s => √c * s ^ (1 / 4 : ℝ))
      fun s hs => ?_]
    · rw [integral_const_mul, ← intervalIntegral.integral_of_le hS.le,
        integral_rpow (Or.inl (by norm_num)), Real.zero_rpow (by norm_num), add_comm,
        Real.rpow_add hS, Real.rpow_one]
      ring
    · simp only [Real.sqrt_mul hc.le, Real.sqrt_eq_rpow, ← Real.rpow_mul hs.1.le]
      norm_num
  have h2 : ∫ s in Set.Ioc 0 S, c * √s = c * (S * (S ^ (1 / 4 : ℝ)) ^ 2) / (3 / 2) := by
    rw [integral_const_mul, ← intervalIntegral.integral_of_le hS.le]
    simp_rw [Real.sqrt_eq_rpow]
    rw [integral_rpow (Or.inl (by norm_num)), Real.zero_rpow (by norm_num), add_comm,
      Real.rpow_add hS, Real.rpow_one, ← Real.rpow_natCast, ← Real.rpow_mul hS.le]
    norm_num
    ring
  rw [paretoGamma, h1, h2]
  simp only [Measure.real, Measure.restrict_apply MeasurableSet.univ, Set.univ_inter,
    Real.volume_Ioc, sub_zero, ENNReal.toReal_ofReal hS.le]
  rw [div_pow, mul_pow, Real.sq_sqrt hc.le]
  field_simp
  ring

end Transformer.NoiseScale

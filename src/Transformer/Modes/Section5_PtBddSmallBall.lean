import Transformer.Modes.Section5_PtBddFejer

/-
# The number of modes of a Gaussian KDE — Fourier decay bounds the mass of small balls

A probability law whose Fourier transform decays like `(1 + |s|)^{-1/2}` gives mass
`O(√r)` to every ball `{|W| ≤ r}` (`measureReal_abs_le_of_decay`).  The Fejér kernel
`Λ_a` of `Section5_PtBddFejer.lean` is nonnegative and `≥ a/4` on `|w| ≤ 1/a`; by
Fubini its integral against the law of `W` is the integral of `(1 - s/a) Re 𝓕(s)` over
`[0, a]`, which the decay bounds by `2C√a`.

This is used in `Section5_PtBddDecayFalse.lean` to refute `eq:uniform-decay` for
`β > 2`: the law of `G'(t)` has mass `≳ r^{2/(β+2) - o(1)}` on `{|G'(t)| ≤ r}`.

Source: not a statement of arXiv:2412.09080v3; a tool for refuting its
`eq:uniform-decay` (§5.5).
-/

open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Modes

/-- `Re 𝔼 e^{-i s W} = 𝔼 cos (s W)`. -/
theorem re_integral_exp_neg_I_mul {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    [IsProbabilityMeasure P] {W : Ω → ℝ} (hW : Measurable W) (s : ℝ) :
    (∫ ω, Complex.exp (-(Complex.I * ((s * W ω : ℝ) : ℂ))) ∂P).re =
      ∫ ω, Real.cos (s * W ω) ∂P := by
  have hi : Integrable (fun ω => Complex.exp (-(Complex.I * ((s * W ω : ℝ) : ℂ)))) P := by
    refine Integrable.of_bound ?_ 1 (Filter.Eventually.of_forall fun ω => ?_)
    · exact (by fun_prop : Measurable fun ω =>
        Complex.exp (-(Complex.I * ((s * W ω : ℝ) : ℂ)))).aestronglyMeasurable
    · simp [Complex.norm_exp]
  have hpt : ∀ ω, (Complex.exp (-(Complex.I * ((s * W ω : ℝ) : ℂ)))).re =
      Real.cos (s * W ω) := by
    intro ω
    have : -(Complex.I * ((s * W ω : ℝ) : ℂ)) = ((-(s * W ω) : ℝ) : ℂ) * Complex.I := by
      push_cast; ring
    rw [this, Complex.exp_ofReal_mul_I_re, Real.cos_neg]
  have h := integral_re hi
  simp only [RCLike.re_to_complex, hpt] at h
  exact h.symm

/-- **Fourier decay bounds small balls.**  If `|𝔼 e^{-i s W}| ≤ C (1 + s)^{-1/2}` for all
`s ≥ 0`, then `ℙ(|W| ≤ 1/a) ≤ 8 C / √a` for every `a > 0`. -/
theorem measureReal_abs_le_of_decay {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    [IsProbabilityMeasure P] {W : Ω → ℝ} (hW : Measurable W) {C : ℝ}
    (hC : ∀ s : ℝ, 0 ≤ s →
      ‖∫ ω, Complex.exp (-(Complex.I * ((s * W ω : ℝ) : ℂ))) ∂P‖ ≤ C / Real.sqrt (1 + s))
    {a : ℝ} (ha : 0 < a) :
    P.real {ω | |W ω| ≤ 1 / a} ≤ 8 * C / Real.sqrt a := by
  have hC0 : 0 ≤ C := by
    have h := (norm_nonneg _).trans (hC 0 le_rfl)
    simpa using h
  set f : ℝ → Ω → ℝ := fun s ω => (1 - s / a) * Real.cos (s * W ω) with hf
  have hWc : Measurable fun p : ℝ × Ω => W p.2 := hW.comp measurable_snd
  have hmeas : Measurable (Function.uncurry f) := by
    simp only [hf, Function.uncurry_def]
    fun_prop
  have hprod : (volume.restrict (Set.Ioc (0 : ℝ) a)).prod P =
      (volume.prod P).restrict (Set.Ioc (0 : ℝ) a ×ˢ Set.univ) := by
    rw [← Measure.prod_restrict, Measure.restrict_univ]
  have hint : Integrable (Function.uncurry f)
      ((volume.restrict (Set.uIoc (0 : ℝ) a)).prod P) := by
    rw [Set.uIoc_of_le ha.le]
    refine Integrable.of_bound hmeas.aestronglyMeasurable 1 ?_
    rw [hprod, ae_restrict_iff' (measurableSet_Ioc.prod MeasurableSet.univ)]
    refine Filter.Eventually.of_forall fun ⟨s, ω⟩ hp => ?_
    have h1 : s / a ≤ 1 := (div_le_one ha).2 hp.1.2
    have h2 : 0 ≤ 1 - s / a := by linarith
    simp only [hf, Function.uncurry_apply_pair, norm_mul, Real.norm_eq_abs, abs_of_nonneg h2]
    calc (1 - s / a) * |Real.cos (s * W ω)| ≤ 1 * 1 :=
          mul_le_mul (by linarith [div_nonneg hp.1.1.le ha.le]) (Real.abs_cos_le_one _)
            (abs_nonneg _) zero_le_one
      _ = 1 := one_mul 1
  have hswap := intervalIntegral_integral_swap hint
  have hint_s : IntervalIntegrable (fun s => ∫ ω, f s ω ∂P) volume 0 a := by
    rw [intervalIntegrable_iff_integrableOn_Ioc_of_le ha.le]
    have := hint.integral_prod_left
    rwa [Set.uIoc_of_le ha.le] at this
  have hint_ω : Integrable (fun ω => fejer a (W ω)) P := by
    have := hint.integral_prod_right
    refine this.congr (Filter.Eventually.of_forall fun ω => ?_)
    simp [fejer, intervalIntegral.integral_of_le ha.le, Set.uIoc_of_le ha.le, hf]
  have hS : MeasurableSet {ω | |W ω| ≤ 1 / a} := measurableSet_le (continuous_abs.measurable.comp hW) measurable_const
  have hlow : a / 4 * P.real {ω | |W ω| ≤ 1 / a} ≤ ∫ ω, fejer a (W ω) ∂P := by
    calc a / 4 * P.real {ω | |W ω| ≤ 1 / a}
        = ∫ ω in {ω | |W ω| ≤ 1 / a}, a / 4 ∂P := by
          rw [setIntegral_const, smul_eq_mul, mul_comm]
      _ ≤ ∫ ω in {ω | |W ω| ≤ 1 / a}, fejer a (W ω) ∂P :=
          setIntegral_mono_on (integrableOn_const (measure_ne_top P _)) hint_ω.integrableOn hS
            fun ω hω => fejer_ge ha hω
      _ ≤ ∫ ω, fejer a (W ω) ∂P :=
          setIntegral_le_integral hint_ω
            (Filter.Eventually.of_forall fun ω => fejer_nonneg ha _)
  have hcont : ContinuousOn (fun s : ℝ => C / Real.sqrt (1 + s)) (Set.uIcc 0 a) := by
    refine continuousOn_const.div (Continuous.continuousOn (by fun_prop)) fun s hs => ?_
    rw [Set.uIcc_of_le ha.le] at hs
    exact (Real.sqrt_pos.2 (by linarith [hs.1])).ne'
  have hint_rhs : IntervalIntegrable (fun s : ℝ => C / Real.sqrt (1 + s)) volume 0 a :=
    hcont.intervalIntegrable
  have hd : ∀ s ∈ Set.uIcc (0 : ℝ) a,
      HasDerivAt (fun s => 2 * C * Real.sqrt (1 + s)) (C / Real.sqrt (1 + s)) s := by
    intro s hs
    rw [Set.uIcc_of_le ha.le] at hs
    have h1 : HasDerivAt (fun s : ℝ => 1 + s) 1 s := (hasDerivAt_id' s).const_add 1
    have h2 := (h1.sqrt (by linarith [hs.1] : (1 + s) ≠ 0)).const_mul (2 * C)
    refine h2.congr_deriv ?_
    have := Real.sqrt_pos.2 (by linarith [hs.1] : 0 < 1 + s)
    field_simp
  have hup : ∫ ω, fejer a (W ω) ∂P ≤ 2 * C * Real.sqrt a := by
    have hswap' : ∫ ω, fejer a (W ω) ∂P = ∫ s in (0 : ℝ)..a, ∫ ω, f s ω ∂P := hswap.symm
    rw [hswap']
    calc ∫ s in (0 : ℝ)..a, ∫ ω, f s ω ∂P
        ≤ ∫ s in (0 : ℝ)..a, C / Real.sqrt (1 + s) := by
          refine intervalIntegral.integral_mono_on ha.le hint_s hint_rhs fun s hs => ?_
          have hinner : ∫ ω, f s ω ∂P =
              (1 - s / a) * (∫ ω, Complex.exp (-(Complex.I * ((s * W ω : ℝ) : ℂ))) ∂P).re := by
            rw [re_integral_exp_neg_I_mul P hW s, ← integral_const_mul]
          rw [hinner]
          have hre : (∫ ω, Complex.exp (-(Complex.I * ((s * W ω : ℝ) : ℂ))) ∂P).re ≤
              C / Real.sqrt (1 + s) := (Complex.re_le_norm _).trans (hC s hs.1)
          have hb : 0 ≤ C / Real.sqrt (1 + s) := div_nonneg hC0 (Real.sqrt_nonneg _)
          have h1 : 0 ≤ 1 - s / a := by linarith [(div_le_one ha).2 hs.2]
          have h2 : 1 - s / a ≤ 1 := by linarith [div_nonneg hs.1 ha.le]
          rcases le_total 0
            (∫ ω, Complex.exp (-(Complex.I * ((s * W ω : ℝ) : ℂ))) ∂P).re with h0 | h0
          · calc _ ≤ 1 * (∫ ω, Complex.exp (-(Complex.I * ((s * W ω : ℝ) : ℂ))) ∂P).re :=
                mul_le_mul_of_nonneg_right h2 h0
              _ ≤ _ := by rw [one_mul]; exact hre
          · exact (mul_nonpos_of_nonneg_of_nonpos h1 h0).trans hb
      _ = 2 * C * Real.sqrt (1 + a) - 2 * C := by
          rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hd hint_rhs]
          simp
      _ ≤ 2 * C * Real.sqrt a := by
          have : Real.sqrt (1 + a) ≤ 1 + Real.sqrt a := by
            rw [Real.sqrt_le_iff]
            exact ⟨by positivity, by nlinarith [Real.sq_sqrt ha.le, Real.sqrt_nonneg a]⟩
          nlinarith
  have hsa : 0 < Real.sqrt a := Real.sqrt_pos.2 ha
  have hsq : Real.sqrt a * Real.sqrt a = a := Real.mul_self_sqrt ha.le
  rw [le_div_iff₀ hsa]
  nlinarith [hlow, hup, hsa, hsq, measureReal_nonneg (μ := P) (s := {ω | |W ω| ≤ 1 / a})]

/-- The hypotheses of `re_integral_exp_neg_I_mul` are satisfiable: the standard Gaussian. -/
example : (∫ x : ℝ, Complex.exp (-(Complex.I * (((1 : ℝ) * x : ℝ) : ℂ))) ∂gaussianReal 0 1).re =
    ∫ x : ℝ, Real.cos (1 * x) ∂gaussianReal 0 1 :=
  re_integral_exp_neg_I_mul (gaussianReal 0 1) measurable_id 1

/-- The hypotheses of `measureReal_abs_le_of_decay` are satisfiable: the standard Gaussian
has `|𝔼 e^{-i s x}| = e^{-s²/2} ≤ 2 (1 + s)^{-1/2}`. -/
example : ∀ s : ℝ, 0 ≤ s →
    ‖∫ x : ℝ, Complex.exp (-(Complex.I * ((s * x : ℝ) : ℂ))) ∂gaussianReal 0 1‖ ≤
      2 / Real.sqrt (1 + s) := by
  intro s hs
  have h : ∫ x : ℝ, Complex.exp (-(Complex.I * ((s * x : ℝ) : ℂ))) ∂gaussianReal 0 1 =
      charFun (gaussianReal 0 1) (-s) := by
    rw [charFun_apply_real]
    congr 1
    ext x
    congr 1
    push_cast
    ring
  rw [h, charFun_gaussianReal, Complex.norm_exp]
  have hre : ((-s : ℝ) * ((0 : ℝ) : ℂ) * Complex.I - ((1 : NNReal) : ℂ) * ((-s : ℝ) : ℂ) ^ 2 / 2).re
      = -(s ^ 2 / 2) := by
    simp [pow_two]
  rw [hre]
  have hpos : 0 < Real.sqrt (1 + s) := Real.sqrt_pos.2 (by linarith)
  rw [le_div_iff₀ hpos]
  have h1 : Real.sqrt (1 + s) ≤ 2 * Real.exp (s ^ 2 / 2) := by
    rw [Real.sqrt_le_iff]
    refine ⟨by positivity, ?_⟩
    have h2 : Real.exp (s ^ 2) = Real.exp (s ^ 2 / 2) ^ 2 := by
      rw [← Real.exp_nat_mul]; ring_nf
    have h3 : 1 + s ≤ 4 * Real.exp (s ^ 2) := by
      rcases le_total s 1 with h | h
      · nlinarith [Real.add_one_le_exp (s ^ 2), sq_nonneg s]
      · nlinarith [Real.add_one_le_exp (s ^ 2), sq_nonneg s]
    nlinarith [h2, h3]
  calc Real.exp (-(s ^ 2 / 2)) * Real.sqrt (1 + s)
      ≤ Real.exp (-(s ^ 2 / 2)) * (2 * Real.exp (s ^ 2 / 2)) :=
        mul_le_mul_of_nonneg_left h1 (Real.exp_pos _).le
    _ = 2 := by
        rw [← mul_assoc, mul_comm _ 2, mul_assoc, ← Real.exp_add]; simp

end Modes
end Transformer

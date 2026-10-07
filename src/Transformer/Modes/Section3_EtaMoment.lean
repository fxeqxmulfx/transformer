/-
# The number of modes of a Gaussian KDE — the moment bound of `lem:eta`

The standardized law has `E‖Y(t)‖² = 2`. The uniform pointwise estimate
`‖Y(t,x)‖ ≤ C(β exp(t²))^{1/4}` therefore gives
`E‖Y(t)‖^s ≤ 2 C^{s-2}(β exp(t²))^{(s-2)/4}` for every integer `s ≥ 2`.
The paper's `s ≥ 3` statement follows with its original hypotheses.

This uses boundedness and the exact second moment in place of the manuscript's
calculation of each higher Gaussian moment. No cumulant estimate or density
existence is required.

Source: arXiv:2412.09080v3, `lem:eta`, §3.1 and its proof in §5.3.
-/

import Transformer.Modes.Section3_SummandUniform
import Mathlib.MeasureTheory.SpecificCodomains.Pi

open Real MeasureTheory ProbabilityTheory Filter

namespace Transformer.Modes

/-- The squared Euclidean norm is integrable for a standardized law.
Source: arXiv:2412.09080v3, the sentence after `eq:Yi`, `lem:eta`, §5.3. -/
theorem integrable_eucl_sq_of_standardized {μ : Measure (ℝ × ℝ)}
    (hμ : IsStandardized μ) : Integrable (fun z => eucl z ^ 2) μ := by
  have hfst := hμ.memLp.fst.integrable_sq
  have hsnd := hμ.memLp.snd.integrable_sq
  convert hfst.add hsnd using 1
  funext z
  exact Real.sq_sqrt (by positivity)

/-- The standard Gaussian law satisfies the integrability hypothesis. -/
example : IsStandardized stdGauss2 := isStandardized_stdGauss2

/-- Standardization fixes the second Euclidean moment at two.
Source: arXiv:2412.09080v3, the sentence after `eq:Yi`, `lem:eta`, §5.3. -/
theorem integral_eucl_sq_of_standardized {μ : Measure (ℝ × ℝ)}
    (hμ : IsStandardized μ) : ∫ z, eucl z ^ 2 ∂μ = 2 := by
  have heq : ∀ z : ℝ × ℝ, eucl z ^ 2 = z.1 ^ 2 + z.2 ^ 2 :=
    fun z => Real.sq_sqrt (by positivity)
  have hfst : Integrable (fun z : ℝ × ℝ => z.1 ^ 2) μ := hμ.memLp.fst.integrable_sq
  have hsnd : Integrable (fun z : ℝ × ℝ => z.2 ^ 2) μ := hμ.memLp.snd.integrable_sq
  simp_rw [heq]
  rw [integral_add hfst hsnd, hμ.var_fst, hμ.var_snd]
  norm_num

/-- The standard Gaussian law witnesses the exact second-moment assertion. -/
example : IsStandardized stdGauss2 := isStandardized_stdGauss2

/-- In particular, the exact second moment of the law in `eq:Yi` is two,
at every positive bandwidth and every location, before taking a regime limit.
Source: arXiv:2412.09080v3, the sentence after `eq:Yi`, `lem:eta`, §5.3. -/
theorem etaMoment_two_eq {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    etaMoment β t 2 = 2 :=
  integral_eucl_sq_of_standardized (isStandardized_lawY hβ t)

/-- Positive bandwidth witnesses the exact second-moment identity's
hypothesis; no restriction on `t` is needed. -/
example : (0 : ℝ) < 1 := one_pos

/-- Boundedness and standardization control every higher Euclidean moment.
Source: arXiv:2412.09080v3, `lem:eta`, §5.3; the bounded-second-moment proof. -/
theorem integral_eucl_pow_le_of_ae_bound {μ : Measure (ℝ × ℝ)}
    [IsProbabilityMeasure μ] (hμ : IsStandardized μ) {M : ℝ}
    (hbound : ∀ᵐ z ∂μ, eucl z ≤ M) (s : ℕ) (hs : 2 ≤ s) :
    ∫ z, eucl z ^ s ∂μ ≤ 2 * M ^ (s - 2) := by
  have hnonneg : ∀ z : ℝ × ℝ, 0 ≤ eucl z := fun z => Real.sqrt_nonneg _
  have hmeas : AEStronglyMeasurable (fun z : ℝ × ℝ => eucl z ^ s) μ := by
    have hc : Continuous (fun z : ℝ × ℝ => eucl z ^ s) := by unfold eucl; fun_prop
    exact hc.aestronglyMeasurable
  have hint : Integrable (fun z => eucl z ^ s) μ :=
    Integrable.of_bound hmeas (M ^ s) (by
      filter_upwards [hbound] with z hz
      rw [Real.norm_eq_abs, abs_of_nonneg (pow_nonneg (hnonneg z) _)]
      exact pow_le_pow_left₀ (hnonneg z) hz s)
  have hsint := integrable_eucl_sq_of_standardized hμ
  have hpoint : ∀ᵐ z ∂μ, eucl z ^ s ≤ M ^ (s - 2) * eucl z ^ 2 := by
    filter_upwards [hbound] with z hz
    calc
      eucl z ^ s = eucl z ^ (s - 2) * eucl z ^ 2 := by
        rw [← pow_add, Nat.sub_add_cancel hs]
      _ ≤ M ^ (s - 2) * eucl z ^ 2 :=
        mul_le_mul_of_nonneg_right (pow_le_pow_left₀ (hnonneg z) hz _) (sq_nonneg _)
  calc
    _ ≤ ∫ z, M ^ (s - 2) * eucl z ^ 2 ∂μ :=
      integral_mono_ae hint (hsint.const_mul _) hpoint
    _ = 2 * M ^ (s - 2) := by
      rw [integral_const_mul, integral_eucl_sq_of_standardized hμ]
      ring

/-- The hypotheses hold for an actual standardized summand with positive
bandwidth at `t = 0`. Source: arXiv:2412.09080v3, `eq:Yi`, `lem:eta`, §5.3. -/
example : ∃ (μ : Measure (ℝ × ℝ)) (M : ℝ),
    IsProbabilityMeasure μ ∧ IsStandardized μ ∧ (∀ᵐ z ∂μ, eucl z ≤ M) ∧ 2 ≤ 3 := by
  obtain ⟨C, _, hbound⟩ :=
    eventually_singleY_le isRegime_succ isSlowGrowth_sqrt_log_log
  obtain ⟨k, hk⟩ := hbound.exists
  let β := ((k + 1 : ℕ) : ℝ)
  refine ⟨lawY β 0, C * (β * Real.exp (0 ^ 2)) ^ ((1 : ℝ) / 4), inferInstance,
    isStandardized_lawY (by dsimp [β]; positivity) 0, ?_, by norm_num⟩
  rw [lawY]
  apply (ae_map_iff (measurable_singleY β 0).aemeasurable ?_).mpr
  · exact ae_of_all _ fun x => hk 0 (zero_mem_intervalT _ _ _) x
  · exact measurableSet_le (by unfold eucl; fun_prop) measurable_const

/-- The quarter-power scale raised to `s - 2` has the exponent printed in
`lem:eta`. Source: arXiv:2412.09080v3, `lem:eta`, §5.3. -/
theorem etaScale_pow {β : ℝ} (hβ : 0 < β) (t : ℝ) (s : ℕ) (hs : 2 ≤ s) :
    ((β * Real.exp (t ^ 2)) ^ ((1 : ℝ) / 4)) ^ (s - 2) =
      (β * Real.exp (t ^ 2)) ^ (((s : ℝ) - 2) / 4) := by
  rw [← Real.rpow_natCast, ← Real.rpow_mul (by positivity), Nat.cast_sub hs]
  norm_num
  congr 1
  ring

/-- Positive bandwidth and order three satisfy the scale identity's inputs. -/
example : (0 : ℝ) < 1 ∧ 2 ≤ 3 := by norm_num

/-- **Lemma (lem:eta), moments.** `η_s = E‖Y(t)‖^s ≲ (β e^{t²})^{(s-2)/4}`,
uniformly on `T`, with the manuscript's hypotheses unchanged. The pointwise
bound and the exact second moment give a constant depending only on `s`.
Source: arXiv:2412.09080v3, `lem:eta` and its proof in §5.3. -/
theorem etaMoment_le {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B)
    {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) (s : ℕ) (hs : 3 ≤ s) :
    ∃ C : ℝ, ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
      etaMoment (B k) t s ≤ C * (B k * Real.exp (t ^ 2)) ^ (((s : ℝ) - 2) / 4) := by
  obtain ⟨C, _, hbound⟩ := eventually_singleY_le hreg hω
  refine ⟨2 * C ^ (s - 2), ?_⟩
  filter_upwards [hbound] with k hk t ht
  have hβ := hreg.B_pos k
  have hs₂ : 2 ≤ s := by omega
  let R := (B k * Real.exp (t ^ 2)) ^ ((1 : ℝ) / 4)
  have hae : ∀ᵐ z ∂lawY (B k) t, eucl z ≤ C * R := by
    rw [lawY]
    apply (ae_map_iff (measurable_singleY (B k) t).aemeasurable ?_).mpr
    · exact ae_of_all _ fun x => hk t ht x
    · exact measurableSet_le (by unfold eucl; fun_prop) measurable_const
  calc
    etaMoment (B k) t s ≤ 2 * (C * R) ^ (s - 2) :=
      integral_eucl_pow_le_of_ae_bound (isStandardized_lawY hβ t) hae s hs₂
    _ = (2 * C ^ (s - 2)) * (B k * Real.exp (t ^ 2)) ^ (((s : ℝ) - 2) / 4) := by
      rw [mul_pow, etaScale_pow hβ t s hs₂]
      ring

/-- The hypotheses of `etaMoment_le` are satisfiable. -/
example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) ∧ 3 ≤ 3 :=
  ⟨isRegime_succ, isSlowGrowth_sqrt_log_log, le_rfl⟩

end Transformer.Modes

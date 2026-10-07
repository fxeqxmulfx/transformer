/-
# The number of modes of a Gaussian KDE — `lem:main-int-phi` on `T`

The first estimate of `lem:main-int-phi` in arXiv:2412.09080v3, §2.3:
the Gaussian proxy Kac–Rice integral over `T` has order `√(β log β)`.
The source passes through `eq:int-phi-final`, whose pointwise form requires
`n ≲ β^{5/2}`.  Here the shift is bounded by `α_t δ_t² ≤ C R`, where
`R = n β^{-3/2} e^{-t²/2}`, and is handled directly.  All of the original
regimes `n^c ≲ β ≲ n^{2-c}` are retained.
-/

import Transformer.Modes.Section2_ProxyKRBounds

open Real Filter Asymptotics MeasureTheory
open scoped Topology ENNReal

namespace Transformer.Modes

/-- The radius of `T` is at most a constant times `√(log β)` in the paper's
regime.  Source: arXiv:2412.09080v3, §2.3, `eq:T`, `lem:main-int-phi`. -/
theorem eventually_window_radius_le {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∃ M : ℝ, 0 < M ∧ ∀ᶠ k in atTop, 1 ≤ Real.log (B k) ∧
      Real.sqrt (2 * Real.log (N k) - Real.log (B k) - ω (B k)) ≤
        M * Real.sqrt (Real.log (B k)) := by
  have hc := hreg.c_pos
  obtain ⟨K₁, K₂, hlog⟩ := hreg.eventually_log
  refine ⟨Real.sqrt (4 / c), by positivity, ?_⟩
  filter_upwards [hlog, hreg.tendsto_B.eventually (hω.eventually_le_log one_pos)] with k hk hw
  obtain ⟨-, hL1, hK₁, -, -, hN, -⟩ := hk
  have hL0 : 0 ≤ Real.log (B k) := by linarith
  refine ⟨hL1, ?_⟩
  rw [← Real.sqrt_mul (by positivity)]
  apply Real.sqrt_le_sqrt
  rw [div_mul_eq_mul_div, le_div_iff₀ hc]
  nlinarith [mul_nonneg hc.le hw.1, mul_nonneg hc.le hL0]

/-- The regime and window in the radius bound have a concrete instance.
Source: arXiv:2412.09080v3, §2.3, `lem:main-int-phi`. -/
example := eventually_window_radius_le isRegime_succ isSlowGrowth_sqrt_log_log

/-- Removing `[-1,1]` retains a positive `√(log β)` lower bound for the rate
integral.  Source: arXiv:2412.09080v3, §2.3, `eq:int-phi-b`,
proof of `lem:main-int-phi`. -/
theorem eventually_integral_exp_phiRate_outside_ge {c : ℝ} {N : ℕ → ℕ}
    {B : ℕ → ℝ} (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω)
    {γ : ℝ} (hγ : 0 < γ) :
    ∃ ε : ℝ, 0 < ε ∧ ∀ᶠ k in atTop,
      ε * Real.sqrt (Real.log (B k)) ≤
        ∫ t in intervalT (N k) (B k) (ω (B k)) \ Set.Icc (-1 : ℝ) 1,
          Real.exp (-γ * phiRate (N k) (B k) t) := by
  obtain ⟨ε, hε, hlow⟩ := isBigO_iff''.mp (integral_exp_phiRate_T hreg hω hγ).2
  have hL := Real.tendsto_sqrt_atTop.comp (Real.tendsto_log_atTop.comp hreg.tendsto_B)
  refine ⟨ε / 2, by positivity, ?_⟩
  filter_upwards [hlow, hL.eventually_ge_atTop (4 / ε)] with k hk hLk
  change 4 / ε ≤ Real.sqrt (Real.log (B k)) at hLk
  let T := intervalT (N k) (B k) (ω (B k))
  let f : ℝ → ℝ := fun t => Real.exp (-γ * phiRate (N k) (B k) t)
  have hf0 : ∀ t, 0 ≤ f t := fun t => (Real.exp_pos _).le
  have hf1 : ∀ t, f t ≤ 1 := fun t =>
    Real.exp_le_one_iff.mpr (by nlinarith [phiRate_nonneg (N k) (hreg.B_pos k) t])
  have hint : IntegrableOn f T := by
    refine Continuous.integrableOn_Icc ?_
    dsimp [f, phiRate]
    fun_prop
  have hI0 : 0 ≤ ∫ t in T, f t := setIntegral_nonneg measurableSet_Icc fun t _ => hf0 t
  rw [Real.norm_of_nonneg (Real.sqrt_nonneg _), Real.norm_of_nonneg hI0] at hk
  have hradius := norm_integral_exp_phiRate_T_le (N k) (hreg.B_pos k) (ω (B k)) hγ
  rw [Real.norm_of_nonneg hI0] at hradius
  have ha : 1 ≤ Real.sqrt (2 * Real.log (N k) - Real.log (B k) - ω (B k)) := by
    have hεL : 4 ≤ ε * Real.sqrt (Real.log (B k)) := by
      simpa only [mul_comm] using (div_le_iff₀ hε).mp hLk
    linarith
  have hsub : Set.Icc (-1 : ℝ) 1 ⊆ T := fun t ht =>
    ⟨by linarith [ht.1], by linarith [ht.2]⟩
  have hcentral : ∫ t in Set.Icc (-1 : ℝ) 1, f t ≤ 2 := by
    have h := norm_setIntegral_le_of_norm_le_const (μ := volume) (C := 1)
      (s := Set.Icc (-1 : ℝ) 1) measure_Icc_lt_top fun t _ => by
        rw [Real.norm_of_nonneg (hf0 t)]
        exact hf1 t
    have h0 : 0 ≤ ∫ t in Set.Icc (-1 : ℝ) 1, f t :=
      setIntegral_nonneg measurableSet_Icc fun t _ => hf0 t
    norm_num [Real.norm_of_nonneg h0, Real.volume_real_Icc] at h
    exact h
  have hdiff := setIntegral_sdiff measurableSet_Icc hint hsub
  change ε / 2 * Real.sqrt (Real.log (B k)) ≤ ∫ t in T \ Set.Icc (-1 : ℝ) 1, f t
  rw [hdiff]
  have hεL : 4 ≤ ε * Real.sqrt (Real.log (B k)) := by
    simpa only [mul_comm] using (div_le_iff₀ hε).mp hLk
  linarith

/-- The lower-bound parameters are simultaneously satisfiable.
Source: arXiv:2412.09080v3, §2.3, `lem:main-int-phi`. -/
example := eventually_integral_exp_phiRate_outside_ge isRegime_succ
  isSlowGrowth_sqrt_log_log (γ := 1) one_pos

/-- **Lemma (lem:main-int-phi), on `T`.**  In the regime
`n^c ≲ β ≲ n^{2-c}`, the Gaussian proxy integral has order `√(β log β)`.
The proof retains the shift rather than using `eq:int-phi-final` uniformly
on `T`; no additional bound on `n/β^{5/2}` is assumed.
Source: arXiv:2412.09080v3, §2.3, `lem:main-int-phi`. -/
theorem main_int_phi_T {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B)
    {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    (fun k => (proxyKR (N k) (B k) (intervalT (N k) (B k) (ω (B k)))).toReal)
      =Θ[atTop] fun k => Real.sqrt (B k * Real.log (B k)) := by
  obtain ⟨C, D, hC, hD, hup⟩ := eventually_proxyKR_T_le hreg hω
  obtain ⟨M, hM, hrad⟩ := eventually_window_radius_le hreg hω
  have hfinite : ∀ᶠ k in atTop,
      proxyKR (N k) (B k) (intervalT (N k) (B k) (ω (B k))) ≠ ∞ := by
    filter_upwards [hup] with k hk
    exact ne_top_of_le_ne_top ENNReal.ofReal_ne_top hk
  refine ⟨IsBigO.of_bound (C * (M + D)) ?_, ?_⟩
  · filter_upwards [hup, hrad] with k hk hr
    have hβ := hreg.B_pos k
    have hroot : 1 ≤ Real.sqrt (Real.log (B k)) := by
      simpa using Real.sqrt_le_sqrt hr.1
    have hreal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hk
    rw [ENNReal.toReal_ofReal (by positivity)] at hreal
    rw [Real.norm_of_nonneg ENNReal.toReal_nonneg,
      Real.norm_of_nonneg (Real.sqrt_nonneg _), Real.sqrt_mul hβ.le]
    refine hreal.trans ?_
    have hwindow : Real.sqrt (2 * Real.log (N k) - Real.log (B k) - ω (B k)) + D ≤
        (M + D) * Real.sqrt (Real.log (B k)) := by
      nlinarith [mul_le_mul_of_nonneg_left hroot hD]
    have := mul_le_mul_of_nonneg_left hwindow
      (by positivity : 0 ≤ C * Real.sqrt (B k))
    nlinarith
  · obtain ⟨L, γ, hL, hγ, hpoint⟩ := eventually_inner_ge_T hreg hω
    obtain ⟨ε, hε, hout⟩ := eventually_integral_exp_phiRate_outside_ge hreg hω hγ
    refine IsBigO.of_bound (1 / (L * ε)) ?_
    filter_upwards [hpoint, hout, hfinite] with k hk hIk hfin
    let T := intervalT (N k) (B k) (ω (B k))
    let S := T \ Set.Icc (-1 : ℝ) 1
    let f : ℝ → ℝ := fun t => Real.exp (-γ * phiRate (N k) (B k) t)
    have hSm : MeasurableSet S := measurableSet_Icc.diff measurableSet_Icc
    have hintT : IntegrableOn f T := by
      refine Continuous.integrableOn_Icc ?_
      dsimp [f, phiRate]
      fun_prop
    have hsub : S ⊆ T := Set.sdiff_subset
    have hI0 : 0 ≤ ∫ t in S, f t := setIntegral_nonneg hSm fun t _ => (Real.exp_pos _).le
    have hlin : ENNReal.ofReal (L * Real.sqrt (B k)) *
        ENNReal.ofReal (∫ t in S, f t) ≤ proxyKR (N k) (B k) T := by
      rw [ofReal_integral_eq_lintegral_ofReal (hintT.mono_set hsub)
        (Eventually.of_forall fun t => (Real.exp_pos _).le), ← lintegral_const_mul' _ _
          ENNReal.ofReal_ne_top]
      refine le_trans (setLIntegral_mono' hSm fun t ht => ?_)
        (lintegral_mono_set hsub)
      apply hk t ht.1
      have ht' : ¬ (-1 ≤ t ∧ t ≤ 1) := ht.2
      rcases lt_or_ge t (-1) with h | h
      · nlinarith
      · have : 1 < t := lt_of_not_ge (fun h1 => ht' ⟨h, h1⟩)
        nlinarith
    have hreal := ENNReal.toReal_mono hfin hlin
    rw [ENNReal.toReal_mul, ENNReal.toReal_ofReal (by positivity),
      ENNReal.toReal_ofReal hI0] at hreal
    have hge : (L * ε) * (Real.sqrt (B k) * Real.sqrt (Real.log (B k))) ≤
        (proxyKR (N k) (B k) T).toReal := by
      calc (L * ε) * (Real.sqrt (B k) * Real.sqrt (Real.log (B k)))
          = (L * Real.sqrt (B k)) * (ε * Real.sqrt (Real.log (B k))) := by ring
        _ ≤ (L * Real.sqrt (B k)) * ∫ t in S, f t := by gcongr
        _ ≤ _ := hreal
    rw [Real.norm_of_nonneg (Real.sqrt_nonneg _),
      Real.norm_of_nonneg ENNReal.toReal_nonneg, Real.sqrt_mul (hreg.B_pos k).le,
      one_div, ← div_eq_inv_mul, le_div_iff₀ (mul_pos hL hε)]
    nlinarith

/-- Both hypotheses of the full-window asymptotic estimate are satisfiable.
Source: arXiv:2412.09080v3, §2.3, `lem:main-int-phi`. -/
example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) :=
  ⟨isRegime_succ, isSlowGrowth_sqrt_log_log⟩

end Transformer.Modes

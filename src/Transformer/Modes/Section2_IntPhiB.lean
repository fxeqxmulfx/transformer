/-
# The number of modes of a Gaussian KDE — `eq:int-phi-b` on `T`

The half of `eq:int-phi-b` of arXiv:2412.09080v3 that lives on `T`:
`∫_T e^{-C β^{-3/2} n t² e^{-t²/2}} dt ≍ √(log β)`.

The upper bound is the source's: the integrand is at most `1`, and `T` has
length `2√(2 log n - log β - ω) ≲ √(log β)`.  For the lower bound the source
integrates over `[t₁, t₂]`, `t_k² = 2 log n - (1 + c/10^k) log β`; here the
window is `t² ∈ [max(2 log n - (5/2) log β, 0), 2 log n - log β - ω]`, where the
exponent is at most `C (log β) β^{-1/4}` and the integrand at least `e^{-1}`.
The same argument, with a window that needs no case split on `c`.

Source: arXiv:2412.09080v3, `eq:int-phi-b`, proof of `lem:main-int-phi`.
-/

import Transformer.Modes.Section2_MainIntPhi

open Real Filter Asymptotics MeasureTheory
open scoped Topology

namespace Transformer
namespace Modes

/-- The rate is nonnegative.  arXiv:2412.09080v3, `eq:int-phi-b`. -/
theorem phiRate_nonneg (n : ℕ) {β : ℝ} (hβ : 0 < β) (t : ℝ) : 0 ≤ phiRate n β t := by
  unfold phiRate; have := Real.rpow_nonneg hβ.le (-(3 : ℝ) / 2); positivity

/-- `β^{-3/2} n t² e^{-t²/2} = t² e^{-(3/2) log β + log n - t²/2}`.
arXiv:2412.09080v3, `eq:int-phi-b`. -/
theorem phiRate_eq {n : ℕ} {β : ℝ} (hn : (0 : ℝ) < n) (hβ : 0 < β) (t : ℝ) :
    phiRate n β t =
      t ^ 2 * Real.exp (Real.log β * (-(3 : ℝ) / 2) + Real.log n + -(t ^ 2) / 2) := by
  rw [phiRate, Real.rpow_def_of_pos hβ, Real.exp_add, Real.exp_add, Real.exp_log hn]; ring

/-- The upper half: the integrand is at most `1`, and `T` has length `2√D`.
arXiv:2412.09080v3, proof of `lem:main-int-phi`. -/
theorem norm_integral_exp_phiRate_T_le (n : ℕ) {β : ℝ} (hβ : 0 < β) (w : ℝ) {C : ℝ}
    (hC : 0 < C) :
    ‖∫ t in intervalT n β w, Real.exp (-C * phiRate n β t)‖ ≤
      2 * Real.sqrt (2 * Real.log n - Real.log β - w) := by
  refine (norm_setIntegral_le_of_norm_le_const (C := 1) measure_Icc_lt_top
    fun t _ => ?_).trans ?_
  · rw [Real.norm_of_nonneg (Real.exp_pos _).le, Real.exp_le_one_iff]
    nlinarith [phiRate_nonneg n hβ t]
  · rw [Real.volume_real_Icc, one_mul]
    exact max_le (by linarith) (by positivity)

/-- The lower half: if `C β^{-3/2} n t² e^{-t²/2} ≤ 1` whenever `s₀ ≤ t² ≤ D`, the
integral over `T` is at least `e^{-1} (√D - √s₀)`.  arXiv:2412.09080v3, proof of
`lem:main-int-phi` (the source's `t_k`). -/
theorem integral_exp_phiRate_T_ge {n : ℕ} {β w C s₀ : ℝ} (hs₀ : 0 ≤ s₀)
    (hs₀D : s₀ ≤ 2 * Real.log n - Real.log β - w)
    (hrate : ∀ t, s₀ ≤ t ^ 2 → t ^ 2 ≤ 2 * Real.log n - Real.log β - w →
      C * phiRate n β t ≤ 1) :
    Real.exp (-1) * (Real.sqrt (2 * Real.log n - Real.log β - w) - Real.sqrt s₀) ≤
      ∫ t in intervalT n β w, Real.exp (-C * phiRate n β t) := by
  set D := 2 * Real.log n - Real.log β - w
  have hsq : Real.sqrt s₀ ≤ Real.sqrt D := Real.sqrt_le_sqrt hs₀D
  have hsub : Set.Icc (Real.sqrt s₀) (Real.sqrt D) ⊆ intervalT n β w := fun t ht =>
    ⟨by linarith [ht.1, Real.sqrt_nonneg s₀, Real.sqrt_nonneg D], ht.2⟩
  have hint : IntegrableOn (fun t => Real.exp (-C * phiRate n β t)) (intervalT n β w) := by
    refine Continuous.integrableOn_Icc ?_; unfold phiRate; fun_prop
  refine le_trans ?_ (setIntegral_mono_set hint
    (Eventually.of_forall fun t => (Real.exp_pos _).le) (Eventually.of_forall hsub))
  have hge := setIntegral_ge_of_const_le (μ := volume) measurableSet_Icc measure_Icc_lt_top.ne
    (fun t (ht : t ∈ Set.Icc (Real.sqrt s₀) (Real.sqrt D)) => ?_) (hint.mono_set hsub)
    (c := Real.exp (-1))
  · rwa [Real.volume_real_Icc, max_eq_left (by linarith), smul_eq_mul, mul_comm] at hge
  · have ht0 : 0 ≤ t := (Real.sqrt_nonneg _).trans ht.1
    have h1 : s₀ ≤ t ^ 2 := by
      rw [← Real.sq_sqrt hs₀]; exact pow_le_pow_left₀ (Real.sqrt_nonneg _) ht.1 2
    have h2 : t ^ 2 ≤ D := by
      rw [← Real.sq_sqrt (hs₀.trans hs₀D)]; exact pow_le_pow_left₀ ht0 ht.2 2
    exact Real.exp_le_exp.mpr (by linarith [hrate t h1 h2])

/-- The regime, in logarithms: eventually `log K₁ ≤ log β`, `log K₂ ≤ (c/8) log β`,
`log K₂ ≤ (log β)/8`, and `c log n ≤ log K₁ + log β`, `log β ≤ log K₂ + (2 - c) log n`.
arXiv:2412.09080v3, `thm:main`, the regime `n^c ≲ β ≲ n^{2-c}`. -/
theorem IsRegime.eventually_log {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B) :
    ∃ K₁ K₂ : ℝ, ∀ᶠ k in atTop, 1 ≤ (N k : ℝ) ∧ 1 ≤ Real.log (B k) ∧
      Real.log K₁ ≤ Real.log (B k) ∧ Real.log K₂ ≤ c * Real.log (B k) / 8 ∧
      Real.log K₂ ≤ Real.log (B k) / 8 ∧
      c * Real.log (N k) ≤ Real.log K₁ + Real.log (B k) ∧
      Real.log (B k) ≤ Real.log K₂ + (2 - c) * Real.log (N k) := by
  have hL : Tendsto (fun k => Real.log (B k)) atTop atTop :=
    Real.tendsto_log_atTop.comp hreg.tendsto_B
  obtain ⟨K₁, hK₁, hlow⟩ := hreg.lower.exists_pos
  obtain ⟨K₂, hK₂, hup⟩ := hreg.upper.exists_pos
  refine ⟨K₁, K₂, ?_⟩
  filter_upwards [hlow.bound, hup.bound, hreg.tendsto_N.eventually_ge_atTop 1,
    hL.eventually_ge_atTop 1, hL.eventually_ge_atTop (Real.log K₁),
    (hL.const_mul_atTop (by linarith [hreg.c_pos] : (0 : ℝ) < c / 8)).eventually_ge_atTop
      (Real.log K₂),
    (hL.atTop_div_const (by norm_num : (0 : ℝ) < 8)).eventually_ge_atTop (Real.log K₂)]
    with k h1 h2 hN hL1 hLK₁ hLK₂ hLK₂'
  have hn : (0 : ℝ) < N k := by linarith
  have hB := hreg.B_pos k
  rw [Real.norm_of_nonneg (Real.rpow_nonneg hn.le _), Real.norm_of_nonneg hB.le] at h1
  rw [Real.norm_of_nonneg hB.le, Real.norm_of_nonneg (Real.rpow_nonneg hn.le _)] at h2
  have l1 := Real.log_le_log (Real.rpow_pos_of_pos hn c) h1
  have l2 := Real.log_le_log hB h2
  rw [Real.log_rpow hn, Real.log_mul hK₁.ne' hB.ne'] at l1
  rw [Real.log_mul hK₂.ne' (Real.rpow_pos_of_pos hn _).ne', Real.log_rpow hn] at l2
  exact ⟨hN, hL1, hLK₁, by linarith, hLK₂', l1, l2⟩

/-- `ω(β) ≤ ε log β` eventually, for every `ε > 0`: `ω ≪ log log β ≤ log β`.
arXiv:2412.09080v3, `sec: sketch`. -/
theorem IsSlowGrowth.eventually_le_log {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) {ε : ℝ}
    (hε : 0 < ε) : ∀ᶠ β in atTop, 0 ≤ ω β ∧ ω β ≤ ε * Real.log β := by
  filter_upwards [hω.upper.bound hε, hω.lower.eventually_ge_atTop 0,
    Real.tendsto_log_atTop.eventually_ge_atTop 1] with β hb h0 hL
  have hll0 : 0 ≤ Real.log (Real.log β) := Real.log_nonneg hL
  have hll : Real.log (Real.log β) ≤ Real.log β :=
    (Real.log_le_sub_one_of_pos (by linarith)).trans (by linarith)
  rw [Real.norm_of_nonneg hll0, Real.norm_of_nonneg h0] at hb
  exact ⟨h0, hb.trans (by gcongr)⟩

/-- **Equation (eq:int-phi-b), on `T`.**  For every `C > 0`,
`∫_T e^{-C β^{-3/2} n t² e^{-t²/2}} dt ≍ √(log β)`.

Source: arXiv:2412.09080v3, `eq:int-phi-b`, proof of `lem:main-int-phi`. -/
theorem integral_exp_phiRate_T {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B)
    {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) {C : ℝ} (hC : 0 < C) :
    (fun k => ∫ t in intervalT (N k) (B k) (ω (B k)), Real.exp (-C * phiRate (N k) (B k) t))
      =Θ[atTop] fun k => Real.sqrt (Real.log (B k)) := by
  have hc := hreg.c_pos
  have hL : Tendsto (fun k => Real.log (B k)) atTop atTop :=
    Real.tendsto_log_atTop.comp hreg.tendsto_B
  obtain ⟨K₁, K₂, hlog⟩ := hreg.eventually_log
  have hw := hreg.tendsto_B.eventually (hω.eventually_le_log (by positivity : (0 : ℝ) < c / 16))
  have hw' := hreg.tendsto_B.eventually (hω.eventually_le_log (by norm_num : (0 : ℝ) < 1 / 2))
  have hsmall : ∀ᶠ k in atTop,
      Real.log (B k) / 4 * Real.exp (-(Real.log (B k) / 4)) ≤ c / (16 * C) := by
    have h := (Real.tendsto_pow_mul_exp_neg_atTop_nhds_zero 1).comp
      (hL.atTop_div_const (by norm_num : (0 : ℝ) < 4))
    simpa using h.eventually (ge_mem_nhds (show (0 : ℝ) < c / (16 * C) by positivity))
  -- the facts both halves use: `c/8 · log β ≤ D ≤ 4/c · log β`
  have hD : ∀ᶠ k in atTop, 1 ≤ (N k : ℝ) ∧ 1 ≤ Real.log (B k) ∧
      0 ≤ ω (B k) ∧ ω (B k) ≤ Real.log (B k) / 2 ∧
      c / 8 * Real.log (B k) ≤ 2 * Real.log (N k) - Real.log (B k) - ω (B k) ∧
      2 * Real.log (N k) - Real.log (B k) - ω (B k) ≤ 4 / c * Real.log (B k) := by
    filter_upwards [hlog, hw, hw'] with k hk hwk hwk'
    obtain ⟨hN, hL1, hK₁, hK₂, hK₂', l1, l2⟩ := hk
    obtain ⟨h0, hwc⟩ := hwk
    obtain ⟨-, hw2⟩ := hwk'
    have hℓ : 0 ≤ Real.log (N k) := Real.log_nonneg hN
    have hℓ' : (Real.log (B k) - Real.log K₂) / 2 ≤ Real.log (N k) := by
      nlinarith [mul_nonneg hc.le hℓ]
    refine ⟨hN, hL1, h0, by linarith, ?_, ?_⟩
    · nlinarith [mul_le_mul_of_nonneg_left hℓ' hc.le]
    · rw [div_mul_eq_mul_div, le_div_iff₀ hc]
      nlinarith [mul_nonneg hc.le h0, mul_nonneg hc.le (by linarith : (0 : ℝ) ≤ Real.log (B k))]
  refine ⟨IsBigO.of_bound (2 * Real.sqrt (4 / c)) ?_,
    IsBigO.of_bound (2 * Real.sqrt (4 / c) / (min 1 (c / 8) * Real.exp (-1))) ?_⟩
  · filter_upwards [hD] with k hk
    obtain ⟨-, hL1, -, -, -, hDu⟩ := hk
    refine (norm_integral_exp_phiRate_T_le _ (hreg.B_pos k) _ hC).trans ?_
    rw [Real.norm_of_nonneg (Real.sqrt_nonneg _), mul_assoc, ← Real.sqrt_mul (by positivity)]
    gcongr
  · filter_upwards [hD, hsmall] with k hk hs
    obtain ⟨hN, hL1, h0, hw2, hDl, hDu⟩ := hk
    set L := Real.log (B k)
    set D := 2 * Real.log (N k) - L - ω (B k)
    set s₀ := max (2 * Real.log (N k) - 5 / 2 * L) 0
    set κ := min 1 (c / 8)
    have hκ : 0 < κ := lt_min one_pos (by positivity)
    have hκD : κ * L ≤ D - s₀ := by
      rcases le_total (2 * Real.log (N k) - 5 / 2 * L) 0 with h | h
      · rw [show s₀ = 0 from max_eq_right h]
        nlinarith [mul_le_mul_of_nonneg_right (min_le_right 1 (c / 8)) (by linarith : 0 ≤ L)]
      · rw [show s₀ = _ from max_eq_left h]
        nlinarith [mul_le_mul_of_nonneg_right (min_le_left 1 (c / 8)) (by linarith : 0 ≤ L)]
    have hs₀ : 0 ≤ s₀ := le_max_right _ _
    have hs₀D : s₀ ≤ D := by nlinarith
    have hn : (0 : ℝ) < N k := by linarith
    have hge := integral_exp_phiRate_T_ge (C := C) hs₀ hs₀D fun t h1 h2 => by
      rw [phiRate_eq hn (hreg.B_pos k)]
      have he : Real.exp (L * (-(3 : ℝ) / 2) + Real.log (N k) + -(t ^ 2) / 2) ≤
          Real.exp (-(L / 4)) :=
        Real.exp_le_exp.mpr (by linarith [le_max_left (2 * Real.log (N k) - 5 / 2 * L) 0])
      calc C * (t ^ 2 * Real.exp _) ≤ C * (4 / c * L * Real.exp (-(L / 4))) := by
            gcongr; linarith
        _ = C * (16 / c) * (L / 4 * Real.exp (-(L / 4))) := by ring
        _ ≤ C * (16 / c) * (c / (16 * C)) := by gcongr
        _ = 1 := by field_simp
    -- `κ L ≤ D - s₀ ≤ 2 √D (√D - √s₀)` and `√D ≤ √(4/c) √L`
    have hsq := Real.sqrt_le_sqrt hs₀D
    have hDs : D - s₀ ≤ 2 * Real.sqrt D * (Real.sqrt D - Real.sqrt s₀) := by
      nlinarith [Real.sq_sqrt hs₀, Real.sq_sqrt (hs₀.trans hs₀D), Real.sqrt_nonneg s₀]
    have hDL : Real.sqrt D ≤ Real.sqrt (4 / c) * Real.sqrt L := by
      rw [← Real.sqrt_mul (by positivity)]; exact Real.sqrt_le_sqrt hDu
    have hLpos : 0 < Real.sqrt L := Real.sqrt_pos.mpr (by linarith)
    have hLL : L = Real.sqrt L * Real.sqrt L := (Real.mul_self_sqrt (by linarith)).symm
    have key : κ * Real.sqrt L ≤ 2 * Real.sqrt (4 / c) * (Real.sqrt D - Real.sqrt s₀) := by
      have h1 : κ * Real.sqrt L * Real.sqrt L ≤
          2 * Real.sqrt (4 / c) * (Real.sqrt D - Real.sqrt s₀) * Real.sqrt L := by
        have := mul_le_mul_of_nonneg_right hDL (by linarith : 0 ≤ 2 * (Real.sqrt D - Real.sqrt s₀))
        nlinarith
      exact le_of_mul_le_mul_right h1 hLpos
    rw [Real.norm_of_nonneg hLpos.le, Real.norm_of_nonneg ((mul_nonneg (Real.exp_pos _).le
      (by linarith)).trans hge), div_mul_eq_mul_div, le_div_iff₀ (by positivity)]
    have hge' : Real.exp (-1) * (Real.sqrt D - Real.sqrt s₀) ≤
        ∫ t in intervalT (N k) (B k) (ω (B k)), Real.exp (-C * phiRate (N k) (B k) t) := hge
    nlinarith [mul_le_mul_of_nonneg_right key (Real.exp_pos (-1)).le,
      mul_le_mul_of_nonneg_left hge' (by positivity : (0 : ℝ) ≤ 2 * Real.sqrt (4 / c))]

/-- The hypotheses of `phiRate_nonneg`, `phiRate_eq` and
`norm_integral_exp_phiRate_T_le` are satisfiable. -/
example : (0 : ℝ) < ((1 : ℕ) : ℝ) ∧ (0 : ℝ) < 1 := ⟨by norm_num, one_pos⟩

/-- The hypotheses of `integral_exp_phiRate_T_ge` are satisfiable: `C = 0`. -/
example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 2 * Real.log (1 : ℕ) - Real.log 1 - 0 ∧
    ∀ t : ℝ, 0 ≤ t ^ 2 → t ^ 2 ≤ 2 * Real.log (1 : ℕ) - Real.log 1 - 0 →
      0 * phiRate 1 1 t ≤ 1 :=
  ⟨le_rfl, by simp, fun _ _ _ => by simp⟩

/-- The hypotheses of `IsRegime.eventually_log`, `IsSlowGrowth.eventually_le_log`
and `integral_exp_phiRate_T` are satisfiable. -/
example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) ∧ (0 : ℝ) < 1 :=
  ⟨isRegime_succ, isSlowGrowth_sqrt_log_log, one_pos⟩

end Modes
end Transformer

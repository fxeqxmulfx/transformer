/-
# The number of modes of a Gaussian KDE — `lem:error-3` on `T`

The absolute Edgeworth contribution is `O(exp(-ω/4) √(β log β))` throughout
the original regime.  Its moment prefactor is uniformly `O(exp(-ω/4))`,
and its Gaussian majorant integrates to `√β` times the window length plus
a constant.  The linear Hermite term and Gaussian shift are both retained.
Source: arXiv:2412.09080v3, §3.1, `lem:error-3` and its proof.
-/

import Transformer.Modes.Section3_PsiRate
import Transformer.Modes.Section3_ErrorHigher
import Transformer.Modes.Section2_MainIntPhiT

open Real MeasureTheory Filter
open scoped ENNReal

namespace Transformer.Modes

/-- The squared radius of `T` is eventually nonnegative in every regime.
This supplies the hypothesis required by `rate_T`.
Source: arXiv:2412.09080v3, §1.2, `eq:T`, and §3.2, `eq:rate`. -/
theorem eventually_window_nonneg {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∀ᶠ k in atTop, 0 ≤ 2 * Real.log (N k) - Real.log (B k) - ω (B k) := by
  have hc := hreg.c_pos
  obtain ⟨K₁, K₂, hlog⟩ := hreg.eventually_log
  filter_upwards [hlog, hreg.tendsto_B.eventually
    (hω.eventually_le_log (by positivity : (0 : ℝ) < c / 16))] with k hk hw
  obtain ⟨hN, hL, -, hK₂, hK₂', -, hupper⟩ := hk
  have hn : 0 ≤ Real.log (N k) := Real.log_nonneg hN
  have hl : 0 ≤ Real.log (B k) := by linarith
  have hhalf : (Real.log (B k) - Real.log K₂) / 2 ≤ Real.log (N k) := by
    nlinarith [mul_nonneg hc.le hn]
  nlinarith [mul_le_mul_of_nonneg_left hhalf hc.le, mul_nonneg hc.le hl]

example := eventually_window_nonneg isRegime_succ isSlowGrowth_sqrt_log_log

/-- On `T`, the third-moment prefactor of the Edgeworth term is
`O(exp(-ω/4))`.  Source: arXiv:2412.09080v3, §3.1, `lem:eta`,
`lem:error-3`, and §3.2, `eq:rate` with `s = 2`. -/
theorem eventually_eta_three_scaled_le_T {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∃ C : ℝ, 0 < C ∧ ∀ᶠ k in atTop,
      ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
        (N k : ℝ) ^ (-(1 : ℝ) / 2) * etaMoment (B k) t 3 ≤
          C * Real.exp (-ω (B k) / 4) := by
  obtain ⟨C, hη⟩ := etaMoment_le hreg hω 3 le_rfl
  refine ⟨|C| + 1, by positivity, ?_⟩
  filter_upwards [hη, eventually_window_nonneg hreg hω,
    hreg.tendsto_N.eventually_ge_atTop 1] with k hk hT hN t ht
  have hn : 0 < N k := by exact_mod_cast (by linarith : (0 : ℝ) < N k)
  have hβ := hreg.B_pos k
  have hmoment : etaMoment (B k) t 3 ≤
      (|C| + 1) * (B k * Real.exp (t ^ 2)) ^ ((1 : ℝ) / 4) := by
    have h := hk t ht
    norm_num only [Nat.cast_ofNat] at h
    refine h.trans ?_
    gcongr
    linarith [le_abs_self C]
  have hrate := rate_T hn hβ hT ht 2 (by norm_num)
  norm_num only [Nat.cast_ofNat, neg_mul, one_mul] at hrate
  calc (N k : ℝ) ^ (-(1 : ℝ) / 2) * etaMoment (B k) t 3
      ≤ (N k : ℝ) ^ (-(1 : ℝ) / 2) *
        ((|C| + 1) * (B k * Real.exp (t ^ 2)) ^ ((1 : ℝ) / 4)) := by gcongr
    _ = (|C| + 1) *
        ((N k : ℝ)⁻¹ * Real.sqrt (B k) * Real.exp (t ^ 2 / 2)) ^ ((1 : ℝ) / 2) := by
      rw [mul_left_comm]
      congr 1
      convert rate_eq hn hβ t (1 / 2) using 1
      norm_num
    _ ≤ _ := by gcongr

example := eventually_eta_three_scaled_le_T isRegime_succ isSlowGrowth_sqrt_log_log

/-- Integrating the Gaussian majorant over `T` gives the decay factor
times `√β` times the radius plus a constant, independently of `n/β^{5/2}`.
Source: arXiv:2412.09080v3, §3.1, proof of `lem:error-3`. -/
theorem eventually_psiKR_T_le {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∃ C D : ℝ, 0 < C ∧ 0 ≤ D ∧ ∀ᶠ k in atTop,
      psiKR (N k) (B k) (intervalT (N k) (B k) (ω (B k))) ≤
        ENNReal.ofReal (C * Real.exp (-ω (B k) / 4) * Real.sqrt (B k) *
          (Real.sqrt (2 * Real.log (N k) - Real.log (B k) - ω (B k)) + D)) := by
  obtain ⟨C, κ, K, hC, hκ, hK, hin⟩ := eventually_psi_inner_le_T hreg hω
  obtain ⟨E, hE, hη⟩ := eventually_eta_three_scaled_le_T hreg hω
  let A := 1 + K * (1 + κ⁻¹)
  let D := 10 * K * Real.sqrt (π / (κ / 2))
  have hA : 0 < A := by dsimp [A]; positivity
  have hD : 0 ≤ D := by dsimp [D]; positivity
  refine ⟨C * E * (2 * A + 1), D, by positivity, hD, ?_⟩
  filter_upwards [hin, hη, hreg.tendsto_N.eventually_ge_atTop 1] with k hk hηk hN
  let a := Real.sqrt (2 * Real.log (N k) - Real.log (B k) - ω (B k))
  let ρ := B k ^ (-(3 : ℝ) / 2) * (N k : ℝ)
  have hβ := hreg.B_pos k
  have hρ : 0 < ρ := by dsimp [ρ]; positivity
  have ha : 0 ≤ a := Real.sqrt_nonneg _
  have hwidth := lintegral_shift_width_le hκ hK hρ ha
  have heq (t : ℝ) : phiRate (N k) (B k) t =
      ρ * Real.exp (-(t ^ 2) / 2) * t ^ 2 := by dsimp [ρ, phiRate]; ring
  have hpoint (t : ℝ) (ht : t ∈ intervalT (N k) (B k) (ω (B k))) :
      ENNReal.ofReal (C * ((N k : ℝ) ^ (-(1 : ℝ) / 2) * etaMoment (B k) t 3) *
          Real.sqrt (B k)) ≤
        ENNReal.ofReal (C * E * Real.exp (-ω (B k) / 4) * Real.sqrt (B k)) := by
    apply ENNReal.ofReal_le_ofReal
    calc C * ((N k : ℝ) ^ (-(1 : ℝ) / 2) * etaMoment (B k) t 3) * Real.sqrt (B k)
        ≤ C * (E * Real.exp (-ω (B k) / 4)) * Real.sqrt (B k) := by
          exact mul_le_mul_of_nonneg_right
            (mul_le_mul_of_nonneg_left (hηk t ht) hC.le) (Real.sqrt_nonneg _)
      _ = _ := by ring
  calc psiKR (N k) (B k) (intervalT (N k) (B k) (ω (B k)))
      ≤ ∫⁻ t in intervalT (N k) (B k) (ω (B k)),
          ENNReal.ofReal (C * E * Real.exp (-ω (B k) / 4) * Real.sqrt (B k)) *
            ENNReal.ofReal (Real.exp (-κ * phiRate (N k) (B k) t) *
              (1 + K * Real.sqrt (rateScale (N k) (B k) t))) := by
        refine setLIntegral_mono' measurableSet_Icc fun t ht => (hk t ht).trans ?_
        exact mul_le_mul' (hpoint t ht) le_rfl
    _ = ENNReal.ofReal (C * E * Real.exp (-ω (B k) / 4) * Real.sqrt (B k)) *
        ∫⁻ t in Set.Icc (-a) a, ENNReal.ofReal
          (Real.exp (-κ * (ρ * Real.exp (-(t ^ 2) / 2) * t ^ 2)) *
            (1 + K * Real.sqrt (ρ * Real.exp (-(t ^ 2) / 2)))) := by
      simp_rw [heq, show rateScale (N k) (B k) =
        fun t => ρ * Real.exp (-(t ^ 2) / 2) from rfl]
      exact lintegral_const_mul' _ _ ENNReal.ofReal_ne_top
    _ ≤ ENNReal.ofReal (C * E * Real.exp (-ω (B k) / 4) * Real.sqrt (B k)) *
        ENNReal.ofReal (A * (2 * a) + D) := by gcongr
    _ ≤ _ := by
      rw [← ENNReal.ofReal_mul (by positivity)]
      apply ENNReal.ofReal_le_ofReal
      have hle : A * (2 * a) + D ≤ (2 * A + 1) * (a + D) := by
        nlinarith [mul_nonneg hA.le hD]
      convert mul_le_mul_of_nonneg_left hle
        (by positivity : 0 ≤ C * E * Real.exp (-ω (B k) / 4) * Real.sqrt (B k)) using 1
      ring

example := eventually_psiKR_T_le isRegime_succ isSlowGrowth_sqrt_log_log

/-- **Lemma (lem:error-3), on `T`.** In the regime `n^c ≲ β ≲ n^{2-c}`,
`∫_T ∫_{Δ_t}^∞ y (n det Σ_t)^{-1/2} |ψ|(…) dy dt ≲ exp(-ω(β)/4) √(β log β)`.
The source states a signed integral; this proves the absolute contribution
required by `eq:error-goal`.  The corrected global Hermite bound and exact
Gaussian shift preserve the original hypotheses and rate.
Source: arXiv:2412.09080v3, §3.1, `lem:error-3`. -/
theorem error_three_T {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B)
    {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∃ C : ℝ, ∀ᶠ k in atTop,
      psiKR (N k) (B k) (intervalT (N k) (B k) (ω (B k)))
        ≤ ENNReal.ofReal (C * (Real.exp (-(ω (B k)) / 4) * Real.sqrt (B k * Real.log (B k)))) := by
  obtain ⟨C, D, hC, hD, hup⟩ := eventually_psiKR_T_le hreg hω
  obtain ⟨M, hM, hrad⟩ := eventually_window_radius_le hreg hω
  refine ⟨C * (M + D), ?_⟩
  filter_upwards [hup, hrad] with k hk hr
  have hβ := hreg.B_pos k
  have hroot : 1 ≤ Real.sqrt (Real.log (B k)) := by
    simpa using Real.sqrt_le_sqrt hr.1
  refine hk.trans (ENNReal.ofReal_le_ofReal ?_)
  rw [Real.sqrt_mul hβ.le]
  have hwindow : Real.sqrt (2 * Real.log (N k) - Real.log (B k) - ω (B k)) + D ≤
      (M + D) * Real.sqrt (Real.log (B k)) := by
    nlinarith [mul_le_mul_of_nonneg_left hroot hD]
  convert mul_le_mul_of_nonneg_left hwindow
    (by positivity : 0 ≤ C * Real.exp (-ω (B k) / 4) * Real.sqrt (B k)) using 1
  ring

example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) :=
  ⟨isRegime_succ, isSlowGrowth_sqrt_log_log⟩

end Transformer.Modes

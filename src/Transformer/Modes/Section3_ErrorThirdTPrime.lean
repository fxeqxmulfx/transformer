/-
# The number of modes of a Gaussian KDE — `lem:error-3` on `T'`

For the standard slow window, `T' ⊆ T` eventually.  The inner Edgeworth
majorant on `T` therefore holds on `T'`, and the shifted Gaussian rate
integrates to a constant there.  In particular the absolute correction is
`O(√β)`, including regimes where `n/β^{5/2}` is unbounded.
The argument bounds the nonnegative integral itself, so it also proves
finiteness.  A third-moment bound of `O(β^{-1/2})` is proved first; its
uniform boundedness is enough for the rate stated by the source.
Source: arXiv:2412.09080v3, §3.1, `lem:error-3`, second display.
-/

import Transformer.Modes.Section3_ErrorThirdT
import Transformer.Modes.Section2_WidthIntegral

open Real MeasureTheory Filter
open scoped ENNReal

namespace Transformer.Modes

/-- On `T'` the third-moment prefactor of `ψ` is `O(β^{-1/2})`.
The moment estimate restricts from `T`; the arithmetic rate uses the
smaller radius of `T'` itself.  This also covers the case of empty `T'`.
Source: arXiv:2412.09080v3, §3.1, `lem:eta`, `lem:error-3`, and
§3.2, `eq:rate` with `s = 2`. -/
theorem eventually_eta_three_scaled_le_T' {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) :
    ∃ C : ℝ, 0 < C ∧ ∀ᶠ k in atTop, ∀ t ∈ intervalT' (N k) (B k),
      (N k : ℝ) ^ (-(1 : ℝ) / 2) * etaMoment (B k) t 3 ≤
        C * B k ^ (-(1 : ℝ) / 2) := by
  obtain ⟨C, hη⟩ := etaMoment_le hreg isSlowGrowth_sqrt_log_log 3 le_rfl
  refine ⟨|C| + 1, by positivity, ?_⟩
  filter_upwards [hη, eventually_intervalT'_subset hreg,
    hreg.tendsto_N.eventually_ge_atTop 1] with k hk hsub hN t ht
  have hn : 0 < N k := by
    exact_mod_cast (by linarith : (0 : ℝ) < N k)
  have hβ := hreg.B_pos k
  have hmoment : etaMoment (B k) t 3 ≤
      (|C| + 1) * (B k * Real.exp (t ^ 2)) ^ ((1 : ℝ) / 4) := by
    have h := hk t (hsub ht)
    norm_num only [Nat.cast_ofNat] at h
    refine h.trans ?_
    gcongr
    linarith [le_abs_self C]
  have hrate := rate_T' hn hβ ht 2 (by norm_num)
  norm_num only [Nat.cast_ofNat] at hrate
  rw [show -(1 / 2 : ℝ) = -(1 : ℝ) / 2 by ring] at hrate
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

example := eventually_eta_three_scaled_le_T' isRegime_succ

/-- The decaying prefactor on `T'` is bounded once `β ≥ 1`.
Source: arXiv:2412.09080v3, §3.1, proof of `lem:error-3`. -/
theorem eventually_eta_three_scaled_bounded_T' {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) :
    ∃ C : ℝ, 0 < C ∧ ∀ᶠ k in atTop, ∀ t ∈ intervalT' (N k) (B k),
      (N k : ℝ) ^ (-(1 : ℝ) / 2) * etaMoment (B k) t 3 ≤ C := by
  obtain ⟨C, hC, hη⟩ := eventually_eta_three_scaled_le_T' hreg
  refine ⟨C, hC, ?_⟩
  filter_upwards [hη, hreg.tendsto_B.eventually_ge_atTop 1] with k hk hβ t ht
  refine (hk t ht).trans ?_
  have hpow : B k ^ (-(1 : ℝ) / 2) ≤ 1 :=
    Real.rpow_le_one_of_one_le_of_nonpos hβ (by norm_num)
  nlinarith

example := eventually_eta_three_scaled_bounded_T' isRegime_succ

/-- The same Gaussian width majorant as for the proxy controls the inner
Edgeworth integral on `T'`.  Source: arXiv:2412.09080v3, §3.1, proof of
`lem:error-3`.  The cutoff, absolute value and Gaussian shift are retained. -/
theorem eventually_psi_inner_le_T' {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) :
    ∃ C κ K : ℝ, 0 < C ∧ 0 < κ ∧ 0 ≤ K ∧
      ∀ᶠ k in atTop, ∀ t ∈ intervalT' (N k) (B k),
        (∫⁻ y in Set.Ioi (deltaCut (N k) (B k) t), ENNReal.ofReal
          (y * ((N k : ℝ) * sigmaDet (B k) t) ^ (-(1 : ℝ) / 2) *
            |psiOf (lawY (B k) t)
              (whiten (sigmaFst (B k) t) (sigmaCov (B k) t) (sigmaSnd (B k) t)
                (-muFst (N k) (B k) t, y - muSnd (N k) (B k) t))|)) ≤
          ENNReal.ofReal (C * Real.sqrt (B k)) * ENNReal.ofReal
            (Real.exp (-κ * (rateScale (N k) (B k) t * t ^ 2)) *
              (1 + K * Real.sqrt (rateScale (N k) (B k) t) * (4 + 6 * t ^ 2))) := by
  obtain ⟨C, κ, K, hC, hκ, hK, hin⟩ :=
    eventually_psi_inner_le_T hreg isSlowGrowth_sqrt_log_log
  obtain ⟨E, hE, hη⟩ := eventually_eta_three_scaled_bounded_T' hreg
  refine ⟨C * E, κ, K, by positivity, hκ, hK, ?_⟩
  filter_upwards [hin, hη, eventually_intervalT'_subset hreg] with k hk hηk hsub t ht
  refine (hk t (hsub ht)).trans ?_
  apply mul_le_mul'
  · apply ENNReal.ofReal_le_ofReal
    have h := mul_le_mul_of_nonneg_left (hηk t ht) hC.le
    exact mul_le_mul_of_nonneg_right h (Real.sqrt_nonneg _)
  · apply ENNReal.ofReal_le_ofReal
    have heq : phiRate (N k) (B k) t = rateScale (N k) (B k) t * t ^ 2 := by
      unfold phiRate rateScale
      ring
    rw [heq]
    apply mul_le_mul_of_nonneg_left ?_ (Real.exp_pos _).le
    have hfactor : 0 ≤ K * Real.sqrt (rateScale (N k) (B k) t) := by positivity
    nlinarith [mul_nonneg hfactor (sq_nonneg t)]

example := eventually_psi_inner_le_T' isRegime_succ

/-- **Lemma (lem:error-3), on `T'`.** In the regime `n^c ≲ β ≲ n^{2-c}`,
`∫_{T'} ∫_{Δ_t}^∞ y (n det Σ_t)^{-1/2} |ψ|(…) dy dt ≲ √β`.
The source states a signed integral; this bounds the absolute contribution.
The corrected Hermite estimate and the exact Gaussian shift leave the
original hypotheses unchanged, with no bound on `n/β^{5/2}`.
Source: arXiv:2412.09080v3, §3.1, `lem:error-3`. -/
theorem error_three_T' {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B) :
    ∃ C : ℝ, ∀ᶠ k in atTop,
      psiKR (N k) (B k) (intervalT' (N k) (B k)) ≤
        ENNReal.ofReal (C * Real.sqrt (B k)) := by
  obtain ⟨C, κ, K, hC, hκ, hK, hin⟩ := eventually_psi_inner_le_T' hreg
  obtain ⟨W, hW, hwidth⟩ := lintegral_width_le hκ hK
  refine ⟨C * W, ?_⟩
  filter_upwards [hin, hreg.tendsto_N.eventually_ge_atTop 1,
    hreg.tendsto_B.eventually_ge_atTop 1] with k hk hN hβ
  have hn : 1 ≤ N k := by exact_mod_cast hN
  calc psiKR (N k) (B k) (intervalT' (N k) (B k))
      ≤ ∫⁻ t in intervalT' (N k) (B k), ENNReal.ofReal (C * Real.sqrt (B k)) *
          ENNReal.ofReal (Real.exp (-κ * (rateScale (N k) (B k) t * t ^ 2)) *
            (1 + K * Real.sqrt (rateScale (N k) (B k) t) * (4 + 6 * t ^ 2))) :=
        setLIntegral_mono' (measurableSet_intervalT' _ _) hk
    _ = ENNReal.ofReal (C * Real.sqrt (B k)) * ∫⁻ t in intervalT' (N k) (B k),
        ENNReal.ofReal (Real.exp (-κ * (rateScale (N k) (B k) t * t ^ 2)) *
          (1 + K * Real.sqrt (rateScale (N k) (B k) t) * (4 + 6 * t ^ 2))) :=
      lintegral_const_mul' _ _ ENNReal.ofReal_ne_top
    _ ≤ ENNReal.ofReal (C * Real.sqrt (B k)) * ENNReal.ofReal W := by
      gcongr
      exact hwidth hn hβ
    _ = _ := by
      rw [← ENNReal.ofReal_mul (by positivity)]
      congr 1
      ring

example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) :=
  isRegime_succ

end Transformer.Modes

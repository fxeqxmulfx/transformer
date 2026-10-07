/-
# The number of modes of a Gaussian KDE — the full-window shift

For arXiv:2412.09080v3, §2.3, `lem:main-int-phi`, the shift satisfies
`α_t δ_t² ≤ C R` with `R = n β^{-3/2} e^{-t²/2}` on the full window.
The majorant `e^{-κ R t²}(1 + K √R)` integrates to a constant times the
window length plus a constant independent of `n β^{-3/2}`.
-/

import Transformer.Modes.Section2_IntPhiB
import Transformer.Modes.Section2_WidthTerms

open Real Filter Asymptotics MeasureTheory
open scoped Topology ENNReal

namespace Transformer.Modes

/-- The polynomial in the shift bound is negligible against the bandwidth,
uniformly on `T`.  Source: arXiv:2412.09080v3, §2.3, `eq:T`, `lem:phi-t`.
-/
theorem eventually_window_polynomial_le {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
      (4 + 6 * t ^ 2) ^ 2 ≤ B k := by
  have hc := hreg.c_pos
  obtain ⟨K₁, K₂, hlog⟩ := hreg.eventually_log
  have hsmall := ((isLittleO_log_rpow_rpow_atTop 2 (by norm_num : (0 : ℝ) < 1)).comp_tendsto
    hreg.tendsto_B).bound (by positivity : (0 : ℝ) < 1 / (144 * (4 / c) ^ 2))
  filter_upwards [hlog, hreg.tendsto_B.eventually (hω.eventually_le_log one_pos),
    hsmall, hreg.tendsto_B.eventually_ge_atTop 64] with k hk hw hs hB t ht
  obtain ⟨-, hL1, hK₁, -, -, hN, -⟩ := hk
  have hL0 : 0 ≤ Real.log (B k) := by linarith
  have hD : 2 * Real.log (N k) - Real.log (B k) - ω (B k) ≤
      (4 / c) * Real.log (B k) := by
    rw [div_mul_eq_mul_div, le_div_iff₀ hc]
    nlinarith [mul_nonneg hc.le hw.1, mul_nonneg hc.le hL0]
  have htbound : t ^ 2 ≤ ((4 / c) * Real.log (B k)) := by
    have hsqrt := Real.sqrt_le_sqrt hD
    have habs : |t| ≤ Real.sqrt ((4 / c) * Real.log (B k)) :=
      (abs_le.mpr ht).trans hsqrt
    have hsq := pow_le_pow_left₀ (abs_nonneg t) habs 2
    rwa [sq_abs, Real.sq_sqrt (by positivity)] at hsq
  have hpow : t ^ 4 ≤ (4 / c) ^ 2 * Real.log (B k) ^ 2 := by
    have := pow_le_pow_left₀ (sq_nonneg t) htbound 2
    nlinarith
  change ‖Real.log (B k) ^ (2 : ℝ)‖ ≤
    1 / (144 * (4 / c) ^ 2) * ‖B k ^ (1 : ℝ)‖ at hs
  rw [Real.rpow_two, Real.rpow_one, Real.norm_of_nonneg (sq_nonneg _),
    Real.norm_of_nonneg (hreg.B_pos k).le] at hs
  have hscale : 72 * ((4 / c) ^ 2 * Real.log (B k) ^ 2) ≤ B k / 2 := by
    have := mul_le_mul_of_nonneg_left hs (by positivity : 0 ≤ 72 * (4 / c) ^ 2)
    field_simp at this ⊢
    nlinarith
  nlinarith [sq_nonneg (6 * t ^ 2 - 4)]

/-- The regime and slow window hypotheses occur at `β = n`.
Source: arXiv:2412.09080v3, §2.3, `lem:main-int-phi`. -/
example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) :=
  ⟨isRegime_succ, isSlowGrowth_sqrt_log_log⟩

/-- The squared shift in Gaussian units is at most `C R` on the full window;
no upper bound on `n/β^{5/2}` is used.
Source: arXiv:2412.09080v3, §2.3, `lem:phi-t`, `lem:main-int-phi`. -/
theorem eventually_shift_sq_le_rateScale {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∃ C : ℝ, 0 < C ∧ ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
      phiAlpha (B k) t * phiDelta (N k) (B k) t ^ 2 ≤
        C * rateScale (N k) (B k) t := by
  obtain ⟨C, hC, hδ⟩ := phiAlpha_mul_phiDelta_sq_le_rate hreg hω
  refine ⟨C, hC, ?_⟩
  filter_upwards [hδ, eventually_window_polynomial_le hreg hω] with k hk hp t ht
  have hβ := hreg.B_pos k
  have hR0 : 0 ≤ rateScale (N k) (B k) t := by unfold rateScale; positivity
  have heq : (N k : ℝ) * B k ^ (-(5 : ℝ) / 2) * Real.exp (-(t ^ 2) / 2) =
      rateScale (N k) (B k) t / B k := by
    have hr : B k ^ (-(5 : ℝ) / 2) = B k ^ (-(3 : ℝ) / 2) * (B k)⁻¹ := by
      rw [← Real.rpow_neg_one, ← Real.rpow_add hβ]
      congr 1
      ring
    rw [hr, rateScale]
    ring
  calc phiAlpha (B k) t * phiDelta (N k) (B k) t ^ 2
      ≤ C * ((N k : ℝ) * B k ^ (-(5 : ℝ) / 2) * Real.exp (-(t ^ 2) / 2)) *
          (4 + 6 * t ^ 2) ^ 2 := hk t ht
    _ = C * (rateScale (N k) (B k) t / B k) * (4 + 6 * t ^ 2) ^ 2 := by rw [heq]
    _ ≤ C * (rateScale (N k) (B k) t / B k) * B k := by gcongr; exact hp t ht
    _ = C * rateScale (N k) (B k) t := by field_simp

/-- The sharpened shift estimate has the same witnessed hypotheses.
Source: arXiv:2412.09080v3, §2.3, `lem:main-int-phi`. -/
example := eventually_shift_sq_le_rateScale isRegime_succ isSlowGrowth_sqrt_log_log

/-- A majorant for the shifted Gaussian proxy, with no requirement `R ≥ 1`.
Source: arXiv:2412.09080v3, §2.3, proof of `lem:main-int-phi`. -/
theorem shift_width_integrand_le {κ K ρ : ℝ} (hκ : 0 < κ) (hK : 0 ≤ K)
    (hρ : 0 < ρ) (t : ℝ) :
    Real.exp (-κ * (ρ * Real.exp (-(t ^ 2) / 2) * t ^ 2)) *
        (1 + K * Real.sqrt (ρ * Real.exp (-(t ^ 2) / 2))) ≤
      (1 + K * (1 + κ⁻¹)) +
        10 * K * Real.sqrt ρ * Real.exp (-(κ / 2 * ρ) * t ^ 2) := by
  set R := ρ * Real.exp (-(t ^ 2) / 2)
  have hR : 0 ≤ R := by positivity
  have hE : 0 ≤ Real.exp (-κ * (R * t ^ 2)) := (Real.exp_pos _).le
  have hE1 : Real.exp (-κ * (R * t ^ 2)) ≤ 1 :=
    Real.exp_le_one_iff.mpr (by nlinarith [mul_nonneg hκ.le (mul_nonneg hR (sq_nonneg t))])
  have hG : 0 ≤ 10 * K * Real.sqrt ρ * Real.exp (-(κ / 2 * ρ) * t ^ 2) := by positivity
  by_cases ht : t ^ 2 ≤ 1
  · have he : Real.exp (-(t ^ 2) / 2) ≤ 1 := Real.exp_le_one_iff.mpr (by nlinarith)
    have he' : 1 / 2 ≤ Real.exp (-(t ^ 2) / 2) := by
      linarith [Real.add_one_le_exp (-(t ^ 2) / 2)]
    have hRρ : R ≤ ρ := by dsimp [R]; nlinarith
    have hρR : ρ / 2 ≤ R := by dsimp [R]; nlinarith
    have hsmall := width_term_small hκ hK (sq_nonneg t) hRρ hρR ht
    have hterm : Real.exp (-κ * (R * t ^ 2)) * (K * Real.sqrt R) ≤
        Real.exp (-κ * (R * t ^ 2)) * (K * Real.sqrt R * (4 + 6 * t ^ 2)) := by
      have : 1 ≤ 4 + 6 * t ^ 2 := by nlinarith
      nlinarith [mul_nonneg hE (mul_nonneg hK (Real.sqrt_nonneg R))]
    have hKinv : 0 ≤ K * (1 + κ⁻¹) := by positivity
    dsimp only [R] at hsmall hterm hE1
    nlinarith
  · have ht1 : 1 ≤ t ^ 2 := (lt_of_not_ge ht).le
    have hroot : Real.sqrt R ≤ 1 + R * t ^ 2 := by
      have hs := Real.sq_sqrt hR
      nlinarith [sq_nonneg (Real.sqrt R - 1), mul_nonneg hR (sub_nonneg.mpr ht1)]
    have hexp := mul_exp_neg_le (x := R * t ^ 2) hκ
    have hterm : Real.exp (-κ * (R * t ^ 2)) * (K * Real.sqrt R) ≤ K * (1 + κ⁻¹) := by
      calc Real.exp (-κ * (R * t ^ 2)) * (K * Real.sqrt R)
          ≤ Real.exp (-κ * (R * t ^ 2)) * (K * (1 + R * t ^ 2)) := by gcongr
        _ = K * (Real.exp (-κ * (R * t ^ 2)) +
            R * t ^ 2 * Real.exp (-κ * (R * t ^ 2))) := by ring
        _ ≤ K * (1 + κ⁻¹) := by gcongr
    dsimp only [R] at hterm hE1
    nlinarith

/-- The parameters in the majorant are simultaneously satisfiable.
Source: arXiv:2412.09080v3, §2.3, `lem:main-int-phi`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 :=
  ⟨one_pos, le_rfl, one_pos⟩

/-- The majorant integrates to a constant times interval length plus a
constant independent of `ρ`.  Source: arXiv:2412.09080v3, §2.3,
proof of `lem:main-int-phi`. -/
theorem lintegral_shift_width_le {κ K ρ a : ℝ} (hκ : 0 < κ) (hK : 0 ≤ K)
    (hρ : 0 < ρ) (ha : 0 ≤ a) :
    (∫⁻ t in Set.Icc (-a) a, ENNReal.ofReal
      (Real.exp (-κ * (ρ * Real.exp (-(t ^ 2) / 2) * t ^ 2)) *
        (1 + K * Real.sqrt (ρ * Real.exp (-(t ^ 2) / 2))))) ≤
      ENNReal.ofReal ((1 + K * (1 + κ⁻¹)) * (2 * a) +
        10 * K * Real.sqrt (π / (κ / 2))) := by
  let M : ℝ → ℝ := fun t => (1 + K * (1 + κ⁻¹)) +
    10 * K * Real.sqrt ρ * Real.exp (-(κ / 2 * ρ) * t ^ 2)
  have hg : Integrable (fun t : ℝ => Real.exp (-(κ / 2 * ρ) * t ^ 2)) :=
    integrable_exp_neg_mul_sq (by positivity)
  have hM : IntegrableOn M (Set.Icc (-a) a) :=
    continuous_const.integrableOn_Icc.add (hg.const_mul _).integrableOn
  have hM0 : ∀ t, 0 ≤ M t := fun t => by dsimp [M]; positivity
  have hsqrt : Real.sqrt ρ * Real.sqrt (π / (κ / 2 * ρ)) =
      Real.sqrt (π / (κ / 2)) := by
    rw [← Real.sqrt_mul hρ.le]
    congr 1
    field_simp
  calc
    _ ≤ ∫⁻ t in Set.Icc (-a) a, ENNReal.ofReal (M t) :=
      lintegral_mono fun t => ENNReal.ofReal_le_ofReal (shift_width_integrand_le hκ hK hρ t)
    _ = ENNReal.ofReal (∫ t in Set.Icc (-a) a, M t) :=
      (ofReal_integral_eq_lintegral_ofReal hM (Eventually.of_forall hM0)).symm
    _ ≤ _ := by
      apply ENNReal.ofReal_le_ofReal
      change (∫ t in Set.Icc (-a) a,
        (1 + K * (1 + κ⁻¹)) + 10 * K * Real.sqrt ρ * Real.exp (-(κ / 2 * ρ) * t ^ 2)) ≤ _
      rw [integral_add continuous_const.integrableOn_Icc (hg.const_mul _).integrableOn,
        integral_const, integral_const_mul, smul_eq_mul,
        measureReal_restrict_apply_univ, Real.volume_real_Icc, max_eq_left (by linarith)]
      have hle := setIntegral_le_integral hg
        (Eventually.of_forall fun t => (Real.exp_pos (-(κ / 2 * ρ) * t ^ 2)).le)
        (s := Set.Icc (-a) a)
      rw [integral_gaussian] at hle
      calc (a - -a) * (1 + K * (1 + κ⁻¹)) +
          10 * K * Real.sqrt ρ * ∫ t in Set.Icc (-a) a, Real.exp (-(κ / 2 * ρ) * t ^ 2)
          ≤ (a - -a) * (1 + K * (1 + κ⁻¹)) +
            10 * K * Real.sqrt ρ * Real.sqrt (π / (κ / 2 * ρ)) := by gcongr
        _ = _ := by rw [mul_assoc (10 * K), hsqrt]; ring

/-- A nonempty interval witnesses the hypotheses of the integral estimate.
Source: arXiv:2412.09080v3, §2.3, `lem:main-int-phi`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 :=
  ⟨one_pos, le_rfl, one_pos, zero_le_one⟩

end Transformer.Modes

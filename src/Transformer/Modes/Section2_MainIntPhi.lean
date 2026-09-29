/-
# The number of modes of a Gaussian KDE — `lem:main-int-phi` and `eq:int-phi-b`

§2.3 of arXiv:2412.09080v3: `lem:main-int-phi` and the two bounds `eq:int-phi-b`
its proof reduces to.  The proxy Kac-Rice integral is `proxyKR`
(`Section2_IntPhiFinal.lean`).

**What the source says and what is carried here.**

* `lem:main-int-phi` is stated as the source has it.  Its proof goes through
  `eq:int-phi-final`, so for `c < 2/5` the source's proof does not cover it.
  The bound on `T'` is proved here regardless, directly and with no bound on
  `n`.  On `T'` the shift `δ_t` of the Gaussian in `y` is large where
  `R = β^{-3/2} n e^{-t²/2} ≥ 1` is large: `√(α_t δ_t²) ≲ √R (4 + 6t²)`, and the
  factor `√R` is paid for by `e^{-A_t/2} ≤ e^{-κ R t²}` together with the width
  `R^{-1/2}` of `∫ e^{-κ R t²} dt` near `t = 0`
  (`Section2_ProxyKRTPrime.lean`, `Section2_WidthIntegral.lean`).  The bound on
  `T`, `main_int_phi_T`, is not proved here.

* `eq:int-phi-b` is carried in the form its proof uses: for every `C > 0`,
  `∫_T e^{-C β^{-3/2} n t² e^{-t²/2}} dt ≍ √(log β)` and
  `∫_{T'} e^{-C β^{-3/2} n t² e^{-t²/2}} dt ≲ 1`.  This form needs neither the
  moments nor `A_t`: `phiA_isTheta` is what ties it to `A_t`.  The bound on
  `T'` is proved here: on `T'` the exponent is at least `C t²`.  The bound on
  `T` is proved in `Section2_IntPhiB.lean`.

Source: arXiv:2412.09080v3, §2.3 (`sec: 2.3`), `lem:main-int-phi`,
`eq:int-phi-b`.
-/

import Transformer.Modes.Section2_IntPhiFinal
import Transformer.Modes.Section2_ProxyKRTPrime
import Transformer.Modes.Section2_WidthIntegral

open Real Filter Asymptotics MeasureTheory
open scoped Topology ENNReal

namespace Transformer
namespace Modes

/-- The rate `β^{-3/2} n t² e^{-t²/2}` that `A_t` is `≍` to. -/
noncomputable def phiRate (n : ℕ) (β t : ℝ) : ℝ :=
  β ^ (-(3 : ℝ) / 2) * n * t ^ 2 * Real.exp (-(t ^ 2) / 2)

/-! ### `eq:int-phi-b` -/

/-- On `T'`, `β^{-3/2} n t² e^{-t²/2} ≥ t²`: there `e^{-t²/2} ≥ β^{3/2}/n`
(`one_le_rateScale`). -/
theorem sq_le_phiRate {n : ℕ} {β t : ℝ} (hn : 1 ≤ n) (hβ : 1 ≤ β)
    (ht : t ∈ intervalT' n β) : t ^ 2 ≤ phiRate n β t := by
  have h := one_le_rateScale hn hβ ht
  calc t ^ 2 = 1 * t ^ 2 := (one_mul _).symm
    _ ≤ rateScale n β t * t ^ 2 := by gcongr
    _ = phiRate n β t := by unfold rateScale phiRate; ring

/-- The hypotheses of `sq_le_phiRate` are satisfiable: `n = β = 1`, `t = 0`.
Those of the regime theorems below are witnessed in `Section2_PhiTDelta.lean`. -/
example : 1 ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℝ) ∈ intervalT' 1 1 := by
  refine ⟨le_rfl, le_rfl, ?_⟩
  unfold intervalT'
  simp only [show (1 : ℝ) ≤ ((1 : ℕ) : ℝ) ^ ((2 : ℝ) / 3) by simp, ite_true]
  exact ⟨by simp, Real.sqrt_nonneg _⟩

/-- **Equation (eq:int-phi-b), on `T'`.**  For every `C > 0`,
`∫_{T'} e^{-C β^{-3/2} n t² e^{-t²/2}} dt ≲ 1`: the integrand is at most
`e^{-Ct²}`, whose integral over `ℝ` is `√(π/C)`.

Source: arXiv:2412.09080v3, `eq:int-phi-b`, proof of `lem:main-int-phi`. -/
theorem integral_exp_phiRate_T' {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B)
    {C : ℝ} (hC : 0 < C) :
    (fun k => ∫ t in intervalT' (N k) (B k), Real.exp (-C * phiRate (N k) (B k) t))
      =O[atTop] fun _ => (1 : ℝ) := by
  refine IsBigO.of_bound (Real.sqrt (π / C)) ?_
  have hN1 : ∀ᶠ k in atTop, (1 : ℝ) ≤ N k := hreg.tendsto_N.eventually_ge_atTop 1
  filter_upwards [hN1, hreg.tendsto_B.eventually_ge_atTop 1] with k hN hB
  have hN' : 1 ≤ N k := by exact_mod_cast hN
  have hmeas : MeasurableSet (intervalT' (N k) (B k)) := measurableSet_intervalT' _ _
  have hcont : Continuous fun t => Real.exp (-C * phiRate (N k) (B k) t) := by
    unfold phiRate; fun_prop
  have hint : IntegrableOn (fun t => Real.exp (-C * phiRate (N k) (B k) t))
      (intervalT' (N k) (B k)) := by
    unfold intervalT'; split_ifs
    · exact hcont.integrableOn_Icc
    · exact integrableOn_empty
  have hg := integrable_exp_neg_mul_sq hC
  rw [norm_one, mul_one, Real.norm_of_nonneg (setIntegral_nonneg hmeas fun t _ => (Real.exp_pos _).le)]
  calc ∫ t in intervalT' (N k) (B k), Real.exp (-C * phiRate (N k) (B k) t)
      ≤ ∫ t in intervalT' (N k) (B k), Real.exp (-C * t ^ 2) :=
        setIntegral_mono_on hint hg.integrableOn hmeas fun t ht =>
          Real.exp_le_exp.mpr (by nlinarith [sq_le_phiRate hN' hB ht])
    _ ≤ ∫ t, Real.exp (-C * t ^ 2) :=
        setIntegral_le_integral hg (Eventually.of_forall fun t => (Real.exp_pos _).le)
    _ = Real.sqrt (π / C) := integral_gaussian C

/-! ### `lem:main-int-phi` -/

/-- **Lemma (lem:main-int-phi), on `T`.**  In the regime `n^c ≲ β ≲ n^{2-c}`,
`∫_T ∫_0^∞ y (det Σ_t)^{-1/2} φ(…) dy dt ≍ √(β log β)`.

Not proved here.

Source: arXiv:2412.09080v3, `lem:main-int-phi`. -/
theorem main_int_phi_T {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B)
    {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    (fun k => (proxyKR (N k) (B k) (intervalT (N k) (B k) (ω (B k)))).toReal)
      =Θ[atTop] fun k => Real.sqrt (B k * Real.log (B k)) := by
  sorry

/-- **Lemma (lem:main-int-phi), on `T'`.**  In the regime `n^c ≲ β ≲ n^{2-c}`,
`∫_{T'} ∫_0^∞ y (det Σ_t)^{-1/2} φ(…) dy dt ≲ √β`.  Finiteness is stated,
since `toReal` reads `∞` as `0`.

The statement is the source's.  Its proof is not: the source goes through
`eq:int-phi-final`, which needs `n ≲ β^{5/2}` (`int_phi_final`), and for
`c < 2/5` the regime allows `n ≫ β^{5/2}`.  Here the shift `δ_t` of the Gaussian
in `y` is handled directly on `T'`, with no bound on `n`
(`eventually_inner_le_T'`, `lintegral_width_le`).

Source: arXiv:2412.09080v3, `lem:main-int-phi`. -/
theorem main_int_phi_T' {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B) :
    (∀ᶠ k in atTop, proxyKR (N k) (B k) (intervalT' (N k) (B k)) ≠ ∞) ∧
    (fun k => (proxyKR (N k) (B k) (intervalT' (N k) (B k))).toReal)
      =O[atTop] fun k => Real.sqrt (B k) := by
  obtain ⟨C, κ, K₁, hC, hκ, hK₁, hin⟩ := eventually_inner_le_T' hreg
  obtain ⟨K, hK0, hK⟩ := lintegral_width_le hκ hK₁
  have hle : ∀ᶠ k in atTop, proxyKR (N k) (B k) (intervalT' (N k) (B k)) ≤
      ENNReal.ofReal (C * Real.sqrt (B k)) * ENNReal.ofReal K := by
    filter_upwards [hin, hreg.tendsto_N.eventually_ge_atTop 1,
      hreg.tendsto_B.eventually_ge_atTop 1] with k hk hN hB
    have hN' : 1 ≤ N k := by exact_mod_cast hN
    unfold proxyKR
    calc ∫⁻ t in intervalT' (N k) (B k), ∫⁻ y in Set.Ioi (0 : ℝ), ENNReal.ofReal
          (y * sigmaDet (B k) t ^ (-(1 : ℝ) / 2) * krPhi (N k) (B k) t 0 y)
        ≤ ∫⁻ t in intervalT' (N k) (B k), ENNReal.ofReal (C * Real.sqrt (B k)) *
            ENNReal.ofReal (Real.exp (-κ * (rateScale (N k) (B k) t * t ^ 2)) *
              (1 + K₁ * Real.sqrt (rateScale (N k) (B k) t) * (4 + 6 * t ^ 2))) :=
          setLIntegral_mono' (measurableSet_intervalT' _ _) hk
      _ = ENNReal.ofReal (C * Real.sqrt (B k)) * ∫⁻ t in intervalT' (N k) (B k),
            ENNReal.ofReal (Real.exp (-κ * (rateScale (N k) (B k) t * t ^ 2)) *
              (1 + K₁ * Real.sqrt (rateScale (N k) (B k) t) * (4 + 6 * t ^ 2))) :=
          lintegral_const_mul' _ _ ENNReal.ofReal_ne_top
      _ ≤ ENNReal.ofReal (C * Real.sqrt (B k)) * ENNReal.ofReal K := by
          gcongr
          exact hK hN' hB
  refine ⟨?_, ?_⟩
  · filter_upwards [hle] with k hk
    exact ne_top_of_le_ne_top
      (ENNReal.mul_ne_top ENNReal.ofReal_ne_top ENNReal.ofReal_ne_top) hk
  · refine IsBigO.of_bound (C * K) ?_
    filter_upwards [hle] with k hk
    have h1 := ENNReal.toReal_mono
      (ENNReal.mul_ne_top ENNReal.ofReal_ne_top ENNReal.ofReal_ne_top) hk
    rw [ENNReal.toReal_mul, ENNReal.toReal_ofReal (by positivity),
      ENNReal.toReal_ofReal hK0] at h1
    rw [Real.norm_of_nonneg ENNReal.toReal_nonneg, Real.norm_of_nonneg (Real.sqrt_nonneg _)]
    calc _ ≤ C * Real.sqrt (B k) * K := h1
      _ = C * K * Real.sqrt (B k) := by ring

/-- The regime hypothesis of `main_int_phi_T'` is satisfiable, with `T'` nonempty at
every `k`: `c = 1/2`, `n = k + 1`, `β = √n ≤ n^{2/3}`.  (For `β = n`, the witness
`isRegime_succ`, `T'` is empty once `n ≥ 2` and the bound is trivial.) -/
example : IsRegime (1 / 2) (fun k => k + 1) (fun k => Real.sqrt ((k + 1 : ℕ) : ℝ)) ∧
    ∀ k : ℕ, (0 : ℝ) ∈ intervalT' (k + 1) (Real.sqrt ((k + 1 : ℕ) : ℝ)) := by
  have hn1 : ∀ k : ℕ, (1 : ℝ) ≤ ((k + 1 : ℕ) : ℝ) := fun k => by
    push_cast; linarith [k.cast_nonneg (α := ℝ)]
  have hn : Tendsto (fun k : ℕ => ((k + 1 : ℕ) : ℝ)) atTop atTop := by
    push_cast; exact tendsto_natSucc_atTop
  refine ⟨⟨by norm_num, fun k => Real.sqrt_pos.2 (by linarith [hn1 k]), hn,
    Real.tendsto_sqrt_atTop.comp hn, ?_, ?_⟩, fun k => ?_⟩
  · refine IsBigO.of_bound 1 ?_
    filter_upwards with k
    have h0 : 0 ≤ ((k + 1 : ℕ) : ℝ) := by linarith [hn1 k]
    rw [one_mul, Real.norm_of_nonneg (Real.rpow_nonneg h0 _),
      Real.norm_of_nonneg (Real.sqrt_nonneg _), Real.sqrt_eq_rpow]
  · refine IsBigO.of_bound 1 ?_
    filter_upwards with k
    rw [one_mul, Real.norm_of_nonneg (Real.sqrt_nonneg _),
      Real.norm_of_nonneg (Real.rpow_nonneg (by linarith [hn1 k]) _), Real.sqrt_eq_rpow]
    exact Real.rpow_le_rpow_of_exponent_le (hn1 k) (by norm_num)
  · unfold intervalT'
    have h : Real.sqrt ((k + 1 : ℕ) : ℝ) ≤ ((k + 1 : ℕ) : ℝ) ^ ((2 : ℝ) / 3) := by
      rw [Real.sqrt_eq_rpow]
      exact Real.rpow_le_rpow_of_exponent_le (hn1 k) (by norm_num)
    simp only [h, ite_true]
    exact ⟨by simp, Real.sqrt_nonneg _⟩

end Modes
end Transformer

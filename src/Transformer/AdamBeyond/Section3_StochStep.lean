import Transformer.AdamBeyond.Section3_StochMean

/-
# Adam and beyond — §3: the lemma of the proof of Theorem 3

For `C` large enough, Adam's step on the stochastic problem drifts upward in
expectation: `E[x_{t+1} - x_t] ≥ 0`.

**What the source says and what is carried here.**  The statement is the
source's, for step sizes `α_t ≥ 0`.  The proof splits `m_t/√v_t` on the last
coin as the source does, `T₁` when it is `C` and `T₂ + T₃` when it is `-1`,
and bounds `E[T₁]` as the source does; it bounds `E[T₂]` by
`β₁ K p/(1-γ)` with no event (`stoch_T2_le`), and `E[T₃]` by the tangent of
`1/√x` in place of Jensen's inequality (`stoch_int_inv`).  The source's
`E[T₃]` display reads `√(β₂(1+δ)C² + 1 - β₂)`, while its own estimate of
`E[v_{t-1}]` is `≤ (1+δ)C`; the constant here is `a = 1 + (1+δ)C`, and `C₀` is
explicit: `C₀ = max(2(1+δ), 4(1+δ)³B²/(1-β₁)²)` with
`B = 1/√(1-β₂) + β₁K/(1-γ)`.

Source: arXiv:1904.09237, Appendix, proof of Theorem 3, the lemma.
-/

open MeasureTheory ProbabilityTheory Finset Function

namespace Transformer
namespace AdamBeyond

open AMSGrad

variable {C β₁ β₂ : ℝ} {α : ℕ → ℝ}

/-- `m_{n+1}/√v_{n+1}` split on the coin `b_{n+1}`: `T₁` if it is `C`, and
`(β₁ m_n - (1-β₁))/√(β₂ v_n + 1 - β₂)` if it is `-1`. -/
theorem coin_step_split (n : ℕ) (u : ℕ → Bool) :
    coinM C β₁ β₂ α (n + 1) u / Real.sqrt (coinV C β₁ β₂ α (n + 1) u) =
      coinInd (u (n + 1)) * ((β₁ * coinM C β₁ β₂ α n u + (1 - β₁) * C) /
          Real.sqrt (β₂ * coinV C β₁ β₂ α n u + (1 - β₂) * C ^ 2)) +
        (1 - coinInd (u (n + 1))) * ((β₁ * coinM C β₁ β₂ α n u - (1 - β₁)) /
          Real.sqrt (β₂ * coinV C β₁ β₂ α n u + (1 - β₂))) := by
  have hm : coinM C β₁ β₂ α (n + 1) u =
      β₁ * coinM C β₁ β₂ α n u + (1 - β₁) * coinGrad C (u (n + 1)) := stoch_m_succ ..
  have hv : coinV C β₁ β₂ α (n + 1) u =
      β₂ * coinV C β₁ β₂ α n u + (1 - β₂) * coinGrad C (u (n + 1)) ^ 2 := stoch_v_succ ..
  rw [hm, hv]
  cases u (n + 1)
  · norm_num [coinGrad, coinInd]
    ring_nf
  · norm_num [coinGrad, coinInd]

/-- The constants of the lemma: with `p = (1+δ)/(C+1)`, `a = 1 + (1+δ)C` and
`C ≥ max(2(1+δ), 4(1+δ)³B²/(1-β₁)²)`, `p B ≤ (1-p)(1-β₁)/√a`. -/
theorem stoch_const_le {B δ : ℝ} (hB : 0 ≤ B) (hβ₁ : β₁ < 1) (hδ : 0 < δ)
    (hC1 : 2 * (1 + δ) ≤ C) (hC2 : 4 * (1 + δ) ^ 3 * B ^ 2 / (1 - β₁) ^ 2 ≤ C) :
    (1 + δ) / (C + 1) * B ≤
      (1 - (1 + δ) / (C + 1)) * (1 - β₁) / Real.sqrt (1 + (1 + δ) * C) := by
  have hu : 0 < C + 1 := by linarith
  set p := (1 + δ) / (C + 1) with hpdef
  have hpu : p * (C + 1) = 1 + δ := div_mul_cancel₀ _ hu.ne'
  have hp0 : 0 ≤ p := by positivity
  have hp2 : p ≤ 1 / 2 := by rw [hpdef, div_le_iff₀ hu]; linarith
  have ha : 0 < 1 + (1 + δ) * C := by nlinarith
  have hs : 0 < Real.sqrt (1 + (1 + δ) * C) := Real.sqrt_pos.2 ha
  have h4 : 4 * (1 + δ) ^ 3 * B ^ 2 ≤ (C + 1) * (1 - β₁) ^ 2 := by
    rw [div_le_iff₀ (by nlinarith)] at hC2
    nlinarith [sq_nonneg (1 - β₁)]
  have k1 : p ^ 2 * B ^ 2 * (1 + (1 + δ) * C) ≤ p * B ^ 2 * (1 + δ) ^ 2 := by
    calc p ^ 2 * B ^ 2 * (1 + (1 + δ) * C) ≤ p ^ 2 * B ^ 2 * ((1 + δ) * (C + 1)) :=
          mul_le_mul_of_nonneg_left (by nlinarith) (by positivity)
      _ = p * B ^ 2 * (1 + δ) * (p * (C + 1)) := by ring
      _ = p * B ^ 2 * (1 + δ) ^ 2 := by rw [hpu]; ring
  have k2 : p * B ^ 2 * (1 + δ) ^ 2 ≤ (1 - β₁) ^ 2 / 4 := by
    refine le_of_mul_le_mul_right ?_ hu
    calc p * B ^ 2 * (1 + δ) ^ 2 * (C + 1) = (p * (C + 1)) * B ^ 2 * (1 + δ) ^ 2 := by ring
      _ = (1 + δ) ^ 3 * B ^ 2 := by rw [hpu]; ring
      _ ≤ (1 - β₁) ^ 2 / 4 * (C + 1) := by linarith
  have k3 : (1 - β₁) ^ 2 / 4 ≤ ((1 - p) * (1 - β₁)) ^ 2 := by
    have : 1 / 4 ≤ (1 - p) ^ 2 := by nlinarith
    rw [mul_pow]
    nlinarith [sq_nonneg (1 - β₁)]
  rw [le_div_iff₀ hs, ← pow_le_pow_iff_left₀ (by positivity)
    (mul_nonneg (by linarith) (by linarith)) two_ne_zero, mul_pow, Real.sq_sqrt ha.le, mul_pow]
  linarith

/-- **The lemma of the proof of Theorem 3.**  For `C` large enough, depending on
`β₁`, `β₂`, `δ`, Adam's step `Δ_t = x_{t+1} - x_t` has `E[Δ_t] ≥ 0`.
arXiv:1904.09237, Appendix, proof of Theorem 3, Lemma. -/
theorem stoch_step (hβ₁ : 0 ≤ β₁) (hβ₂ : β₂ < 1) (hγ : β₁ < Real.sqrt β₂) {δ : ℝ}
    (hδ : 0 < δ) : ∃ C₀ : ℝ, ∀ C ≥ C₀, ∀ (Ω : Type) [MeasurableSpace Ω] (μ : Measure Ω)
      [IsProbabilityMeasure μ] (b : ℕ → Ω → Bool), IsBernoulliSeq μ b ((1 + δ) / (C + 1)) →
      ∀ α : ℕ → ℝ, (∀ t, 0 ≤ α t) → ∀ t, 1 ≤ t →
        Integrable (fun ω => stochX C β₁ β₂ α b (t + 1) ω - stochX C β₁ β₂ α b t ω) μ ∧
        0 ≤ ∫ ω, (stochX C β₁ β₂ α b (t + 1) ω - stochX C β₁ β₂ α b t ω) ∂μ := by
  have hβ₂0 : 0 < β₂ := Real.sqrt_pos.1 (hβ₁.trans_lt hγ)
  have hs1 : Real.sqrt β₂ < 1 := by
    rw [show (1 : ℝ) = Real.sqrt 1 by simp]; exact Real.sqrt_lt_sqrt hβ₂0.le hβ₂
  have hβ₁1 : β₁ < 1 := hγ.trans hs1
  have hγ0 : 0 ≤ β₁ / Real.sqrt β₂ := by positivity
  have hγ1 : β₁ / Real.sqrt β₂ < 1 := (div_lt_one (Real.sqrt_pos.2 hβ₂0)).2 hγ
  set K := 2 * (1 - β₁) / Real.sqrt (β₂ * (1 - β₂)) with hK
  have hK0 : 0 ≤ K := div_nonneg (by linarith) (Real.sqrt_nonneg _)
  have hL0 : 0 ≤ β₁ * K / (1 - β₁ / Real.sqrt β₂) := div_nonneg (by positivity) (by linarith)
  set B := 1 / Real.sqrt (1 - β₂) + β₁ * K / (1 - β₁ / Real.sqrt β₂) with hB
  have hB0 : 0 ≤ B := by positivity
  refine ⟨max (2 * (1 + δ)) (4 * (1 + δ) ^ 3 * B ^ 2 / (1 - β₁) ^ 2),
    fun C hC Ω _ μ _ b hb α hα t ht => ?_⟩
  obtain ⟨n, rfl⟩ : ∃ n, t = n + 1 := ⟨t - 1, by omega⟩
  have hC1 : 2 * (1 + δ) ≤ C := le_of_max_le_left hC
  have hC0 : 1 ≤ C := by linarith
  set p := (1 + δ) / (C + 1) with hpdef
  have hp0 : 0 ≤ p := by positivity
  have hp1 : p ≤ 1 := by rw [hpdef, div_le_one (by linarith)]; linarith
  set U : Ω → ℕ → Bool := fun ω j => b j ω
  have e : (fun ω => stochX C β₁ β₂ α b (n + 1 + 1) ω - stochX C β₁ β₂ α b (n + 1) ω) =
      fun ω => -(α (n + 1) * (coinM C β₁ β₂ α (n + 1) (U ω) /
        Real.sqrt (coinV C β₁ β₂ α (n + 1) (U ω)))) :=
    funext fun ω => stochX_succ C β₁ β₂ α b ω n
  rw [e]
  refine ⟨((coin_integrable hb (coinMV_dep (n + 1) fun m v => m / Real.sqrt v)).const_mul
    _).neg, ?_⟩
  rw [integral_neg, integral_const_mul, neg_nonneg]
  refine mul_nonpos_of_nonneg_of_nonpos (hα _) ?_
  simp_rw [coin_step_split]
  have dT := coinMV_dep (C := C) (β₁ := β₁) (β₂ := β₂) (α := α) n
    fun m v => (β₁ * m + (1 - β₁) * C) / Real.sqrt (β₂ * v + (1 - β₂) * C ^ 2)
  have dF := coinMV_dep (C := C) (β₁ := β₁) (β₂ := β₂) (α := α) n
    fun m v => (β₁ * m - (1 - β₁)) / Real.sqrt (β₂ * v + (1 - β₂))
  have hn : n + 1 ∉ Icc 1 n := by simp
  rw [integral_add (coin_integrable hb (dependsOn_coin_mul dT (n + 1) coinInd))
    (coin_integrable hb (dependsOn_coin_mul dF (n + 1) fun c => 1 - coinInd c)),
    coin_integral_ind_mul hb hp0 dT hn, coin_integral_not_mul hb hp0 dF hn]
  have d1 := coinMV_dep (C := C) (β₁ := β₁) (β₂ := β₂) (α := α) n
    fun m v => β₁ * m / Real.sqrt (β₂ * v + (1 - β₂))
  have d2 := coinMV_dep (C := C) (β₁ := β₁) (β₂ := β₂) (α := α) n
    fun _ v => 1 / Real.sqrt (β₂ * v + (1 - β₂))
  have eF : ∀ m v : ℝ, (β₁ * m - (1 - β₁)) / Real.sqrt (β₂ * v + (1 - β₂)) =
      β₁ * m / Real.sqrt (β₂ * v + (1 - β₂)) - (1 - β₁) * (1 / Real.sqrt (β₂ * v + (1 - β₂))) :=
    fun _ _ => by ring
  simp_rw [eF]
  rw [integral_sub (coin_integrable hb d1) ((coin_integrable hb d2).const_mul _),
    integral_const_mul]
  have h1 := stoch_int_T1 (α := α) hb hC0 hβ₁ hβ₁1.le hβ₂0.le hβ₂ n
  have h2 := stoch_int_T2 (α := α) hb hp0 hC0 hβ₁ hβ₁1.le hβ₂0 hβ₂ n
  have hpC : 1 + p * C ^ 2 ≤ 1 + (1 + δ) * C := by
    have : p * C ^ 2 * (C + 1) ≤ (1 + δ) * C * (C + 1) := by
      rw [show p * C ^ 2 * (C + 1) = (p * (C + 1)) * C ^ 2 by ring,
        div_mul_cancel₀ _ (by linarith : C + 1 ≠ 0)]
      nlinarith
    nlinarith [le_of_mul_le_mul_right this (by linarith : (0 : ℝ) < C + 1)]
  have h3 := stoch_int_inv (α := α) (β₁ := β₁) hb hp0 hβ₂0.le hβ₂ hpC n
  have hS := sum_pow_sub_le hγ0 hγ1 n
  have key := stoch_const_le hB0 hβ₁1 hδ hC1 (le_of_max_le_right hC)
  rw [← hpdef] at key
  set E1 := ∫ ω, (β₁ * coinM C β₁ β₂ α n (U ω) + (1 - β₁) * C) /
    Real.sqrt (β₂ * coinV C β₁ β₂ α n (U ω) + (1 - β₂) * C ^ 2) ∂μ
  set E2 := ∫ ω, β₁ * coinM C β₁ β₂ α n (U ω) /
    Real.sqrt (β₂ * coinV C β₁ β₂ α n (U ω) + (1 - β₂)) ∂μ
  set E3 := ∫ ω, 1 / Real.sqrt (β₂ * coinV C β₁ β₂ α n (U ω) + (1 - β₂)) ∂μ
  set s := Real.sqrt (1 + (1 + δ) * C)
  have a1 : p * E1 ≤ p * (1 / Real.sqrt (1 - β₂)) := mul_le_mul_of_nonneg_left h1 hp0
  have a2 : E2 ≤ p * (β₁ * K / (1 - β₁ / Real.sqrt β₂)) := by
    refine h2.trans ?_
    have := mul_le_mul_of_nonneg_left hS (by positivity : 0 ≤ β₁ * K * p)
    rw [mul_one_div] at this
    calc β₁ * (K * (p * ∑ j ∈ Icc 1 n, (β₁ / Real.sqrt β₂) ^ (n - j)))
        = β₁ * K * p * ∑ j ∈ Icc 1 n, (β₁ / Real.sqrt β₂) ^ (n - j) := by ring
      _ ≤ β₁ * K * p / (1 - β₁ / Real.sqrt β₂) := this
      _ = _ := by ring
  have a2' : (1 - p) * E2 ≤ p * (β₁ * K / (1 - β₁ / Real.sqrt β₂)) := by
    have := mul_le_mul_of_nonneg_left a2 (by linarith : 0 ≤ 1 - p)
    nlinarith [mul_nonneg hp0 hL0]
  have a3 : (1 - p) * (1 - β₁) * (1 / s) ≤ (1 - p) * ((1 - β₁) * E3) := by
    have := mul_le_mul_of_nonneg_left h3 (mul_nonneg (by linarith : 0 ≤ 1 - p)
      (by linarith : 0 ≤ 1 - β₁))
    linarith
  have key' : p * B ≤ (1 - p) * (1 - β₁) * (1 / s) := by rw [mul_one_div]; exact key
  rw [hB] at key'
  linarith

/-- The hypotheses of `stoch_const_le` are satisfiable: `B = β₁ = 0`, `δ = 1`,
`C = 4`; those of `stoch_step` at `β₁ = 0`, `β₂ = 1/2`, `δ = 1`. -/
example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ 2 * (1 + 1) ≤ (4 : ℝ) ∧
    4 * (1 + 1) ^ 3 * (0 : ℝ) ^ 2 / (1 - 0) ^ 2 ≤ 4 ∧ (1 / 2 : ℝ) < 1 ∧
    (0 : ℝ) < Real.sqrt (1 / 2) :=
  ⟨le_rfl, one_pos, one_pos, by norm_num, by norm_num, by norm_num,
    Real.sqrt_pos.mpr (by norm_num)⟩

end AdamBeyond
end Transformer

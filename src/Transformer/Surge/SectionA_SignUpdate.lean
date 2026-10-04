/-
# The sign update

arXiv:2405.14578, Appendix A, eqs. (28) and (29).  As `β₁, β₂ → 1` the update of Adam tends to
`E_t[G]/√E_t[G²] = sign(E_t[G])/√(1 + var_t(G)/E_t[G]²)` (`tendsto_adamUpdate_one`,
`adamLimit_eq_sign`).  "When the variance of `G_est` is small, the update amount is
approximately `sign(G_est)`": once `t var_t(G) < E_t[G]²`, every `G_i`, `1 ≤ i ≤ t`, has the sign
of `E_t[G]` (`sign_eq_of_mul_iterVar_lt`), and the limit is within `var_t(G)/(2E_t[G]²)` of it
(`abs_adamLimit_sub_sign_le`).

Without `ε_Adam`, as `β₁, β₂ → 0` the update tends to `G_t/√(G_t²) = sign(G_t)`, eq. (29)
(`tendsto_adamUpdate_zero`), and at `β₁ = β₂ = 0` it is `sign(G_t)` exactly (`adamUpdate_zero`).
Eq. (29) leaves implicit `G_t ≠ 0`, where its `G_est/√(G_est²)` is defined: for `G_1 = 1`,
`G_2 = 0` the update at `t = 2` has no limit (`not_tendsto_adamUpdate_zero`).
-/

import Transformer.Surge.SectionA_AdamMoments

open Filter Topology

namespace Transformer.Surge

/-- `|m/√s - sign(m)| ≤ (s - m²)/(2m²)` when `m² ≤ s`. -/
theorem abs_div_sqrt_sub_sign_le {m s : ℝ} (h : m ^ 2 ≤ s) :
    |m / √s - Real.sign m| ≤ (s - m ^ 2) / (2 * m ^ 2) := by
  rcases eq_or_ne m 0 with rfl | hm
  · simp
  have hr : 0 ≤ (s - m ^ 2) / m ^ 2 := div_nonneg (by linarith) (sq_nonneg m)
  rw [div_sqrt_eq_sign, mul_comm, ← div_div]
  generalize (s - m ^ 2) / m ^ 2 = r at hr ⊢
  have hq1 : 1 ≤ √(1 + r) := Real.one_le_sqrt.2 (by linarith)
  have hq2 : √(1 + r) ≤ 1 + r / 2 := Real.sqrt_le_iff.2 ⟨by linarith, by nlinarith⟩
  have hq0 : 0 < √(1 + r) := by linarith
  have key : (√(1 + r) - 1) / √(1 + r) ≤ √(1 + r) - 1 := div_le_self (by linarith) hq1
  rw [sub_div, div_self hq0.ne'] at key
  have hle : 1 / √(1 + r) ≤ 1 := (div_le_one hq0).2 hq1
  rcases Real.sign_apply_eq_of_ne_zero m hm with h | h <;> rw [h, abs_le]
  · rw [neg_div]
    constructor <;> linarith
  · constructor <;> linarith

/-- The hypothesis of `abs_div_sqrt_sub_sign_le` is satisfiable: `m = s = 1`. -/
example := abs_div_sqrt_sub_sign_le (m := 1) (s := 1) (by norm_num)

/-- If `t var_t(G) < E_t[G]²`, every `G_i`, `1 ≤ i ≤ t`, has the sign of `E_t[G]`. -/
theorem sign_eq_of_mul_iterVar_lt {g : ℕ → ℝ} {t : ℕ} (h : t * iterVar g t < iterMean g t ^ 2)
    {i : ℕ} (hi : i ∈ Finset.Icc 1 t) : Real.sign (g i) = Real.sign (iterMean g t) := by
  rw [mul_iterVar] at h
  have hi' : (g i - iterMean g t) ^ 2 < iterMean g t ^ 2 :=
    (Finset.single_le_sum (fun j _ => sq_nonneg (g j - iterMean g t)) hi).trans_lt h
  rw [sq_lt_sq, abs_lt] at hi'
  rcases lt_trichotomy (iterMean g t) 0 with hm | hm | hm
  · rw [abs_of_neg hm] at hi'
    rw [Real.sign_of_neg hm, Real.sign_of_neg (by linarith [hi'.2])]
  · rw [hm, abs_zero] at hi'
    exfalso
    linarith [hi'.1, hi'.2]
  · rw [abs_of_pos hm] at hi'
    rw [Real.sign_of_pos hm, Real.sign_of_pos (by linarith [hi'.1])]

/-- The hypotheses of `sign_eq_of_mul_iterVar_lt` are satisfiable: `G = 1`, `t = i = 1`. -/
example := sign_eq_of_mul_iterVar_lt (g := fun _ => 1) (t := 1)
  (by norm_num [iterVar, iterMean]) (i := 1) (by simp)

/-- **Eq. (28)**, "when the variance of `G_est` is small, the update amount is approximately
`sign(G_est)`": if `t var_t(G) < E_t[G]²`, then `|E_t[G]/√E_t[G²] - sign(G_i)| ≤
var_t(G)/(2E_t[G]²)` for every `1 ≤ i ≤ t`. -/
theorem abs_adamLimit_sub_sign_le {g : ℕ → ℝ} {t : ℕ} (h : t * iterVar g t < iterMean g t ^ 2)
    {i : ℕ} (hi : i ∈ Finset.Icc 1 t) :
    |iterMean g t / √(iterMean (fun i => g i ^ 2) t) - Real.sign (g i)| ≤
      iterVar g t / (2 * iterMean g t ^ 2) := by
  rw [sign_eq_of_mul_iterVar_lt h hi]
  exact abs_div_sqrt_sub_sign_le (sub_nonneg.1 (iterVar_nonneg g t))

/-- The hypotheses of `abs_adamLimit_sub_sign_le` are satisfiable: `G = 1`, `t = i = 1`. -/
example := abs_adamLimit_sub_sign_le (g := fun _ => 1) (t := 1)
  (by norm_num [iterVar, iterMean]) (i := 1) (by simp)

/-- **Eq. (29)** at `β₁ = β₂ = 0`: without `ε_Adam`, the update of Adam is `sign(G_t)`. -/
theorem adamUpdate_zero (g : ℕ → ℝ) {t : ℕ} (ht : t ≠ 0) :
    adamUpdate 0 0 0 g t = Real.sign (g t) := by
  simp only [adamUpdate, adamMoment_zero _ ht, zero_pow ht, sub_zero, div_one, add_zero,
    Real.sqrt_sq_eq_abs]
  rcases lt_trichotomy (g t) 0 with h | h | h
  · rw [abs_of_neg h, Real.sign_of_neg h, div_neg, div_self h.ne]
  · simp [h]
  · rw [abs_of_pos h, Real.sign_of_pos h, div_self h.ne']

/-- The hypothesis of `adamUpdate_zero` is satisfiable: `t = 1`. -/
example (g : ℕ → ℝ) := adamUpdate_zero g one_ne_zero

/-- **Eq. (29)**: without `ε_Adam`, as `β₁, β₂ → 0` the update of Adam tends to
`G_t/√(G_t²) = sign(G_t)`.  The source's `G_est/√(G_est²)` leaves implicit `G_t ≠ 0`. -/
theorem tendsto_adamUpdate_zero {g : ℕ → ℝ} {t : ℕ} (ht : t ≠ 0) (hg : g t ≠ 0) :
    Tendsto (fun β : ℝ × ℝ => adamUpdate β.1 β.2 0 g t) (𝓝 0 ×ˢ 𝓝 0)
      (𝓝 (Real.sign (g t))) := by
  have hF (f : ℕ → ℝ) : Tendsto (fun β => adamMoment β f t / (1 - β ^ t)) (𝓝 0) (𝓝 (f t)) := by
    have hc : Continuous fun β => adamMoment β f t := by simp_rw [adamMoment_eq]; fun_prop
    have h1 : Tendsto (fun β : ℝ => 1 - β ^ t) (𝓝 0) (𝓝 1) := by
      simpa [zero_pow ht] using ((continuous_pow t).tendsto (0 : ℝ)).const_sub 1
    have := (hc.tendsto 0).div h1 one_ne_zero
    rw [adamMoment_zero f ht, div_one] at this
    exact this
  have := ((hF g).comp tendsto_fst).div ((hF fun i => g i ^ 2).comp tendsto_snd).sqrt
    (by simpa [Real.sqrt_sq_eq_abs] using hg)
  rw [← adamUpdate_zero g ht, show adamUpdate 0 0 0 g t = g t / √(g t ^ 2) by
    simp [adamUpdate, adamMoment_zero _ ht, zero_pow ht]]
  simp only [adamUpdate, add_zero]
  exact this

/-- The hypotheses of `tendsto_adamUpdate_zero` are satisfiable: `G = 1`, `t = 1`. -/
example := tendsto_adamUpdate_zero (g := fun _ => 1) one_ne_zero one_ne_zero

/-- Eq. (29) needs `G_t ≠ 0`: for `G_1 = 1`, `G_2 = 0`, the update of Adam at `t = 2` has no limit
as `β₁, β₂ → 0`.  It tends to `0` along `β₁ = β₂`, and to `1` along `β₂ = β₁²`, `β₁ > 0`. -/
theorem not_tendsto_adamUpdate_zero : ¬∃ L, Tendsto
    (fun β : ℝ × ℝ => adamUpdate β.1 β.2 0 (fun i => if i = 1 then 1 else 0) 2) (𝓝 0 ×ˢ 𝓝 0)
    (𝓝 L) := by
  rintro ⟨L, hL⟩
  have hm (β : ℝ) : adamMoment β (fun i => if i = 1 then 1 else 0) 2 = β * (1 - β) := by
    simp [adamMoment]
  have hm2 (β : ℝ) :
      adamMoment β (fun i => (if i = 1 then (1 : ℝ) else 0) ^ 2) 2 = β * (1 - β) := by
    simp [adamMoment]
  simp only [adamUpdate, hm, hm2, add_zero] at hL
  have h1 : Tendsto (fun s : ℝ => s * (1 - s) / (1 - s ^ 2)) (𝓝 0) (𝓝 0) := by
    have : ContinuousAt (fun s : ℝ => s * (1 - s) / (1 - s ^ 2)) 0 := by
      fun_prop (disch := norm_num)
    simpa using this.tendsto
  have h2 : Tendsto (fun s : ℝ => √(1 + s ^ 2) / (1 + s)) (𝓝[>] 0) (𝓝 1) := by
    have : ContinuousAt (fun s : ℝ => √(1 + s ^ 2) / (1 + s)) 0 := by
      fun_prop (disch := norm_num)
    simpa using this.tendsto.mono_left nhdsWithin_le_nhds
  have e1 : L = 0 := by
    have h : Tendsto (fun s : ℝ => s * (1 - s) / (1 - s ^ 2) / √(s * (1 - s) / (1 - s ^ 2)))
        (𝓝 0) (𝓝 0) := by
      simp only [Real.div_sqrt]
      simpa using h1.sqrt
    exact tendsto_nhds_unique (hL.comp (tendsto_id.prodMk tendsto_id)) h
  have e2 : L = 1 := by
    have hp : Tendsto (fun s : ℝ => (s, s ^ 2)) (𝓝[>] 0) (𝓝 0 ×ˢ 𝓝 0) := by
      refine (tendsto_id.prodMk ?_).mono_left nhdsWithin_le_nhds
      simpa using (continuous_pow 2).tendsto (0 : ℝ)
    refine tendsto_nhds_unique (hL.comp hp) (h2.congr' ?_)
    filter_upwards [Ioo_mem_nhdsGT one_pos] with s ⟨hs0, hs1⟩
    show √(1 + s ^ 2) / (1 + s) = s * (1 - s) / (1 - s ^ 2) /
      √(s ^ 2 * (1 - s ^ 2) / (1 - (s ^ 2) ^ 2))
    have hs2 : s ^ 2 < 1 := by nlinarith
    rw [show s * (1 - s) / (1 - s ^ 2) = s / (1 + s) by
        rw [div_eq_div_iff (by nlinarith : (0 : ℝ) < 1 - s ^ 2).ne' (by positivity)]; ring,
      show s ^ 2 * (1 - s ^ 2) / (1 - (s ^ 2) ^ 2) = s ^ 2 / (1 + s ^ 2) by
        rw [div_eq_div_iff (by nlinarith : (0 : ℝ) < 1 - (s ^ 2) ^ 2).ne' (by positivity)]; ring,
      Real.sqrt_div' _ (by positivity), Real.sqrt_sq hs0.le]
    field_simp
  exact zero_ne_one (e1.symm.trans e2)

end Transformer.Surge

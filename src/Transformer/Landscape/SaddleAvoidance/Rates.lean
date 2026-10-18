/-
# Saddle avoidance — rates of convergence of gradient descent

Along an orbit `x_k` of gradient descent converging to `w`, where the
Łojasiewicz inequality holds with `0 < a < 1`, `r_k = f x_k - f w ≥ 0`
eventually drops by `c r_k ^ (2a)` per step and `‖x_k - w‖ ≤ K r_k ^ (1 - a)`
(`exists_rate_estimates_of_isLojasiewiczAt`).  By `Landscape.SaddleAvoidance.Decay`,
`f x_k - f w` and `‖x_k - w‖` decay geometrically for `a ≤ 1/2`
(`exists_le_mul_pow_of_isLojasiewiczAt`), and as `(k + 1) ^ (-1/(2a - 1))` and
`(k + 1) ^ (-(1 - a)/(2a - 1))` for `1/2 < a < 1`
(`exists_le_mul_rpow_of_isLojasiewiczAt`).

This is the rate part of `prop:rates` of arXiv:1602.04915: at a limit `x*`
where the inequality holds, `‖x_k - x*‖ ≤ C b^k` with `b < 1` for
`0 < a ≤ 1/2`, and `‖x_k - x*‖ ≤ C / k^((1-a)/(2a-1))` for `1/2 < a < 1`.
Here the limit need not be a local minimizer, `a ≤ 1/2` may be `0` or negative,
and the values of `f` decay too.
-/

import Transformer.Landscape.SaddleAvoidance.Decay
import Transformer.Landscape.SaddleAvoidance.Trap

open Filter Metric
open scoped Topology NNReal

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ} {L : ℝ≥0} {η a : ℝ}

/-- **The two estimates behind the rates.**  Let an orbit of gradient descent
converge to `w`, where the Łojasiewicz inequality holds with `0 < a < 1`.
Then `f` stays above `f w` along it, and for some `c > 0` and `K ≥ 0`, from
some step on, `r_k = f x_k - f w` satisfies `r_{k+1} ≤ r_k - c r_k ^ (2a)` and
`‖x_k - w‖ ≤ K r_k ^ (1 - a)`. -/
theorem exists_rate_estimates_of_isLojasiewiczAt (hf : Differentiable ℝ f)
    (hL : LipschitzWith L (gradient f)) (hη : 0 < η) (hηL : η * L < 1) (ha0 : 0 < a)
    (ha1 : a < 1) {w x : E} (hw : IsLojasiewiczAt f w a)
    (hx : Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w)) :
    (∀ k, f w ≤ f ((gradientStep f η)^[k] x)) ∧ ∃ c > 0, ∃ K ≥ 0, ∀ᶠ k in atTop,
      f ((gradientStep f η)^[k + 1] x) - f w ≤ f ((gradientStep f η)^[k] x) - f w -
          c * (f ((gradientStep f η)^[k] x) - f w) ^ (2 * a) ∧
        ‖(gradientStep f η)^[k] x - w‖ ≤
          K * (f ((gradientStep f η)^[k] x) - f w) ^ (1 - a) := by
  set u : ℕ → E := fun k => (gradientStep f η)^[k] x with hu
  obtain ⟨m, hm, ε, hε, hev⟩ := hw
  obtain ⟨ρ, hρ, hloj⟩ := Metric.eventually_nhds_iff_ball.1 hev
  have hflim : Tendsto (fun k => f (u k)) atTop (𝓝 (f w)) := (hf.continuous.tendsto w).comp hx
  have hfw : ∀ k, f w ≤ f (u k) := (antitone_iterate hf hL hη.le hηL.le x).le_of_tendsto hflim
  have hc : 0 < 1 - η * L := by linarith
  have hκ : 0 < m * (1 - a) * (1 - η * L) := mul_pos (mul_pos hm (by linarith)) hc
  obtain ⟨N, hN⟩ := eventually_atTop.1 ((hx.eventually (ball_mem_nhds w hρ)).and
    (hflim.eventually (gt_mem_nhds (lt_add_of_pos_right (f w) hε))))
  have hl : ∀ k, N ≤ k → f w < f (u k) → m * |f (u k) - f w| ^ a ≤ ‖gradient f (u k)‖ :=
    fun k hk h => hloj _ (hN k hk).1 h (hN k hk).2
  have hsucc : ∀ k, u (k + 1) = gradientStep f η (u k) := fun k =>
    Function.iterate_succ_apply' _ _ _
  have hstep : ∀ k, N ≤ k → m * (1 - a) * (1 - η * L) * ‖u (k + 1) - u k‖ ≤
      (f (u k) - f w) ^ (1 - a) - (f (u (k + 1)) - f w) ^ (1 - a) := fun k hk => by
    have h := hfw (k + 1)
    rw [hsucc] at h ⊢
    exact mul_norm_gradientStep_sub_le hf hL hη hηL ha0.le ha1 (hl k hk) h
  refine ⟨hfw, η * (1 - η * L) * m ^ 2, by positivity, (m * (1 - a) * (1 - η * L))⁻¹,
    inv_nonneg.2 hκ.le, eventually_atTop.2 ⟨N, fun k hk => ⟨?_, ?_⟩⟩⟩
  · have hdec := le_sub_of_gradientStep hf hL η (u k)
    rw [← hsucc] at hdec
    have hr : 0 ≤ f (u k) - f w := sub_nonneg.2 (hfw k)
    have hG := mul_nonneg (mul_nonneg hη.le hc.le) (sq_nonneg ‖gradient f (u k)‖)
    have hsq : (f (u k) - f w) ^ (2 * a) = ((f (u k) - f w) ^ a) ^ 2 := by
      rw [← Real.rpow_natCast, ← Real.rpow_mul hr]
      norm_num [mul_comm]
    rw [hsq]
    rcases hr.eq_or_lt with h0 | hpos
    · rw [← h0, Real.zero_rpow ha0.ne', zero_pow two_ne_zero, mul_zero, sub_zero]
      linarith
    · have hl' := hl k hk (sub_pos.1 hpos)
      rw [abs_of_pos hpos] at hl'
      have hsq' := pow_le_pow_left₀ (mul_nonneg hm.le (Real.rpow_nonneg hr a)) hl' 2
      nlinarith [mul_le_mul_of_nonneg_left hsq' (mul_nonneg hη.le hc.le)]
  · have hlen : ∀ n, m * (1 - a) * (1 - η * L) * ‖u (k + n) - u k‖ ≤
        (f (u k) - f w) ^ (1 - a) := fun n => by
      have h := mul_norm_sub_le_of_step (u := fun j => u (k + j))
        (φ := fun j => (f (u (k + j)) - f w) ^ (1 - a)) hκ.le
        (fun j => hstep (k + j) (hk.trans le_self_add)) (Nat.zero_le n)
      simp only [add_zero] at h
      linarith [Real.rpow_nonneg (sub_nonneg.2 (hfw (k + n))) (1 - a)]
    have hlim : Tendsto (fun n => m * (1 - a) * (1 - η * L) * ‖u (k + n) - u k‖) atTop
        (𝓝 (m * (1 - a) * (1 - η * L) * ‖w - u k‖)) :=
      (((hx.comp ((tendsto_add_atTop_nat k).congr fun n => add_comm n k)).sub_const
        (u k)).norm.const_mul _)
    have h := le_of_tendsto' hlim hlen
    rw [norm_sub_rev, ← div_eq_inv_mul, le_div_iff₀ hκ]
    linarith

/-- **Geometric rate** (`prop:rates` of arXiv:1602.04915, `0 < a ≤ 1/2`).  Let
`f` have an `L`-Lipschitz gradient, `0 < η`, `η L < 1`, and let an orbit of
gradient descent converge to `w`, where the Łojasiewicz inequality holds with
an exponent `a ≤ 1/2`.  Then for some `C` and `0 < b < 1`,
`f x_k - f w ≤ C b ^ k` and `‖x_k - w‖ ≤ C b ^ k` for every `k`. -/
theorem exists_le_mul_pow_of_isLojasiewiczAt (hf : Differentiable ℝ f)
    (hL : LipschitzWith L (gradient f)) (hη : 0 < η) (hηL : η * L < 1) (ha : a ≤ 1 / 2)
    {w x : E} (hw : IsLojasiewiczAt f w a)
    (hx : Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w)) :
    ∃ C b : ℝ, 0 < b ∧ b < 1 ∧ ∀ k : ℕ, f ((gradientStep f η)^[k] x) - f w ≤ C * b ^ k ∧
      ‖(gradientStep f η)^[k] x - w‖ ≤ C * b ^ k := by
  obtain ⟨hfw, c, hc, K, hK, hev⟩ := exists_rate_estimates_of_isLojasiewiczAt hf hL hη hηL
    (by norm_num : (0 : ℝ) < 1 / 2) (by norm_num) (hw.mono ha) hx
  set r : ℕ → ℝ := fun k => f ((gradientStep f η)^[k] x) - f w with hr_def
  have hr : ∀ k, 0 ≤ r k := fun k => sub_nonneg.2 (hfw k)
  replace hev : ∀ᶠ k in atTop, r (k + 1) ≤ r k - c * r k ∧
      ‖(gradientStep f η)^[k] x - w‖ ≤ K * r k ^ (1 / 2 : ℝ) := hev.mono fun k hk => by
    rwa [show 2 * (1 / 2 : ℝ) = 1 by norm_num, Real.rpow_one,
      show (1 : ℝ) - 1 / 2 = 1 / 2 by norm_num] at hk
  set q := max (1 - c) (1 / 2) with hq
  have hq0 : 0 < q := lt_max_of_lt_right (by norm_num)
  have hq1 : q < 1 := max_lt (by linarith) (by norm_num)
  obtain ⟨C₁, hC₁⟩ := exists_le_mul_pow_of_le_mul hq0 hr (hev.mono fun k hk => by
    calc r (k + 1) ≤ r k - c * r k := hk.1
      _ = (1 - c) * r k := by ring
      _ ≤ q * r k := mul_le_mul_of_nonneg_right (le_max_left _ _) (hr k))
  obtain ⟨C, hC0, hC⟩ := exists_le_mul_and_le_mul_rpow hr (fun k => norm_nonneg _)
    (fun k => pow_pos hq0 k) hK (by norm_num) hC₁ (hev.mono fun k hk => hk.2)
  have hqb : q ≤ q ^ (1 / 2 : ℝ) := by
    calc q = q ^ (1 : ℝ) := (Real.rpow_one q).symm
      _ ≤ q ^ (1 / 2 : ℝ) := Real.rpow_le_rpow_of_exponent_ge hq0 hq1.le (by norm_num)
  refine ⟨C, q ^ (1 / 2 : ℝ), Real.rpow_pos_of_pos hq0 _,
    Real.rpow_lt_one hq0.le hq1 (by norm_num), fun k => ⟨?_, ?_⟩⟩
  · exact (hC k).1.trans (mul_le_mul_of_nonneg_left (pow_le_pow_left₀ hq0.le hqb k) hC0)
  · rw [Real.rpow_pow_comm hq0.le]
    exact (hC k).2

/-- **Polynomial rate** (`prop:rates` of arXiv:1602.04915, `1/2 < a < 1`).
Under the hypotheses of `exists_le_mul_pow_of_isLojasiewiczAt`, with an
exponent `1/2 < a < 1` instead, for some `C` and every `k`,
`f x_k - f w ≤ C (k + 1) ^ (-1/(2a - 1))` and
`‖x_k - w‖ ≤ C (k + 1) ^ (-(1 - a)/(2a - 1))`.  The source's bound
`C / k^((1-a)/(2a-1))` for `k ≥ 1` follows, since `k + 1 > k`. -/
theorem exists_le_mul_rpow_of_isLojasiewiczAt (hf : Differentiable ℝ f)
    (hL : LipschitzWith L (gradient f)) (hη : 0 < η) (hηL : η * L < 1) (ha : 1 / 2 < a)
    (ha1 : a < 1) {w x : E} (hw : IsLojasiewiczAt f w a)
    (hx : Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w)) :
    ∃ C : ℝ, ∀ k : ℕ,
      f ((gradientStep f η)^[k] x) - f w ≤ C * ((k : ℝ) + 1) ^ (-(1 / (2 * a - 1))) ∧
        ‖(gradientStep f η)^[k] x - w‖ ≤ C * ((k : ℝ) + 1) ^ (-((1 - a) / (2 * a - 1))) := by
  obtain ⟨hfw, c, hc, K, hK, hev⟩ := exists_rate_estimates_of_isLojasiewiczAt hf hL hη hηL
    (by linarith) ha1 hw hx
  set r : ℕ → ℝ := fun k => f ((gradientStep f η)^[k] x) - f w with hr_def
  have hr : ∀ k, 0 ≤ r k := fun k => sub_nonneg.2 (hfw k)
  replace hev : ∀ᶠ k in atTop, r (k + 1) ≤ r k - c * r k ^ (1 + (2 * a - 1)) ∧
      ‖(gradientStep f η)^[k] x - w‖ ≤ K * r k ^ (1 - a) := hev.mono fun k hk => by
    rwa [show 1 + (2 * a - 1) = 2 * a by ring]
  obtain ⟨C₁, hC₁⟩ := exists_le_mul_rpow_of_le_sub (by linarith : 0 < 2 * a - 1)
    (by linarith) hc hr (hev.mono fun k hk => hk.1)
  obtain ⟨C, -, hC⟩ := exists_le_mul_and_le_mul_rpow hr (fun k => norm_nonneg _)
    (fun k => Real.rpow_pos_of_pos (by positivity) _) hK (by linarith) hC₁
    (hev.mono fun k hk => hk.2)
  refine ⟨C, fun k => ⟨(hC k).1, ?_⟩⟩
  rw [show -((1 - a) / (2 * a - 1)) = -(1 / (2 * a - 1)) * (1 - a) by ring,
    Real.rpow_mul (by positivity)]
  exact (hC k).2

/-- The hypotheses of the theorems above hold for `f(x) = ‖x‖² / 2` on the
line with `η = 1/2`, `L = 1` and the orbit `2⁻ᵏ` of `1`, which converges to
`0`: there the Łojasiewicz inequality holds with `a = 1/2`, and so with
`a = 3/4`. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    Differentiable ℝ f ∧ LipschitzWith 1 (gradient f) ∧ (0 : ℝ) < 1 / 2 ∧
      (1 / 2 : ℝ) * ((1 : ℝ≥0) : ℝ) < 1 ∧ (1 / 2 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) < 3 / 4 ∧
      (3 / 4 : ℝ) < 1 ∧ IsLojasiewiczAt f 0 (1 / 2) ∧ IsLojasiewiczAt f 0 (3 / 4) ∧
      Tendsto (fun k => (gradientStep f (1 / 2))^[k] (1 : ℝ)) atTop (𝓝 0) := by
  intro f
  have hL : LipschitzWith 1 (gradient f) := by
    rw [show gradient f = id from gradient_half_mul_norm_sq]
    exact LipschitzWith.id
  have horb : ∀ k, (gradientStep f (1 / 2))^[k] (1 : ℝ) = (1 / 2) ^ k := fun k => by
    simp only [f, iterate_gradientStep_half_mul_norm_sq]
    norm_num
  refine ⟨(contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero, hL,
    by norm_num, by norm_num, le_rfl, by norm_num, by norm_num,
    isLojasiewiczAt_half_mul_norm_sq, isLojasiewiczAt_half_mul_norm_sq.mono (by norm_num), ?_⟩
  simp only [horb]
  exact tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)

end Transformer.Landscape.SaddleAvoidance

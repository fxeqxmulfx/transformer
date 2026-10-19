module

public import Transformer.Landscape.SaddleAvoidance.Convergence

/-!
# Saddle avoidance — the estimates behind the rates of gradient descent

Let an orbit `x_k` of gradient descent converge to `w`, where `f` is continuous and
the Łojasiewicz inequality holds with `0 < a < 1`, and let every step lower `f` by
at least `η κ ‖∇f(x_k)‖²` for some `κ > 0`. Then `r_k = f x_k - f w ≥ 0`
eventually drops by `c r_k ^ (2a)` per step and `‖x_k - w‖ ≤ K r_k ^ (1 - a)`
(`exists_rate_estimates_of_isLojasiewiczAt_of_le_sub`).

The decrease holds with `κ = 1 - η L` for an `L`-Lipschitz gradient
(`iterate_succ_le_sub`) and under a bound `L` on the Hessian along the segments of
the orbit (`iterate_succ_le_sub_of_segment`). The rates of `prop:rates` of
arXiv:1602.04915 follow from the two estimates, in `Landscape.SaddleAvoidance.Rates`
and `Landscape.SaddleAvoidance.OrbitRates`.
-/

@[expose] public section

open Filter Metric
open scoped Topology

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ} {η κ a : ℝ}

/-- **The two estimates behind the rates, from a decrease along the orbit.**  Let
every step of an orbit of gradient descent lower `f` by at least `η κ ‖∇f(x_k)‖²`,
`0 < η`, `0 < κ`, and let the orbit converge to `w`, where `f` is continuous and the
Łojasiewicz inequality holds with `0 < a < 1`. Then `f` stays above `f w` along
it, and for some `c > 0` and `K ≥ 0`, from some step on, `r_k = f x_k - f w`
satisfies `r_{k+1} ≤ r_k - c r_k ^ (2a)` and `‖x_k - w‖ ≤ K r_k ^ (1 - a)`. -/
theorem exists_rate_estimates_of_isLojasiewiczAt_of_le_sub (hη : 0 < η) (hκ : 0 < κ)
    (ha0 : 0 < a) (ha1 : a < 1) {w x : E} (hw : IsLojasiewiczAt f w a)
    (hfc : ContinuousAt f w)
    (hdec : ∀ k, f ((gradientStep f η)^[k + 1] x) ≤
      f ((gradientStep f η)^[k] x) - η * κ * ‖gradient f ((gradientStep f η)^[k] x)‖ ^ 2)
    (hx : Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w)) :
    (∀ k, f w ≤ f ((gradientStep f η)^[k] x)) ∧ ∃ c > 0, ∃ K ≥ 0, ∀ᶠ k in atTop,
      f ((gradientStep f η)^[k + 1] x) - f w ≤ f ((gradientStep f η)^[k] x) - f w -
          c * (f ((gradientStep f η)^[k] x) - f w) ^ (2 * a) ∧
        ‖(gradientStep f η)^[k] x - w‖ ≤
          K * (f ((gradientStep f η)^[k] x) - f w) ^ (1 - a) := by
  set u : ℕ → E := fun k => (gradientStep f η)^[k] x with hu
  obtain ⟨m, hm, ε, hε, hev⟩ := hw
  obtain ⟨ρ, hρ, hloj⟩ := Metric.eventually_nhds_iff_ball.1 hev
  have hflim : Tendsto (fun k => f (u k)) atTop (𝓝 (f w)) := hfc.tendsto.comp hx
  have hfw : ∀ k, f w ≤ f (u k) :=
    (antitone_iterate_of_le_sub hη.le hκ.le hdec).le_of_tendsto hflim
  have hmκ : 0 < m * (1 - a) * κ := mul_pos (mul_pos hm (by linarith)) hκ
  obtain ⟨N, hN⟩ := eventually_atTop.1 ((hx.eventually (ball_mem_nhds w hρ)).and
    (hflim.eventually (gt_mem_nhds (lt_add_of_pos_right (f w) hε))))
  have hl : ∀ k, N ≤ k → f w < f (u k) → m * |f (u k) - f w| ^ a ≤ ‖gradient f (u k)‖ :=
    fun k hk h => hloj _ (hN k hk).1 h (hN k hk).2
  have hsucc : ∀ k, u (k + 1) = gradientStep f η (u k) := fun k =>
    Function.iterate_succ_apply' _ _ _
  have hd : ∀ k, f (u (k + 1)) ≤ f (u k) - η * κ * ‖gradient f (u k)‖ ^ 2 := hdec
  have hstep : ∀ k, N ≤ k → m * (1 - a) * κ * ‖u (k + 1) - u k‖ ≤
      (f (u k) - f w) ^ (1 - a) - (f (u (k + 1)) - f w) ^ (1 - a) := fun k hk => by
    have h := hfw (k + 1)
    have hdk := hd k
    rw [hsucc] at h hdk ⊢
    exact mul_norm_gradientStep_sub_le_of_le_sub hη hκ ha0.le ha1 hdk (hl k hk) h
  refine ⟨hfw, η * κ * m ^ 2, by positivity, (m * (1 - a) * κ)⁻¹,
    inv_nonneg.2 hmκ.le, eventually_atTop.2 ⟨N, fun k hk => ⟨?_, ?_⟩⟩⟩
  · have hdk := hd k
    have hr : 0 ≤ f (u k) - f w := sub_nonneg.2 (hfw k)
    have hG := mul_nonneg (mul_nonneg hη.le hκ.le) (sq_nonneg ‖gradient f (u k)‖)
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
      nlinarith [mul_le_mul_of_nonneg_left hsq' (mul_nonneg hη.le hκ.le)]
  · have hlen : ∀ n, m * (1 - a) * κ * ‖u (k + n) - u k‖ ≤ (f (u k) - f w) ^ (1 - a) :=
      fun n => by
        have h := mul_norm_sub_le_of_step (u := fun j => u (k + j))
          (φ := fun j => (f (u (k + j)) - f w) ^ (1 - a)) hmκ.le
          (fun j => hstep (k + j) (hk.trans le_self_add)) (Nat.zero_le n)
        simp only [add_zero] at h
        linarith [Real.rpow_nonneg (sub_nonneg.2 (hfw (k + n))) (1 - a)]
    have hlim : Tendsto (fun n => m * (1 - a) * κ * ‖u (k + n) - u k‖) atTop
        (𝓝 (m * (1 - a) * κ * ‖w - u k‖)) :=
      (((hx.comp ((tendsto_add_atTop_nat k).congr fun n => add_comm n k)).sub_const
        (u k)).norm.const_mul _)
    have h := le_of_tendsto' hlim hlen
    rw [norm_sub_rev, ← div_eq_inv_mul, le_div_iff₀ hmκ]
    linarith

/-- Along the orbit `(-1/2)ᵏ` of `1` for `‖x‖² / 2` on the line and `η = 3/2`, beyond
`η L < 1` for every Lipschitz constant `L ≥ 1` of the gradient, every step lowers `f`
by `η κ ‖∇f‖²` with `κ = 1 - η / 2 = 1/4`; the orbit converges to `0`, where the
Łojasiewicz inequality holds with `a = 1/2`. -/
example := exists_rate_estimates_of_isLojasiewiczAt_of_le_sub
  (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2) (η := 3 / 2) (κ := 1 - 3 / 2 / 2) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) isLojasiewiczAt_half_mul_norm_sq (by fun_prop)
  (x := 1)
  (fun _ => by rw [Function.iterate_succ_apply']; exact (half_mul_norm_sq_gradientStep _ _).le)
  (tendsto_iterate_gradientStep_half_mul_norm_sq (E := ℝ) (η := 3 / 2) (by norm_num) 1)

end Transformer.Landscape.SaddleAvoidance

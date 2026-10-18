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

These theorems use the `L`-Lipschitz gradient and `η L < 1` only through the
decrease `f x_{k+1} ≤ f x_k - η (1 - η L) ‖∇f(x_k)‖²` (`iterate_succ_le_sub`).  Their
`_of_le_sub` forms ask instead that every step lower `f` by at least
`η κ ‖∇f(x_k)‖²` for some `κ > 0`, and that `f` be continuous at `w`
(`exists_rate_estimates_of_isLojasiewiczAt_of_le_sub` in
`Landscape.SaddleAvoidance.RateEstimates`, `exists_le_mul_pow_of_isLojasiewiczAt_of_le_sub`,
`exists_le_mul_rpow_of_isLojasiewiczAt_of_le_sub`); the present ones follow with
`κ = 1 - η L`.  `Landscape.SaddleAvoidance.OrbitRates` gives them under a Hessian bound
on the segments of the orbit.

This is the rate part of `prop:rates` of arXiv:1602.04915: at a limit `x*`
where the inequality holds, `‖x_k - x*‖ ≤ C b^k` with `b < 1` for
`0 < a ≤ 1/2`, and `‖x_k - x*‖ ≤ C / k^((1-a)/(2a-1))` for `1/2 < a < 1`.
Here the limit need not be a local minimizer, `a ≤ 1/2` may be `0` or negative,
and the values of `f` decay too.
-/

import Transformer.Landscape.SaddleAvoidance.Decay
import Transformer.Landscape.SaddleAvoidance.RateEstimates

open Filter Metric
open scoped Topology NNReal

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ} {L : ℝ≥0} {η κ a : ℝ}

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
          K * (f ((gradientStep f η)^[k] x) - f w) ^ (1 - a) :=
  exists_rate_estimates_of_isLojasiewiczAt_of_le_sub hη (by linarith) ha0 ha1 hw
    hf.continuous.continuousAt (iterate_succ_le_sub hf hL η x) hx

/-- **Geometric rate from a decrease along the orbit** (`0 < a ≤ 1/2`).  Under the
hypotheses of `exists_rate_estimates_of_isLojasiewiczAt_of_le_sub`, with an exponent
`a ≤ 1/2` instead, for some `C` and `0 < b < 1`, `f x_k - f w ≤ C b ^ k` and
`‖x_k - w‖ ≤ C b ^ k` for every `k`. -/
theorem exists_le_mul_pow_of_isLojasiewiczAt_of_le_sub (hη : 0 < η) (hκ : 0 < κ)
    (ha : a ≤ 1 / 2) {w x : E} (hw : IsLojasiewiczAt f w a) (hfc : ContinuousAt f w)
    (hdec : ∀ k, f ((gradientStep f η)^[k + 1] x) ≤
      f ((gradientStep f η)^[k] x) - η * κ * ‖gradient f ((gradientStep f η)^[k] x)‖ ^ 2)
    (hx : Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w)) :
    ∃ C b : ℝ, 0 < b ∧ b < 1 ∧ ∀ k : ℕ, f ((gradientStep f η)^[k] x) - f w ≤ C * b ^ k ∧
      ‖(gradientStep f η)^[k] x - w‖ ≤ C * b ^ k := by
  obtain ⟨hfw, c, hc, K, hK, hev⟩ := exists_rate_estimates_of_isLojasiewiczAt_of_le_sub hη hκ
    (by norm_num : (0 : ℝ) < 1 / 2) (by norm_num) (hw.mono ha) hfc hdec hx
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
      ‖(gradientStep f η)^[k] x - w‖ ≤ C * b ^ k :=
  exists_le_mul_pow_of_isLojasiewiczAt_of_le_sub hη (by linarith) ha hw
    hf.continuous.continuousAt (iterate_succ_le_sub hf hL η x) hx

/-- **Polynomial rate from a decrease along the orbit** (`1/2 < a < 1`).  Under the
hypotheses of `exists_le_mul_pow_of_isLojasiewiczAt_of_le_sub`, with an exponent
`1/2 < a < 1` instead, for some `C` and every `k`,
`f x_k - f w ≤ C (k + 1) ^ (-1/(2a - 1))` and
`‖x_k - w‖ ≤ C (k + 1) ^ (-(1 - a)/(2a - 1))`. -/
theorem exists_le_mul_rpow_of_isLojasiewiczAt_of_le_sub (hη : 0 < η) (hκ : 0 < κ)
    (ha : 1 / 2 < a) (ha1 : a < 1) {w x : E} (hw : IsLojasiewiczAt f w a)
    (hfc : ContinuousAt f w)
    (hdec : ∀ k, f ((gradientStep f η)^[k + 1] x) ≤
      f ((gradientStep f η)^[k] x) - η * κ * ‖gradient f ((gradientStep f η)^[k] x)‖ ^ 2)
    (hx : Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w)) :
    ∃ C : ℝ, ∀ k : ℕ,
      f ((gradientStep f η)^[k] x) - f w ≤ C * ((k : ℝ) + 1) ^ (-(1 / (2 * a - 1))) ∧
        ‖(gradientStep f η)^[k] x - w‖ ≤ C * ((k : ℝ) + 1) ^ (-((1 - a) / (2 * a - 1))) := by
  obtain ⟨hfw, c, hc, K, hK, hev⟩ := exists_rate_estimates_of_isLojasiewiczAt_of_le_sub hη hκ
    (by linarith) ha1 hw hfc hdec hx
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
        ‖(gradientStep f η)^[k] x - w‖ ≤ C * ((k : ℝ) + 1) ^ (-((1 - a) / (2 * a - 1))) :=
  exists_le_mul_rpow_of_isLojasiewiczAt_of_le_sub hη (by linarith) ha ha1 hw
    hf.continuous.continuousAt (iterate_succ_le_sub hf hL η x) hx

/-- `‖x‖² / 2` on the line has the `1`-Lipschitz gradient `x`; for `η = 1/2` the
orbit `2⁻ᵏ` of `1` converges to `0`, where the Łojasiewicz inequality holds with
`a = 1/2`, and so with `a = 3/4`. -/
example := exists_rate_estimates_of_isLojasiewiczAt (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2)
  (L := 1) ((contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero)
  (by rw [gradient_half_mul_norm_sq]; exact LipschitzWith.id) (η := 1 / 2) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) isLojasiewiczAt_half_mul_norm_sq
  (tendsto_iterate_gradientStep_half_mul_norm_sq (E := ℝ) (η := 1 / 2) (by norm_num) 1)

example := exists_le_mul_pow_of_isLojasiewiczAt (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2)
  (L := 1) ((contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero)
  (by rw [gradient_half_mul_norm_sq]; exact LipschitzWith.id) (η := 1 / 2) (by norm_num)
  (by norm_num) le_rfl isLojasiewiczAt_half_mul_norm_sq
  (tendsto_iterate_gradientStep_half_mul_norm_sq (E := ℝ) (η := 1 / 2) (by norm_num) 1)

example := exists_le_mul_rpow_of_isLojasiewiczAt (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2)
  (L := 1) ((contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero)
  (by rw [gradient_half_mul_norm_sq]; exact LipschitzWith.id) (η := 1 / 2) (a := 3 / 4)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (isLojasiewiczAt_half_mul_norm_sq.mono (by norm_num))
  (tendsto_iterate_gradientStep_half_mul_norm_sq (E := ℝ) (η := 1 / 2) (by norm_num) 1)

/-- With `η = 3/2`, beyond `η L < 1`, the orbit `(-1/2)ᵏ` of `1` lowers `‖x‖² / 2`
by `η κ ‖∇f‖²` at every step, `κ = 1 - η / 2 = 1/4`, and converges to `0`. -/
example := exists_le_mul_pow_of_isLojasiewiczAt_of_le_sub
  (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2) (η := 3 / 2) (κ := 1 - 3 / 2 / 2) (by norm_num)
  (by norm_num) le_rfl isLojasiewiczAt_half_mul_norm_sq (by fun_prop) (x := 1)
  (fun _ => by rw [Function.iterate_succ_apply']; exact (half_mul_norm_sq_gradientStep _ _).le)
  (tendsto_iterate_gradientStep_half_mul_norm_sq (E := ℝ) (η := 3 / 2) (by norm_num) 1)

example := exists_le_mul_rpow_of_isLojasiewiczAt_of_le_sub
  (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2) (η := 3 / 2) (κ := 1 - 3 / 2 / 2) (a := 3 / 4)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (isLojasiewiczAt_half_mul_norm_sq.mono (by norm_num)) (by fun_prop) (x := 1)
  (fun _ => by rw [Function.iterate_succ_apply']; exact (half_mul_norm_sq_gradientStep _ _).le)
  (tendsto_iterate_gradientStep_half_mul_norm_sq (E := ℝ) (η := 3 / 2) (by norm_num) 1)

end Transformer.Landscape.SaddleAvoidance

module

public import Transformer.Grokking.Protocol.AdamGrowth

/-!
# How far the author's Adam and AdamW runs can move a parameter

Source: openai/grok at 3d64b1d8, `grok/training.py`, `CustomAdamW.step`;
`experiments/grok_division/PROTOCOL.md`, *The optimizer*; the constants of arXiv:2201.02177v1
(Power et al.), appendix *Additional experimental details*: learning rate `10⁻³`, `β₁ = 0.9`,
`β₂ = 0.98`, weight decay `1` or none.

Consequences of `abs_adamRun_step_le_rate` of `AdamGrowth`, for every gradient oracle:

* without weight decay a parameter is at most `t · rate · stepBound` from its initial value
  after `t` updates (`abs_adamRun_params_sub_le`): the parameters may grow, at most linearly;
* with weight decay `wd > 0` and `rate · wd ≤ 1` a parameter stays within
  `max(|θ₀|, stepBound / wd)` of `0` (`abs_adamRun_params_le`).

For the source's constants `momentRatio = 49/17` and `stepBound < 17`: the Adam run of
Figure 1 moves a parameter by less than `0.017 t` in `t` updates, and under AdamW with weight
decay `1` a parameter that starts in `[-17, 17]` stays there. Nothing here bounds the Adam
run's parameters uniformly in `t`, and no such bound is assumed anywhere.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

variable {ι : Type*} {h : Hyper} {grad : ℕ → (ι → ℝ) → ι → ℝ} {θ₀ : ι → ℝ}

/-- **Without weight decay a parameter moves at most linearly**: after `t` updates it is at
most `t · rate · stepBound` from its initial value, whatever the gradients. -/
theorem abs_adamRun_params_sub_le (hr : 0 ≤ h.rate) (h1 : 0 ≤ h.beta1) (h1' : h.beta1 < 1)
    (h2 : 0 ≤ h.beta2) (h2' : h.beta2 < 1) (h12 : h.beta1 ^ 2 < h.beta2) (he : 0 < h.eps)
    (hd : h.decay = 0) (i : ι) (t : ℕ) :
    |(adamRun h grad θ₀ t).params i - θ₀ i| ≤ t * (h.rate * h.stepBound) := by
  induction t with
  | zero => simp [adamRun]
  | succ t ih =>
    have hstep := abs_adamRun_step_le_rate (grad := grad) (θ₀ := θ₀) hr h1 h1' h2 h2' h12 he i t
    rw [hd, mul_zero, sub_zero, one_mul] at hstep
    have hrate := mul_le_mul_of_nonneg_right (h.rateAt_le hr (t + 1)) (h.stepBound_nonneg h1')
    have htri := abs_sub_le ((adamRun h grad θ₀ (t + 1)).params i)
      ((adamRun h grad θ₀ t).params i) (θ₀ i)
    push_cast
    linarith

/-- **With weight decay a parameter stays in a fixed interval**: within
`max(|θ₀|, stepBound / wd)` of `0`, whatever the gradients. -/
theorem abs_adamRun_params_le (hr : 0 ≤ h.rate) (h1 : 0 ≤ h.beta1) (h1' : h.beta1 < 1)
    (h2 : 0 ≤ h.beta2) (h2' : h.beta2 < 1) (h12 : h.beta1 ^ 2 < h.beta2) (he : 0 < h.eps)
    (hd : 0 < h.decay) (hrd : h.rate * h.decay ≤ 1) (i : ι) (t : ℕ) :
    |(adamRun h grad θ₀ t).params i| ≤ max |θ₀ i| (h.stepBound / h.decay) := by
  induction t with
  | zero => exact le_max_left _ _
  | succ t ih =>
    have hstep := abs_adamRun_step_le_rate (grad := grad) (θ₀ := θ₀) hr h1 h1' h2 h2' h12 he i t
    have hlr := h.rateAt_nonneg hr (t + 1)
    have ha : 0 ≤ 1 - h.rateAt (t + 1) * h.decay :=
      sub_nonneg.2 ((mul_le_mul_of_nonneg_right (h.rateAt_le hr (t + 1)) hd.le).trans hrd)
    have hB : h.stepBound ≤ h.decay * max |θ₀ i| (h.stepBound / h.decay) := by
      rw [← div_le_iff₀' hd]
      exact le_max_right _ _
    have hsplit : |(adamRun h grad θ₀ (t + 1)).params i| ≤
        (1 - h.rateAt (t + 1) * h.decay) * |(adamRun h grad θ₀ t).params i| +
          h.rateAt (t + 1) * h.stepBound := by
      have := abs_sub_abs_le_abs_sub ((adamRun h grad θ₀ (t + 1)).params i)
        ((1 - h.rateAt (t + 1) * h.decay) * (adamRun h grad θ₀ t).params i)
      rw [abs_mul, abs_of_nonneg ha] at this
      linarith
    nlinarith [mul_le_mul_of_nonneg_left ih ha, mul_le_mul_of_nonneg_left hB hlr]

/-! The witnesses: the source's constants. `momentRatio = 49/17`, and `stepBound < 17` since
`49/17 < 2.89 = 1.7²`. -/

lemma momentRatio_source (decay : ℝ) : (Hyper.source decay).momentRatio = 49 / 17 := by
  norm_num [Hyper.momentRatio, Hyper.source]

lemma stepBound_source_lt (decay : ℝ) : (Hyper.source decay).stepBound < 17 := by
  have hroot : √(49 / 17 : ℝ) < 17 / 10 := by
    rw [Real.sqrt_lt' (by norm_num)]
    norm_num
  have hb : (1 : ℝ) - (Hyper.source decay).beta1 = 1 / 10 := by norm_num [Hyper.source]
  rw [Hyper.stepBound, momentRatio_source, hb, div_lt_iff₀ (by norm_num)]
  linarith

example (grad : ℕ → (Unit → ℝ) → Unit → ℝ) (t : ℕ) :=
  sq_adamRun_first_le (h := Hyper.source 0) (grad := grad) (θ₀ := 0) (by norm_num [Hyper.source])
    (by norm_num [Hyper.source]) (by norm_num [Hyper.source]) t ()

example (grad : ℕ → (Unit → ℝ) → Unit → ℝ) (t : ℕ) :=
  abs_adamRun_step_le_rate (h := Hyper.source 1) (grad := grad) (θ₀ := 0)
    (by norm_num [Hyper.source]) (by norm_num [Hyper.source]) (by norm_num [Hyper.source])
    (by norm_num [Hyper.source]) (by norm_num [Hyper.source]) (by norm_num [Hyper.source])
    (by norm_num [Hyper.source]) () t

/-- The Adam run of Figure 1: after `t` updates a parameter is less than `0.017 t` from its
initial value, here in the form `t · (10⁻³ · stepBound)`. -/
example (grad : ℕ → (Unit → ℝ) → Unit → ℝ) (t : ℕ) :
    |(adamRun (Hyper.source 0) grad 0 t).params () - 0| ≤
      t * ((Hyper.source 0).rate * (Hyper.source 0).stepBound) :=
  abs_adamRun_params_sub_le (by norm_num [Hyper.source]) (by norm_num [Hyper.source])
    (by norm_num [Hyper.source]) (by norm_num [Hyper.source]) (by norm_num [Hyper.source])
    (by norm_num [Hyper.source]) (by norm_num [Hyper.source]) rfl () t

/-- The AdamW run from `0` keeps its parameter within `stepBound`, which is less than `17`. -/
example (grad : ℕ → (Unit → ℝ) → Unit → ℝ) (t : ℕ) :
    |(adamRun (Hyper.source 1) grad 0 t).params ()| ≤
      max |(0 : Unit → ℝ) ()| ((Hyper.source 1).stepBound / (Hyper.source 1).decay) :=
  abs_adamRun_params_le (by norm_num [Hyper.source]) (by norm_num [Hyper.source])
    (by norm_num [Hyper.source]) (by norm_num [Hyper.source]) (by norm_num [Hyper.source])
    (by norm_num [Hyper.source]) (by norm_num [Hyper.source]) (by norm_num [Hyper.source])
    (by norm_num [Hyper.source]) () t

example := Hyper.momentRatio_nonneg (h := Hyper.source 0) (by norm_num [Hyper.source])
  (by norm_num [Hyper.source]) (by norm_num [Hyper.source])

example := Hyper.stepBound_nonneg (h := Hyper.source 0) (by norm_num [Hyper.source])

/-- One update keeps `m² ≤ K v`, at the source's `β` and the values `m = 1`, `v = 1`, `g = 2`. -/
example := sq_moment_step_le (β₁ := 9 / 10) (β₂ := 49 / 50) (K := 49 / 17) (m := 1) (v := 1)
  (g := 2) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.Protocol

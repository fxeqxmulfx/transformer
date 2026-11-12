module

public import Transformer.Grokking.Protocol.AdamBounds
public import Transformer.Grokking.Protocol.AdamLimitWitness

/-!
# A coordinate that converges under Adam without weight decay sees gradients that tend to zero

Source: openai/grok at 3d64b1d8, `grok/training.py`, `CustomAdamW.step`;
`experiments/grok_division/PROTOCOL.md`, *The optimizer*. arXiv:2201.02177v1 (Power et al.),
appendix *Additional experimental details*: the experiment of Figure 1 "used Adam optimizer
with no weight decay".

`AdamLimit` describes the limit of a run that sees one continuous gradient field at every
update.
A run of the protocol sees another batch at every update, and what the backward pass returns
is not continuous in the parameters. This file assumes neither.

Without weight decay the update of a coordinate is `− lr / (1 − β₁ᵗ) · m / (√v̂ + ε)`. If the
gradients seen in the coordinate are at most `G` in absolute value, the denominator is at most
`G + ε` (`adamRun_second_le`), so after warmup

    |rate| · |m_t| ≤ (G + ε) · |θ_{t−1} − θ_t|

(`abs_adamRun_first_mul_rate_le`). Hence if the coordinate converges, its first moment tends
to `0` (`tendsto_adamRun_first_zero`), and since `g_t = (m_t − β₁ m_{t−1}) / (1 − β₁)` so do
the gradients that the run sees in it (`tendsto_adamRun_grad_zero`), whatever fields the
updates see and whatever the other coordinates do.

With weight decay the argument does not apply: the update has the further term
`− lr · wd · θ`, and for one field `AdamLimit` gives `wd · θ = − g / (|g| + ε)` at a limit,
which does not make `g` vanish. Nothing here says that a run converges.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology

variable {ι : Type*} {h : Hyper} {grad : ℕ → (ι → ℝ) → ι → ℝ} {θ₀ : ι → ℝ} {i : ι} {a G : ℝ}

/-- **After warmup and without weight decay the first moment of a coordinate is bounded by the
step of the coordinate**: `|rate| · |m_{t+1}| ≤ (G + ε) · |θ_t − θ_{t+1}|` when the gradients
seen in it so far are at most `G` in absolute value. -/
theorem abs_adamRun_first_mul_rate_le (hh : h.Regular) (hd : h.decay = 0) {t : ℕ}
    (ht : h.warmup < t) (hG : ∀ s < t + 1, |grad (s + 1) (adamRun h grad θ₀ s).params i| ≤ G) :
    |(adamRun h grad θ₀ (t + 1)).first i| * |h.rate| ≤ (G + h.eps) *
      |(adamRun h grad θ₀ t).params i - (adamRun h grad θ₀ (t + 1)).params i| := by
  obtain ⟨-, h1, h1', h2, h2', he⟩ := hh
  have hG0 : 0 ≤ G := (abs_nonneg _).trans (hG 0 (Nat.succ_pos t))
  have hc := one_sub_pow_pos h1 h1' (Nat.succ_le_succ (Nat.zero_le t))
  have hc1 : 1 - h.beta1 ^ (t + 1) ≤ 1 := sub_le_self 1 (pow_nonneg h1 _)
  have hc2 := one_sub_pow_pos h2 h2' (Nat.succ_le_succ (Nat.zero_le t))
  have hDpos := adamDenominator_pos he ((adamRun h grad θ₀ (t + 1)).second i) (t + 1)
  have hDle : √((adamRun h grad θ₀ (t + 1)).second i) / √(1 - h.beta2 ^ (t + 1)) + h.eps ≤
      G + h.eps := by
    have hv := adamRun_second_le (G := G) h2 h2'.le i (t + 1) hG
    have h3 : √((adamRun h grad θ₀ (t + 1)).second i) ≤ √(1 - h.beta2 ^ (t + 1)) * G :=
      (Real.sqrt_le_sqrt hv).trans_eq (by rw [Real.sqrt_mul hc2.le, Real.sqrt_sq hG0])
    linarith [(div_le_iff₀' (Real.sqrt_pos.2 hc2)).2 h3]
  have hstep : (adamRun h grad θ₀ t).params i - (adamRun h grad θ₀ (t + 1)).params i =
      h.rate / (1 - h.beta1 ^ (t + 1)) * ((adamRun h grad θ₀ (t + 1)).first i /
        (√((adamRun h grad θ₀ (t + 1)).second i) / √(1 - h.beta2 ^ (t + 1)) + h.eps)) := by
    rw [adamRun_succ_params, hd, Hyper.rateAt_of_lt (by simpa using ht)]
    ring
  rw [hstep]
  generalize √((adamRun h grad θ₀ (t + 1)).second i) / √(1 - h.beta2 ^ (t + 1)) + h.eps = D
    at hDpos hDle
  generalize (adamRun h grad θ₀ (t + 1)).first i = m
  generalize 1 - h.beta1 ^ (t + 1) = c at hc hc1
  have hcD : c * D ≤ G + h.eps := (mul_le_of_le_one_left hDpos.le hc1).trans hDle
  have hcDpos : 0 < c * D := mul_pos hc hDpos
  have key : (G + h.eps) * |h.rate / c * (m / D)| = |m| * |h.rate| * ((G + h.eps) / (c * D)) := by
    rw [abs_mul, abs_div, abs_div, abs_of_pos hc, abs_of_pos hDpos]
    field_simp
  rw [key]
  exact le_mul_of_one_le_right (mul_nonneg (abs_nonneg _) (abs_nonneg _))
    ((one_le_div hcDpos).2 hcD)

/-- **The first moment of a coordinate that converges tends to zero**, without weight decay
and with bounded gradients seen in the coordinate. -/
theorem tendsto_adamRun_first_zero (hh : h.Regular) (hd : h.decay = 0)
    (hlim : Tendsto (fun t => (adamRun h grad θ₀ t).params i) atTop (𝓝 a))
    (hG : ∀ t, |grad (t + 1) (adamRun h grad θ₀ t).params i| ≤ G) :
    Tendsto (fun t => (adamRun h grad θ₀ t).first i) atTop (𝓝 0) := by
  have hr : 0 < |h.rate| := abs_pos.2 hh.1
  have hstep : Tendsto (fun t => (G + h.eps) / |h.rate| *
      |(adamRun h grad θ₀ t).params i - (adamRun h grad θ₀ (t + 1)).params i|) atTop (𝓝 0) := by
    simpa using
      (hlim.sub ((tendsto_add_atTop_iff_nat 1).2 hlim)).abs.const_mul ((G + h.eps) / |h.rate|)
  have hsq : Tendsto (fun t => |(adamRun h grad θ₀ (t + 1)).first i|) atTop (𝓝 0) :=
    squeeze_zero' (Eventually.of_forall fun _ => abs_nonneg _)
      ((eventually_gt_atTop h.warmup).mono fun t ht => by
        rw [div_mul_eq_mul_div, le_div_iff₀ hr]
        exact abs_adamRun_first_mul_rate_le hh hd ht fun s _ => hG s) hstep
  exact (tendsto_add_atTop_iff_nat 1).1 ((tendsto_zero_iff_abs_tendsto_zero _).2 hsq)

/-- **The gradients seen in a coordinate that converges tend to zero**, without weight decay
and if they are bounded: update `t + 1` sees a gradient that tends to `0` as `t` grows,
whatever fields the updates see. -/
theorem tendsto_adamRun_grad_zero (hh : h.Regular) (hd : h.decay = 0)
    (hlim : Tendsto (fun t => (adamRun h grad θ₀ t).params i) atTop (𝓝 a))
    (hG : ∀ t, |grad (t + 1) (adamRun h grad θ₀ t).params i| ≤ G) :
    Tendsto (fun t => grad (t + 1) (adamRun h grad θ₀ t).params i) atTop (𝓝 0) := by
  have hm := tendsto_adamRun_first_zero hh hd hlim hG
  have h1 : 1 - h.beta1 ≠ 0 := (sub_pos.2 hh.2.2.1).ne'
  have key : ∀ t, grad (t + 1) (adamRun h grad θ₀ t).params i =
      ((adamRun h grad θ₀ (t + 1)).first i - h.beta1 * (adamRun h grad θ₀ t).first i) /
        (1 - h.beta1) := fun t => by
    rw [adamRun_succ_first]
    field_simp
    ring
  have h0 := (((tendsto_add_atTop_iff_nat 1).2 hm).sub (hm.const_mul h.beta1)).div_const
    (1 - h.beta1)
  simp only [mul_zero, sub_zero, zero_div] at h0
  exact h0.congr fun t => (key t).symm

/-! The witnesses: the source's constants without weight decay. At the constant gradient `1`
the bound of the first moment holds after update 12; the run that stays at its start converges
and sees gradients bounded by `0`. -/

example := abs_adamRun_first_mul_rate_le (h := Hyper.source 0)
  (grad := fun _ _ (_ : Unit) => (1 : ℝ)) (θ₀ := 0) (i := ()) (G := 1) (t := 11)
  (Hyper.source_regular 0) rfl (by norm_num [Hyper.source]) fun _ _ => by simp

example : Tendsto (fun t => (adamRun (Hyper.source 0) (fun _ _ => 0) (0 : Unit → ℝ) t).first ())
    atTop (𝓝 0) :=
  tendsto_adamRun_first_zero (a := 0) (G := 0) (Hyper.source_regular 0) rfl
    (by simp [adamRun_zero]) fun t => by simp

example := tendsto_adamRun_grad_zero (h := Hyper.source 0) (grad := fun _ _ => 0)
  (θ₀ := (0 : Unit → ℝ)) (i := ()) (a := 0) (G := 0) (Hyper.source_regular 0) rfl
  (by simp [adamRun_zero]) fun t => by simp

end Transformer.Grokking.Protocol

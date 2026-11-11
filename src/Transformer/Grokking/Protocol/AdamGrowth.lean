module

public import Transformer.Grokking.Protocol.AdamBounds

/-!
# Bounds on the author's Adam run that do not look at the gradients

Source: openai/grok at 3d64b1d8, `grok/training.py`, `CustomAdamW.step`;
`experiments/grok_division/PROTOCOL.md`, *The optimizer*; the constants `β₁ = 0.9`,
`β₂ = 0.98` of arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*.
The run is `adamRun` of `Transformer.Grokking.Protocol.Adam`.

For `β₁² < β₂ < 1` the two moments of a parameter satisfy `m_t² ≤ K v_t` at every update,
whatever the gradients were (`sq_adamRun_first_le`), with
`K = (1 - β₁)² β₂ / ((1 - β₂)(β₂ - β₁²))` (`Hyper.momentRatio`). So an update moves a decayed
parameter by at most `lr_t · √K / (1 - β₁)` (`abs_adamRun_step_le_rate`, `Hyper.stepBound`).
`AdamBox` draws the consequences for the parameters of a run and evaluates the constants.

The proof of the first bound is one inequality for one update (`sq_moment_step_le`): if
`m² ≤ K v` then `(β₁ m + (1 - β₁) g)² ≤ K (β₂ v + (1 - β₂) g²)` for every `g`, the difference
being `β₂ (K v - m²)` plus a square over `β₂ - β₁²`. These are bounds on the moments and the
parameters of a run. They say nothing of its loss, its predictions or its limit.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

variable {ι : Type*} {h : Hyper} {grad : ℕ → (ι → ℝ) → ι → ℝ} {θ₀ : ι → ℝ}

/-- The bound of `m² / v`: `(1 - β₁)² β₂ / ((1 - β₂)(β₂ - β₁²))`. -/
noncomputable def Hyper.momentRatio (h : Hyper) : ℝ :=
  (1 - h.beta1) ^ 2 * h.beta2 / ((1 - h.beta2) * (h.beta2 - h.beta1 ^ 2))

/-- The bound of an update in units of its learning rate: `√K / (1 - β₁)`. -/
noncomputable def Hyper.stepBound (h : Hyper) : ℝ := √h.momentRatio / (1 - h.beta1)

lemma Hyper.momentRatio_nonneg (h2 : 0 ≤ h.beta2) (h2' : h.beta2 < 1)
    (h12 : h.beta1 ^ 2 < h.beta2) : 0 ≤ h.momentRatio :=
  div_nonneg (mul_nonneg (sq_nonneg _) h2)
    (mul_nonneg (sub_nonneg.2 h2'.le) (sub_nonneg.2 h12.le))

lemma Hyper.stepBound_nonneg (h1' : h.beta1 < 1) : 0 ≤ h.stepBound :=
  div_nonneg (Real.sqrt_nonneg _) (sub_nonneg.2 h1'.le)

/-- One update keeps `m² ≤ K v`. -/
lemma sq_moment_step_le {β₁ β₂ K m v g : ℝ} (h2 : 0 ≤ β₂) (h2' : β₂ < 1) (h12 : β₁ ^ 2 < β₂)
    (hK : K = (1 - β₁) ^ 2 * β₂ / ((1 - β₂) * (β₂ - β₁ ^ 2))) (hm : m ^ 2 ≤ K * v) :
    (β₁ * m + (1 - β₁) * g) ^ 2 ≤ K * (β₂ * v + (1 - β₂) * g ^ 2) := by
  have hδ : 0 < β₂ - β₁ ^ 2 := sub_pos.2 h12
  have hδ' := hδ.ne'
  have h1 : (1 - β₂) ≠ 0 := (sub_pos.2 h2').ne'
  have key : K * (β₂ * v + (1 - β₂) * g ^ 2) - (β₁ * m + (1 - β₁) * g) ^ 2 =
      β₂ * (K * v - m ^ 2) + ((β₂ - β₁ ^ 2) * m - β₁ * (1 - β₁) * g) ^ 2 / (β₂ - β₁ ^ 2) := by
    rw [hK]
    field_simp
    ring
  have hnn : 0 ≤ β₂ * (K * v - m ^ 2) +
      ((β₂ - β₁ ^ 2) * m - β₁ * (1 - β₁) * g) ^ 2 / (β₂ - β₁ ^ 2) :=
    add_nonneg (mul_nonneg h2 (sub_nonneg.2 hm)) (div_nonneg (sq_nonneg _) hδ.le)
  linarith

/-- **The first moment is controlled by the second, whatever the gradients**:
`m_t² ≤ K v_t`. -/
theorem sq_adamRun_first_le (h2 : 0 ≤ h.beta2) (h2' : h.beta2 < 1) (h12 : h.beta1 ^ 2 < h.beta2)
    (t : ℕ) (i : ι) :
    (adamRun h grad θ₀ t).first i ^ 2 ≤ h.momentRatio * (adamRun h grad θ₀ t).second i := by
  induction t with
  | zero => simp [adamRun]
  | succ t ih => exact sq_moment_step_le h2 h2' h12 rfl ih

/-- **An update moves a decayed parameter by at most `lr_t √K / (1 - β₁)`**, whatever the
gradients. -/
theorem abs_adamRun_step_le_rate (hr : 0 ≤ h.rate) (h1 : 0 ≤ h.beta1) (h1' : h.beta1 < 1)
    (h2 : 0 ≤ h.beta2) (h2' : h.beta2 < 1) (h12 : h.beta1 ^ 2 < h.beta2) (he : 0 < h.eps)
    (i : ι) (t : ℕ) :
    |(adamRun h grad θ₀ (t + 1)).params i -
        (1 - h.rateAt (t + 1) * h.decay) * (adamRun h grad θ₀ t).params i| ≤
      h.rateAt (t + 1) * h.stepBound := by
  have hc := one_sub_pow_pos h1 h1' (Nat.succ_le_succ (Nat.zero_le t))
  have hc2 := one_sub_pow_pos h2 h2' (Nat.succ_le_succ (Nat.zero_le t))
  have hlr := h.rateAt_nonneg hr (t + 1)
  have hK := h.momentRatio_nonneg h2 h2' h12
  have hmv := sq_adamRun_first_le (grad := grad) (θ₀ := θ₀) h2 h2' h12 (t + 1) i
  have hv := adamRun_second_nonneg (grad := grad) (θ₀ := θ₀) h2 h2'.le (t + 1) i
  set m := (adamRun h grad θ₀ (t + 1)).first i with hmdef
  set v := (adamRun h grad θ₀ (t + 1)).second i with hvdef
  have hden := adamDenominator_pos he v (t + 1)
  have hstep : (adamRun h grad θ₀ (t + 1)).params i -
      (1 - h.rateAt (t + 1) * h.decay) * (adamRun h grad θ₀ t).params i =
      -(h.rateAt (t + 1) / (1 - h.beta1 ^ (t + 1)) *
        (m / (√v / √(1 - h.beta2 ^ (t + 1)) + h.eps))) := by
    simp only [hmdef, hvdef, adamRun, adamStep]
    ring
  have hm : |m| ≤ √h.momentRatio * √v := by
    rw [← Real.sqrt_mul hK, ← Real.sqrt_sq_eq_abs]
    exact Real.sqrt_le_sqrt hmv
  have hroot : √v ≤ √v / √(1 - h.beta2 ^ (t + 1)) + h.eps := by
    have hle : √(1 - h.beta2 ^ (t + 1)) ≤ 1 :=
      Real.sqrt_le_one.2 (sub_le_self _ (pow_nonneg h2 _))
    have := div_le_div_of_nonneg_left (Real.sqrt_nonneg v) (Real.sqrt_pos.2 hc2) hle
    rw [div_one] at this
    linarith
  have hquot : |m| / (√v / √(1 - h.beta2 ^ (t + 1)) + h.eps) ≤ √h.momentRatio := by
    rw [div_le_iff₀ hden]
    exact hm.trans (mul_le_mul_of_nonneg_left hroot (Real.sqrt_nonneg _))
  have hb : 1 - h.beta1 ≤ 1 - h.beta1 ^ (t + 1) :=
    sub_le_sub_left (pow_le_of_le_one h1 h1'.le (Nat.succ_ne_zero t)) 1
  rw [hstep, abs_neg, abs_mul, abs_div, abs_div, abs_of_nonneg hlr, abs_of_pos hc, abs_of_pos hden,
    Hyper.stepBound, div_mul_eq_mul_div, mul_div_assoc]
  refine mul_le_mul_of_nonneg_left ?_ hlr
  exact div_le_div₀ (Real.sqrt_nonneg _) hquot (sub_pos.2 h1') hb

end Transformer.Grokking.Protocol

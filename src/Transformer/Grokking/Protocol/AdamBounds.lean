module

public import Transformer.Grokking.Protocol.Adam

/-!
# Bounds on the moments and the step of the author's Adam run

Source: openai/grok at 3d64b1d8, `grok/training.py`, `CustomAdamW.step`;
`experiments/grok_division/PROTOCOL.md`, *The optimizer*. The run is `adamRun` of
`Transformer.Grokking.Protocol.Adam`.

If the gradients seen up to update `t` are at most `G` in absolute value, then
`|m_t| ≤ (1 - β₁ᵗ) G` and `v_t ≤ (1 - β₂ᵗ) G²` (`abs_adamRun_first_le`, `adamRun_second_le`):
the bias-corrected moments are at most `G` and `G²`. The update then moves a decayed parameter
by at most `lr_t G / ε` (`abs_adamRun_step_le`). The bounds use a finite history of the run and
say nothing of its limit.

The witness is the run of the source's AdamW constants on one parameter at the constant
gradient `1`.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

variable {ι : Type*} {h : Hyper} {grad : ℕ → (ι → ℝ) → ι → ℝ} {θ₀ : ι → ℝ} {G : ℝ}

/-- **The first moment is at most `(1 - β₁ᵗ) G`** when the gradients seen so far are at most
`G` in absolute value. -/
theorem abs_adamRun_first_le (h1 : 0 ≤ h.beta1) (h1' : h.beta1 ≤ 1) (i : ι) (t : ℕ)
    (hg : ∀ s < t, |grad (s + 1) (adamRun h grad θ₀ s).params i| ≤ G) :
    |(adamRun h grad θ₀ t).first i| ≤ (1 - h.beta1 ^ t) * G := by
  induction t with
  | zero => simp [adamRun]
  | succ t ih =>
    have ih' := ih fun s hs => hg s (Nat.lt_succ_of_lt hs)
    have hgt := hg t (Nat.lt_succ_self t)
    change |h.beta1 * (adamRun h grad θ₀ t).first i + (1 - h.beta1) * _| ≤ _
    calc _ ≤ h.beta1 * |(adamRun h grad θ₀ t).first i| + (1 - h.beta1) * G := by
          refine (abs_add_le _ _).trans (add_le_add ?_ ?_)
          · rw [abs_mul, abs_of_nonneg h1]
          · rw [abs_mul, abs_of_nonneg (sub_nonneg.2 h1')]
            exact mul_le_mul_of_nonneg_left hgt (sub_nonneg.2 h1')
      _ ≤ h.beta1 * ((1 - h.beta1 ^ t) * G) + (1 - h.beta1) * G := by gcongr
      _ = (1 - h.beta1 ^ (t + 1)) * G := by ring

/-- **The second moment is at most `(1 - β₂ᵗ) G²`** under the same bound. -/
theorem adamRun_second_le (h2 : 0 ≤ h.beta2) (h2' : h.beta2 ≤ 1) (i : ι) (t : ℕ)
    (hg : ∀ s < t, |grad (s + 1) (adamRun h grad θ₀ s).params i| ≤ G) :
    (adamRun h grad θ₀ t).second i ≤ (1 - h.beta2 ^ t) * G ^ 2 := by
  induction t with
  | zero => simp [adamRun]
  | succ t ih =>
    have ih' := ih fun s hs => hg s (Nat.lt_succ_of_lt hs)
    have hgt : grad (t + 1) (adamRun h grad θ₀ t).params i ^ 2 ≤ G ^ 2 :=
      sq_le_sq' (neg_le_of_abs_le (hg t (Nat.lt_succ_self t)))
        (le_of_abs_le (hg t (Nat.lt_succ_self t)))
    change h.beta2 * (adamRun h grad θ₀ t).second i + (1 - h.beta2) * _ ≤ _
    calc _ ≤ h.beta2 * ((1 - h.beta2 ^ t) * G ^ 2) + (1 - h.beta2) * G ^ 2 :=
          add_le_add (mul_le_mul_of_nonneg_left ih' h2)
            (mul_le_mul_of_nonneg_left hgt (sub_nonneg.2 h2'))
      _ = (1 - h.beta2 ^ (t + 1)) * G ^ 2 := by ring

/-- **An update moves a decayed parameter by at most `lr_t G / ε`** when the gradients seen up
to it are at most `G` in absolute value. -/
theorem abs_adamRun_step_le (hr : 0 ≤ h.rate) (h1 : 0 ≤ h.beta1) (h1' : h.beta1 < 1)
    (he : 0 < h.eps) (i : ι) (t : ℕ)
    (hg : ∀ s < t + 1, |grad (s + 1) (adamRun h grad θ₀ s).params i| ≤ G) :
    |(adamRun h grad θ₀ (t + 1)).params i -
        (1 - h.rateAt (t + 1) * h.decay) * (adamRun h grad θ₀ t).params i| ≤
      h.rateAt (t + 1) * G / h.eps := by
  have hc := one_sub_pow_pos h1 h1' (Nat.succ_le_succ (Nat.zero_le t))
  have hm := abs_adamRun_first_le h1 h1'.le i (t + 1) hg
  have hlr := h.rateAt_nonneg hr (t + 1)
  have hG : 0 ≤ G := (abs_nonneg _).trans (hg 0 (Nat.succ_pos t))
  set m := (adamRun h grad θ₀ (t + 1)).first i with hmdef
  set v := (adamRun h grad θ₀ (t + 1)).second i with hvdef
  have hden := adamDenominator_pos he v (t + 1)
  have hstep : (adamRun h grad θ₀ (t + 1)).params i -
      (1 - h.rateAt (t + 1) * h.decay) * (adamRun h grad θ₀ t).params i =
      -(h.rateAt (t + 1) / (1 - h.beta1 ^ (t + 1)) *
        (m / (√v / √(1 - h.beta2 ^ (t + 1)) + h.eps))) := by
    simp only [hmdef, hvdef, adamRun, adamStep]
    ring
  rw [hstep, abs_neg, abs_mul, abs_div, abs_div, abs_of_nonneg hlr, abs_of_pos hc, abs_of_pos hden]
  calc h.rateAt (t + 1) / (1 - h.beta1 ^ (t + 1)) * (|m| / (√v / √(1 - h.beta2 ^ (t + 1)) + h.eps))
      ≤ h.rateAt (t + 1) / (1 - h.beta1 ^ (t + 1)) * ((1 - h.beta1 ^ (t + 1)) * G / h.eps) := by
        refine mul_le_mul_of_nonneg_left ?_ (div_nonneg hlr hc.le)
        exact div_le_div₀ (mul_nonneg hc.le hG) hm he
          (le_add_of_nonneg_left (div_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)))
    _ = h.rateAt (t + 1) * G / h.eps := by
        field_simp

/-! The witness: the source's AdamW constants, one parameter, the constant gradient `1`. -/

example (t : ℕ) : |(adamRun (Hyper.source 1) (fun _ _ (_ : Unit) => (1 : ℝ)) 0 t).first ()| ≤
    (1 - (9 / 10 : ℝ) ^ t) * 1 :=
  abs_adamRun_first_le (h := Hyper.source 1) (by norm_num [Hyper.source])
    (by norm_num [Hyper.source]) () t fun _ _ => by simp

example (t : ℕ) : (adamRun (Hyper.source 1) (fun _ _ (_ : Unit) => (1 : ℝ)) 0 t).second () ≤
    (1 - (49 / 50 : ℝ) ^ t) * 1 ^ 2 :=
  adamRun_second_le (h := Hyper.source 1) (by norm_num [Hyper.source])
    (by norm_num [Hyper.source]) () t fun _ _ => by simp

/-- After warmup every update of that run moves the decayed parameter by at most
`10⁻³ · 1 / 10⁻⁸ = 10⁵`. -/
example (t : ℕ) : |(adamRun (Hyper.source 1) (fun _ _ (_ : Unit) => (1 : ℝ)) 0 (t + 1)).params () -
    (1 - (Hyper.source 1).rateAt (t + 1) * 1) *
      (adamRun (Hyper.source 1) (fun _ _ (_ : Unit) => (1 : ℝ)) 0 t).params ()| ≤
    (Hyper.source 1).rateAt (t + 1) * 1 / (1 / 10 ^ 8) :=
  abs_adamRun_step_le (h := Hyper.source 1) (by norm_num [Hyper.source])
    (by norm_num [Hyper.source]) (by norm_num [Hyper.source]) (by norm_num [Hyper.source]) () t
    fun _ _ => by simp

end Transformer.Grokking.Protocol

module

public import Transformer.CRASP.GramsPad

/-!
# Positional logits read the last tokens: ALiBi, KERPLE, SSMax, and the scale of the logit

In no paper. The construction of `Transformer.CRASP.Grams` needs `ReadsPrev L b` of its logit `L`,
with the dot product `b` between differing grams. It holds for
* ALiBi (arXiv:2506.16055v3, `eq:alibi`) of the slope `b`, the logit divided by `c` with
  `0 < c ≤ b` (`readsPrev_scale_alibi`). Dividing the logit by `c` divides the slope by `c` too, so
  these are also ALiBi of the slope `b / c` added to the dot product divided by `c`;
* KERPLE (arXiv:2205.09921v2, `eq:krpe`, §4 Practical Choice), whose logit is
  `(q·k + k̃(m, n)) / √d`, under the future mask: the logarithmic kernel
  `k̃ = −r₁ log(1 + r₂|m − n|)` with `r₁ = b / log 2` and `r₂ = 1`, the logit divided by `c` with
  `0 < 2c ≤ b` (`readsPrev_scale_kerpleLog`); the power kernel `k̃ = −r₁|m − n|^{r₂}` with `r₁ = b`
  and any `r₂ > 0`, the logit divided by `c` with `0 < c ≤ b` and `c ≤ b(2^{r₂} − 1)`
  (`readsPrev_scale_kerplePower`);
* SSMax (arXiv:2501.19399v1, `eq:ssmax`) of any logit that meets `ReadsPrev`, when `s log 2 ≥ 1`
  (`readsPrev_ssmax`): the query at the position `i` sees `i + 1` keys, and SSMax multiplies its
  logits by `s log(i + 1)`, which is `0` at the first position and at least `1` after it.

The divisor `c` is `1` in `def:transformer` and `eq:alibi`, `√d` in `eq:krpe` and in standard
attention, and `d` in a third variant (`Transformer.CRASP.scaleLogit`). With `√d` or `d` the dot
product `b` grows with the width `d`; `Transformer.CRASP.Grams.exists_recognizes` absorbs it into
the precision.
-/

@[expose] public section

namespace Transformer.CRASP.Grams

/-- **Scaled ALiBi**: ALiBi of the slope `b`, the logit divided by `c` with `0 < c ≤ b`, meets
`ReadsPrev` with the dot product `b`. -/
theorem readsPrev_scale_alibi {b : ℕ} {c : ℝ} (hc : 0 < c) (hcb : c ≤ b) :
    ReadsPrev (scaleLogit c (alibiLogit b)) b where
  self i := by simp [scaleLogit, alibiLogit]
  prev i hi := by
    simp only [scaleLogit, alibiLogit]
    rw [Nat.cast_sub (by omega : 1 ≤ i), Nat.cast_one, sub_sub_cancel, mul_one, sub_self,
      zero_div]
  prevSame i hi := by
    simp only [scaleLogit, alibiLogit]
    rw [Nat.cast_sub (by omega : 1 ≤ i), Nat.cast_one, sub_sub_cancel, mul_one, div_le_iff₀ hc]
    linarith
  far i j hij := by
    have h2 : (2 : ℝ) ≤ (i : ℝ) - j := by
      have : (j : ℝ) + 2 ≤ i := by exact_mod_cast hij
      linarith
    have h3 : (b : ℝ) * 2 ≤ b * ((i : ℝ) - j) := mul_le_mul_of_nonneg_left h2 (Nat.cast_nonneg b)
    simp only [scaleLogit, alibiLogit]
    constructor <;> rw [div_le_iff₀ hc] <;> linarith
  run i := Or.inl fun j hj => by
    have h0 : (0 : ℝ) ≤ (i : ℝ) - j := by
      have : (j : ℝ) ≤ i := by exact_mod_cast hj
      linarith
    have h1 : (0 : ℝ) ≤ b * ((i : ℝ) - j) := mul_nonneg (Nat.cast_nonneg b) h0
    simp only [scaleLogit, alibiLogit]
    exact div_neg_of_neg_of_pos (by linarith) hc

/-- The hypotheses of `readsPrev_scale_alibi` hold for the width `4`, the divisor `√4 = 2` and the
slope `2`. -/
example : ReadsPrev (scaleLogit (Real.sqrt 4) (alibiLogit (2 : ℕ))) 2 := by
  have h : Real.sqrt 4 = 2 := by
    rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  exact readsPrev_scale_alibi (by rw [h]; norm_num) (by rw [h]; norm_num)

/-- `3 log 2 ≤ 2 log 3`, as `8 ≤ 9`. -/
theorem three_log_two_le : 3 * Real.log 2 ≤ 2 * Real.log 3 := by
  have h8 : Real.log 8 = 3 * Real.log 2 := by
    rw [show (8 : ℝ) = 2 ^ 3 by norm_num, Real.log_pow]
    norm_num
  have h9 : Real.log 9 = 2 * Real.log 3 := by
    rw [show (9 : ℝ) = 3 ^ 2 by norm_num, Real.log_pow]
    norm_num
  rw [← h8, ← h9]
  exact Real.log_le_log (by norm_num) (by norm_num)

/-- The distance from a position to the one before it is `1`. -/
theorem abs_sub_pred {i : ℕ} (hi : 0 < i) : |(i : ℝ) - ((i - 1 : ℕ) : ℝ)| = 1 := by
  rw [Nat.cast_sub (by omega : 1 ≤ i), Nat.cast_one, sub_sub_cancel, abs_one]

/-- The hypothesis of `abs_sub_pred` holds at the position `1`. -/
example : |((1 : ℕ) : ℝ) - ((1 - 1 : ℕ) : ℝ)| = 1 := abs_sub_pred one_pos

/-- **Scaled KERPLE, logarithmic**: the kernel `−(b / log 2) log(1 + |m − n|)`, the logit divided
by `c` with `0 < 2c ≤ b`, meets `ReadsPrev` with the dot product `b`. -/
theorem readsPrev_scale_kerpleLog {b : ℕ} {c : ℝ} (hc : 0 < c) (hcb : 2 * c ≤ b) :
    ReadsPrev (scaleLogit c (kerpleLogLogit (b / Real.log 2) 1)) b := by
  have hl : 0 < Real.log 2 := Real.log_pos one_lt_two
  have hr : (b : ℝ) / Real.log 2 * Real.log 2 = b := div_mul_cancel₀ _ hl.ne'
  have hr0 : 0 ≤ (b : ℝ) / Real.log 2 := div_nonneg (Nat.cast_nonneg b) hl.le
  have h2 : Real.log (1 + 1 * 1) = Real.log 2 := by norm_num
  refine ⟨fun i => ?_, fun i hi => ?_, fun i hi => ?_, fun i j hij => ?_, fun i => ?_⟩
  · simp [scaleLogit, kerpleLogLogit, KERPLE.logKernel]
  · simp only [scaleLogit, kerpleLogLogit, KERPLE.logKernel, abs_sub_pred hi, h2, neg_mul, hr,
      add_neg_cancel, zero_div]
  · simp only [scaleLogit, kerpleLogLogit, KERPLE.logKernel, abs_sub_pred hi, h2, neg_mul, hr,
      zero_add, div_le_iff₀ hc]
    linarith
  · have hji : (j : ℝ) + 2 ≤ i := by exact_mod_cast hij
    have habs : |(i : ℝ) - j| = (i : ℝ) - j := abs_of_nonneg (by linarith)
    have hlog : Real.log 3 ≤ Real.log (1 + 1 * |(i : ℝ) - j|) :=
      Real.log_le_log (by norm_num) (by rw [habs]; linarith)
    have hk : (b : ℝ) / Real.log 2 * Real.log 3 ≤
        (b : ℝ) / Real.log 2 * Real.log (1 + 1 * |(i : ℝ) - j|) :=
      mul_le_mul_of_nonneg_left hlog hr0
    have h3 : 3 * (b : ℝ) ≤ 2 * ((b : ℝ) / Real.log 2 * Real.log 3) := by
      calc 3 * (b : ℝ) = 3 * ((b : ℝ) / Real.log 2 * Real.log 2) := by rw [hr]
        _ = (b : ℝ) / Real.log 2 * (3 * Real.log 2) := by ring
        _ ≤ (b : ℝ) / Real.log 2 * (2 * Real.log 3) :=
          mul_le_mul_of_nonneg_left three_log_two_le hr0
        _ = 2 * ((b : ℝ) / Real.log 2 * Real.log 3) := by ring
    simp only [scaleLogit, kerpleLogLogit, KERPLE.logKernel, neg_mul]
    constructor <;> rw [div_le_iff₀ hc] <;> linarith
  · refine Or.inl fun j hj => ?_
    have h0 : 0 ≤ Real.log (1 + 1 * |(i : ℝ) - j|) :=
      Real.log_nonneg (by have := abs_nonneg ((i : ℝ) - j); linarith)
    have h1 : 0 ≤ (b : ℝ) / Real.log 2 * Real.log (1 + 1 * |(i : ℝ) - j|) := mul_nonneg hr0 h0
    simp only [scaleLogit, kerpleLogLogit, KERPLE.logKernel, neg_mul]
    exact div_neg_of_neg_of_pos (by linarith) hc

/-- The hypotheses of `readsPrev_scale_kerpleLog` hold for the divisor `1` and `b = 2`. -/
example : ReadsPrev (scaleLogit 1 (kerpleLogLogit ((2 : ℕ) / Real.log 2) 1)) 2 :=
  readsPrev_scale_kerpleLog one_pos (by norm_num)

/-- **Scaled KERPLE, power**: the kernel `−b|m − n|^{r₂}` with `r₂ > 0`, the logit divided by `c`
with `0 < c ≤ b` and `c ≤ b(2^{r₂} − 1)`, meets `ReadsPrev` with the dot product `b`. -/
theorem readsPrev_scale_kerplePower {b : ℕ} {r₂ c : ℝ} (hr₂ : 0 < r₂) (hc : 0 < c)
    (hcb : c ≤ b) (hcb' : c ≤ b * (2 ^ r₂ - 1)) :
    ReadsPrev (scaleLogit c (kerplePowerLogit b r₂)) b := by
  have hb0 : (0 : ℝ) ≤ b := Nat.cast_nonneg b
  rw [mul_sub, mul_one] at hcb'
  refine ⟨fun i => ?_, fun i hi => ?_, fun i hi => ?_, fun i j hij => ?_, fun i => ?_⟩
  · simp [scaleLogit, kerplePowerLogit, KERPLE.powerKernel, Real.zero_rpow hr₂.ne']
  · simp only [scaleLogit, kerplePowerLogit, KERPLE.powerKernel, abs_sub_pred hi, Real.one_rpow,
      mul_one, add_neg_cancel, zero_div]
  · simp only [scaleLogit, kerplePowerLogit, KERPLE.powerKernel, abs_sub_pred hi, Real.one_rpow,
      mul_one, zero_add, div_le_iff₀ hc]
    linarith
  · have hji : (j : ℝ) + 2 ≤ i := by exact_mod_cast hij
    have habs : (2 : ℝ) ≤ |(i : ℝ) - j| := by
      rw [abs_of_nonneg (by linarith)]
      linarith
    have hpow : (2 : ℝ) ^ r₂ ≤ |(i : ℝ) - j| ^ r₂ := Real.rpow_le_rpow (by norm_num) habs hr₂.le
    have hk : (b : ℝ) * 2 ^ r₂ ≤ b * |(i : ℝ) - j| ^ r₂ := mul_le_mul_of_nonneg_left hpow hb0
    simp only [scaleLogit, kerplePowerLogit, KERPLE.powerKernel, neg_mul]
    constructor <;> rw [div_le_iff₀ hc] <;> linarith
  · refine Or.inl fun j hj => ?_
    have h1 : 0 ≤ (b : ℝ) * |(i : ℝ) - j| ^ r₂ :=
      mul_nonneg hb0 (Real.rpow_nonneg (abs_nonneg _) r₂)
    simp only [scaleLogit, kerplePowerLogit, KERPLE.powerKernel, neg_mul]
    exact div_neg_of_neg_of_pos (by linarith) hc

/-- The hypotheses of `readsPrev_scale_kerplePower` hold for `r₂ = 1`, the divisor `1` and
`b = 1`. -/
example : ReadsPrev (scaleLogit 1 (kerplePowerLogit (1 : ℕ) 1)) 1 :=
  readsPrev_scale_kerplePower one_pos one_pos (by norm_num) (by rw [Real.rpow_one]; norm_num)

/-- **SSMax** of a logit that meets `ReadsPrev`, with `s log 2 ≥ 1`, meets it too. -/
theorem readsPrev_ssmax {L : ℕ → ℕ → ℝ → ℝ} {b : ℕ} {s : ℝ} (hs : 1 ≤ s * Real.log 2)
    (hL : ReadsPrev L b) : ReadsPrev (ssmaxLogit s L) b := by
  have hl : 0 < Real.log 2 := Real.log_pos one_lt_two
  have hs0 : 0 ≤ s := by
    by_contra h
    have : s * Real.log 2 < 0 := mul_neg_of_neg_of_pos (lt_of_not_ge h) hl
    linarith
  have hgrow : ∀ i : ℕ, 0 < i → 1 ≤ s * Real.log ((i : ℝ) + 1) := fun i hi => by
    have h2 : (2 : ℝ) ≤ (i : ℝ) + 1 := by
      have : (1 : ℝ) ≤ i := by exact_mod_cast hi
      linarith
    exact hs.trans (mul_le_mul_of_nonneg_left (Real.log_le_log (by norm_num) h2) hs0)
  have hmul : ∀ i : ℕ, 0 < i → ∀ z : ℝ, z ≤ -1 → s * Real.log ((i : ℝ) + 1) * z ≤ -1 :=
    fun i hi z hz => by
      have h := hgrow i hi
      calc s * Real.log ((i : ℝ) + 1) * z ≤ s * Real.log ((i : ℝ) + 1) * (-1) :=
            mul_le_mul_of_nonneg_left hz (by linarith)
        _ ≤ -1 := by linarith
  refine ⟨fun i => ?_, fun i hi => ?_, fun i hi => ?_, fun i j hij => ?_, fun i => ?_⟩
  · simp [ssmaxLogit, hL.self i]
  · simp [ssmaxLogit, hL.prev i hi]
  · exact hmul i hi _ (hL.prevSame i hi)
  · exact ⟨hmul i (by omega) _ (hL.far i j hij).1, hmul i (by omega) _ (hL.far i j hij).2⟩
  · rcases Nat.eq_zero_or_pos i with rfl | hi
    · exact Or.inr fun j _ => by simp [ssmaxLogit]
    · have hf : 0 < s * Real.log ((i : ℝ) + 1) := lt_of_lt_of_le one_pos (hgrow i hi)
      rcases hL.run i with h | h
      · exact Or.inl fun j hj => mul_neg_of_pos_of_neg hf (h j hj)
      · exact Or.inr fun j hj => by simp only [ssmaxLogit, h j hj, mul_zero]

/-- The hypotheses of `readsPrev_ssmax` hold for `s = 2` and ALiBi of slope `1`. -/
example : ReadsPrev (ssmaxLogit 2 (alibiLogit (1 : ℕ))) 1 :=
  readsPrev_ssmax (by have := Real.log_two_gt_d9; linarith) (readsPrev_alibi le_rfl)

end Transformer.CRASP.Grams

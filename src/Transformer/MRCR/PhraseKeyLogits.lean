module

public import Transformer.MRCR.PhraseKeyGrams

/-!
# Rounded transformers with ALiBi, KERPLE or SSMax answer spelled asks

In no paper. Source of the task: MRCR of arXiv:2409.12640v2, `eval_framework.tex`, lines 35–37,
and MRCR-V2 of arXiv:2507.06261v6, line 75 of `8_appendix/1_evaluation_additional_details.tex`, as
in `Transformer.MRCR.PhraseKey`. No rounded transformer without positions, with periodic or with
rotary positions answers the query of a conversation whose asks are spelled by words; one with
ALiBi does.

Five layers suffice, at a precision and width that do not depend on the length: three read the
last four tokens into a gram (`Transformer.CRASP.Grams`), and two run the transformer of
`thm:TLCl_to_rtfr` of arXiv:2506.16055v3 for the answer formula over grams
(`Transformer.MRCR.phraseAnswerG`). The logit can be ALiBi of `eq:alibi`
(`exists_alibi_phraseAnswer`), or any logit of the positions that reads the token before
(`exists_phraseAnswer`):
* scaled ALiBi, the logit divided by `c d` (`exists_scale_alibi_phraseAnswer`);
* KERPLE (arXiv:2205.09921v2, `eq:krpe`, §4 Practical Choice) with the logarithmic or the power
  kernel, the logit divided by `c d` (`exists_kerpleLog_phraseAnswer`,
  `exists_kerplePower_phraseAnswer`);
* SSMax (arXiv:2501.19399v1, `eq:ssmax`) of each with `s log 2 ≥ 1`, scaling the whole logit
  (`exists_ssmax_alibi_phraseAnswer`, `exists_ssmax_kerpleLog_phraseAnswer`,
  `exists_ssmax_kerplePower_phraseAnswer`).

The divisor `c d` of the width `d` is any positive one: `1` as in `def:transformer` and
`eq:alibi`, `√d` as in `eq:krpe` and standard attention, or `d`. SSMax that scales the dot product
but not a positional term added after it is another model, not treated here.
-/

@[expose] public section

namespace Transformer.MRCR

open Transformer.CRASP Transformer.CRASP.Grams

variable {F N V R : ℕ}

/-- **Some transformer with the logit `L d` answers spelled asks at every length**, when the logits
`L d` and dot products `b d` meet `ReadsPrev` at every width `d`: five layers. -/
theorem exists_phraseAnswer (L : ℕ → ℕ → ℕ → ℝ → ℝ) (b : ℕ → ℕ)
    (hL : ∀ d, 0 < d → ReadsPrev (L d) (b d)) (v : Fin V) :
    ∃ (p d : ℕ) (T : LTfr (Option (PhraseTok F N V R)) p 0 d 5), 0 < d ∧ T.logit = L d ∧
      T.Recognizes {w | PhraseAnswer w v} := by
  obtain ⟨p, d, T, hd, hT, hR⟩ :=
    exists_recognizes (n := 3) (k := 2) L b hL id (phraseAnswerG_mem (F := F) (N := N) (R := R) v)
  exact ⟨p, d, T, hd, hT, fun w => (hR w).trans (mem_lang_phraseAnswerG v w)⟩

/-- The hypothesis of `exists_phraseAnswer` holds for ALiBi of slope `1` at every width. -/
example := exists_phraseAnswer (F := 1) (N := 1) (R := 1) (fun _ => alibiLogit (1 : ℕ))
  (fun _ => 1) (fun _ _ => readsPrev_alibi le_rfl) (0 : Fin 1)

/-- **ALiBi**: some rounded transformer with ALiBi of slope `1` (`eq:alibi`) answers spelled asks
at every length. -/
theorem exists_alibi_phraseAnswer (v : Fin V) :
    ∃ (p d : ℕ) (T : PTfr (Option (PhraseTok F N V R)) p 0 d 5), T.pe = .alibi 1 ∧
      T.Recognizes {w | PhraseAnswer w v} := by
  obtain ⟨p, d, T, hT, hR⟩ :=
    exists_alibi_recognizes (n := 3) (k := 2) id (phraseAnswerG_mem (F := F) (N := N) (R := R) v)
  exact ⟨p, d, T, hT, fun w => (hR w).trans (mem_lang_phraseAnswerG v w)⟩

/-- An answer for one format, one topic and one value. -/
example := exists_alibi_phraseAnswer (F := 1) (N := 1) (R := 1) (0 : Fin 1)

/-- **Scaled ALiBi**: with the logit divided by `c d`, some slope `a > 0` does. -/
theorem exists_scale_alibi_phraseAnswer (c : ℕ → ℝ) (hc : ∀ d, 0 < d → 0 < c d) (v : Fin V) :
    ∃ (p d : ℕ) (a : ℝ) (T : LTfr (Option (PhraseTok F N V R)) p 0 d 5), 0 < a ∧
      T.logit = scaleLogit (c d) (alibiLogit a) ∧ T.Recognizes {w | PhraseAnswer w v} := by
  obtain ⟨p, d, T, hd, hT, hR⟩ := exists_phraseAnswer (F := F) (N := N) (R := R)
    (fun d => scaleLogit (c d) (alibiLogit ⌈c d⌉₊)) (fun d => ⌈c d⌉₊)
    (fun d hd => readsPrev_scale_alibi (hc d hd) (Nat.le_ceil _)) v
  exact ⟨p, d, _, T, (hc d hd).trans_le (Nat.le_ceil _), hT, hR⟩

/-- The hypothesis of `exists_scale_alibi_phraseAnswer` holds for the divisors `√d` and `d`. -/
example := exists_scale_alibi_phraseAnswer (F := 1) (N := 1) (R := 1) (fun d => Real.sqrt d)
  (fun _ hd => Real.sqrt_pos.mpr (by exact_mod_cast hd)) (0 : Fin 1)

example := exists_scale_alibi_phraseAnswer (F := 1) (N := 1) (R := 1) (fun d => (d : ℝ))
  (fun _ hd => by exact_mod_cast hd) (0 : Fin 1)

/-- **SSMax and scaled ALiBi**: SSMax with `s log 2 ≥ 1` of the logit divided by `c d`. -/
theorem exists_ssmax_alibi_phraseAnswer (c : ℕ → ℝ) (hc : ∀ d, 0 < d → 0 < c d) {s : ℝ}
    (hs : 1 ≤ s * Real.log 2) (v : Fin V) :
    ∃ (p d : ℕ) (a : ℝ) (T : LTfr (Option (PhraseTok F N V R)) p 0 d 5), 0 < a ∧
      T.logit = ssmaxLogit s (scaleLogit (c d) (alibiLogit a)) ∧
      T.Recognizes {w | PhraseAnswer w v} := by
  obtain ⟨p, d, T, hd, hT, hR⟩ := exists_phraseAnswer (F := F) (N := N) (R := R)
    (fun d => ssmaxLogit s (scaleLogit (c d) (alibiLogit ⌈c d⌉₊))) (fun d => ⌈c d⌉₊)
    (fun d hd => readsPrev_ssmax hs (readsPrev_scale_alibi (hc d hd) (Nat.le_ceil _))) v
  exact ⟨p, d, _, T, (hc d hd).trans_le (Nat.le_ceil _), hT, hR⟩

/-- The hypotheses of `exists_ssmax_alibi_phraseAnswer` hold for `s = 2` and the divisor `√d`. -/
example := exists_ssmax_alibi_phraseAnswer (F := 1) (N := 1) (R := 1) (fun d => Real.sqrt d)
  (fun _ hd => Real.sqrt_pos.mpr (by exact_mod_cast hd)) (s := 2)
  (by have := Real.log_two_gt_d9; linarith) (0 : Fin 1)

/-- The dot product `2⌈c⌉` for the logarithmic kernel. -/
theorem kerpleLog_bound {c : ℝ} (hc : 0 < c) :
    2 * c ≤ ((2 * ⌈c⌉₊ : ℕ) : ℝ) ∧ 0 < ((2 * ⌈c⌉₊ : ℕ) : ℝ) / Real.log 2 := by
  have h := Nat.le_ceil c
  refine ⟨by push_cast; linarith, div_pos ?_ (Real.log_pos one_lt_two)⟩
  push_cast
  linarith

/-- The hypothesis of `kerpleLog_bound` holds for `c = 1`. -/
example := kerpleLog_bound (c := 1) one_pos

/-- **Scaled KERPLE, logarithmic**: with the logit divided by `c d`, the kernel
`−r₁ log(1 + |m − n|)` of some `r₁ > 0` does. -/
theorem exists_kerpleLog_phraseAnswer (c : ℕ → ℝ) (hc : ∀ d, 0 < d → 0 < c d) (v : Fin V) :
    ∃ (p d : ℕ) (r₁ : ℝ) (T : LTfr (Option (PhraseTok F N V R)) p 0 d 5), 0 < r₁ ∧
      T.logit = scaleLogit (c d) (kerpleLogLogit r₁ 1) ∧ T.Recognizes {w | PhraseAnswer w v} := by
  obtain ⟨p, d, T, hd, hT, hR⟩ := exists_phraseAnswer (F := F) (N := N) (R := R)
    (fun d => scaleLogit (c d) (kerpleLogLogit ((2 * ⌈c d⌉₊ : ℕ) / Real.log 2) 1))
    (fun d => 2 * ⌈c d⌉₊)
    (fun d hd => readsPrev_scale_kerpleLog (hc d hd) (kerpleLog_bound (hc d hd)).1) v
  exact ⟨p, d, _, T, (kerpleLog_bound (hc d hd)).2, hT, hR⟩

/-- **SSMax and scaled KERPLE, logarithmic.** -/
theorem exists_ssmax_kerpleLog_phraseAnswer (c : ℕ → ℝ) (hc : ∀ d, 0 < d → 0 < c d) {s : ℝ}
    (hs : 1 ≤ s * Real.log 2) (v : Fin V) :
    ∃ (p d : ℕ) (r₁ : ℝ) (T : LTfr (Option (PhraseTok F N V R)) p 0 d 5), 0 < r₁ ∧
      T.logit = ssmaxLogit s (scaleLogit (c d) (kerpleLogLogit r₁ 1)) ∧
      T.Recognizes {w | PhraseAnswer w v} := by
  obtain ⟨p, d, T, hd, hT, hR⟩ := exists_phraseAnswer (F := F) (N := N) (R := R)
    (fun d => ssmaxLogit s (scaleLogit (c d) (kerpleLogLogit ((2 * ⌈c d⌉₊ : ℕ) / Real.log 2) 1)))
    (fun d => 2 * ⌈c d⌉₊) (fun d hd => readsPrev_ssmax hs
      (readsPrev_scale_kerpleLog (hc d hd) (kerpleLog_bound (hc d hd)).1)) v
  exact ⟨p, d, _, T, (kerpleLog_bound (hc d hd)).2, hT, hR⟩

/-- The hypotheses of the logarithmic theorems hold for `s = 2` and the divisor `√d`. -/
example := exists_kerpleLog_phraseAnswer (F := 1) (N := 1) (R := 1) (fun d => Real.sqrt d)
  (fun _ hd => Real.sqrt_pos.mpr (by exact_mod_cast hd)) (0 : Fin 1)

example := exists_ssmax_kerpleLog_phraseAnswer (F := 1) (N := 1) (R := 1) (fun d => Real.sqrt d)
  (fun _ hd => Real.sqrt_pos.mpr (by exact_mod_cast hd)) (s := 2)
  (by have := Real.log_two_gt_d9; linarith) (0 : Fin 1)

/-- The dot product `⌈c / (2^{r₂} − 1)⌉ + ⌈c⌉` for the power kernel. -/
theorem kerplePower_bound {c r₂ : ℝ} (hc : 0 < c) (hr₂ : 0 < r₂) :
    c ≤ ((⌈c / (2 ^ r₂ - 1)⌉₊ + ⌈c⌉₊ : ℕ) : ℝ) ∧
      c ≤ ((⌈c / (2 ^ r₂ - 1)⌉₊ + ⌈c⌉₊ : ℕ) : ℝ) * (2 ^ r₂ - 1) ∧
      0 < ((⌈c / (2 ^ r₂ - 1)⌉₊ + ⌈c⌉₊ : ℕ) : ℝ) := by
  have h2 : 0 < (2 : ℝ) ^ r₂ - 1 := by
    have := Real.one_lt_rpow (by norm_num : (1 : ℝ) < 2) hr₂
    linarith
  have h1 := Nat.le_ceil (c / (2 ^ r₂ - 1))
  have h3 := Nat.le_ceil c
  have h4 : (0 : ℝ) ≤ ⌈c / (2 ^ r₂ - 1)⌉₊ := Nat.cast_nonneg _
  have h5 : (0 : ℝ) ≤ ⌈c⌉₊ := Nat.cast_nonneg _
  push_cast
  refine ⟨by linarith, ?_, by linarith⟩
  calc c = c / (2 ^ r₂ - 1) * (2 ^ r₂ - 1) := (div_mul_cancel₀ _ h2.ne').symm
    _ ≤ (⌈c / (2 ^ r₂ - 1)⌉₊ + ⌈c⌉₊) * (2 ^ r₂ - 1) :=
      mul_le_mul_of_nonneg_right (by linarith) h2.le

/-- The hypotheses of `kerplePower_bound` hold for `c = 1` and `r₂ = 1`. -/
example := kerplePower_bound (c := 1) (r₂ := 1) one_pos one_pos

/-- **Scaled KERPLE, power**: with the logit divided by `c d` and any `r₂ > 0`, the kernel
`−r₁|m − n|^{r₂}` of some `r₁ > 0` does. -/
theorem exists_kerplePower_phraseAnswer (c : ℕ → ℝ) (hc : ∀ d, 0 < d → 0 < c d) {r₂ : ℝ}
    (hr₂ : 0 < r₂) (v : Fin V) :
    ∃ (p d : ℕ) (r₁ : ℝ) (T : LTfr (Option (PhraseTok F N V R)) p 0 d 5), 0 < r₁ ∧
      T.logit = scaleLogit (c d) (kerplePowerLogit r₁ r₂) ∧
      T.Recognizes {w | PhraseAnswer w v} := by
  obtain ⟨p, d, T, hd, hT, hR⟩ := exists_phraseAnswer (F := F) (N := N) (R := R)
    (fun d => scaleLogit (c d) (kerplePowerLogit (⌈c d / (2 ^ r₂ - 1)⌉₊ + ⌈c d⌉₊ : ℕ) r₂))
    (fun d => ⌈c d / (2 ^ r₂ - 1)⌉₊ + ⌈c d⌉₊) (fun d hd => readsPrev_scale_kerplePower hr₂
      (hc d hd) (kerplePower_bound (hc d hd) hr₂).1 (kerplePower_bound (hc d hd) hr₂).2.1) v
  exact ⟨p, d, _, T, (kerplePower_bound (hc d hd) hr₂).2.2, hT, hR⟩

/-- **SSMax and scaled KERPLE, power.** -/
theorem exists_ssmax_kerplePower_phraseAnswer (c : ℕ → ℝ) (hc : ∀ d, 0 < d → 0 < c d)
    {r₂ s : ℝ} (hr₂ : 0 < r₂) (hs : 1 ≤ s * Real.log 2) (v : Fin V) :
    ∃ (p d : ℕ) (r₁ : ℝ) (T : LTfr (Option (PhraseTok F N V R)) p 0 d 5), 0 < r₁ ∧
      T.logit = ssmaxLogit s (scaleLogit (c d) (kerplePowerLogit r₁ r₂)) ∧
      T.Recognizes {w | PhraseAnswer w v} := by
  obtain ⟨p, d, T, hd, hT, hR⟩ := exists_phraseAnswer (F := F) (N := N) (R := R)
    (fun d => ssmaxLogit s
      (scaleLogit (c d) (kerplePowerLogit (⌈c d / (2 ^ r₂ - 1)⌉₊ + ⌈c d⌉₊ : ℕ) r₂)))
    (fun d => ⌈c d / (2 ^ r₂ - 1)⌉₊ + ⌈c d⌉₊) (fun d hd => readsPrev_ssmax hs
      (readsPrev_scale_kerplePower hr₂ (hc d hd) (kerplePower_bound (hc d hd) hr₂).1
        (kerplePower_bound (hc d hd) hr₂).2.1)) v
  exact ⟨p, d, _, T, (kerplePower_bound (hc d hd) hr₂).2.2, hT, hR⟩

/-- The hypotheses of the power theorems hold for `r₂ = 1`, `s = 2` and the divisor `d`. -/
example := exists_kerplePower_phraseAnswer (F := 1) (N := 1) (R := 1) (fun d => (d : ℝ))
  (fun _ hd => by exact_mod_cast hd) one_pos (0 : Fin 1)

example := exists_ssmax_kerplePower_phraseAnswer (F := 1) (N := 1) (R := 1) (fun d => (d : ℝ))
  (fun _ hd => by exact_mod_cast hd) one_pos (s := 2) (by have := Real.log_two_gt_d9; linarith)
  (0 : Fin 1)

end Transformer.MRCR

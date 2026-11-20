module

public import Transformer.MRCR.PhraseKeyStyleGrams
public import Transformer.MRCR.PhraseKeyReRopeSu

/-!
# Rounded transformers of six layers answer asks with a style

In no paper. Source of the task: MRCR-V2 of arXiv:2507.06261v6, line 75 of
`8_appendix/1_evaluation_additional_details.tex`, keys of depth three spelled by words, as in
`Transformer.MRCR.PhraseKeyStyle`. Six layers suffice, at a precision and width that do not
depend on the length: four read the last five tokens into a gram (`Transformer.CRASP.Grams`), and
two run the transformer of `thm:TLCl_to_rtfr` of arXiv:2506.16055v3 for the answer formula over
grams (`Transformer.MRCR.styleAnswerG`). That is one layer more than keys of depth two take
(`Transformer.MRCR.PhraseKeyLogits`). The logit is any logit of the positions that reads the
token before (`exists_styleAnswer`):
* ALiBi of `eq:alibi` (`exists_alibi_styleAnswer`);
* ReRoPE's clamped logit of `Transformer.MRCR.PhraseKeyReRope` at every window `W ≥ 2`, alone
  (`exists_rerope_styleAnswer`), under SSMax with `s log 2 ≥ 1`
  (`exists_ssmax_rerope_styleAnswer`), and multiplied by Su's scale `max(1, log_T n)`
  (`exists_logScale_rerope_styleAnswer`).
-/

@[expose] public section

namespace Transformer.MRCR

open Transformer.CRASP Transformer.CRASP.Grams
open Transformer.RoPE.ReRoPE (logScale)

variable {K F N S V R : ℕ}

/-- **Some transformer with the logit `L d` answers asks with a style at every length**, when the
logits `L d` and dot products `b d` meet `ReadsPrev` at every width `d`: six layers. -/
theorem exists_styleAnswer (L : ℕ → ℕ → ℕ → ℝ → ℝ) (b : ℕ → ℕ)
    (hL : ∀ d, 0 < d → ReadsPrev (L d) (b d)) (v : Fin V) :
    ∃ (p d : ℕ) (T : LTfr (Option (StyleTok F N S V R)) p 0 d 6), 0 < d ∧ T.logit = L d ∧
      T.Recognizes {w | StyleAnswer w v} := by
  obtain ⟨p, d, T, hd, hT, hR⟩ := exists_recognizes (n := 4) (k := 2) L b hL id
    (styleAnswerG_mem (F := F) (N := N) (S := S) (R := R) v)
  exact ⟨p, d, T, hd, hT, fun w => (hR w).trans (mem_lang_styleAnswerG v w)⟩

/-- The hypothesis of `exists_styleAnswer` holds for ALiBi of slope `1` at every width. -/
example := exists_styleAnswer (F := 1) (N := 1) (S := 1) (R := 8) (fun _ => alibiLogit (1 : ℕ))
  (fun _ => 1) (fun _ _ => readsPrev_alibi le_rfl) (0 : Fin 1)

/-- **ALiBi**: some rounded transformer with ALiBi of slope `1` (`eq:alibi`) answers asks with a
style at every length. -/
theorem exists_alibi_styleAnswer (v : Fin V) :
    ∃ (p d : ℕ) (T : PTfr (Option (StyleTok F N S V R)) p 0 d 6), T.pe = .alibi 1 ∧
      T.Recognizes {w | StyleAnswer w v} := by
  obtain ⟨p, d, T, hT, hR⟩ := exists_alibi_recognizes (n := 4) (k := 2) id
    (styleAnswerG_mem (F := F) (N := N) (S := S) (R := R) v)
  exact ⟨p, d, T, hT, fun w => (hR w).trans (mem_lang_styleAnswerG v w)⟩

/-- Eight needles, with one format, topic, style and value. -/
example := exists_alibi_styleAnswer (F := 1) (N := 1) (S := 1) (R := 8) (0 : Fin 1)

/-- **ReRoPE**: the clamped logit of `readsPrev_rerope`, the dot product divided by any positive
`c d`, at every window `W ≥ 2` and frequency with `θ_m (W + 2) ≤ π`. -/
theorem exists_rerope_styleAnswer (c : ℕ → ℝ) (hc : ∀ d, 0 < d → 0 < c d) {θ : Fin K → ℝ}
    {m : Fin K} (hθ : 0 < θ m) {W : ℕ} (hW : 2 ≤ W) (hθW : θ m * (W + 2) ≤ Real.pi)
    (v : Fin V) :
    ∃ (p d : ℕ) (b : Fin K → ℝ) (T : LTfr (Option (StyleTok F N S V R)) p 0 d 6),
      T.logit = reropeLogit (c d) θ 0 b W ∧ T.Recognizes {w | StyleAnswer w v} := by
  obtain ⟨p, d, T, -, hT, hR⟩ := exists_styleAnswer (F := F) (N := N) (S := S) (R := R)
    (fun d => reropeLogit (c d) θ 0
      (Pi.single m (-(((3 * ⌈c d⌉₊ : ℕ) : ℝ) / c d) / Real.sin (θ m))) W)
    (fun d => 3 * ⌈c d⌉₊) (fun d hd => readsPrev_rerope (hc d hd)
      (by push_cast; linarith [Nat.le_ceil (c d)]) hθ hW hθW) v
  exact ⟨p, d, _, T, hT, hR⟩

/-- The hypotheses of `exists_rerope_styleAnswer` hold for eight needles, the divisor `√d`,
`θ = π/4` and the window `2`. -/
example := exists_rerope_styleAnswer (F := 1) (N := 1) (S := 1) (R := 8) (fun d => Real.sqrt d)
  (fun _ hd => Real.sqrt_pos.mpr (by exact_mod_cast hd)) (θ := fun _ : Fin 1 => Real.pi / 4)
  (m := 0) (by positivity) (W := 2) le_rfl (by push_cast; linarith) (0 : Fin 1)

/-- **SSMax and ReRoPE**: SSMax with `s log 2 ≥ 1` (arXiv:2501.19399v1, `eq:ssmax`) of the same
logit, scaling the whole logit. -/
theorem exists_ssmax_rerope_styleAnswer (c : ℕ → ℝ) (hc : ∀ d, 0 < d → 0 < c d) {s : ℝ}
    (hs : 1 ≤ s * Real.log 2) {θ : Fin K → ℝ} {m : Fin K} (hθ : 0 < θ m) {W : ℕ} (hW : 2 ≤ W)
    (hθW : θ m * (W + 2) ≤ Real.pi) (v : Fin V) :
    ∃ (p d : ℕ) (b : Fin K → ℝ) (T : LTfr (Option (StyleTok F N S V R)) p 0 d 6),
      T.logit = ssmaxLogit s (reropeLogit (c d) θ 0 b W) ∧
      T.Recognizes {w | StyleAnswer w v} := by
  obtain ⟨p, d, T, -, hT, hR⟩ := exists_styleAnswer (F := F) (N := N) (S := S) (R := R)
    (fun d => ssmaxLogit s (reropeLogit (c d) θ 0
      (Pi.single m (-(((3 * ⌈c d⌉₊ : ℕ) : ℝ) / c d) / Real.sin (θ m))) W))
    (fun d => 3 * ⌈c d⌉₊) (fun d hd => readsPrev_ssmax hs (readsPrev_rerope (hc d hd)
      (by push_cast; linarith [Nat.le_ceil (c d)]) hθ hW hθW)) v
  exact ⟨p, d, _, T, hT, hR⟩

/-- The hypotheses of `exists_ssmax_rerope_styleAnswer` hold for eight needles, `s = 2`, the
divisor `d`, `θ = π/4` and the window `2`. -/
example := exists_ssmax_rerope_styleAnswer (F := 1) (N := 1) (S := 1) (R := 8)
  (fun d => (d : ℝ)) (fun _ hd => by exact_mod_cast hd) (s := 2)
  (by have := Real.log_two_gt_d9; linarith) (θ := fun _ : Fin 1 => Real.pi / 4) (m := 0)
  (by positivity) (W := 2) le_rfl (by push_cast; linarith) (0 : Fin 1)

/-- **Su's scale and ReRoPE**: the same logit multiplied at the query position `i` by
`max(1, log_T (i + 1))`, the scale of the `log n` runs of kexue.fm/archives/9708 and of
`rerope_patch.py`, at every training length `T`. -/
theorem exists_logScale_rerope_styleAnswer (c : ℕ → ℝ) (hc : ∀ d, 0 < d → 0 < c d) (T : ℝ)
    {θ : Fin K → ℝ} {m : Fin K} (hθ : 0 < θ m) {W : ℕ} (hW : 2 ≤ W)
    (hθW : θ m * (W + 2) ≤ Real.pi) (v : Fin V) :
    ∃ (p d : ℕ) (b : Fin K → ℝ) (Tr : LTfr (Option (StyleTok F N S V R)) p 0 d 6),
      Tr.logit = (fun (i j : ℕ) (x : ℝ) => logScale T (i + 1) * reropeLogit (c d) θ 0 b W i j x) ∧
      Tr.Recognizes {w | StyleAnswer w v} := by
  obtain ⟨p, d, Tr, -, hT, hR⟩ := exists_styleAnswer (F := F) (N := N) (S := S) (R := R)
    (fun d i j x => logScale T ((i : ℝ) + 1) * reropeLogit (c d) θ 0
      (Pi.single m (-(((3 * ⌈c d⌉₊ : ℕ) : ℝ) / c d) / Real.sin (θ m))) W i j x)
    (fun d => 3 * ⌈c d⌉₊) (fun d hd => readsPrev_mul (readsPrev_rerope (hc d hd)
      (by push_cast; linarith [Nat.le_ceil (c d)]) hθ hW hθW)
      (fun i => zero_le_one.trans (one_le_logScale _ _)) (fun i _ => one_le_logScale _ _)) v
  exact ⟨p, d, _, Tr, hT, hR⟩

/-- The hypotheses of `exists_logScale_rerope_styleAnswer` hold for eight needles, Su's training
length `4096` and window `512`, the frequency `π/1024` and the divisor `√d`. -/
example := exists_logScale_rerope_styleAnswer (F := 1) (N := 1) (S := 1) (R := 8)
  (fun d => Real.sqrt d) (fun _ hd => Real.sqrt_pos.mpr (by exact_mod_cast hd)) 4096
  (θ := fun _ : Fin 1 => Real.pi / 1024) (m := 0) (by positivity) (W := 512) (by norm_num)
  (by push_cast; linarith [Real.pi_pos]) (0 : Fin 1)

end Transformer.MRCR

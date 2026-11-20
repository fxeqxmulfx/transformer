module

public import Transformer.RoPE.YaRN.Section3_Pressure

/-!
# YaRN §4: the arithmetic of the experiments

Peng et al., arXiv:2309.00071v3, §§4.4–4.5, Tables 3, 4 and 9. The scores, perplexities and
training times are measurements; the arithmetic the text draws from them is checked here.

* `table3_mean_drop`: §4.4, "there was on average a 0.49% drop in scores between the YaRN
  `s = 16` and `s = 32` models": over the eight scores of Table 3 the mean drop is `0.4875`
  points, so the `0.49%` counts percentage points; relative to the `s = 16` scores the pooled
  drop is `0.88%`;
* `codeLlama_scale`: Tables 4 and 9 give Code Llama, whose base is raised from `b = 10000` (§2.1)
  to `1M` (footnote *), the NTK-aware scale `s = 88.6`. Eqs. (19)–(22) give
  `s = 100^{(|D|−2)/|D|}`, which is `93.06` at `|D| = 128` and `86.60` at `|D| = 64`;
* `together_scale`: Table 9 gives Together's PI model, of context window `32k` from Llama 2's
  `4k`, the scale factor `4`; Eq. (20)'s `s = L'/L` and Table 4's `4k × 8` give `8`.
-/

@[expose] public section

namespace Transformer.RoPE.YaRN

/-- Table 3, the scores of the YaRN `s = 16` models: 7B, then 13B; ARC-c, Hellaswag, MMLU and
TruthfulQA. -/
noncomputable def table3Scale16 : Fin 8 → ℝ := ![52.3, 78.8, 42.5, 38.2, 58.1, 82.3, 52.8, 37.8]

/-- Table 3, the scores of the YaRN `s = 32` models, in the order of `table3Scale16`. -/
noncomputable def table3Scale32 : Fin 8 → ℝ := ![52.1, 78.4, 41.7, 37.3, 58.0, 82.2, 51.9, 37.3]

/-- §4.4: "on average a 0.49% drop in scores between the YaRN `s = 16` and `s = 32` models". The
mean drop is `0.4875` points; the pooled drop relative to the `s = 16` scores, `3.9/442.8`, lies
between `0.88%` and `0.89%`. -/
theorem table3_mean_drop :
    (∑ i, (table3Scale16 i - table3Scale32 i)) / 8 = 0.4875 ∧
      0.0088 < (∑ i, (table3Scale16 i - table3Scale32 i)) / ∑ i, table3Scale16 i ∧
      (∑ i, (table3Scale16 i - table3Scale32 i)) / ∑ i, table3Scale16 i < 0.0089 := by
  simp only [Fin.sum_univ_eight, table3Scale16, table3Scale32]
  norm_num

/-- Tables 4 and 9 give Code Llama, whose base `b = 10000` (§2.1) is raised to `1M`
(footnote *), the NTK-aware scale `s = 88.6`. By Eqs. (19)–(22), `s = 100^{(|D|−2)/|D|}`: it lies
in `(93, 93.1)` at `|D| = 128`, the head dimension of LLaMA 7B and 13B, which the manuscript does
not state, and in `(86.5, 86.6)` at `|D| = 64`; `88.6` matches neither. -/
theorem codeLlama_scale {s : ℝ} (hs : 0 < s) :
    (adjustedBase 10000 s 128 = 1000000 → 93 < s ∧ s < 93.1) ∧
      (adjustedBase 10000 s 64 = 1000000 → 86.5 < s ∧ s < 86.6) := by
  have h100 : (1000000 : ℝ) / 10000 = 100 := by norm_num
  constructor
  · intro h
    rw [adjustedBase_eq_iff (base := 10000) (base' := 1000000) (D := 128) (by norm_num)
      (by norm_num) hs (by norm_num), h100,
      show ((128 : ℝ) - 2) / 128 = ((63 : ℕ) : ℝ) / ((64 : ℕ) : ℝ) by norm_num] at h
    rw [h]
    exact ⟨lt_rpow_div (by norm_num) (by norm_num) (by norm_num) (by norm_num),
      rpow_div_lt (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩
  · intro h
    rw [adjustedBase_eq_iff (base := 10000) (base' := 1000000) (D := 64) (by norm_num)
      (by norm_num) hs (by norm_num), h100,
      show ((64 : ℝ) - 2) / 64 = ((31 : ℕ) : ℝ) / ((32 : ℕ) : ℝ) by norm_num] at h
    rw [h]
    exact ⟨lt_rpow_div (by norm_num) (by norm_num) (by norm_num) (by norm_num),
      rpow_div_lt (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

example : (adjustedBase 10000 8 128 = 1000000 → 93 < (8 : ℝ) ∧ (8 : ℝ) < 93.1) ∧
    (adjustedBase 10000 8 64 = 1000000 → 86.5 < (8 : ℝ) ∧ (8 : ℝ) < 86.6) :=
  codeLlama_scale (by norm_num)

/-- Table 9 gives Together's PI model, of context window `32k` from Llama 2's `4k`, the scale
factor `4`; Eq. (20)'s `s = L'/L` and Table 4's `4k × 8` give `8`. -/
theorem together_scale : contextScale 4096 32768 = 8 ∧ contextScale 4096 32768 ≠ 4 := by
  have h : contextScale 4096 32768 = 8 := by
    rw [contextScale, max_eq_right (by norm_num)]
    norm_num
  exact ⟨h, by rw [h]; norm_num⟩

end Transformer.RoPE.YaRN

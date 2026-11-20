module

public import Transformer.RoPE.LongRoPE2.Section3_Mixed

/-!
# LongRoPE2 abstract, §1, §4.1–4.2 and §5: the token budget and Table 4

Shang et al., *LongRoPE2*, arXiv:2502.20082v1. The scores are measurements; the arithmetic the
text draws from them is checked here. Table 4 lists MMLU, MMLU-Pro, HellaSwag, TruthfulQA and
GSM8K, and an average `Avg.` to one decimal.

* `token_budget`: §4.1, "we sample 4.5B, 2.5B, and 2B tokens from RedPajama-v1, RedPajama-v2,
  and StarCoder, covering 8k–200k sequence lengths. For short context windows, we sample 1B
  tokens from Fineweb-Edu", "a 10B-token dataset", against Table 1's `3B` (`≤ L_train`), `3B`
  (`L_train`–100k) and `4B` (100k–200k): both add up to `10B`, but the short windows hold `1B` in
  the text and `3B` in Table 1; "80x fewer than Meta's approach" (abstract), with "800B tokens for
  LLaMA3.1" (§1);
* `mmlu_drops`: §1, "As shown in Fig. 2(c), extending Phi3-mini … to 128k results in MMLU score
  drops of 7.56, 4.34, and 3.52 points for YaRN, NTK, and LongRoPE": Table 4(a)'s MMLU column
  gives `7.56`, `4.35` and `3.52`;
* `table4_averages`: every printed average lies within `0.1` of its row's mean, rounded or
  truncated, except YaRN's on Phi3-mini, `53.6` for the mean `53.804`;
* `retention`: §4.2, "LongRoPE2 retains 97.6% and 98.6% of the pre-trained performance on
  Phi3-mini-3.8B and LLaMA3-8B", §5, "retaining 97.6%", and the abstract, "retaining over 98.5% of
  short-context performance" (LLaMA3-8B): the printed averages give `97.6%` and `98.6%`; the means
  of the rows give `97.6%` and `98.45%`, below `98.5%`;
* `only_over_97`: §1, "LongRoPE2 is the only RoPE rescaling method that can retain over 97% of
  the original short-context performance on standard tasks": YaRN, NTK and LongRoPE stay below
  `97%` on both models and LongRoPE2 exceeds it, from the printed averages and from the means;
* `phi3_declines`: §4.2, "YaRN and NTK show performance drop of -15.2% and -9.3% on Phi3-mini,
  with declines of -21.15 and -14.55 absolute points on GSM8K": `-15.2%` follows from the printed
  `53.6`, while the mean `53.804` gives `-14.9%`; the other three hold.
-/

@[expose] public section

namespace Transformer.RoPE.LongRoPE2

/-- The mean of a row of `n` scores. -/
noncomputable def rowMean {n : ℕ} (row : Fin n → ℝ) : ℝ := (∑ i, row i) / n

/-- Table 4(a), original Phi3-mini (2k): MMLU, MMLU-Pro, HellaSwag, TruthfulQA, GSM8K. -/
noncomputable def table4Phi3 : Fin 5 → ℝ := ![70.78, 41.17, 77.96, 47.82, 78.54]
/-- Table 4(a), YaRN. -/
noncomputable def table4Phi3YaRN : Fin 5 → ℝ := ![63.22, 30.95, 75.27, 42.19, 57.39]
/-- Table 4(a), NTK. -/
noncomputable def table4Phi3NTK : Fin 5 → ℝ := ![66.43, 36.09, 76.92, 43.34, 63.99]
/-- Table 4(a), LongRoPE. -/
noncomputable def table4Phi3LongRoPE : Fin 5 → ℝ := ![67.26, 36.28, 75.73, 46.26, 67.17]
/-- Table 4(a), LongRoPE2. -/
noncomputable def table4Phi3LongRoPE2 : Fin 5 → ℝ := ![70.04, 40.30, 77.07, 47.61, 73.62]
/-- Table 4(b), LLaMA3.1-8B. -/
noncomputable def table4LLaMA31 : Fin 5 → ℝ := ![66.33, 36.79, 81.71, 45.17, 56.18]
/-- Table 4(b), original LLaMA3-8B (8k). -/
noncomputable def table4LLaMA3 : Fin 5 → ℝ := ![66.62, 35.87, 82.08, 44.04, 54.05]
/-- Table 4(b), YaRN. -/
noncomputable def table4LLaMA3YaRN : Fin 5 → ℝ := ![62.25, 31.88, 81.25, 42.61, 42.45]
/-- Table 4(b), NTK∗. -/
noncomputable def table4LLaMA3NTK : Fin 5 → ℝ := ![63.84, 34.14, 82.11, 43.45, 46.92]
/-- Table 4(b), LongRoPE. -/
noncomputable def table4LLaMA3LongRoPE : Fin 5 → ℝ := ![64.69, 33.74, 82.14, 43.65, 48.90]
/-- Table 4(b), LongRoPE2. -/
noncomputable def table4LLaMA3LongRoPE2 : Fin 5 → ℝ := ![65.01, 34.61, 81.69, 46.17, 50.80]

/-- §4.1, Table 1 and the abstract: the token counts, in billions. -/
theorem token_budget :
    (4.5 + 2.5 + 2 + 1 : ℝ) = 10 ∧ (3 + 3 + 4 : ℝ) = 10 ∧ (1 : ℝ) ≠ 3 ∧
      (4.5 + 2.5 + 2 : ℝ) ≠ 3 + 4 ∧ (800 : ℝ) / 10 = 80 := by
  norm_num

/-- §1: Phi3-mini's MMLU drops under YaRN, NTK and LongRoPE; the second is `4.35`. -/
theorem mmlu_drops :
    table4Phi3 0 - table4Phi3YaRN 0 = 7.56 ∧ table4Phi3 0 - table4Phi3NTK 0 = 4.35 ∧
      table4Phi3 0 - table4Phi3LongRoPE 0 = 3.52 := by
  norm_num [table4Phi3, table4Phi3YaRN, table4Phi3NTK, table4Phi3LongRoPE]

/-- Table 4: the printed averages against the means of the rows. -/
theorem table4_averages :
    |rowMean table4Phi3 - 63.2| < 0.1 ∧ rowMean table4Phi3YaRN - 53.6 = 0.204 ∧
      |rowMean table4Phi3NTK - 57.3| < 0.1 ∧ |rowMean table4Phi3LongRoPE - 58.5| < 0.1 ∧
      |rowMean table4Phi3LongRoPE2 - 61.7| < 0.1 ∧ |rowMean table4LLaMA31 - 57.2| < 0.1 ∧
      |rowMean table4LLaMA3 - 56.5| < 0.1 ∧ |rowMean table4LLaMA3YaRN - 52.1| < 0.1 ∧
      |rowMean table4LLaMA3NTK - 54.0| < 0.1 ∧ |rowMean table4LLaMA3LongRoPE - 54.6| < 0.1 ∧
      |rowMean table4LLaMA3LongRoPE2 - 55.7| < 0.1 := by
  norm_num [rowMean, Fin.sum_univ_five, table4Phi3, table4Phi3YaRN, table4Phi3NTK,
    table4Phi3LongRoPE, table4Phi3LongRoPE2, table4LLaMA31, table4LLaMA3, table4LLaMA3YaRN,
    table4LLaMA3NTK, table4LLaMA3LongRoPE, table4LLaMA3LongRoPE2, abs_lt]

/-- §4.2, §5 and the abstract: LongRoPE2's retained share of the original average. -/
theorem retention :
    0.9755 ≤ (61.7 : ℝ) / 63.2 ∧ (61.7 : ℝ) / 63.2 < 0.9765 ∧
      0.9855 ≤ (55.7 : ℝ) / 56.5 ∧ (55.7 : ℝ) / 56.5 < 0.9865 ∧
      0.9755 ≤ rowMean table4Phi3LongRoPE2 / rowMean table4Phi3 ∧
      rowMean table4Phi3LongRoPE2 / rowMean table4Phi3 < 0.9765 ∧
      rowMean table4LLaMA3LongRoPE2 / rowMean table4LLaMA3 < 0.985 := by
  norm_num [rowMean, Fin.sum_univ_five, table4Phi3, table4Phi3LongRoPE2, table4LLaMA3,
    table4LLaMA3LongRoPE2]

/-- §1: only LongRoPE2 retains over `97%` of the original average, on both models, from the
printed averages and from the means of the rows. -/
theorem only_over_97 :
    (53.6 / 63.2 < (0.97 : ℝ) ∧ 57.3 / 63.2 < (0.97 : ℝ) ∧ 58.5 / 63.2 < (0.97 : ℝ) ∧
      0.97 < (61.7 : ℝ) / 63.2) ∧
    (52.1 / 56.5 < (0.97 : ℝ) ∧ 54.0 / 56.5 < (0.97 : ℝ) ∧ 54.6 / 56.5 < (0.97 : ℝ) ∧
      0.97 < (55.7 : ℝ) / 56.5) ∧
    (rowMean table4Phi3YaRN / rowMean table4Phi3 < 0.97 ∧
      rowMean table4Phi3NTK / rowMean table4Phi3 < 0.97 ∧
      rowMean table4Phi3LongRoPE / rowMean table4Phi3 < 0.97 ∧
      0.97 < rowMean table4Phi3LongRoPE2 / rowMean table4Phi3) ∧
    (rowMean table4LLaMA3YaRN / rowMean table4LLaMA3 < 0.97 ∧
      rowMean table4LLaMA3NTK / rowMean table4LLaMA3 < 0.97 ∧
      rowMean table4LLaMA3LongRoPE / rowMean table4LLaMA3 < 0.97 ∧
      0.97 < rowMean table4LLaMA3LongRoPE2 / rowMean table4LLaMA3) := by
  norm_num [rowMean, Fin.sum_univ_five, table4Phi3, table4Phi3YaRN, table4Phi3NTK,
    table4Phi3LongRoPE, table4Phi3LongRoPE2, table4LLaMA3, table4LLaMA3YaRN, table4LLaMA3NTK,
    table4LLaMA3LongRoPE, table4LLaMA3LongRoPE2]

/-- §4.2: Phi3-mini's relative declines under YaRN and NTK and their GSM8K drops. -/
theorem phi3_declines :
    -0.1525 ≤ (53.6 : ℝ) / 63.2 - 1 ∧ (53.6 : ℝ) / 63.2 - 1 < -0.1515 ∧
      -0.1495 ≤ rowMean table4Phi3YaRN / rowMean table4Phi3 - 1 ∧
      rowMean table4Phi3YaRN / rowMean table4Phi3 - 1 < -0.1485 ∧
      -0.0935 ≤ (57.3 : ℝ) / 63.2 - 1 ∧ (57.3 : ℝ) / 63.2 - 1 < -0.0925 ∧
      -0.0935 ≤ rowMean table4Phi3NTK / rowMean table4Phi3 - 1 ∧
      rowMean table4Phi3NTK / rowMean table4Phi3 - 1 < -0.0925 ∧
      table4Phi3 4 - table4Phi3YaRN 4 = 21.15 ∧ table4Phi3 4 - table4Phi3NTK 4 = 14.55 := by
  norm_num [rowMean, Fin.sum_univ_five, table4Phi3, table4Phi3YaRN, table4Phi3NTK]

end Transformer.RoPE.LongRoPE2

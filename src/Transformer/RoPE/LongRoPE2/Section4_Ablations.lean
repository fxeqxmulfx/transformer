module

public import Transformer.RoPE.LongRoPE2.Section4_RealWorld

/-!
# LongRoPE2 §4.3 and Appendix B: Tables 5, 6, 7 and 10

Shang et al., *LongRoPE2*, arXiv:2502.20082v1. Tables 5, 7 and 10 list MMLU, MMLU-Pro, GSM8K and
RULER at `4k, …, 128k` (indices `0, 1, 2` and `3, …, 8`); Table 6 lists RULER only, and its
PG19-Needle rows repeat Table 2's LongRoPE2 rows.

* `rcd_long_context`: §4.3, "correcting `d_rcd` improves long-context performance for both
  methods": YaRN-rcd and NTK-rcd beat YaRN and NTK at `32k`, `64k` and `128k` on both models, but
  at `8k` three of the four score lower, and at `16k` two (YaRN-rcd on Phi3-mini, NTK-rcd on
  LLaMA3-8B);
* `longrope2_over_rcd`: §4.3, "LongRoPE2 consistently outperforms YaRN-rcd and NTK-rcd on both
  short- and long-context benchmarks": except on Phi3-mini at `4k`, `90.41` against NTK-rcd's
  `90.51`;
* `needle_search`: §4.3, Table 6, "naive PPL-guided search fails to ensure effective rescaling
  factors": the needle-driven search leads from `16k` on Phi3-mini and at every length on
  LLaMA3-8B; plain PG19 leads at `4k` and `8k` on Phi3-mini;
* `mixed_window_ablation`: §4.3, Table 7, removing mixed context window training "results in a
  significant drop in performance on regular short-context tasks" and the mixed training "also
  improves long-context performance (8k–128k)": both hold on both models, and at `4k` the run
  without it scores higher on both; Table 7's Phi3 row ("Phi3 June") has `86.87` at `8k`, Table 5's
  `87.22`;
* `searched_dims`: Appendix B, Table 10, searching all dimensions "underperforms compared to our
  proposed method": lower on MMLU, MMLU-Pro and RULER, higher on GSM8K, on both models;
* `phi3_mmlu_tables`: Phi3-mini's LongRoPE2 MMLU is `70.04` in Table 4 and `70.07` in Tables 5, 7
  and 10.
-/

@[expose] public section

namespace Transformer.RoPE.LongRoPE2

/-- Table 5 (and Table 10), Phi3-mini, LongRoPE2. -/
noncomputable def table5Phi3LongRoPE2 : Fin 9 → ℝ :=
  ![70.07, 40.30, 73.62, 90.41, 87.22, 83.33, 76.51, 65.37, 58.81]
/-- Table 5, Phi3-mini, YaRN. -/
noncomputable def table5Phi3YaRN : Fin 9 → ℝ :=
  ![63.22, 30.95, 57.39, 85.74, 78.68, 75.97, 65.22, 52.16, 39.37]
/-- Table 5, Phi3-mini, YaRN-rcd. -/
noncomputable def table5Phi3YaRNrcd : Fin 9 → ℝ :=
  ![62.30, 30.24, 56.48, 86.56, 77.66, 74.48, 67.73, 52.73, 44.39]
/-- Table 5, Phi3-mini, NTK. -/
noncomputable def table5Phi3NTK : Fin 9 → ℝ :=
  ![66.43, 36.09, 63.99, 91.34, 87.02, 80.57, 72.81, 61.91, 49.37]
/-- Table 5, Phi3-mini, NTK-rcd. -/
noncomputable def table5Phi3NTKrcd : Fin 9 → ℝ :=
  ![65.31, 35.09, 59.29, 90.51, 85.32, 81.80, 73.89, 63.59, 54.42]
/-- Table 5 (and Tables 7, 10), LLaMA3-8B, LongRoPE2. -/
noncomputable def table5LLaMA3LongRoPE2 : Fin 9 → ℝ :=
  ![65.01, 34.61, 50.80, 94.61, 93.68, 92.31, 90.49, 85.62, 82.03]
/-- Table 5, LLaMA3-8B, YaRN. -/
noncomputable def table5LLaMA3YaRN : Fin 9 → ℝ :=
  ![62.25, 31.88, 42.45, 91.86, 87.87, 84.67, 68.80, 62.51, 49.39]
/-- Table 5, LLaMA3-8B, YaRN-rcd. -/
noncomputable def table5LLaMA3YaRNrcd : Fin 9 → ℝ :=
  ![64.30, 33.17, 50.34, 94.22, 92.02, 89.20, 82.56, 76.37, 71.46]
/-- Table 5, LLaMA3-8B, NTK. -/
noncomputable def table5LLaMA3NTK : Fin 9 → ℝ :=
  ![63.84, 34.14, 46.92, 94.38, 92.64, 91.93, 87.33, 79.26, 73.19]
/-- Table 5, LLaMA3-8B, NTK-rcd. -/
noncomputable def table5LLaMA3NTKrcd : Fin 9 → ℝ :=
  ![64.70, 34.23, 45.87, 94.39, 92.35, 91.43, 88.82, 83.22, 77.25]
/-- Table 6, Phi3-mini, search on PG19-128k PPL. -/
noncomputable def table6Phi3PG19 : Fin 6 → ℝ := ![91.16, 87.93, 83.05, 75.27, 62.72, 50.23]
/-- Table 6, LLaMA3-8B, search on PG19-128k PPL. -/
noncomputable def table6LLaMA3PG19 : Fin 6 → ℝ := ![94.46, 93.36, 91.67, 90.28, 84.55, 78.68]
/-- Table 7, Phi3 June, LongRoPE2. -/
noncomputable def table7Phi3 : Fin 9 → ℝ :=
  ![70.07, 40.30, 73.62, 90.41, 86.87, 83.33, 76.51, 65.37, 58.81]
/-- Table 7, Phi3 June, LongRoPE2 without mixed context window training. -/
noncomputable def table7Phi3Without : Fin 9 → ℝ :=
  ![66.56, 34.86, 64.67, 90.55, 85.77, 81.08, 73.31, 63.75, 56.22]
/-- Table 7, LLaMA3-8B, LongRoPE2 without mixed context window training. -/
noncomputable def table7LLaMA3Without : Fin 9 → ℝ :=
  ![64.57, 33.83, 48.37, 94.67, 93.15, 91.24, 89.38, 83.53, 80.18]
/-- Table 10, Phi3-mini, LongRoPE2 searching all dimensions. -/
noncomputable def table10Phi3All : Fin 9 → ℝ :=
  ![69.96, 39.84, 74.83, 90.02, 87.21, 82.42, 74.86, 63.95, 57.34]
/-- Table 10, LLaMA3-8B, LongRoPE2 searching all dimensions. -/
noncomputable def table10LLaMA3All : Fin 9 → ℝ :=
  ![64.34, 33.83, 51.55, 93.92, 92.61, 91.41, 89.30, 83.11, 78.07]

/-- §4.3, Table 5: the rcd variants gain from `32k` on; at `8k` three of the four lose, at `16k`
two. -/
theorem rcd_long_context :
    (∀ j : Fin 9, 6 ≤ j → table5Phi3YaRN j < table5Phi3YaRNrcd j ∧
      table5Phi3NTK j < table5Phi3NTKrcd j ∧ table5LLaMA3YaRN j < table5LLaMA3YaRNrcd j ∧
      table5LLaMA3NTK j < table5LLaMA3NTKrcd j) ∧
    table5Phi3YaRNrcd 4 < table5Phi3YaRN 4 ∧ table5Phi3NTKrcd 4 < table5Phi3NTK 4 ∧
    table5LLaMA3NTKrcd 4 < table5LLaMA3NTK 4 ∧ table5LLaMA3YaRN 4 < table5LLaMA3YaRNrcd 4 ∧
    table5Phi3YaRNrcd 5 < table5Phi3YaRN 5 ∧ table5Phi3NTK 5 < table5Phi3NTKrcd 5 ∧
    table5LLaMA3YaRN 5 < table5LLaMA3YaRNrcd 5 ∧ table5LLaMA3NTKrcd 5 < table5LLaMA3NTK 5 := by
  refine ⟨fun j hj => ?_, by norm_num [table5Phi3YaRNrcd, table5Phi3YaRN, table5Phi3NTKrcd,
    table5Phi3NTK, table5LLaMA3NTKrcd, table5LLaMA3NTK, table5LLaMA3YaRN, table5LLaMA3YaRNrcd]⟩
  fin_cases j <;> simp_all [table5Phi3YaRN, table5Phi3YaRNrcd, table5Phi3NTK, table5Phi3NTKrcd,
    table5LLaMA3YaRN, table5LLaMA3YaRNrcd, table5LLaMA3NTK, table5LLaMA3NTKrcd, Fin.le_def] <;>
    norm_num

/-- §4.3, Table 5: LongRoPE2 leads YaRN-rcd and NTK-rcd everywhere but on Phi3-mini at `4k`. -/
theorem longrope2_over_rcd :
    (∀ j : Fin 9, table5Phi3YaRNrcd j < table5Phi3LongRoPE2 j) ∧
    (∀ j : Fin 9, j ≠ 3 → table5Phi3NTKrcd j < table5Phi3LongRoPE2 j) ∧
    table5Phi3LongRoPE2 3 < table5Phi3NTKrcd 3 ∧
    ∀ j : Fin 9, table5LLaMA3YaRNrcd j < table5LLaMA3LongRoPE2 j ∧
      table5LLaMA3NTKrcd j < table5LLaMA3LongRoPE2 j := by
  refine ⟨fun j => ?_, fun j hj => ?_, by norm_num [table5Phi3LongRoPE2, table5Phi3NTKrcd],
    fun j => ?_⟩ <;>
    fin_cases j <;> simp_all [table5Phi3YaRNrcd, table5Phi3NTKrcd, table5Phi3LongRoPE2,
      table5LLaMA3YaRNrcd, table5LLaMA3NTKrcd, table5LLaMA3LongRoPE2] <;> norm_num

/-- §4.3, Table 6: the needle-driven search against plain PG19 perplexity. -/
theorem needle_search :
    (∀ j : Fin 6, 2 ≤ j → table6Phi3PG19 j < table2Phi3LongRoPE2 j) ∧
    table2Phi3LongRoPE2 0 < table6Phi3PG19 0 ∧ table2Phi3LongRoPE2 1 < table6Phi3PG19 1 ∧
    ∀ j : Fin 6, table6LLaMA3PG19 j < table2LLaMA3LongRoPE2 j := by
  refine ⟨fun j hj => ?_, by norm_num [table2Phi3LongRoPE2, table6Phi3PG19],
    by norm_num [table2Phi3LongRoPE2, table6Phi3PG19], fun j => ?_⟩ <;>
    fin_cases j <;> simp_all [table6Phi3PG19, table2Phi3LongRoPE2, table6LLaMA3PG19,
      table2LLaMA3LongRoPE2, Fin.le_def] <;> norm_num

/-- §4.3, Table 7: mixed context window training raises the short tasks and RULER from `8k` on;
at `4k` the run without it scores higher. -/
theorem mixed_window_ablation :
    (∀ j : Fin 9, j ≠ 3 → table7Phi3Without j < table7Phi3 j ∧
      table7LLaMA3Without j < table5LLaMA3LongRoPE2 j) ∧
    table7Phi3 3 < table7Phi3Without 3 ∧ table5LLaMA3LongRoPE2 3 < table7LLaMA3Without 3 ∧
    table7Phi3 4 = 86.87 ∧ table5Phi3LongRoPE2 4 = 87.22 := by
  refine ⟨fun j hj => ?_, by norm_num [table7Phi3, table7Phi3Without],
    by norm_num [table5LLaMA3LongRoPE2, table7LLaMA3Without], by norm_num [table7Phi3],
    by norm_num [table5Phi3LongRoPE2]⟩
  fin_cases j <;> simp_all [table7Phi3Without, table7Phi3, table7LLaMA3Without,
    table5LLaMA3LongRoPE2] <;> norm_num

/-- Appendix B, Table 10: searching all dimensions scores lower everywhere but on GSM8K. -/
theorem searched_dims :
    (∀ j : Fin 9, j ≠ 2 → table10Phi3All j < table5Phi3LongRoPE2 j ∧
      table10LLaMA3All j < table5LLaMA3LongRoPE2 j) ∧
    table5Phi3LongRoPE2 2 < table10Phi3All 2 ∧ table5LLaMA3LongRoPE2 2 < table10LLaMA3All 2 := by
  refine ⟨fun j hj => ?_, by norm_num [table5Phi3LongRoPE2, table10Phi3All],
    by norm_num [table5LLaMA3LongRoPE2, table10LLaMA3All]⟩
  fin_cases j <;> simp_all [table10Phi3All, table5Phi3LongRoPE2, table10LLaMA3All,
    table5LLaMA3LongRoPE2] <;> norm_num

/-- Tables 4, 5 and 7: Phi3-mini's LongRoPE2 MMLU is `70.04` in one and `70.07` in the others. -/
theorem phi3_mmlu_tables :
    table4Phi3LongRoPE2 0 = 70.04 ∧ table5Phi3LongRoPE2 0 = 70.07 ∧ table7Phi3 0 = 70.07 := by
  norm_num [table4Phi3LongRoPE2, table5Phi3LongRoPE2, table7Phi3]

end Transformer.RoPE.LongRoPE2

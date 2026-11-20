module

public import Transformer.RoPE.LongRoPE2.Section4_Ablations

/-!
# LongRoPE2 Appendix B: Tables 8 and 9, RULER task by task

Shang et al., *LongRoPE2*, arXiv:2502.20082v1, Tables 8 (Phi3-mini) and 9 (LLaMA3-8B): thirteen
RULER tasks (eight NIAH variants, VT, CWE at index `9`, FEW, single-hop and multi-hop QA) at
`4096, …, 131072` tokens (indices `0, …, 5`), with an average printed to two decimals.

* `table8_averages`: every average of Table 8 is its row's mean rounded; all but the `8192` one
  repeat Table 2's LongRoPE2 row for Phi3-mini, and at `8192` Tables 8, 2 and 7 print three values,
  `87.34`, `87.22` and `86.87`;
* `table9_averages`: every average of Table 9 repeats Table 2's LongRoPE2 row for LLaMA3-8B, and
  all but the `4096` one are their rows' means rounded; the `4096` row's mean is
  `94.61 + 0.45 / 13 ≈ 94.64`;
* `phi3_cwe`: §4.2, "the 128k average score of 58.81 is skewed by tasks with low scores on smaller
  LLMs, such as CWE, which achieves only 1% accuracy": Table 8 prints `0.3` for CWE at `131072`,
  and without it the twelve other tasks average `63.69`.
-/

@[expose] public section

namespace Transformer.RoPE.LongRoPE2

/-- Table 8, Phi3-mini with LongRoPE2: the thirteen RULER tasks at each length. -/
noncomputable def table8Phi3 : Fin 6 → Fin 13 → ℝ :=
  ![![100, 100, 99, 91, 96, 97, 97.75, 97.75, 85.8, 93.7, 85.33, 82, 50],
    ![100, 100, 100, 90, 93, 97, 89.5, 93.75, 84, 87.2, 86, 68, 47],
    ![100, 100, 99, 87, 88, 82, 91.25, 89, 85, 55.4, 91.67, 70, 45],
    ![100, 100, 99, 86, 86, 57, 87, 78, 76.8, 33.2, 91.67, 56, 44],
    ![100, 100, 99, 85, 71, 32, 67.75, 69.25, 66.8, 0.4, 71.67, 50, 37],
    ![100, 98, 95, 92, 40, 18, 56.75, 59, 35.2, 0.3, 89.33, 47, 34]]
/-- Table 8, the printed averages. -/
noncomputable def table8Phi3Avg : Fin 6 → ℝ := ![90.41, 87.34, 83.33, 76.51, 65.37, 58.81]
/-- Table 9, LLaMA3-8B with LongRoPE2: the thirteen RULER tasks at each length. -/
noncomputable def table9LLaMA3 : Fin 6 → Fin 13 → ℝ :=
  ![![100, 100, 99, 100, 100, 100, 99, 99.75, 98.8, 98.5, 96.33, 79, 60],
    ![100, 100, 100, 100, 100, 100, 99, 99.75, 99.8, 95.9, 91.33, 74, 58],
    ![100, 100, 100, 99, 100, 98, 95, 98.25, 99.6, 86.8, 96.33, 69, 58],
    ![100, 100, 100, 99, 98, 100, 98, 96.25, 98.6, 63.9, 95.67, 72, 55],
    ![100, 100, 100, 98, 98, 95, 95.75, 99.75, 98.6, 33.6, 80.33, 62, 52],
    ![100, 100, 99, 96, 91, 94, 96.5, 97, 92.6, 9, 85.33, 56, 50]]
/-- Table 9, the printed averages. -/
noncomputable def table9LLaMA3Avg : Fin 6 → ℝ := ![94.61, 93.68, 92.31, 90.49, 85.62, 82.03]

/-- Appendix B, Table 8: the averages are the rows' means; at `8192` they disagree with Tables 2
and 7. -/
theorem table8_averages :
    (∀ j : Fin 6, |rowMean (table8Phi3 j) - table8Phi3Avg j| < 0.005) ∧
      (∀ j : Fin 6, j ≠ 1 → table8Phi3Avg j = table2Phi3LongRoPE2 j) ∧
      table8Phi3Avg 1 = 87.34 ∧ table2Phi3LongRoPE2 1 = 87.22 ∧ table7Phi3 4 = 86.87 := by
  refine ⟨fun j => ?_, fun j hj => ?_, by norm_num [table8Phi3Avg],
    by norm_num [table2Phi3LongRoPE2], by norm_num [table7Phi3]⟩
  · fin_cases j <;> norm_num [rowMean, Fin.sum_univ_succ, table8Phi3, table8Phi3Avg, abs_lt]
  · fin_cases j <;> simp_all [table8Phi3Avg, table2Phi3LongRoPE2]

/-- Appendix B, Table 9: the averages repeat Table 2; the `4096` one is not its row's mean. -/
theorem table9_averages :
    (∀ j : Fin 6, j ≠ 0 → |rowMean (table9LLaMA3 j) - table9LLaMA3Avg j| < 0.005) ∧
      rowMean (table9LLaMA3 0) - table9LLaMA3Avg 0 = 0.45 / 13 ∧
      ∀ j : Fin 6, table9LLaMA3Avg j = table2LLaMA3LongRoPE2 j := by
  refine ⟨fun j hj => ?_, ?_, fun j => ?_⟩
  · fin_cases j <;> simp_all <;>
      norm_num [rowMean, Fin.sum_univ_succ, table9LLaMA3, table9LLaMA3Avg, abs_lt]
  · norm_num [rowMean, Fin.sum_univ_succ, table9LLaMA3, table9LLaMA3Avg]
  · fin_cases j <;> simp [table9LLaMA3Avg, table2LLaMA3LongRoPE2]

/-- §4.2 and Table 8: CWE on Phi3-mini at `131072` scores `0.3`; the other twelve tasks average
`63.69`. -/
theorem phi3_cwe :
    table8Phi3 5 9 = 0.3 ∧ (∑ i, table8Phi3 5 i - table8Phi3 5 9) / 12 = 63.69 := by
  norm_num [Fin.sum_univ_succ, table8Phi3]

end Transformer.RoPE.LongRoPE2

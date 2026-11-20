module

public import Transformer.RoPE.LongRoPE2.Section4_RULER

/-!
# LongRoPE2 §4.2 and Table 3: real-world long-context benchmarks

Shang et al., *LongRoPE2*, arXiv:2502.20082v1, Table 3: LOFT's seven tasks (ArguAna, FEVER,
HotPotQA, MS MACRO, NQ, Quora, SciFact) with their average, and InfiniteBench–LongBench's six (KV
retrieval, En.MC, TriviaQA, TREC, LCC, RepoBench-P) with theirs, printed to two decimals.

* `table3_averages`: every printed average lies within `0.01` of its row's mean, except
  InfiniteBench–LongBench's for YaRN on LLaMA3-8B: `51.81` for the mean `53.808…`;
* `table3_best`: §4.2, "LongRoPE2 consistently improves performance across all benchmarks": its
  two averages lead on both models; task by task, on Phi3-mini LongRoPE leads SciFact (`16`
  against `14`) and YaRN leads LCC (`63.98` against `62.67`).
-/

@[expose] public section

namespace Transformer.RoPE.LongRoPE2

/-- Table 3, LOFT, Phi3-mini, YaRN. -/
noncomputable def loftPhi3YaRN : Fin 7 → ℝ := ![4, 4, 0, 8, 12, 1, 12]
/-- Table 3, LOFT, Phi3-mini, NTK. -/
noncomputable def loftPhi3NTK : Fin 7 → ℝ := ![0, 21, 0, 6, 13, 4, 9]
/-- Table 3, LOFT, Phi3-mini, LongRoPE. -/
noncomputable def loftPhi3LongRoPE : Fin 7 → ℝ := ![5, 64, 3, 17, 35, 8, 16]
/-- Table 3, LOFT, Phi3-mini, LongRoPE2. -/
noncomputable def loftPhi3LongRoPE2 : Fin 7 → ℝ := ![5, 70, 4, 19, 39, 10, 14]
/-- Table 3, LOFT, LLaMA3-8B, YaRN. -/
noncomputable def loftLLaMA3YaRN : Fin 7 → ℝ := ![7, 62, 15, 21, 43, 23, 12]
/-- Table 3, LOFT, LLaMA3-8B, NTK. -/
noncomputable def loftLLaMA3NTK : Fin 7 → ℝ := ![22, 96, 53, 75, 89, 71, 64]
/-- Table 3, LOFT, LLaMA3-8B, LongRoPE. -/
noncomputable def loftLLaMA3LongRoPE : Fin 7 → ℝ := ![22, 96, 25, 57, 90, 74, 62]
/-- Table 3, LOFT, LLaMA3-8B, LongRoPE2. -/
noncomputable def loftLLaMA3LongRoPE2 : Fin 7 → ℝ := ![28, 96, 70, 80, 94, 79, 73]
/-- Table 3, InfiniteBench–LongBench, Phi3-mini, YaRN. -/
noncomputable def benchPhi3YaRN : Fin 6 → ℝ := ![5.8, 31.44, 84.35, 61.00, 63.98, 59.23]
/-- Table 3, InfiniteBench–LongBench, Phi3-mini, NTK. -/
noncomputable def benchPhi3NTK : Fin 6 → ℝ := ![5.1, 37.55, 84.01, 65.00, 62.36, 59.82]
/-- Table 3, InfiniteBench–LongBench, Phi3-mini, LongRoPE. -/
noncomputable def benchPhi3LongRoPE : Fin 6 → ℝ := ![5.6, 35.81, 86.47, 62.50, 55.25, 58.43]
/-- Table 3, InfiniteBench–LongBench, Phi3-mini, LongRoPE2. -/
noncomputable def benchPhi3LongRoPE2 : Fin 6 → ℝ := ![12.0, 42.36, 87.27, 67.00, 62.67, 60.10]
/-- Table 3, InfiniteBench–LongBench, LLaMA3-8B, YaRN. -/
noncomputable def benchLLaMA3YaRN : Fin 6 → ℝ := ![2.2, 30.57, 88.97, 73.50, 65.40, 62.21]
/-- Table 3, InfiniteBench–LongBench, LLaMA3-8B, NTK. -/
noncomputable def benchLLaMA3NTK : Fin 6 → ℝ := ![66.0, 42.79, 90.87, 74.00, 68.67, 65.55]
/-- Table 3, InfiniteBench–LongBench, LLaMA3-8B, LongRoPE. -/
noncomputable def benchLLaMA3LongRoPE : Fin 6 → ℝ := ![74.0, 45.85, 89.99, 76.00, 69.13, 67.38]
/-- Table 3, InfiniteBench–LongBench, LLaMA3-8B, LongRoPE2. -/
noncomputable def benchLLaMA3LongRoPE2 : Fin 6 → ℝ :=
  ![88.0, 46.72, 91.13, 76.50, 70.47, 67.39]

/-- Table 3: the printed averages against the means of the rows. -/
theorem table3_averages :
    |rowMean loftPhi3YaRN - 5.86| < 0.01 ∧ |rowMean loftPhi3NTK - 7.57| < 0.01 ∧
      |rowMean loftPhi3LongRoPE - 21.14| < 0.01 ∧ |rowMean loftPhi3LongRoPE2 - 23.00| < 0.01 ∧
      |rowMean loftLLaMA3YaRN - 26.14| < 0.01 ∧ |rowMean loftLLaMA3NTK - 67.14| < 0.01 ∧
      |rowMean loftLLaMA3LongRoPE - 60.85| < 0.01 ∧ |rowMean loftLLaMA3LongRoPE2 - 74.28| < 0.01 ∧
      |rowMean benchPhi3YaRN - 50.96| < 0.01 ∧ |rowMean benchPhi3NTK - 52.31| < 0.01 ∧
      |rowMean benchPhi3LongRoPE - 50.67| < 0.01 ∧ |rowMean benchPhi3LongRoPE2 - 55.23| < 0.01 ∧
      1.99 < rowMean benchLLaMA3YaRN - 51.81 ∧ |rowMean benchLLaMA3NTK - 67.98| < 0.01 ∧
      |rowMean benchLLaMA3LongRoPE - 70.39| < 0.01 ∧
      |rowMean benchLLaMA3LongRoPE2 - 73.37| < 0.01 := by
  norm_num [rowMean, Fin.sum_univ_seven, Fin.sum_univ_six, loftPhi3YaRN, loftPhi3NTK,
    loftPhi3LongRoPE, loftPhi3LongRoPE2, loftLLaMA3YaRN, loftLLaMA3NTK, loftLLaMA3LongRoPE,
    loftLLaMA3LongRoPE2, benchPhi3YaRN, benchPhi3NTK, benchPhi3LongRoPE, benchPhi3LongRoPE2,
    benchLLaMA3YaRN, benchLLaMA3NTK, benchLLaMA3LongRoPE, benchLLaMA3LongRoPE2, abs_lt]

/-- §4.2: LongRoPE2's averages lead on both models; on Phi3-mini LongRoPE leads SciFact and YaRN
leads LCC. -/
theorem table3_best :
    (rowMean loftPhi3YaRN < rowMean loftPhi3LongRoPE2 ∧
      rowMean loftPhi3NTK < rowMean loftPhi3LongRoPE2 ∧
      rowMean loftPhi3LongRoPE < rowMean loftPhi3LongRoPE2) ∧
    (rowMean benchPhi3YaRN < rowMean benchPhi3LongRoPE2 ∧
      rowMean benchPhi3NTK < rowMean benchPhi3LongRoPE2 ∧
      rowMean benchPhi3LongRoPE < rowMean benchPhi3LongRoPE2) ∧
    (rowMean loftLLaMA3YaRN < rowMean loftLLaMA3LongRoPE2 ∧
      rowMean loftLLaMA3NTK < rowMean loftLLaMA3LongRoPE2 ∧
      rowMean loftLLaMA3LongRoPE < rowMean loftLLaMA3LongRoPE2) ∧
    (rowMean benchLLaMA3YaRN < rowMean benchLLaMA3LongRoPE2 ∧
      rowMean benchLLaMA3NTK < rowMean benchLLaMA3LongRoPE2 ∧
      rowMean benchLLaMA3LongRoPE < rowMean benchLLaMA3LongRoPE2) ∧
    loftPhi3LongRoPE2 6 < loftPhi3LongRoPE 6 ∧ benchPhi3LongRoPE2 4 < benchPhi3YaRN 4 := by
  norm_num [rowMean, Fin.sum_univ_seven, Fin.sum_univ_six, loftPhi3YaRN, loftPhi3NTK,
    loftPhi3LongRoPE, loftPhi3LongRoPE2, loftLLaMA3YaRN, loftLLaMA3NTK, loftLLaMA3LongRoPE,
    loftLLaMA3LongRoPE2, benchPhi3YaRN, benchPhi3NTK, benchPhi3LongRoPE, benchPhi3LongRoPE2,
    benchLLaMA3YaRN, benchLLaMA3NTK, benchLLaMA3LongRoPE, benchLLaMA3LongRoPE2]

end Transformer.RoPE.LongRoPE2

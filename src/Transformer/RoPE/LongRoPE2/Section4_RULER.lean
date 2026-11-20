module

public import Transformer.RoPE.LongRoPE2.Section4_ShortTasks

/-!
# LongRoPE2 §4.2 and Table 2: RULER

Shang et al., *LongRoPE2*, arXiv:2502.20082v1, Table 2, the RULER average over 13 tasks at
`4k, 8k, 16k, 32k, 64k, 128k` (indices `0, …, 5`).

* `ruler_phi3_4k`: §4.2, "Across Phi3-mini-3.8B and LLaMA3-8B, LongRoPE2 consistently outperforms
  prior methods, achieving superior results across all evaluation lengths within the 128k
  window": on Phi3-mini at `4k` NTK scores `91.34` against LongRoPE2's `90.41`; at the five longer
  lengths LongRoPE2 leads;
* `ruler_llama3`: on LLaMA3-8B LongRoPE2 leads at every length, "maintaining a strong score of
  82.03 at 128k", and "LongRoPE, the prior best, drops from 81.23 (64k) to 73.40 at 128k": LongRoPE
  leads YaRN and NTK at `64k` and `128k` and loses `7.83` points between them;
* `ruler_phi3_ntk_longrope`: §4.2, "NTK performs well below 32k and declines sharply beyond, while
  LongRoPE underperforms at shorter contexts": on Phi3-mini NTK leads LongRoPE up to `32k` and
  LongRoPE leads at `64k` and `128k`.
-/

@[expose] public section

namespace Transformer.RoPE.LongRoPE2

/-- Table 2, Phi3-mini, YaRN. -/
noncomputable def table2Phi3YaRN : Fin 6 → ℝ := ![85.74, 78.68, 75.97, 65.22, 52.16, 39.37]
/-- Table 2, Phi3-mini, NTK. -/
noncomputable def table2Phi3NTK : Fin 6 → ℝ := ![91.34, 87.02, 80.57, 72.81, 61.91, 49.37]
/-- Table 2, Phi3-mini, LongRoPE. -/
noncomputable def table2Phi3LongRoPE : Fin 6 → ℝ := ![88.40, 83.23, 79.46, 71.20, 64.63, 53.71]
/-- Table 2, Phi3-mini, LongRoPE2. -/
noncomputable def table2Phi3LongRoPE2 : Fin 6 → ℝ := ![90.41, 87.22, 83.33, 76.51, 65.37, 58.81]
/-- Table 2, LLaMA3-8B, YaRN. -/
noncomputable def table2LLaMA3YaRN : Fin 6 → ℝ := ![91.86, 87.87, 84.67, 68.80, 62.51, 49.39]
/-- Table 2, LLaMA3-8B, NTK. -/
noncomputable def table2LLaMA3NTK : Fin 6 → ℝ := ![94.38, 92.64, 91.93, 87.33, 79.26, 73.19]
/-- Table 2, LLaMA3-8B, LongRoPE. -/
noncomputable def table2LLaMA3LongRoPE : Fin 6 → ℝ := ![94.60, 92.70, 91.01, 86.60, 81.23, 73.40]
/-- Table 2, LLaMA3-8B, LongRoPE2. -/
noncomputable def table2LLaMA3LongRoPE2 : Fin 6 → ℝ :=
  ![94.61, 93.68, 92.31, 90.49, 85.62, 82.03]

/-- §4.2: on Phi3-mini NTK leads LongRoPE2 at `4k`; LongRoPE2 leads every prior method at the
longer lengths. -/
theorem ruler_phi3_4k :
    table2Phi3LongRoPE2 0 < table2Phi3NTK 0 ∧
      ∀ j : Fin 6, j ≠ 0 → table2Phi3YaRN j < table2Phi3LongRoPE2 j ∧
        table2Phi3NTK j < table2Phi3LongRoPE2 j ∧ table2Phi3LongRoPE j < table2Phi3LongRoPE2 j := by
  refine ⟨by norm_num [table2Phi3LongRoPE2, table2Phi3NTK], fun j hj => ?_⟩
  fin_cases j <;> simp_all [table2Phi3YaRN, table2Phi3NTK, table2Phi3LongRoPE,
    table2Phi3LongRoPE2] <;> norm_num

/-- §4.2: on LLaMA3-8B LongRoPE2 leads at every length; LongRoPE leads YaRN and NTK at `64k` and
`128k` and drops `7.83` points between them. -/
theorem ruler_llama3 :
    (∀ j : Fin 6, table2LLaMA3YaRN j < table2LLaMA3LongRoPE2 j ∧
      table2LLaMA3NTK j < table2LLaMA3LongRoPE2 j ∧
      table2LLaMA3LongRoPE j < table2LLaMA3LongRoPE2 j) ∧
    table2LLaMA3LongRoPE2 5 = 82.03 ∧
    (∀ j : Fin 6, 4 ≤ j → table2LLaMA3YaRN j < table2LLaMA3LongRoPE j ∧
      table2LLaMA3NTK j < table2LLaMA3LongRoPE j) ∧
    table2LLaMA3LongRoPE 4 - table2LLaMA3LongRoPE 5 = 7.83 := by
  refine ⟨fun j => ?_, by norm_num [table2LLaMA3LongRoPE2], fun j hj => ?_,
    by norm_num [table2LLaMA3LongRoPE]⟩
  · fin_cases j <;> norm_num [table2LLaMA3YaRN, table2LLaMA3NTK, table2LLaMA3LongRoPE,
      table2LLaMA3LongRoPE2]
  · fin_cases j <;> simp_all [table2LLaMA3YaRN, table2LLaMA3NTK, table2LLaMA3LongRoPE,
      Fin.le_def] <;> norm_num

/-- §4.2: on Phi3-mini NTK leads LongRoPE up to `32k`, LongRoPE leads at `64k` and `128k`. -/
theorem ruler_phi3_ntk_longrope :
    (∀ j : Fin 6, j ≤ 3 → table2Phi3LongRoPE j < table2Phi3NTK j) ∧
      ∀ j : Fin 6, 4 ≤ j → table2Phi3NTK j < table2Phi3LongRoPE j := by
  refine ⟨fun j hj => ?_, fun j hj => ?_⟩ <;>
    fin_cases j <;> simp_all [table2Phi3NTK, table2Phi3LongRoPE, Fin.le_def] <;> norm_num

end Transformer.RoPE.LongRoPE2

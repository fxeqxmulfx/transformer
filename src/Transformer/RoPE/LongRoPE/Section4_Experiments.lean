module

public import Transformer.RoPE.LongRoPE.Section3_Progressive

/-!
# LongRoPE §4: the arithmetic of Tables 8 and 10

Ding et al., arXiv:2402.13753v1, §§4.2–4.3. The scores are measurements; the arithmetic the text
and Table 10 draw from them is checked here. Table 8 lists ARC-c, HellaSwag, MMLU and TruthfulQA.

* `truthfulQA_gain`: §4.2, "our models achieve comparable results on the original benchmark …,
  and even outperform the original Mistral on TruthfulQA by +0.5%": LongRoPE-Mistral-2048k
  (ft=128k) scores `43.1` against `42.6`, `0.5` points, and lower on the other three benchmarks;
* `recovered_average`: §4.3, Table 10's average benchmark accuracies after the short-window
  recovery, `52.9` (ft=128k) and `50.8` (ft=256k), are the means of the four scores of the
  LongRoPE-LLaMA2-2048k rows of Table 8.
-/

@[expose] public section

namespace Transformer.RoPE.LongRoPE

/-- Table 8, the original Mistral-7B. -/
noncomputable def table8Mistral : Fin 4 → ℝ := ![60.6, 83.2, 63.6, 42.6]

/-- Table 8, LongRoPE-Mistral-2048k (ft=128k). -/
noncomputable def table8LongMistral : Fin 4 → ℝ := ![59.0, 81.2, 61.3, 43.1]

/-- Table 8, LongRoPE-LLaMA2-2048k (ft=128k). -/
noncomputable def table8LongLLaMA128 : Fin 4 → ℝ := ![52.9, 76.5, 43.4, 38.8]

/-- Table 8, LongRoPE-LLaMA2-2048k (ft=256k). -/
noncomputable def table8LongLLaMA256 : Fin 4 → ℝ := ![51.0, 75.3, 39.6, 37.3]

/-- §4.2: on TruthfulQA LongRoPE-Mistral-2048k (ft=128k) gains `0.5` points over Mistral-7B; on
ARC-c, HellaSwag and MMLU it scores lower. -/
theorem truthfulQA_gain :
    table8LongMistral 3 - table8Mistral 3 = 0.5 ∧
      ∀ i, i ≠ 3 → table8LongMistral i < table8Mistral i := by
  refine ⟨by norm_num [table8LongMistral, table8Mistral], fun i hi => ?_⟩
  fin_cases i <;> simp_all [table8LongMistral, table8Mistral] <;> norm_num

/-- §4.3: Table 10's average accuracies after recovery are the means of Table 8's rows. -/
theorem recovered_average :
    (∑ i, table8LongLLaMA128 i) / 4 = 52.9 ∧ (∑ i, table8LongLLaMA256 i) / 4 = 50.8 := by
  simp only [Fin.sum_univ_four, table8LongLLaMA128, table8LongLLaMA256]
  norm_num

end Transformer.RoPE.LongRoPE

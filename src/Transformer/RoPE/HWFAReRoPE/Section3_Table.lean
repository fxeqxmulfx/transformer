module

public import Transformer.RoPE.HWFAReRoPE.Section3_Windows

/-!
# HWFA meets ReRoPE, 实验 (Experiments) and 小结 (Summary): what the table shows

Su, *Transformer升级之路：14、当HWFA遇见ReRoPE*, kexue.fm/archives/9731, sections 实验
(Experiments) and 小结 (Summary), on the table of `Section3_Windows`: the eight HFWA-ReRoPE runs
`runs` (`w64-f2` is the run `5`) and the whole table `table14`, columns `512` (training), `4096`
repeated, `4096` not repeated.

* `train_gap`, `train_full_layers`: "after increasing `w̃` the training performance quickly aligns
  with the baseline, and adding full attention even exceeds it": HWFA trails the baseline's `49.41`
  by `0.71` at `512`, each run by at most `0.41`, the `f1` runs at `w32` and `w64` by `0.12` and
  `0.09`; each `f2` run leads its `f1` run, and the `f2` runs at `w32` and `w64` exceed the
  baseline, those at `w128` and `w256` do not;
* `hwfa_lt_w32_w64`: "the `w32` and `w64` cases are both quite good, clearly exceeding HFWA": on
  both `4096` columns they lead HWFA by more than one point; `w128-f1` and both `w256` runs trail
  it on the repeated column;
* `w64f2_best`: "the best combination is `w64-f2`": among the eight runs it has the largest sum of
  the three columns, `183.25`; at `512` only `w32-f2` leads it, on the repeated column none, on the
  other column only `w64-f1`, so it leads every other run in two columns at least;
  `w64f2_rerope`: "its training performance and its extrapolation without repetition both exceed
  the original ReRoPE": it leads the three ReRoPE-w256 rows at `512` and without repetition, while
  on the repeated column ReRoPE-w256-log n's `85.12` stays ahead of its `84.43`;
* `bold`: the bold `85.12` and `49.37` are the largest entries of their columns; the bold `49.50`
  of `w32-f2` at `512` is the largest of the eight runs, not of the table, where part 13's `b8` run
  has `49.62`;
* `summary`: Summary, "without losing training performance it reaches nearly the best
  extrapolation": `w64-f2` is at least the baseline at `512` and within `0.69` and `0.01` of the
  largest entries of the two `4096` columns. The Summary's lower inference cost is
  `cost_lt_rerope`.
-/

@[expose] public section

namespace Transformer.RoPE.HWFAReRoPE

open InvLeakyReRoPE

/-- Experiments: at `512` HWFA trails the baseline by `0.71`, each run by at most `0.41`, and the
`f1` runs at `w32` and `w64` by `0.12` and `0.09`. -/
theorem train_gap :
    tableBaseline 0 - tableHWFA 0 = 0.71 ∧ (∀ r : Fin 8, tableBaseline 0 - runs r 0 ≤ 0.41) ∧
      tableBaseline 0 - tableW32F1 0 = 0.12 ∧ tableBaseline 0 - tableW64F1 0 = 0.09 := by
  refine ⟨by norm_num [tableBaseline, tableHWFA], fun r => ?_,
    by norm_num [tableBaseline, tableW32F1], by norm_num [tableBaseline, tableW64F1]⟩
  fin_cases r <;> norm_num [runs, tableBaseline, tableW32F1, tableW64F1, tableW128F1,
    tableW256F1, tableW32F2, tableW64F2, tableW128F2, tableW256F2]

/-- Experiments: at `512` each `f2` run leads its `f1` run; the `f2` runs at `w32` and `w64`
exceed the baseline, those at `w128` and `w256` do not. -/
theorem train_full_layers :
    tableW32F1 0 < tableW32F2 0 ∧ tableW64F1 0 < tableW64F2 0 ∧
      tableW128F1 0 < tableW128F2 0 ∧ tableW256F1 0 < tableW256F2 0 ∧
      tableBaseline 0 < tableW32F2 0 ∧ tableBaseline 0 < tableW64F2 0 ∧
      tableW128F2 0 < tableBaseline 0 ∧ tableW256F2 0 < tableBaseline 0 := by
  norm_num [tableBaseline, tableW32F1, tableW64F1, tableW128F1, tableW256F1, tableW32F2,
    tableW64F2, tableW128F2, tableW256F2]

/-- Experiments: on both `4096` columns the four runs at `w32` and `w64` lead HWFA by more than
one point; on the repeated column `w128-f1` and both `w256` runs trail it. -/
theorem hwfa_lt_w32_w64 :
    (∀ j : Fin 3, j ≠ 0 → tableHWFA j + 1 < tableW32F1 j ∧ tableHWFA j + 1 < tableW64F1 j ∧
      tableHWFA j + 1 < tableW32F2 j ∧ tableHWFA j + 1 < tableW64F2 j) ∧
      tableW128F1 1 < tableHWFA 1 ∧ tableW256F1 1 < tableHWFA 1 ∧
      tableW256F2 1 < tableHWFA 1 := by
  refine ⟨fun j hj => ?_, by norm_num [tableW128F1, tableHWFA],
    by norm_num [tableW256F1, tableHWFA], by norm_num [tableW256F2, tableHWFA]⟩
  fin_cases j <;> simp_all [tableHWFA, tableW32F1, tableW64F1, tableW32F2, tableW64F2] <;>
    norm_num

/-- Experiments, "the best combination is `w64-f2`": among the eight runs it has the largest sum
of the three columns, `183.25`; at `512` only `w32-f2` (run `4`) leads it, on the repeated column
none, without repetition only `w64-f1` (run `1`). -/
theorem w64f2_best :
    ∑ j, tableW64F2 j = 183.25 ∧ (∀ r : Fin 8, r ≠ 5 → ∑ j, runs r j < ∑ j, tableW64F2 j) ∧
      (∀ r : Fin 8, tableW64F2 0 < runs r 0 → r = 4) ∧ (∀ r : Fin 8, runs r 1 ≤ tableW64F2 1) ∧
      ∀ r : Fin 8, tableW64F2 2 < runs r 2 → r = 1 := by
  refine ⟨by norm_num [Fin.sum_univ_three, tableW64F2], fun r => ?_, fun r => ?_, fun r => ?_,
    fun r => ?_⟩ <;>
    fin_cases r <;> norm_num [Fin.sum_univ_three, runs, tableW32F1, tableW64F1, tableW128F1,
      tableW256F1, tableW32F2, tableW64F2, tableW128F2, tableW256F2]

/-- Experiments: `w64-f2` leads the three ReRoPE-w256 rows at `512` and without repetition; on the
repeated column it leads ReRoPE-w256 and ReRoPE-w256-log n† and trails ReRoPE-w256-log n. -/
theorem w64f2_rerope :
    (∀ j : Fin 3, j ≠ 1 → tableReRoPE j < tableW64F2 j ∧ tableReRoPELogDagger j < tableW64F2 j ∧
      tableReRoPELog j < tableW64F2 j) ∧
      tableReRoPE 1 < tableW64F2 1 ∧ tableReRoPELogDagger 1 < tableW64F2 1 ∧
      tableW64F2 1 < tableReRoPELog 1 := by
  refine ⟨fun j hj => ?_, by norm_num [tableReRoPE, tableW64F2],
    by norm_num [tableReRoPELogDagger, tableW64F2], by norm_num [tableReRoPELog, tableW64F2]⟩
  fin_cases j <;> simp_all [tableReRoPE, tableReRoPELogDagger, tableReRoPELog, tableW64F2] <;>
    norm_num

/-- Experiments, the bold entries: `85.12` of ReRoPE-w256-log n and `49.37` of `w64-f1` are the
largest of their columns; `49.50` of `w32-f2` is the largest at `512` among the runs, below part
13's `b8` run's `49.62`. -/
theorem bold :
    (∀ r : Fin 16, table14 r 1 ≤ tableReRoPELog 1 ∧ table14 r 2 ≤ tableW64F1 2) ∧
      (∀ r : Fin 8, runs r 0 ≤ tableW32F2 0) ∧ tableW32F2 0 < tableInvLeakyB8 0 := by
  refine ⟨fun r => ?_, fun r => ?_, by norm_num [tableW32F2, tableInvLeakyB8]⟩
  · fin_cases r <;> norm_num [table14, tableBaseline, tableBaselineLog, tableReRoPE,
      tableReRoPELogDagger, tableReRoPELog, tableInvLeaky, tableInvLeakyB8, tableHWFA, tableW32F1,
      tableW64F1, tableW128F1, tableW256F1, tableW32F2, tableW64F2, tableW128F2, tableW256F2]
  · fin_cases r <;> norm_num [runs, tableW32F1, tableW64F1, tableW128F1, tableW256F1, tableW32F2,
      tableW64F2, tableW128F2, tableW256F2]

/-- Summary: `w64-f2` loses no training performance against the baseline and comes within `0.69`
and `0.01` of the largest entries of the two `4096` columns (`bold`). -/
theorem summary :
    tableBaseline 0 ≤ tableW64F2 0 ∧ tableReRoPELog 1 - tableW64F2 1 = 0.69 ∧
      tableW64F1 2 - tableW64F2 2 = 0.01 := by
  norm_num [tableBaseline, tableW64F2, tableReRoPELog, tableW64F1]

end Transformer.RoPE.HWFAReRoPE

module

public import Transformer.RoPE.HWFAReRoPE.Section2_Combination
public import Transformer.RoPE.InvLeakyReRoPE.Section3_Experiments

/-!
# HWFA meets ReRoPE, 实验 (Experiments): the windows and the table

Su, *Transformer升级之路：14、当HWFA遇见ReRoPE*, kexue.fm/archives/9731, section 实验
(Experiments): the GAU model of the earlier parts, `10⁸` parameters, `L = 24` layers (stated with
the guess below), training length `N = 512`; ReRoPE's `w = 256`; the windows
`w̃ ∈ {32, 64, 128, 256}`; `f ∈ {1, 2}` full layers, at the layer `num_layers / 2` for `f = 1`
and at `num_layers / 3` and `2 · num_layers / 3` for `f = 2`. The comparison runs behind the
placement ("in the middle trains better than at the end") are not reported and are not formalized.

* `constraint_iff`: at `L = 24` and `N = 512` HWFA's constraint `(w̃ − 1)(L − 1) + 1 = αN`,
  `0 < α ≤ 1`, holds exactly for `w̃ ≤ 23`. "The original HFWA, because of the constraints, took
  `w̃` only up to `16`": its receptive field is `346`, `α = 173/256` (`hwfa_window`). The windows of
  the runs break it (`not_constraint_runs`): "after introducing ReRoPE the window can be taken
  somewhat larger". That larger windows lower HWFA's extrapolation is part 9's experiment;
* `guess_window`: "combined with `N = 512` and `L = 24`, the best `w̃` should be about 2–4 times
  `N/L`": `64 = 3 · 512/24` lies in `[2N/L, 4N/L]`; of the four windows tested, `32` lies below
  and `128`, `256` above;
* the table: its first eight rows are part 13's, reprinted unchanged
  (`InvLeakyReRoPE.tableBaseline`, …), then the eight HFWA-ReRoPE runs (the source's spelling),
  `w` the window `w̃`, `f` the number of full layers. The columns are part 13's: per-token accuracy
  in percent at `512` (training), `4096` repeated, `4096` not repeated. Its claims are
  `Section3_Table`.
-/

@[expose] public section

namespace Transformer.RoPE.HWFAReRoPE

open InvLeakyReRoPE

/-- Experiments: at `L = 24` and `N = 512` HWFA's constraint holds for some `0 < α ≤ 1` exactly
when `w̃ ≤ 23`. -/
theorem constraint_iff (W : ℕ) :
    (∃ α : ℝ, 0 < α ∧ α ≤ 1 ∧ (((W - 1) * (24 - 1) + 1 : ℕ) : ℝ) = α * (512 : ℕ)) ↔ W ≤ 23 := by
  rw [exists_alpha_iff W 24 (by norm_num)]
  omega

/-- Experiments: HWFA's window `16` gives the receptive field `15 · 23 + 1 = 346` and
`α = 346/512 = 173/256`. -/
theorem hwfa_window :
    (16 - 1) * (24 - 1) + 1 = 346 ∧
      (((16 - 1) * (24 - 1) + 1 : ℕ) : ℝ) = 173 / 256 * (512 : ℕ) := by
  norm_num

/-- Experiments: the windows `32, 64, 128, 256` of the runs break HWFA's constraint. -/
theorem not_constraint_runs (W : ℕ) (hW : W ∈ ({32, 64, 128, 256} : Finset ℕ)) :
    ¬ ∃ α : ℝ, 0 < α ∧ α ≤ 1 ∧ (((W - 1) * (24 - 1) + 1 : ℕ) : ℝ) = α * (512 : ℕ) := by
  rw [constraint_iff]
  simp only [Finset.mem_insert, Finset.mem_singleton] at hW
  omega

example : ¬ ∃ α : ℝ, 0 < α ∧ α ≤ 1 ∧ (((64 - 1) * (24 - 1) + 1 : ℕ) : ℝ) = α * (512 : ℕ) :=
  not_constraint_runs 64 (by decide)

/-- Experiments: "the best `w̃` should be about 2–4 times `N/L`": at `N = 512` and `L = 24`,
`64 = 3 · 512/24` lies in `[2N/L, 4N/L]`, `32` below it and `128` above it. -/
theorem guess_window :
    (64 : ℝ) = 3 * (512 / 24) ∧ 2 * (512 / 24 : ℝ) ≤ 64 ∧ (64 : ℝ) ≤ 4 * (512 / 24) ∧
      (32 : ℝ) < 2 * (512 / 24) ∧ 4 * (512 / 24 : ℝ) < 128 := by
  norm_num

/-- The table, HFWA-ReRoPE-w32-f1. -/
noncomputable def tableW32F1 : Fin 3 → ℝ := ![49.29, 83.13, 49.34]
/-- The table, HFWA-ReRoPE-w64-f1. -/
noncomputable def tableW64F1 : Fin 3 → ℝ := ![49.32, 82.41, 49.37]
/-- The table, HFWA-ReRoPE-w128-f1. -/
noncomputable def tableW128F1 : Fin 3 → ℝ := ![49.21, 80.18, 48.99]
/-- The table, HFWA-ReRoPE-w256-f1. -/
noncomputable def tableW256F1 : Fin 3 → ℝ := ![49.00, 54.94, 47.64]
/-- The table, HFWA-ReRoPE-w32-f2. -/
noncomputable def tableW32F2 : Fin 3 → ℝ := ![49.50, 84.09, 49.35]
/-- The table, HFWA-ReRoPE-w64-f2. -/
noncomputable def tableW64F2 : Fin 3 → ℝ := ![49.46, 84.43, 49.36]
/-- The table, HFWA-ReRoPE-w128-f2. -/
noncomputable def tableW128F2 : Fin 3 → ℝ := ![49.35, 83.09, 48.97]
/-- The table, HFWA-ReRoPE-w256-f2. -/
noncomputable def tableW256F2 : Fin 3 → ℝ := ![49.37, 75.24, 48.42]

/-- The eight HFWA-ReRoPE runs in the table's order: `w32, w64, w128, w256` with `f1`, then with
`f2`; `w64-f2` is the run `5`. -/
noncomputable def runs : Fin 8 → Fin 3 → ℝ :=
  ![tableW32F1, tableW64F1, tableW128F1, tableW256F1, tableW32F2, tableW64F2, tableW128F2,
    tableW256F2]

/-- The table's sixteen rows in its order: part 13's Baseline, Baseline-log n, the three
ReRoPE-w256 rows, the two InvLeaky ReRoPE rows and HFWA, then the eight runs. -/
noncomputable def table14 : Fin 16 → Fin 3 → ℝ :=
  ![tableBaseline, tableBaselineLog, tableReRoPE, tableReRoPELogDagger, tableReRoPELog,
    tableInvLeaky, tableInvLeakyB8, tableHWFA, tableW32F1, tableW64F1, tableW128F1, tableW256F1,
    tableW32F2, tableW64F2, tableW128F2, tableW256F2]

end Transformer.RoPE.HWFAReRoPE

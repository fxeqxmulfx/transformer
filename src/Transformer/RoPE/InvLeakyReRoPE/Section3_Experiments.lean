module

public import Transformer.RoPE.InvLeakyReRoPE.Section2_Inversion

/-!
# InvLeaky ReRoPE, 实验 (Experiments) and 小结 (Summary)

Su, *Transformer升级之路：13、逆用Leaky ReRoPE*, kexue.fm/archives/9728, sections 实验
(Experiments) and 小结 (Summary). The model is the GAU language model with Deep Norm and the
Tiger optimizer of parts 11 and 12, `10⁸` parameters, trained at the length `512` with Leaky
ReRoPE at `k = 1/16`, `w = 128` and tested with RoPE. The table gives the per-token accuracy in
percent at `512` (training), at `4096` on repeated text and at `4096` on text without repetition,
the columns `0, 1, 2`. Its rows other than the two InvLeaky ReRoPE runs are part 12's, and "HFWA",
the source's spelling, is the HWFA of part 9, kexue.fm/archives/9603, with the same numbers there.
The rule: "`k` the reciprocal of twice the extension factor, `w` a quarter of the training
length, `b` optionally multiplied by the extension factor", `b` RoPE's base.

* `rule_covers`: with `k = 1/(2s)` and `w = N/4`, `s, N > 0`, every distance `D ≤ sN − 1` of an
  input `s` times the training length `N` is the training position of a training distance below
  `3N/4`. `rule_runs`: at the extension `4096/512 = 8` the rule gives the runs' `k = 1/16`,
  `w = 128` and the base `80000` of `b8`; the training distances needed are at most `375.9375`
  (`trainDistance_le`), the largest training position is `6256` (`trainPosition_511`). Read in
  training distances, inference is Leaky ReRoPE with `k = 16`, whose positions exceed the Review's
  bound `N/2 = 256` (`not_trainDistance_le_half`): the rule keeps them below `3N/4` instead;
* `hwfa_lt_invLeaky`: "it still beats HFWA": both InvLeaky ReRoPE runs lead HWFA everywhere;
* `invLeaky_lt_rerope`: "though its effect is below 'RoPE → ReRoPE/Leaky ReRoPE'": on both `4096`
  columns both runs trail ReRoPE-w256-log n, and InvLeaky ReRoPE-w128-log n trails part 12's
  Leaky ReRoPE-w128-k16-log n, the same `w`, `k⁻¹` and scale used the other way round.
  `invLeaky_like_for_like`: the comparison holds like for like only: ReRoPE-w256 without the
  scale trails both runs on the repeated column, the `b8` run leads ReRoPE-w256-log n at `512`,
  and part 12's Leaky ReRoPE-w128-k8-log n trails both runs on both `4096` columns;
* `table_le`: the bold `85.12` and `49.07` are the largest entries of their columns; at `512` the
  largest is the `b8` run's `49.62`;
* `training_time`: "per 1000 steps from 330 s to 350 s, less than `10%`": `20/330 = 2/33`.

Not formalized: the tuning of `k`, `w` and `b` behind the rule, which is not reported, and the
estimate that the slowdown stays below `50%` with multi-head attention or longer training.
-/

@[expose] public section

namespace Transformer.RoPE.InvLeakyReRoPE

open ReRoPE

/-- Experiments' rule, `k = 1/(2s)` and `w = N/4` with `s, N > 0`: every distance `D ≤ sN − 1`
reads as a training distance below `3N/4`. -/
theorem rule_trainDistance_lt {s N D : ℝ} (hs : 0 < s) (hN : 0 < N) (hD : D ≤ s * N - 1) :
    leakyPosition (N / 4) (2 * s) D < 3 * N / 4 := by
  unfold leakyPosition
  split_ifs with h
  · linarith
  · have h1 : (D - N / 4) / (2 * s) < N / 2 := by
      rw [div_lt_iff₀ (by linarith), show N / 2 * (2 * s) = s * N by ring]
      linarith
    linarith

example : leakyPosition (512 / 4) (2 * 8) 4095 < 3 * 512 / 4 :=
  rule_trainDistance_lt (by norm_num) (by norm_num) (by norm_num)

/-- Experiments' rule: every distance `D ≤ sN − 1` is the training position, under
`k = (2s)⁻¹` and `w = N/4`, of a training distance below `3N/4`. -/
theorem rule_covers {s N D : ℝ} (hs : 0 < s) (hN : 0 < N) (hD : D ≤ s * N - 1) :
    ∃ t, t < 3 * N / 4 ∧ leakyPosition (N / 4) (2 * s)⁻¹ t = D := by
  refine ⟨leakyPosition (N / 4) (2 * s) D, rule_trainDistance_lt hs hN hD, ?_⟩
  have h := leakyPosition_inv (N / 4) (k := (2 * s)⁻¹) (inv_pos.mpr (by linarith)) D
  rwa [inv_inv] at h

example : ∃ t, t < 3 * 512 / 4 ∧ leakyPosition (512 / 4) (2 * 8)⁻¹ t = 4095 :=
  rule_covers (s := 8) (by norm_num) (by norm_num) (by norm_num)

/-- Experiments' rule at the extension `4096/512 = 8`: `k = (2 · 8)⁻¹ = 1/16`, `w = 512/4 = 128`
and the base `10000 · 8 = 80000` of `b8`; every test distance `D ≤ 4095` is then the training
position of a training distance below `384 = 3 · 512/4`. -/
theorem rule_runs {D : ℝ} (hD : D ≤ 4095) :
    (2 * ((4096 : ℝ) / 512))⁻¹ = 1 / 16 ∧ (512 : ℝ) / 4 = 128 ∧
      10000 * ((4096 : ℝ) / 512) = 80000 ∧ ∃ t, t < 384 ∧ leakyPosition 128 (1 / 16) t = D := by
  obtain ⟨t, ht, htD⟩ := rule_covers (s := 8) (N := 512) (D := D) (by norm_num) (by norm_num)
    (by linarith)
  refine ⟨by norm_num, by norm_num, by norm_num, t, by linarith, ?_⟩
  rwa [show (512 : ℝ) / 4 = 128 by norm_num, show (2 * 8 : ℝ)⁻¹ = 1 / 16 by norm_num] at htD

example : (2 * ((4096 : ℝ) / 512))⁻¹ = 1 / 16 ∧ (512 : ℝ) / 4 = 128 ∧
    10000 * ((4096 : ℝ) / 512) = 80000 ∧ ∃ t, t < 384 ∧ leakyPosition 128 (1 / 16) t = 4095 :=
  rule_runs le_rfl

/-- Experiments' numbers, `w = 128` and `k = 1/16`: every test distance `D ≤ 4095` is the
training position of the training distance `leakyPosition 128 16 D ≤ 375.9375 < 511`. -/
theorem trainDistance_le {D : ℝ} (hD : D ≤ 4095) :
    leakyPosition 128 (1 / 16) (leakyPosition 128 16 D) = D ∧
      leakyPosition 128 16 D ≤ 375.9375 := by
  refine ⟨by simpa using leakyPosition_inv 128 (k := 1 / 16) (by norm_num) D, ?_⟩
  refine (leakyPosition_monotone 128 (by norm_num) hD).trans ?_
  rw [leakyPosition, ite_eq_right (by norm_num)]
  norm_num

example : leakyPosition 128 (1 / 16) (leakyPosition 128 16 4095) = 4095 ∧
    leakyPosition 128 16 4095 ≤ 375.9375 :=
  trainDistance_le le_rfl

/-- Read in training distances, inference is Leaky ReRoPE with `w = 128` and `k = 16`; its
positions on `4096` tokens exceed the Review's bound, half the training length `256`, which needs
`k ≥ 3967/128` (`leakyPosition_w128_le_iff`). -/
theorem not_trainDistance_le_half : ¬ ∀ d, d ≤ 4095 → leakyPosition 128 16 d ≤ 512 / 2 := by
  rw [leakyPosition_w128_le_iff (by norm_num)]
  norm_num

/-- Experiments' numbers: the largest training position, at the distance `511` of the training
length `512`, is `128 + 16 · 383 = 6256`. -/
theorem trainPosition_511 : leakyPosition 128 (1 / 16) 511 = 6256 := by
  rw [leakyPosition, ite_eq_right (by norm_num)]
  norm_num

/-- The table, Baseline. -/
noncomputable def tableBaseline : Fin 3 → ℝ := ![49.41, 24.17, 23.16]
/-- The table, Baseline-log n. -/
noncomputable def tableBaselineLog : Fin 3 → ℝ := ![49.40, 24.60, 24.02]
/-- The table, NTK-RoPE-fixed. -/
noncomputable def tableNTKFixed : Fin 3 → ℝ := ![49.41, 51.86, 39.61]
/-- The table, NTK-RoPE-log n†-fixed. -/
noncomputable def tableNTKLogDaggerFixed : Fin 3 → ℝ := ![49.41, 55.94, 41.11]
/-- The table, NTK-RoPE-log n-fixed. -/
noncomputable def tableNTKLogFixed : Fin 3 → ℝ := ![49.40, 62.85, 44.14]
/-- The table, NTK-RoPE-mixed. -/
noncomputable def tableNTKMixed : Fin 3 → ℝ := ![49.41, 53.09, 40.12]
/-- The table, NTK-RoPE-log n†-mixed. -/
noncomputable def tableNTKLogDaggerMixed : Fin 3 → ℝ := ![49.41, 59.11, 42.38]
/-- The table, NTK-RoPE-log n-mixed. -/
noncomputable def tableNTKLogMixed : Fin 3 → ℝ := ![49.40, 68.91, 45.41]
/-- The table, ReRoPE-w256. -/
noncomputable def tableReRoPE : Fin 3 → ℝ := ![49.41, 77.90, 48.48]
/-- The table, ReRoPE-w256-log n†. -/
noncomputable def tableReRoPELogDagger : Fin 3 → ℝ := ![49.41, 82.40, 48.85]
/-- The table, ReRoPE-w256-log n. -/
noncomputable def tableReRoPELog : Fin 3 → ℝ := ![49.40, 85.12, 49.07]
/-- The table, InvLeaky ReRoPE-w128-log n. -/
noncomputable def tableInvLeaky : Fin 3 → ℝ := ![49.38, 82.25, 48.32]
/-- The table, InvLeaky ReRoPE-w128-b8-log n, RoPE's base `80000` in place of `10000`. -/
noncomputable def tableInvLeakyB8 : Fin 3 → ℝ := ![49.62, 81.15, 48.85]
/-- The table, HFWA, part 9's HWFA. -/
noncomputable def tableHWFA : Fin 3 → ℝ := ![48.70, 80.84, 48.15]

/-- The table's fourteen rows, in its order. -/
noncomputable def table : Fin 14 → Fin 3 → ℝ :=
  ![tableBaseline, tableBaselineLog, tableNTKFixed, tableNTKLogDaggerFixed, tableNTKLogFixed,
    tableNTKMixed, tableNTKLogDaggerMixed, tableNTKLogMixed, tableReRoPE, tableReRoPELogDagger,
    tableReRoPELog, tableInvLeaky, tableInvLeakyB8, tableHWFA]

/-- Part 12, kexue.fm/archives/9708, the table comparing ReRoPE and Leaky ReRoPE,
Leaky ReRoPE-w128-k16-log n. -/
noncomputable def leakyW128K16Log : Fin 3 → ℝ := ![49.40, 85.15, 49.10]
/-- Part 12, the same table, Leaky ReRoPE-w128-k8-log n. -/
noncomputable def leakyW128K8Log : Fin 3 → ℝ := ![49.40, 80.00, 48.11]

/-- Experiments: "it still beats HFWA": both InvLeaky ReRoPE runs lead HWFA in every column. -/
theorem hwfa_lt_invLeaky (j : Fin 3) :
    tableHWFA j < tableInvLeaky j ∧ tableHWFA j < tableInvLeakyB8 j := by
  fin_cases j <;> norm_num [tableHWFA, tableInvLeaky, tableInvLeakyB8]

/-- Experiments: "though its effect is below 'RoPE → ReRoPE/Leaky ReRoPE'": on both `4096`
columns both runs trail ReRoPE-w256-log n, and InvLeaky ReRoPE-w128-log n trails part 12's Leaky
ReRoPE-w128-k16-log n. -/
theorem invLeaky_lt_rerope (j : Fin 3) (hj : j ≠ 0) :
    tableInvLeaky j < tableReRoPELog j ∧ tableInvLeakyB8 j < tableReRoPELog j ∧
      tableInvLeaky j < leakyW128K16Log j := by
  fin_cases j <;> simp_all [tableInvLeaky, tableInvLeakyB8, tableReRoPELog, leakyW128K16Log] <;>
    norm_num

example : tableInvLeaky 2 < tableReRoPELog 2 ∧ tableInvLeakyB8 2 < tableReRoPELog 2 ∧
    tableInvLeaky 2 < leakyW128K16Log 2 :=
  invLeaky_lt_rerope 2 (by decide)

/-- Experiments: the comparison holds like for like only. Without the scale ReRoPE-w256 trails
both runs on the repeated column; the `b8` run leads ReRoPE-w256-log n at `512`; part 12's Leaky
ReRoPE-w128-k8-log n trails both runs on both `4096` columns. -/
theorem invLeaky_like_for_like :
    tableReRoPE 1 < tableInvLeaky 1 ∧ tableReRoPE 1 < tableInvLeakyB8 1 ∧
      tableReRoPELog 0 < tableInvLeakyB8 0 ∧
      ∀ j : Fin 3, j ≠ 0 → leakyW128K8Log j < tableInvLeaky j ∧
        leakyW128K8Log j < tableInvLeakyB8 j := by
  refine ⟨by norm_num [tableReRoPE, tableInvLeaky], by norm_num [tableReRoPE, tableInvLeakyB8],
    by norm_num [tableReRoPELog, tableInvLeakyB8], fun j hj => ?_⟩
  fin_cases j <;> simp_all [leakyW128K8Log, tableInvLeaky, tableInvLeakyB8] <;> norm_num

/-- Experiments: the bold `85.12` and `49.07` of ReRoPE-w256-log n are the largest entries of
the two `4096` columns; at `512` the largest is the `b8` run's `49.62`. -/
theorem table_le (r : Fin 14) :
    table r 0 ≤ tableInvLeakyB8 0 ∧ table r 1 ≤ tableReRoPELog 1 ∧
      table r 2 ≤ tableReRoPELog 2 := by
  fin_cases r <;> norm_num [table, tableBaseline, tableBaselineLog, tableNTKFixed,
    tableNTKLogDaggerFixed, tableNTKLogFixed, tableNTKMixed, tableNTKLogDaggerMixed,
    tableNTKLogMixed, tableReRoPE, tableReRoPELogDagger, tableReRoPELog, tableInvLeaky,
    tableInvLeakyB8, tableHWFA]

/-- Experiments: "the training time per 1000 steps rose from 330 s to 350 s, an increase of less
than 10%": `(350 − 330)/330 = 2/33`. -/
theorem training_time : ((350 : ℝ) - 330) / 330 = 2 / 33 ∧ ((350 : ℝ) - 330) / 330 < 1 / 10 := by
  norm_num

end Transformer.RoPE.InvLeakyReRoPE

import Transformer.ScheduleFree.AdamW

/-!
# Published Schedule-Free AdamW 1.4.1 checkpoints

The same rational values appear in optimizers/tests/test_schedule_free_adamw.py.
They exercise the actual optimized Lean recurrence, bias-corrected variance,
warmup, old-query decay, evaluation recovery and maximum-rate averaging.
The Python suite also compares directly with the hash-pinned published source.
-/

open Transformer.ScheduleFree Transformer.ScheduleFree.AdamW

noncomputable section

private def parameters : Parameters := ⟨⟨1/2, 1/8, 2, 1/2, 1/2, 1⟩, 0, 2⟩
private def gradients : ℕ → (Fin 1 → ℝ) → (Fin 1 → ℝ) :=
  fun t _ => if t = 0 then ![1] else if t = 1 then ![11] else ![9]

example : let s := trainingRun parameters gradients ![2] 1
    s.point 0 = 29/16 ∧ s.base 0 = 29/16 ∧ s.history.second 0 = 1/2 ∧
    evaluation parameters s 0 = 29/16 ∧ s.history.total = 1/16 ∧
    s.history.maximumRate = 1/4 := by
  norm_num [parameters, gradients, trainingRun, trainingStep, direction, denominator,
    nextHistory, nextSecond, coefficient, weight, nextMaximum, rate, warmupFactor,
    correction, evaluation, Real.rpow_two]

example : let s := trainingRun parameters gradients ![2] 2
    s.point 0 = 15559/12800 ∧ s.base 0 = 1471/1280 ∧ s.history.second 0 = 243/4 ∧
    evaluation parameters s 0 = 2051/1600 ∧ s.history.total = 5/16 ∧
    s.history.maximumRate = 1/2 := by
  norm_num [parameters, gradients, trainingRun, trainingStep, direction, denominator,
    nextHistory, nextSecond, coefficient, weight, nextMaximum, rate, warmupFactor,
    correction, evaluation, Real.rpow_two]

example : let s := trainingRun parameters gradients ![2] 3
    s.point 0 = 2971973/3686400 ∧ s.base 0 = 127641/204800 ∧
    s.history.second 0 = 567/8 ∧ evaluation parameters s 0 = 455801/460800 ∧
    s.history.total = 9/16 ∧ s.history.maximumRate = 1/2 := by
  norm_num [parameters, gradients, trainingRun, trainingStep, direction, denominator,
    nextHistory, nextSecond, coefficient, weight, nextMaximum, rate, warmupFactor,
    correction, evaluation, Real.rpow_two]

private def fallingParameters (t : ℕ) : Parameters :=
  ⟨⟨if t = 0 then 1/2 else if t = 1 then 1/4 else 0, 1/8, 0, 1/2, 0, 1⟩, 0, 2⟩

private def fallingRun : ℕ → TrainingState (Fin 1)
  | 0 => ⟨![2], ![2], ⟨fun _ => 0, 0, -1⟩⟩
  | t + 1 => trainingStep (fallingParameters t) t (fallingRun t) ![1]

example : let s := fallingRun 2
    s.point 0 = 1529/1024 ∧ s.base 0 = 371/256 ∧
    evaluation (fallingParameters 1) s 0 = 787/512 ∧
    s.history.total = 1/2 ∧ s.history.maximumRate = 1/2 := by
  norm_num [fallingRun, fallingParameters, trainingStep, direction, denominator,
    nextHistory, nextSecond, coefficient, weight, nextMaximum, rate, warmupFactor,
    correction, evaluation, Real.rpow_two]

example : let s := fallingRun 3
    s.point 0 = 757/512 ∧ s.base 0 = 371/256 ∧ s.history.second 0 = 1 ∧
    evaluation (fallingParameters 2) s 0 = 193/128 ∧
    s.history.total = 3/4 ∧ s.history.maximumRate = 1/2 := by
  norm_num [fallingRun, fallingParameters, trainingStep, direction, denominator,
    nextHistory, nextSecond, coefficient, weight, nextMaximum, rate, warmupFactor,
    correction, evaluation, Real.rpow_two]

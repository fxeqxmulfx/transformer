module

public import Transformer.Rotate90.Memory.Defs

/-!
# The exact memory clock of stopped Schedule-Free AdamW

Source: `review/flat_spectrum.md`, *Optimizer memory*, M2. This is an identity
for the source two-buffer trajectory, including the real gradient oracle and
raw second-moment updates. Positive stored total prevents a zero denominator.
The effective time scale is W/M², not the zero current learning rate.
-/

@[expose] public section

noncomputable section

namespace Transformer.Rotate90.Memory

open ScheduleFree Learning

variable {I : Type*}

/-- M2: the exact evaluation point after any number of zero-rate source updates. -/
theorem stopped_evaluation (p : AdamParameters) (offset : ℕ)
    (oracle : ℕ → (I → ℝ) → (I → ℝ)) (s : AdamW.TrainingState I)
    (hW : 0 < s.history.total) (hM : 0 ≤ s.history.maximumRate) (n : ℕ) (i : I) :
    AdamW.evaluation (halt p) (stoppedRun p offset oracle s n) i =
      (s.history.total * AdamW.evaluation (halt p) s i +
        n * s.history.maximumRate ^ 2 * s.base i) /
        (s.history.total + n * s.history.maximumRate ^ 2) := by
  induction n with
  | zero => simp [stoppedRun, ne_of_gt hW]
  | succ n ih =>
    have hn : 0 ≤ (n : ℝ) := by positivity
    have hden : s.history.total + n * s.history.maximumRate ^ 2 ≠ 0 := by positivity
    have hnext : s.history.total + n * s.history.maximumRate ^ 2 +
        s.history.maximumRate ^ 2 ≠ 0 := by positivity
    have hs := stopped_history p offset oracle s hM n
    change AdamW.evaluation _ (AdamW.trainingStep _ _ _ _) i = _
    rw [halt_evaluation p _ _ _ (by rw [hs.2.1]; exact hM), hs.2.1, hs.2.2, hs.1, ih]
    push_cast
    have hnext' : s.history.total + (n + 1) * s.history.maximumRate ^ 2 ≠ 0 := by positivity
    field_simp
    ring

/-- The closed form in the displacement coordinates used by the actual loss. -/
theorem stopped_displacement (p : AdamParameters) (offset : ℕ)
    (oracle : ℕ → (I → ℝ) → (I → ℝ)) (s : AdamW.TrainingState I)
    (hW : 0 < s.history.total) (hM : 0 ≤ s.history.maximumRate) (n : ℕ) :
    evaluated (halt p) (stoppedRun p offset oracle s n) - WithLp.toLp 2 s.base =
      (s.history.total / (s.history.total + n * s.history.maximumRate ^ 2)) •
        (evaluated (halt p) s - WithLp.toLp 2 s.base) := by
  ext i
  change AdamW.evaluation _ _ i - s.base i = _
  rw [stopped_evaluation p offset oracle s hW hM]
  simp only [PiLp.smul_apply, PiLp.sub_apply, smul_eq_mul, evaluated]
  have hden : s.history.total + n * s.history.maximumRate ^ 2 ≠ 0 := by positivity
  field_simp
  ring

variable [Fintype I]

/-- The Euclidean distance to the base decays at the exact historical clock. -/
theorem stopped_distance (p : AdamParameters) (offset : ℕ)
    (oracle : ℕ → (I → ℝ) → (I → ℝ)) (s : AdamW.TrainingState I)
    (hW : 0 < s.history.total) (hM : 0 ≤ s.history.maximumRate) (n : ℕ) :
    ‖evaluated (halt p) (stoppedRun p offset oracle s n) - WithLp.toLp 2 s.base‖ =
      (s.history.total / (s.history.total + n * s.history.maximumRate ^ 2)) *
        ‖evaluated (halt p) s - WithLp.toLp 2 s.base‖ := by
  rw [stopped_displacement p offset oracle s hW hM,
    norm_smul_of_nonneg (by positivity)]

/-- Positive stored maximum makes stopped evaluations converge to the base,
regardless of the future gradient sequence or changing raw second moments. -/
theorem stopped_tendsto (p : AdamParameters) (offset : ℕ)
    (oracle : ℕ → (I → ℝ) → (I → ℝ)) (s : AdamW.TrainingState I)
    (hW : 0 < s.history.total) (hM : 0 < s.history.maximumRate) :
    Filter.Tendsto (fun n => evaluated (halt p) (stoppedRun p offset oracle s n))
      Filter.atTop (nhds (WithLp.toLp 2 s.base)) := by
  have hclock : Filter.Tendsto
      (fun n : ℕ => s.history.total + n * s.history.maximumRate ^ 2)
      Filter.atTop Filter.atTop :=
    Filter.tendsto_atTop_add_const_left _ _
      (Filter.Tendsto.atTop_mul_const (by positivity) tendsto_natCast_atTop_atTop)
  have hz := (hclock.const_div_atTop s.history.total).smul_const
    (evaluated (halt p) s - WithLp.toLp 2 s.base)
  have heq (n : ℕ) : evaluated (halt p) (stoppedRun p offset oracle s n) =
      (s.history.total / (s.history.total + n * s.history.maximumRate ^ 2)) •
        (evaluated (halt p) s - WithLp.toLp 2 s.base) + WithLp.toLp 2 s.base := by
    rw [← stopped_displacement p offset oracle s hW (le_of_lt hM), sub_add_cancel]
  simpa only [← heq, zero_smul, zero_add] using hz.add_const (WithLp.toLp 2 s.base)

example := stopped_evaluation (I := Fin 2)
  ⟨3/1000, 1, 10, 9/10, 98/100, 1/100000000⟩ 750 (fun _ y => y)
  ⟨fun _ => 1, fun i => i.val, ⟨fun _ => 1, 1, 3/1000⟩⟩
  (by norm_num) (by norm_num) 50000 0
example := stopped_displacement (I := Fin 2)
  ⟨3/1000, 1, 10, 9/10, 98/100, 1/100000000⟩ 750 (fun _ y => y)
  ⟨fun _ => 1, fun i => i.val, ⟨fun _ => 1, 1, 3/1000⟩⟩
  (by norm_num) (by norm_num) 50000
example := stopped_distance (I := Fin 2)
  ⟨3/1000, 1, 10, 9/10, 98/100, 1/100000000⟩ 750 (fun _ y => y)
  ⟨fun _ => 1, fun i => i.val, ⟨fun _ => 1, 1, 3/1000⟩⟩
  (by norm_num) (by norm_num) 50000
example := stopped_tendsto (I := Fin 2)
  ⟨3/1000, 1, 10, 9/10, 98/100, 1/100000000⟩ 750 (fun _ y => y)
  ⟨fun _ => 1, fun i => i.val, ⟨fun _ => 1, 1, 3/1000⟩⟩
  (by norm_num) (by norm_num)

end Transformer.Rotate90.Memory

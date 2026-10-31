module

public import Transformer.Rotate90.Learning.Basic

/-!
# A stopped base with live Schedule-Free averaging history

Source: `review/flat_spectrum.md`, *Optimizer memory*, M1--M2, and the revised
source `trainingStep`. Setting the learning rate to zero removes base updates
and decay, but does not reset the stored maximum or averaging total. Raw second
moments still follow the supplied gradients. The default averaging powers are
zero and two. This models a continuation from a saved state, not a restart.
-/

@[expose] public section

noncomputable section

namespace Transformer.Rotate90.Memory

open ScheduleFree Learning

/-- Preserve all Adam parameters except the rate, using default source averaging. -/
def halt (p : AdamParameters) : AdamW.Parameters :=
  ⟨{p with learningRate := 0}, 0, 2⟩

variable {I : Type*}

/-- The real source updates continue; their gradients still update raw variance. -/
def stoppedRun (p : AdamParameters) (offset : ℕ)
    (oracle : ℕ → (I → ℝ) → (I → ℝ)) (initial : AdamW.TrainingState I) :
    ℕ → AdamW.TrainingState I
  | 0 => initial
  | n + 1 =>
    let s := stoppedRun p offset oracle initial n
    AdamW.trainingStep (halt p) (offset + n) s (oracle (offset + n) s.point)

/-- Zero rate survives the source warmup factor at every continuation time. -/
theorem halt_rate (p : AdamParameters) (t : ℕ) : AdamW.rate (halt p) t = 0 := by
  simp [halt, AdamW.rate]

/-- A nonnegative historical maximum is preserved by a zero current rate. -/
theorem halt_maximum (p : AdamParameters) (t : ℕ) (h : AdamW.History I)
    (hM : 0 ≤ h.maximumRate) :
    AdamW.nextMaximum (halt p) t h = h.maximumRate := by
  rw [AdamW.nextMaximum, halt_rate, max_eq_right hM]

/-- M1: every zero-rate update still contributes the square of the stored maximum. -/
theorem halt_weight (p : AdamParameters) (t : ℕ) (h : AdamW.History I)
    (hM : 0 ≤ h.maximumRate) :
    AdamW.weight (halt p) t h = h.maximumRate ^ 2 := by
  change AdamW.weight ⟨{p with learningRate := 0}, 0, 2⟩ t h = _
  rw [AdamW.default_weight]
  exact congrArg (fun x : ℝ => x ^ 2) (halt_maximum p t h hM)

/-- The source base is exactly fixed, independently of gradients and decay. -/
theorem halt_base (p : AdamParameters) (t : ℕ) (s : AdamW.TrainingState I) (g : I → ℝ) :
    (AdamW.trainingStep (halt p) t s g).base = s.base := by
  funext i
  simp [AdamW.trainingStep, halt_rate]

/-- The actual evaluation buffer averages toward the unchanged base. Its
algebraic identity even allows beta1=0; the executable API requires beta1>0. -/
theorem halt_evaluation (p : AdamParameters) (t : ℕ) (s : AdamW.TrainingState I)
    (g : I → ℝ) (hM : 0 ≤ s.history.maximumRate) (i : I) :
    AdamW.evaluation (halt p) (AdamW.trainingStep (halt p) t s g) i =
      (1 - s.history.maximumRate ^ 2 / (s.history.total + s.history.maximumRate ^ 2)) *
        AdamW.evaluation (halt p) s i +
      (s.history.maximumRate ^ 2 / (s.history.total + s.history.maximumRate ^ 2)) * s.base i := by
  simp only [AdamW.evaluation, AdamW.trainingStep, halt_rate, zero_mul, add_zero,
    sub_zero, AdamW.coefficient, halt_weight p t s.history hM]
  ring

/-- All continuation states preserve the original base and maximum, while
the actual source total advances at exactly the stored squared-maximum rate. -/
theorem stopped_history (p : AdamParameters) (offset : ℕ)
    (oracle : ℕ → (I → ℝ) → (I → ℝ)) (s : AdamW.TrainingState I)
    (hM : 0 ≤ s.history.maximumRate) (n : ℕ) :
    (stoppedRun p offset oracle s n).base = s.base ∧
    (stoppedRun p offset oracle s n).history.maximumRate = s.history.maximumRate ∧
    (stoppedRun p offset oracle s n).history.total =
      s.history.total + n * s.history.maximumRate ^ 2 := by
  induction n with
  | zero => simp [stoppedRun]
  | succ n ih =>
    constructor
    · change (AdamW.trainingStep _ _ _ _).base = _
      rw [halt_base, ih.1]
    constructor
    · change AdamW.nextMaximum _ _ _ = _
      rw [halt_maximum p _ _ (by rw [ih.2.1]; exact hM), ih.2.1]
    · change (stoppedRun p offset oracle s n).history.total +
        AdamW.weight (halt p) (offset + n) (stoppedRun p offset oracle s n).history = _
      rw [halt_weight p _ _ (by rw [ih.2.1]; exact hM), ih.2.1, ih.2.2]
      push_cast
      ring

example := halt_maximum (I := Fin 2) ⟨3/1000, 1, 10, 9/10, 98/100, 1/100000000⟩
  750 ⟨fun _ => 1, 1, 3/1000⟩ (by norm_num)
example := halt_weight (I := Fin 2) ⟨3/1000, 1, 10, 9/10, 98/100, 1/100000000⟩
  750 ⟨fun _ => 1, 1, 3/1000⟩ (by norm_num)
example := halt_evaluation (I := Fin 2) ⟨3/1000, 1, 10, 9/10, 98/100, 1/100000000⟩
  750 ⟨fun _ => 1, fun i => i.val, ⟨fun _ => 1, 1, 3/1000⟩⟩ (fun _ => 2) (by norm_num) 0
example := stopped_history (I := Fin 2) ⟨3/1000, 1, 10, 9/10, 98/100, 1/100000000⟩
  750 (fun _ y => y) ⟨fun _ => 1, fun i => i.val, ⟨fun _ => 1, 1, 3/1000⟩⟩
  (by norm_num) 50000

end Transformer.Rotate90.Memory

module

public import Transformer.Rotate90.Memory

/-!
# Actual source matrices and normalized-gradient forcing

Source: `review/flat_spectrum.md`, *Gradient forcing*, F1--F2;
`optimizers/src/lean_optim/schedule_free_adamw.py`, `step`.
The homogeneous part contains the old-query decay and historical averaging.
Only its scalar clock is independent of gradients: the forcing retains the
actual query, sampled gradient, changing raw second moment and bias correction.
The network is not linearized, and future forcing is not assumed known.
-/

@[expose] public section

noncomputable section

namespace Transformer.Rotate90.Forcing

open ScheduleFree

/-- The source's query/base pair, with every scalar parameter retained. -/
abbrev Buffers (I : Type*) := (I → ℝ) × (I → ℝ)

variable {I : Type*}

/-- Forget only raw variance and scalar history when reading the two buffers. -/
def buffers (s : AdamW.TrainingState I) : Buffers I := (s.point, s.base)

/-- F1's homogeneous 2x2 scalar matrix, acting identically on all coordinates. -/
def transition (p : AdamW.Parameters) (t : ℕ) (h : AdamW.History I) :
    Buffers I →ₗ[ℝ] Buffers I :=
  let c := AdamW.coefficient p t h
  let η := AdamW.rate p t
  let a := 1 - c + η * p.adam.decay * (p.adam.betaOne * (1 - c) - 1)
  ((a • LinearMap.fst ℝ (I → ℝ) (I → ℝ)) +
    (c • LinearMap.snd ℝ (I → ℝ) (I → ℝ))).prod
    ((-η * p.adam.decay) • LinearMap.fst ℝ (I → ℝ) (I → ℝ) +
      LinearMap.snd ℝ (I → ℝ) (I → ℝ))

/-- The coordinatewise Adam input, using the real source's updated variance. -/
def normalizedInput (p : AdamW.Parameters) (t : ℕ)
    (s : AdamW.TrainingState I) (g : I → ℝ) : I → ℝ :=
  fun i => g i / AdamW.denominator p t s.history g i

/-- The real source's positive epsilon makes zero-gradient input zero; the
identity also holds for Lean's total real division without that hypothesis. -/
theorem normalizedInput_zero (p : AdamW.Parameters) (t : ℕ) (s : AdamW.TrainingState I) :
    normalizedInput p t s 0 = 0 := by
  funext i
  simp [normalizedInput]

/-- F1's gain applied to a normalized gradient; decay belongs to transition. -/
def impulse (p : AdamW.Parameters) (t : ℕ) (h : AdamW.History I) (v : I → ℝ) : Buffers I :=
  ((AdamW.rate p t * (p.adam.betaOne * (1 - AdamW.coefficient p t h) - 1)) • v,
    (-AdamW.rate p t) • v)

/-- The matrix/forcing split is an identity of the actual two-buffer program. -/
theorem step_split (p : AdamW.Parameters) (t : ℕ)
    (s : AdamW.TrainingState I) (g : I → ℝ) :
    buffers (AdamW.trainingStep p t s g) =
      transition p t s.history (buffers s) + impulse p t s.history (normalizedInput p t s g) := by
  ext i <;>
    simp [buffers, AdamW.trainingStep, AdamW.direction, normalizedInput,
      transition, impulse] <;> ring

/-- The scalar coefficient depends only on total and maximum, not raw variance. -/
theorem coefficient_eq (p : AdamW.Parameters) (t : ℕ) (h k : AdamW.History I)
    (hW : h.total = k.total) (hM : h.maximumRate = k.maximumRate) :
    AdamW.coefficient p t h = AdamW.coefficient p t k := by
  simp only [AdamW.coefficient, AdamW.weight, AdamW.nextMaximum, hW, hM]

/-- Real source continuation at an absolute time, with gradients at the query. -/
def run (p : AdamW.Parameters) (offset : ℕ)
    (oracle : ℕ → (I → ℝ) → (I → ℝ)) (initial : AdamW.TrainingState I) :
    ℕ → AdamW.TrainingState I
  | 0 => initial
  | n + 1 =>
    let s := run p offset oracle initial n
    AdamW.trainingStep p (offset + n) s (oracle (offset + n) s.point)

/-- An independent source clock with zero-gradient moment updates. Only its
total and maximum are used for the matrices, never its raw variance. -/
def clock (p : AdamW.Parameters) (offset : ℕ) (initial : AdamW.History I) :
    ℕ → AdamW.History I
  | 0 => initial
  | n + 1 => AdamW.nextHistory p (offset + n) (clock p offset initial n) 0

/-- Both scalar histories agree for every oracle; their raw variances may differ. -/
theorem clock_agrees (p : AdamW.Parameters) (offset : ℕ)
    (oracle : ℕ → (I → ℝ) → (I → ℝ)) (s : AdamW.TrainingState I) (n : ℕ) :
    (run p offset oracle s n).history.total = (clock p offset s.history n).total ∧
    (run p offset oracle s n).history.maximumRate = (clock p offset s.history n).maximumRate := by
  induction n with
  | zero => exact ⟨rfl, rfl⟩
  | succ n ih =>
    constructor
    · change (run p offset oracle s n).history.total +
        AdamW.weight p (offset + n) (run p offset oracle s n).history =
        (clock p offset s.history n).total + AdamW.weight p (offset + n) (clock p offset s.history n)
      simp only [AdamW.weight, AdamW.nextMaximum, ih.1, ih.2]
    · change AdamW.nextMaximum p (offset + n) (run p offset oracle s n).history =
        AdamW.nextMaximum p (offset + n) (clock p offset s.history n)
      simp only [AdamW.nextMaximum, ih.2]

/-- Consequently all source transition matrices are fixed by the initial clock. -/
theorem transition_clock (p : AdamW.Parameters) (offset : ℕ)
    (oracle : ℕ → (I → ℝ) → (I → ℝ)) (s : AdamW.TrainingState I) (n : ℕ) :
    transition p (offset + n) (run p offset oracle s n).history =
      transition p (offset + n) (clock p offset s.history n) := by
  have hc := clock_agrees p offset oracle s n
  simp only [transition, coefficient_eq p _ _ _ hc.1 hc.2]

/-- Evaluation is the source linear recovery of x from the two real buffers. -/
def readEvaluation (p : AdamW.Parameters) : Buffers I →ₗ[ℝ] (I → ℝ) :=
  LinearMap.fst ℝ (I → ℝ) (I → ℝ) + (1 - 1 / p.adam.betaOne) •
    (LinearMap.snd ℝ (I → ℝ) (I → ℝ) - LinearMap.fst ℝ (I → ℝ) (I → ℝ))

/-- No extra evaluation or train/eval mode transition is inserted. -/
theorem readEvaluation_buffers (p : AdamW.Parameters) (s : AdamW.TrainingState I) :
    readEvaluation p (buffers s) = AdamW.evaluation p s := by
  funext i
  rfl

example := coefficient_eq (I := Fin 3)
  ⟨⟨3/1000, 1, 0, 9/10, 98/100, 1/100000000⟩, 0, 2⟩ 400
  ⟨fun _ => 1, 1, 3/1000⟩ ⟨fun i => i.val, 1, 3/1000⟩ rfl rfl
example := step_split (I := Fin 3)
  ⟨⟨3/1000, 1, 0, 9/10, 98/100, 1/100000000⟩, 0, 2⟩ 400
  ⟨fun i => i.val, fun _ => 1, ⟨fun _ => 1, 1, 3/1000⟩⟩ (fun _ => 2)

end Transformer.Rotate90.Forcing

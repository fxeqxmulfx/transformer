module

public import Transformer.Rotate90.Forcing.Actual

/-!
# Sampled learning for arbitrary finite parameterized predictors

Source: `review/flat_spectrum.md`, *General positional models*, G1. The input
type may be a tuple of any finite arity, or a variable-length token sequence.
The forward function retains the complete architecture. No factorization,
fixed rotation, linearized score, or claim of convergence is imposed.

The total Lean gradient is a classical gradient only at differentiable
points. Source identities also apply to any supplied update oracle, including
a separately specified derivative convention at an activation boundary.
-/

@[expose] public section

noncomputable section

namespace Transformer.SequenceModel

open ScheduleFree Rotate90

variable {I X V : Type*} [Fintype I] [Fintype V]

/-- A full vector of independently variable scalar trainable parameters. -/
abbrev Weights (I : Type*) := EuclideanSpace ℝ I

/-- Any architecture's ordinary sampled softmax cross-entropy. The same
parameter vector determines the entire nonlinear score table. -/
def loss (forward : Weights I → X → V → ℝ) (rows : Finset X)
    (target : X → V) (w : Weights I) : ℝ :=
  Landscape.Neural.meanCrossEntropy rows (forward w) target

/-- Sample gradients are taken at the optimizer's query, not its evaluation
average. All trainable coordinates of `forward` vary in this derivative. -/
def sampledOracle (forward : Weights I → X → V → ℝ) (samples : ℕ → Finset X)
    (target : X → V) (t : ℕ) (y : I → ℝ) : I → ℝ :=
  (gradient (loss forward (samples t) target) (WithLp.toLp 2 y)).ofLp

/-- The actual source recurrence for an arbitrary architecture's sampled loss. -/
def trajectory (forward : Weights I → X → V → ℝ) (samples : ℕ → Finset X)
    (target : X → V) (p : AdamW.Parameters) (offset : ℕ)
    (initial : AdamW.TrainingState I) : ℕ → AdamW.TrainingState I
  | 0 => initial
  | n + 1 =>
      let s := trajectory forward samples target p offset initial n
      AdamW.trainingStep p (offset + n) s
        (sampledOracle forward samples target (offset + n) s.point)

/-- G1: the independently stated sampled trajectory is the same source run
used by the architecture-independent matrix/forcing theorems. -/
theorem trajectory_eq_run (forward : Weights I → X → V → ℝ)
    (samples : ℕ → Finset X) (target : X → V) (p : AdamW.Parameters)
    (offset : ℕ) (s : AdamW.TrainingState I) (n : ℕ) :
    trajectory forward samples target p offset s n =
      Forcing.run p offset (sampledOracle forward samples target) s n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [trajectory, Forcing.run, ih]

/-- At a differentiable query the oracle is the ordinary full loss gradient.
No differentiability of a new architecture is hidden inside its definition. -/
theorem oracle_hasGradientAt (forward : Weights I → X → V → ℝ)
    (samples : ℕ → Finset X) (target : X → V) (t : ℕ) (y : I → ℝ)
    (h : DifferentiableAt ℝ (loss forward (samples t) target) (WithLp.toLp 2 y)) :
    HasGradientAt (loss forward (samples t) target)
      (WithLp.toLp 2 (sampledOracle forward samples target t y)) (WithLp.toLp 2 y) := by
  exact h.hasGradientAt

/-- Strict classification, with no tie-breaking convention supplying success. -/
def Correct (forward : Weights I → X → V → ℝ) (w : Weights I)
    (target : X → V) (x : X) : Prop :=
  ∀ v, v ≠ target x → forward w x v < forward w x (target x)

omit [Fintype I] [Fintype V] in
/-- G1: identical score rows cannot strictly predict different labels. This
applies to information discarded by any positional encoding or mixer. -/
theorem collision_not_correct (forward : Weights I → X → V → ℝ)
    (w : Weights I) (target : X → V) (a b : X)
    (hscores : ∀ v, forward w a v = forward w b v) (hlabel : target a ≠ target b) :
    ¬ (Correct forward w target a ∧ Correct forward w target b) := by
  rintro ⟨ha, hb⟩
  have hab := ha (target b) hlabel.symm
  have hba := hb (target a) hlabel
  rw [hscores (target b), hscores (target a)] at hab
  exact (lt_asymm hab hba)

example := oracle_hasGradientAt
  (fun w (r : Fin 3 × Fin 3) q => Rotate90.Learning.score (K := Fin 2) (H := Fin 4)
    (1 / 8388608) (List.finRange 2) w r.1 r.2 q)
  (fun _ => ({(0, 1)} : Finset (Fin 3 × Fin 3))) (fun r => r.1) 0 (fun _ => 1)
  (by
    change DifferentiableAt ℝ
      (Rotate90.Learning.loss (1 / 8388608) (List.finRange 2) {(0, 1)} (fun r => r.1)) _
    exact ((Rotate90.Learning.contDiff_loss (by norm_num) _ _ _).differentiable
      (by decide)).differentiableAt)

example := collision_not_correct (I := Fin 2)
  (fun w (_ : Fin 2) (v : Fin 2) => w.ofLp v) (WithLp.toLp 2 ![1, 2])
  id 0 1 (fun _ => rfl) (by decide)

end Transformer.SequenceModel

module

public import Transformer.ScheduleFree.Section2_Averaging

/-!
# Schedule-Free AdamW

arXiv:2405.15682v4, §4.4, Algorithm 1. This is the actual coordinatewise
algorithm, including interpolation, second moments, bias correction,
warmup, decay at `y`, and squared-learning-rate averaging. Source time
`t=1` is time `0` here. A zero warmup duration means no warmup; the
paper's displayed `t/T_warmup` does not define that conventional case.
No convergence theorem for nonconvex AdamW is claimed by the paper.
-/

@[expose] public section

noncomputable section

namespace Transformer.ScheduleFree

/-- §4.4: the three stored buffers and cumulative averaging weight. -/
structure AdamState (d : ℕ) where
  evaluation : Fin d → ℝ
  base : Fin d → ℝ
  second : Fin d → ℝ
  total : ℝ

/-- §4.4: the optimizer's ordinary hyperparameters. -/
structure AdamParameters where
  learningRate : ℝ
  decay : ℝ
  warmup : ℕ
  betaOne : ℝ
  betaTwo : ℝ
  epsilon : ℝ

/-- Algorithm 1's linear warmup, with an explicit no-warmup convention. -/
def warmupFactor (warmup t : ℕ) : ℝ :=
  if warmup = 0 then 1 else min 1 (((t : ℝ) + 1) / warmup)

/-- Algorithm 1's learning rate, including the second-moment bias correction. -/
def adamLearningRate (p : AdamParameters) (t : ℕ) : ℝ :=
  p.learningRate * Real.sqrt (1 - p.betaTwo ^ (t + 1)) * warmupFactor p.warmup t

/-- Algorithm 1's query point: `α=1-β₁`. -/
def adamQuery {d : ℕ} (p : AdamParameters) (s : AdamState d) : Fin d → ℝ :=
  fun i => (1 - p.betaOne) * s.base i + p.betaOne * s.evaluation i

/-- Algorithm 1, one full step after the oracle has been queried at `adamQuery`. -/
def adamStep {d : ℕ} (p : AdamParameters) (t : ℕ) (s : AdamState d)
    (g : Fin d → ℝ) : AdamState d :=
  let y := adamQuery p s
  let v := fun i => p.betaTwo * s.second i + (1 - p.betaTwo) * g i ^ 2
  let η := adamLearningRate p t
  let total := s.total + η ^ 2
  let c := η ^ 2 / total
  let z := fun i => s.base i - η * g i / (Real.sqrt (v i) + p.epsilon) - η * p.decay * y i
  ⟨fun i => (1 - c) * s.evaluation i + c * z i, z, v, total⟩

/-- Algorithm 1: initialization and all subsequent oracle locations. -/
def adamRun {d : ℕ} (p : AdamParameters) (oracle : ℕ → (Fin d → ℝ) → (Fin d → ℝ))
    (initial : Fin d → ℝ) : ℕ → AdamState d
  | 0 => ⟨initial, initial, fun _ => 0, 0⟩
  | t + 1 =>
    let s := adamRun p oracle initial t
    adamStep p t s (oracle t (adamQuery p s))

/-- Algorithm 1: nonnegative second moments remain nonnegative. -/
theorem adam_second_nonneg {d : ℕ} (p : AdamParameters) (t : ℕ) (s : AdamState d)
    (g : Fin d → ℝ) (hβ : 0 ≤ p.betaTwo) (hβ1 : p.betaTwo ≤ 1)
    (hv : ∀ i, 0 ≤ s.second i) (i : Fin d) : 0 ≤ (adamStep p t s g).second i := by
  exact add_nonneg (mul_nonneg hβ (hv i))
    (mul_nonneg (sub_nonneg.mpr hβ1) (sq_nonneg _))

example (d : ℕ) (i : Fin d) :
    0 ≤ (adamStep ⟨1, 0, 0, 0, 0, 1⟩ 0 ⟨fun _ => 0, fun _ => 0, fun _ => 0, 0⟩
      (fun _ => 0)).second i :=
  adam_second_nonneg _ _ _ _ le_rfl zero_le_one (fun _ => le_rfl) i

/-- Algorithm 1: cumulative averaging weights never become negative. -/
theorem adam_total_nonneg {d : ℕ} (p : AdamParameters) (t : ℕ) (s : AdamState d)
    (g : Fin d → ℝ) (h : 0 ≤ s.total) : 0 ≤ (adamStep p t s g).total :=
  add_nonneg h (sq_nonneg _)

example : 0 ≤ (adamStep (d := 1) ⟨1, 0, 0, 0, 0, 1⟩ 0
    ⟨fun _ => 0, fun _ => 0, fun _ => 0, 0⟩ (fun _ => 0)).total :=
  adam_total_nonneg _ _ _ _ le_rfl

/-- The paper's denominator `√v+ε` is nonsingular for `ε>0`. -/
theorem adam_denominator_pos (v ε : ℝ) (hε : 0 < ε) : 0 < Real.sqrt v + ε :=
  add_pos_of_nonneg_of_pos (Real.sqrt_nonneg v) hε

example : 0 < Real.sqrt (0 : ℝ) + 1 := adam_denominator_pos 0 1 zero_lt_one

end Transformer.ScheduleFree

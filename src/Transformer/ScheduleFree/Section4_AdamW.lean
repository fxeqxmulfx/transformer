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

/-- Algorithm 1: the cumulative weight increases by a square, even when
the externally supplied rate is zero or changes sign. -/
theorem adam_total_monotone {d : ℕ} (p : AdamParameters) (t : ℕ) (s : AdamState d)
    (g : Fin d → ℝ) : s.total ≤ (adamStep p t s g).total := by
  change s.total ≤ s.total + adamLearningRate p t ^ 2
  nlinarith [sq_nonneg (adamLearningRate p t)]

/-- Algorithm 1: every reachable cumulative weight is nonnegative,
without an extra oracle assumption. -/
theorem adamRun_total_nonneg {d : ℕ} (p : AdamParameters)
    (oracle : ℕ → (Fin d → ℝ) → (Fin d → ℝ)) (initial : Fin d → ℝ) (t : ℕ) :
    0 ≤ (adamRun p oracle initial t).total := by
  induction t with
  | zero => exact le_rfl
  | succ t ih => exact adam_total_nonneg p t _ _ ih

/-- Algorithm 1: the actual query-driven run preserves nonnegative
second moments when its second-moment coefficient is in `[0,1]`. -/
theorem adamRun_second_nonneg {d : ℕ} (p : AdamParameters)
    (oracle : ℕ → (Fin d → ℝ) → (Fin d → ℝ)) (initial : Fin d → ℝ)
    (hβ : 0 ≤ p.betaTwo) (hβ1 : p.betaTwo ≤ 1) (t : ℕ) (i : Fin d) :
    0 ≤ (adamRun p oracle initial t).second i := by
  induction t generalizing i with
  | zero => exact le_rfl
  | succ t ih => exact adam_second_nonneg p t _ _ hβ hβ1 ih i

example (t : ℕ) :
    0 ≤ (adamRun (d := 1) ⟨1, 0, 0, 0, 0, 1⟩ (fun _ _ _ => 2) (fun _ => 1) t).second 0 :=
  adamRun_second_nonneg _ _ _ le_rfl zero_le_one t 0

/-- Algorithm 1: its squared-rate averaging coefficient is a convex
weight for a nonnegative previous total, including a zero-rate prefix.
The zero-total case uses total real division and gives coefficient zero. -/
theorem adam_averaging_bounds {d : ℕ} (p : AdamParameters) (t : ℕ) (s : AdamState d)
    (h : 0 ≤ s.total) :
    0 ≤ adamLearningRate p t ^ 2 / (s.total + adamLearningRate p t ^ 2) ∧
      adamLearningRate p t ^ 2 / (s.total + adamLearningRate p t ^ 2) ≤ 1 := by
  have htotal : 0 ≤ s.total + adamLearningRate p t ^ 2 :=
    add_nonneg h (sq_nonneg _)
  constructor
  · exact div_nonneg (sq_nonneg _) htotal
  · by_cases hz : s.total + adamLearningRate p t ^ 2 = 0
    · simp [hz]
    · rw [div_le_one (lt_of_le_of_ne htotal (Ne.symm hz))]
      linarith

example :
    0 ≤ adamLearningRate ⟨1, 0, 0, 0, 0, 1⟩ 0 ^ 2 /
        ((0 : ℝ) + adamLearningRate ⟨1, 0, 0, 0, 0, 1⟩ 0 ^ 2) ∧
      adamLearningRate ⟨1, 0, 0, 0, 0, 1⟩ 0 ^ 2 /
        ((0 : ℝ) + adamLearningRate ⟨1, 0, 0, 0, 0, 1⟩ 0 ^ 2) ≤ 1 :=
  adam_averaging_bounds (d := 1) _ _ ⟨fun _ => 0, fun _ => 0, fun _ => 0, 0⟩ le_rfl

/-- Algorithm 1: the new evaluation point has exactly the old total
weight and the new squared effective-rate weight. Nonzero new total is
explicit; no first-moment EMA or unweighted mean is substituted. -/
theorem adam_weighted_balance {d : ℕ} (p : AdamParameters) (t : ℕ) (s : AdamState d)
    (g : Fin d → ℝ) (h : s.total + adamLearningRate p t ^ 2 ≠ 0) (i : Fin d) :
    (adamStep p t s g).total * (adamStep p t s g).evaluation i =
      s.total * s.evaluation i + adamLearningRate p t ^ 2 * (adamStep p t s g).base i := by
  let q := adamStep p t s g
  change (s.total + adamLearningRate p t ^ 2) *
      ((1 - adamLearningRate p t ^ 2 / (s.total + adamLearningRate p t ^ 2)) * s.evaluation i +
        adamLearningRate p t ^ 2 / (s.total + adamLearningRate p t ^ 2) * q.base i) = _
  field_simp
  ring

example :
    let p : AdamParameters := ⟨1, 0, 0, 0, 0, 1⟩
    let s : AdamState 1 := ⟨fun _ => 1, fun _ => 1, fun _ => 0, 0⟩
    (adamStep p 0 s (fun _ => 2)).total * (adamStep p 0 s (fun _ => 2)).evaluation 0 =
      s.total * s.evaluation 0 + adamLearningRate p 0 ^ 2 * (adamStep p 0 s (fun _ => 2)).base 0 :=
  adam_weighted_balance _ _ _ _ (by norm_num [adamLearningRate, warmupFactor]) 0

end Transformer.ScheduleFree

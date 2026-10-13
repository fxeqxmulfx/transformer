import Transformer.Grokking.AdamW.SecondStep

/-!
# Reachable momentum can turn the next update into ascent

Source: native PyTorch 2.14.1 AdamW at lab commit 43d4d66, zero initial
moment buffers and its exact first/second-step recurrence. Comparison:
the proposed repeated-step extension of first-step loss descent used
when transferring grokking effective models to the actual optimizer.

Use the actual derivative of a strictly convex scalar quadratic. All
optimizer parameters match the experiment: betas `(0.9, 0.98)`, epsilon
`1e-8`, decay `0.1`, first rate `0.001`. At initial parameter `0.00075`,
the first update lowers the loss and crosses its minimum. The retained
first moment makes the next direction oppose the current true gradient.
The complete second update, including decay, raises the loss for every
positive second rate, including arbitrarily small ones.

This is a deterministic same-objective, exact-real reachable history;
no arbitrary buffer is planted. It refutes unconditional repeated-step
descent, not grokking or convergence of GPTMini. The quadratic is an
explicit deviation from transformer CE and the parallelogram quotient.
-/

namespace Transformer.Grokking.AdamW

/-- A scalar test objective with one shared minimum. Source comparison:
native AdamW at 43d4d66; this replaces transformer CE explicitly. -/
noncomputable def quadraticLoss (p : ℝ) : ℝ := p ^ 2 / 2

/-- The objective's actual gradient, rather than a supplied sign rule.
Source: the gradient evaluation in PyTorch 2.14.1's native AdamW equations. -/
noncomputable def quadraticGradient (p : ℝ) : ℝ := deriv quadraticLoss p

/-- The actual first point at the experiment's optimizer parameters.
Source: native AdamW at 43d4d66; both initial moment buffers are zero. -/
noncomputable def quadraticFirstPoint : ℝ :=
  firstUpdate (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000)
    (3 / 4000) (quadraticGradient (3 / 4000))

/-- Native second direction from the actual first and current gradients.
Source: PyTorch 2.14.1 second moment recurrence at 43d4d66; the first
buffers are computed from the preceding true gradient, not reset. -/
noncomputable def quadraticSecondDirection : ℝ :=
  secondDirection (9 / 10) (49 / 50) (1 / 100000000)
    (quadraticGradient (3 / 4000)) (quadraticGradient quadraticFirstPoint)

/-- Exact actual derivative of the quadratic test objective. Source:
native AdamW's gradient input at 43d4d66; the objective is explicitly
the scalar quadratic, not GPTMini CE. -/
theorem quadratic_gradient_eq (p : ℝ) : quadraticGradient p = p := by
  have h : HasDerivAt quadraticLoss p p := by
    convert ((hasDerivAt_id p).pow 2).mul_const (1 / 2) using 1
    · funext x
      unfold quadraticLoss
      dsimp
      ring
    · dsimp
      ring
  exact h.deriv

/-- Strict Jensen inequality confirms that ascent is not explained by
nonconvexity of the chosen objective. Source: the quadratic test of
native repeated-step AdamW descent at 43d4d66. -/
theorem quadratic_strict_jensen (x y t : ℝ) (ht : 0 < t) (h1 : t < 1) (hxy : x ≠ y) :
    quadraticLoss (t * x + (1 - t) * y) < t * quadraticLoss x + (1 - t) * quadraticLoss y := by
  have hd : x - y ≠ 0 := by intro h; apply hxy; linarith
  have hp : 0 < t * (1 - t) * (x - y) ^ 2 / 2 := by positivity
  have heq : t * quadraticLoss x + (1 - t) * quadraticLoss y -
      quadraticLoss (t * x + (1 - t) * y) = t * (1 - t) * (x - y) ^ 2 / 2 := by
    unfold quadraticLoss
    ring
  linarith

example : 0 < (1 / 2 : ℝ) ∧ (1 / 2 : ℝ) < 1 ∧ (0 : ℝ) ≠ 1 := by norm_num

/-- The first point is exactly reachable from zero moments. Source:
native first update at 43d4d66, all experimental hyperparameters retained. -/
theorem quadratic_first_point_eq : quadraticFirstPoint = -750195003 / 3000040000000 := by
  unfold quadraticFirstPoint firstUpdate
  rw [quadratic_gradient_eq, firstDirection_eq _ _ _ _ (by norm_num) (by norm_num)]
  norm_num

/-- The actual first update strictly improves this objective. Source:
the native two-step test at 43d4d66; later ascent does not require an
already-increasing first step. -/
theorem quadratic_first_loss_decreases : quadraticLoss quadraticFirstPoint < quadraticLoss (3 / 4000) := by
  rw [quadratic_first_point_eq]
  norm_num [quadraticLoss]

/-- The reachable second adaptive direction remains positive although
the current gradient has become negative. Source: native moment
recurrence and second bias correction at 43d4d66. -/
theorem quadratic_second_direction_lower_bound : (1 / 10 : ℝ) < quadraticSecondDirection := by
  unfold quadraticSecondDirection
  rw [quadratic_gradient_eq, quadratic_gradient_eq, quadratic_first_point_eq,
    secondDirection_eq _ _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)]
  let v : ℝ := ((49 / 50) * (3 / 4000) ^ 2 +
    (-750195003 / 3000040000000) ^ 2) / (1 + 49 / 50)
  have hv : v ≤ (1 / 1000 : ℝ) ^ 2 := by norm_num [v]
  have hs : Real.sqrt v ≤ (1 / 1000 : ℝ) :=
    Real.sqrt_le_iff.mpr ⟨by norm_num, hv⟩
  have hd : 0 < Real.sqrt v + (1 / 100000000 : ℝ) := by positivity
  apply (lt_div_iff₀ hd).mpr
  change (1 / 10 : ℝ) * (Real.sqrt v + 1 / 100000000) <
    ((9 / 10) * (3 / 4000) + (-750195003 / 3000040000000)) / (1 + 9 / 10)
  nlinarith

/-- Decoupled decay is included and is insufficient to reverse the
reachable second velocity. Source: native AdamW at 43d4d66, decay 0.1. -/
theorem quadratic_second_velocity_negative :
    -(1 / 10 : ℝ) * quadraticFirstPoint - quadraticSecondDirection < 0 := by
  have h := quadratic_second_direction_lower_bound
  rw [quadratic_first_point_eq]
  linarith

/-- Current true gradient and retained adaptive direction are opposed
on this actual history. Source: native AdamW at 43d4d66; a squared
gradient-energy identity from the first step cannot replace this dot. -/
theorem quadratic_reachable_alignment_negative :
    quadraticGradient quadraticFirstPoint * quadraticSecondDirection < 0 := by
  have hg : quadraticGradient quadraticFirstPoint < 0 := by
    rw [quadratic_gradient_eq, quadratic_first_point_eq]
    norm_num
  have hd : 0 < quadraticSecondDirection := by
    have h := quadratic_second_direction_lower_bound
    linarith
  exact mul_neg_iff.mpr (Or.inr ⟨hg, hd⟩)

/-- The full second-update loss has a strictly positive derivative
with respect to its rate at zero. Source: native AdamW at 43d4d66;
the first point and both gradients are the actual reachable history. -/
theorem quadratic_second_loss_derivative_positive :
    ∃ a : ℝ, 0 < a ∧ HasDerivAt (fun eta => quadraticLoss
      (secondUpdate (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) eta
        quadraticFirstPoint (quadraticGradient (3 / 4000))
        (quadraticGradient quadraticFirstPoint))) a 0 := by
  let velocity := -(1 / 10 : ℝ) * quadraticFirstPoint - quadraticSecondDirection
  have hv : velocity < 0 := quadratic_second_velocity_negative
  have hp : quadraticFirstPoint < 0 := by rw [quadratic_first_point_eq]; norm_num
  have h : HasDerivAt (fun eta => quadraticFirstPoint + eta * velocity) velocity 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const velocity).const_add quadraticFirstPoint
  have hd := (h.pow 2).mul_const (1 / 2)
  refine ⟨quadraticFirstPoint * velocity, mul_pos_iff.mpr (Or.inr ⟨hp, hv⟩), ?_⟩
  convert hd using 1
  · funext eta
    unfold quadraticLoss
    rw [secondUpdate_as_linear_path]
    dsimp [velocity, quadraticSecondDirection]
    ring
  · dsimp
    ring

/-- Every positive second rate raises the loss on this history, even
an arbitrarily small rate. Source: native AdamW at 43d4d66; this is
momentum ascent, not merely overshoot from choosing a large second rate. -/
theorem quadratic_second_loss_increases (eta : ℝ) (heta : 0 < eta) :
    quadraticLoss quadraticFirstPoint < quadraticLoss
      (secondUpdate (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) eta
        quadraticFirstPoint (quadraticGradient (3 / 4000))
        (quadraticGradient quadraticFirstPoint)) := by
  let velocity := -(1 / 10 : ℝ) * quadraticFirstPoint - quadraticSecondDirection
  have hv : velocity < 0 := quadratic_second_velocity_negative
  have hp : quadraticFirstPoint < 0 := by rw [quadratic_first_point_eq]; norm_num
  have hm : eta * velocity < 0 := mul_neg_iff.mpr (Or.inl ⟨heta, hv⟩)
  have hprod : 0 < quadraticFirstPoint * (eta * velocity) := mul_pos_iff.mpr (Or.inr ⟨hp, hm⟩)
  rw [secondUpdate_as_linear_path]
  change quadraticLoss quadraticFirstPoint < quadraticLoss (quadraticFirstPoint + eta * velocity)
  unfold quadraticLoss
  nlinarith [sq_nonneg (eta * velocity)]

example : 0 < (1 / 1000 : ℝ) := by norm_num

end Transformer.Grokking.AdamW

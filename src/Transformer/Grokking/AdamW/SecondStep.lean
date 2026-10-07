import Transformer.Grokking.AdamW.LossDescent

/-!
# Native second-step direction and conditional effective-loss descent

Source: PyTorch 2.14.1 AdamW at lab commit 43d4d66, moment recurrence
and bias correction at step two, starting from zero buffers. Comparison:
Liu et al., arXiv:2205.10343v2, section 3.2, eq:l_eff, whose effective
quotient is retained without transferring its assumed Euclidean flow.

Derive the actual adaptive direction from the previous and current
gradients. Differentiate the finite second-update effective loss with
its current representation and history held fixed. Decoupled decay
cancels from that derivative, but the remaining gradient/direction
alignment need not be positive. It is an explicit observable premise
for the conditional local-descent result, not a claimed learned property.
The general formulas do not assert reachability of arbitrary inputs;
the companion counterexample constructs an actual two-step history.
-/

namespace Transformer.Grokking.AdamW

open Transformer.Grokking.EffectiveTheory

/-- Bias-corrected second direction after exactly one prior gradient
and zero initial moments. Source: PyTorch 2.14.1 AdamW equations
at 43d4d66, ordinary non-AMSGrad minimization. -/
noncomputable def secondDirection (b1 b2 eps previous current : ℝ) : ℝ :=
  ((b1 * ((1 - b1) * previous) + (1 - b1) * current) / (1 - b1 ^ 2)) /
    (Real.sqrt ((b2 * ((1 - b2) * previous ^ 2) + (1 - b2) * current ^ 2) /
      (1 - b2 ^ 2)) + eps)

/-- Actual coordinate update using that retained first-gradient
history. Source: native decoupled AdamW at 43d4d66; current gradients
are evaluated before decay, and the past moment is not reset. -/
noncomputable def secondUpdate (b1 b2 eps decay eta p previous current : ℝ) : ℝ :=
  (1 - eta * decay) * p - eta * secondDirection b1 b2 eps previous current

/-- Current effective gradient paired with the actual second direction.
Source: arXiv:2205.10343v2, eq:l_eff, and native AdamW at 43d4d66.
Its sign is not imposed by the definition. -/
noncomputable def secondAlignment (b1 b2 eps x y z : ℝ) (previous : ℝ × ℝ × ℝ) : ℝ :=
  (quotientGradient x y z).1 * secondDirection b1 b2 eps previous.1 (quotientGradient x y z).1 +
    (quotientGradient x y z).2.1 * secondDirection b1 b2 eps previous.2.1 (quotientGradient x y z).2.1 +
    (quotientGradient x y z).2.2 * secondDirection b1 b2 eps previous.2.2 (quotientGradient x y z).2.2

/-- Actual effective loss after the finite second update. Source:
arXiv:2205.10343v2, eq:l_eff, using the native moment recurrence
at 43d4d66; the previous gradient is supplied, not inferred from logits. -/
noncomputable def lossAfterSecondStep (b1 b2 eps decay eta x y z : ℝ)
    (previous : ℝ × ℝ × ℝ) : ℝ :=
  normalizedLoss (secondUpdate b1 b2 eps decay eta x previous.1 (quotientGradient x y z).1)
    (secondUpdate b1 b2 eps decay eta y previous.2.1 (quotientGradient x y z).2.1)
    (secondUpdate b1 b2 eps decay eta z previous.2.2 (quotientGradient x y z).2.2)

/-- Exact bias-correction simplification at step two. Source: native
AdamW at 43d4d66, PyTorch 2.14.1. Excluding both signs of one ensures
the second bias-correction denominators are nonzero. Native betas
in `[0,1)` satisfy these stronger algebraic domain checks. -/
theorem secondDirection_eq (b1 b2 eps previous current : ℝ)
    (h1 : b1 ≠ 1) (h1n : b1 ≠ -1) (h2 : b2 ≠ 1) (h2n : b2 ≠ -1) :
    secondDirection b1 b2 eps previous current =
      ((b1 * previous + current) / (1 + b1)) /
        (Real.sqrt ((b2 * previous ^ 2 + current ^ 2) / (1 + b2)) + eps) := by
  have hn (b : ℝ) (hp : b ≠ 1) (hm : b ≠ -1) : 1 - b ^ 2 ≠ 0 := by
    intro h
    have hf : (1 - b) * (1 + b) = 0 := by nlinarith
    rcases mul_eq_zero.mp hf with hzero | hzero
    · apply hp; linarith
    · apply hm; linarith
  have hd1 := hn b1 h1 h1n
  have hd2 := hn b2 h2 h2n
  have hs1 : 1 + b1 ≠ 0 := by intro h; apply h1n; linarith
  have hs2 : 1 + b2 ≠ 0 := by intro h; apply h2n; linarith
  have hm : (b1 * ((1 - b1) * previous) + (1 - b1) * current) / (1 - b1 ^ 2) =
      (b1 * previous + current) / (1 + b1) := by field_simp; ring
  have hv : (b2 * ((1 - b2) * previous ^ 2) + (1 - b2) * current ^ 2) / (1 - b2 ^ 2) =
      (b2 * previous ^ 2 + current ^ 2) / (1 + b2) := by field_simp; ring
  unfold secondDirection
  rw [hm, hv]

example : (9 / 10 : ℝ) ≠ 1 ∧ (9 / 10 : ℝ) ≠ -1 ∧
    (49 / 50 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ -1 := by norm_num

/-- The second update is affine in its own learning rate with the
current representation and history held fixed. Source: native AdamW
at 43d4d66; this is not variation of the first step's earlier rate. -/
theorem secondUpdate_as_linear_path (b1 b2 eps decay eta p previous current : ℝ) :
    secondUpdate b1 b2 eps decay eta p previous current =
      p + eta * (-decay * p - secondDirection b1 b2 eps previous current) := by
  unfold secondUpdate
  ring

/-- Zero second-step rate leaves the current effective loss unchanged.
Source: native update at 43d4d66 and arXiv:2205.10343v2, eq:l_eff. -/
theorem second_loss_after_zero_rate (b1 b2 eps decay x y z : ℝ)
    (previous : ℝ × ℝ × ℝ) :
    lossAfterSecondStep b1 b2 eps decay 0 x y z previous = normalizedLoss x y z := by
  unfold lossAfterSecondStep secondUpdate
  norm_num

/-- Actual second-rate derivative is minus the gradient/direction
alignment, rather than minus a weighted gradient-square sum. Source:
native second moments at 43d4d66 and arXiv:2205.10343v2, eq:l_eff. -/
theorem second_step_loss_deriv (b1 b2 eps decay x y z : ℝ) (previous : ℝ × ℝ × ℝ)
    (hZ : squaredNorm x y z ≠ 0) :
    HasDerivAt (fun eta => lossAfterSecondStep b1 b2 eps decay eta x y z previous)
      (-secondAlignment b1 b2 eps x y z previous) 0 := by
  have h := normalizedLoss_linear_path_deriv x y z
    (-decay * x - secondDirection b1 b2 eps previous.1 (quotientGradient x y z).1)
    (-decay * y - secondDirection b1 b2 eps previous.2.1 (quotientGradient x y z).2.1)
    (-decay * z - secondDirection b1 b2 eps previous.2.2 (quotientGradient x y z).2.2) hZ
  convert h using 1
  · funext eta
    unfold lossAfterSecondStep
    rw [secondUpdate_as_linear_path, secondUpdate_as_linear_path, secondUpdate_as_linear_path]
  · unfold secondAlignment
    have hr := quotientGradient_radial_zero x y z hZ
    linear_combination decay * hr

example : squaredNorm 1 2 0 ≠ 0 := by norm_num [squaredNorm]

/-- Positive actual alignment suffices for local finite-step decrease.
Source: native AdamW at 43d4d66 and arXiv:2205.10343v2, eq:l_eff.
No claim supplies this optimizer-dependent premise along training. -/
theorem second_step_locally_lowers_effective_loss (b1 b2 eps decay x y z : ℝ)
    (previous : ℝ × ℝ × ℝ) (hZ : squaredNorm x y z ≠ 0)
    (ha : 0 < secondAlignment b1 b2 eps x y z previous) :
    ∃ delta : ℝ, 0 < delta ∧ ∀ eta : ℝ, 0 < eta → eta < delta →
      lossAfterSecondStep b1 b2 eps decay eta x y z previous < normalizedLoss x y z := by
  have hd := second_step_loss_deriv b1 b2 eps decay x y z previous hZ
  have hn : -secondAlignment b1 b2 eps x y z previous < 0 := by linarith
  obtain ⟨delta, hp, hl⟩ := locally_decreases_of_negative_derivative _ _ hd hn
  refine ⟨delta, hp, ?_⟩
  intro eta heta hsmall
  simpa [second_loss_after_zero_rate] using hl eta heta hsmall

example : squaredNorm 1 2 0 ≠ 0 ∧ 0 < secondAlignment (9 / 10) (49 / 50) 1 1 2 0 (0, 0, 0) := by
  constructor
  · norm_num [squaredNorm]
  · have hs (g : ℝ) (hg : g ≠ 0) :
        0 < g * secondDirection (9 / 10) (49 / 50) 1 0 g := by
      let d := Real.sqrt (((49 / 50 : ℝ) * 0 ^ 2 + g ^ 2) / (1 + 49 / 50)) + 1
      have hd : 0 < d := by dsimp [d]; positivity
      have heq : g * secondDirection (9 / 10) (49 / 50) 1 0 g =
          g ^ 2 / ((1 + (9 / 10 : ℝ)) * d) := by
        rw [secondDirection_eq _ _ _ _ _
          (by norm_num) (by norm_num) (by norm_num) (by norm_num)]
        have hdn : d ≠ 0 := by positivity
        change g * (((9 / 10 : ℝ) * 0 + g) / (1 + 9 / 10) / d) =
          g ^ 2 / ((1 + (9 / 10 : ℝ)) * d)
        field_simp
        ring
      rw [heq]
      exact div_pos (sq_pos_of_ne_zero hg) (by positivity)
    have h0 := hs (-48 / 25) (by norm_num)
    have h1 := hs (24 / 25) (by norm_num)
    have h2 := hs (-6 / 5) (by norm_num)
    unfold secondAlignment
    rw [quotientGradient_test_point.1]
    dsimp
    linarith

end Transformer.Grokking.AdamW

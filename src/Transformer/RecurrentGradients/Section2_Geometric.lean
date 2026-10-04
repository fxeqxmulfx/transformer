/-
# The geometrical interpretation

arXiv:1211.5063, §2.3.  The one hidden unit model of eq. (8),
`x_t = w σ(x_{t-1}) + b`, with no input (`unitStates`), simplified to be
linear with `b = 0`, has `x_t = x₀ w^t` (`unitStates_id`), from which
`∂x_t/∂w = t x₀ w^{t-1}` (`deriv_unitStates_id`) and
`∂²x_t/∂w² = t(t-1) x₀ w^{t-2}` (`deriv_deriv_unitStates_id`), "implying that
when the first derivative explodes, so does the second derivative": from
`t ≥ |w| + 1` on, `|∂x_t/∂w| ≤ |∂²x_t/∂w²|` (`eventually_abs_deriv_le`), so
an eventual lower bound `C α^t` on the first is one on the second
(`eventually_le_abs_deriv_deriv`).
-/

import Transformer.RecurrentGradients.Section1_Recurrence
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Order.Filter.AtTopBot.Basic

open Filter

namespace Transformer.RecurrentGradients

/-- The states of the one hidden unit model of eq. (8), `x_t = w σ(x_{t-1}) + b`,
from `x₀`, with no input. -/
def unitStates (σ : ℝ → ℝ) (w b x₀ : ℝ) (t : ℕ) : ℝ :=
  states (fun x (_ : Unit) => w * σ x + b) (fun _ => ()) x₀ t

/-- **§2.3**: the linear model with `b = 0` has `x_t = x₀ w^t`. -/
theorem unitStates_id (w x₀ : ℝ) (t : ℕ) : unitStates id w 0 x₀ t = x₀ * w ^ t := by
  induction t with
  | zero => simp [unitStates, states]
  | succ t ih =>
    change w * id (unitStates id w 0 x₀ t) + 0 = _
    rw [id, ih, add_zero, pow_succ]
    ring

/-- **§2.3**: `∂x_t/∂w = t x₀ w^{t-1}`. -/
theorem hasDerivAt_unitStates_id (x₀ w : ℝ) (t : ℕ) :
    HasDerivAt (fun w => unitStates id w 0 x₀ t) (t * x₀ * w ^ (t - 1)) w := by
  have e : (fun w => unitStates id w 0 x₀ t) = fun w => x₀ * w ^ t :=
    funext fun w => unitStates_id w x₀ t
  rw [e]
  convert (hasDerivAt_pow t w).const_mul x₀ using 1
  ring

theorem deriv_unitStates_id (x₀ w : ℝ) (t : ℕ) :
    deriv (fun w => unitStates id w 0 x₀ t) w = t * x₀ * w ^ (t - 1) :=
  (hasDerivAt_unitStates_id x₀ w t).deriv

/-- **§2.3**: `∂²x_t/∂w² = t(t-1) x₀ w^{t-2}`. -/
theorem deriv_deriv_unitStates_id (x₀ w : ℝ) (t : ℕ) :
    deriv (deriv fun w => unitStates id w 0 x₀ t) w = t * (t - 1) * x₀ * w ^ (t - 2) := by
  have e : deriv (fun w => unitStates id w 0 x₀ t) = fun w => t * x₀ * w ^ (t - 1) :=
    funext fun w => deriv_unitStates_id x₀ w t
  rw [e, ((hasDerivAt_pow (t - 1) w).const_mul ((t : ℝ) * x₀)).deriv]
  rcases t with _ | t
  · simp
  · rw [Nat.add_sub_cancel, show t + 1 - 2 = t - 1 by omega]
    push_cast
    ring

/-- From `t ≥ |w| + 1` on, `|∂x_t/∂w| ≤ |∂²x_t/∂w²|`: the second derivative is
the first times `(t - 1)/w`. -/
theorem eventually_abs_deriv_le (x₀ w : ℝ) :
    ∀ᶠ t in atTop, |deriv (fun w => unitStates id w 0 x₀ t) w| ≤
      |deriv (deriv fun w => unitStates id w 0 x₀ t) w| := by
  filter_upwards [eventually_ge_atTop (⌈|w|⌉₊ + 2)] with t ht
  obtain ⟨s, rfl⟩ : ∃ s, t = s + 2 := ⟨t - 2, by omega⟩
  have hw : |w| ≤ (s : ℝ) + 1 :=
    (Nat.le_ceil _).trans (by exact_mod_cast (by omega : ⌈|w|⌉₊ ≤ s + 1))
  rw [deriv_unitStates_id, deriv_deriv_unitStates_id, show s + 2 - 1 = s + 1 by omega,
    show s + 2 - 2 = s by omega]
  push_cast
  rw [show ((s : ℝ) + 2) * x₀ * w ^ (s + 1) = ((s + 2) * x₀ * w ^ s) * w by ring,
    show ((s : ℝ) + 2) * (s + 2 - 1) * x₀ * w ^ s = ((s + 2) * x₀ * w ^ s) * (s + 1) by ring,
    abs_mul _ w, abs_mul _ ((s : ℝ) + 1), abs_of_nonneg (by positivity : (0 : ℝ) ≤ s + 1)]
  exact mul_le_mul_of_nonneg_left hw (abs_nonneg _)

/-- **§2.3: "when the first derivative explodes, so does the second
derivative"**: a lower bound `C α^t` that eventually holds for `|∂x_t/∂w|`
eventually holds for `|∂²x_t/∂w²|`. -/
theorem eventually_le_abs_deriv_deriv {x₀ w C α : ℝ}
    (h : ∀ᶠ t in atTop, C * α ^ t ≤ |deriv (fun w => unitStates id w 0 x₀ t) w|) :
    ∀ᶠ t in atTop, C * α ^ t ≤ |deriv (deriv fun w => unitStates id w 0 x₀ t) w| := by
  filter_upwards [h, eventually_abs_deriv_le x₀ w] with t h₁ h₂
  exact h₁.trans h₂

/-- The hypothesis of `eventually_le_abs_deriv_deriv` is satisfiable: `x₀ = 1`,
`w = 2`, whose first derivative `t 2^{t-1}` is at least `2^t / 2`. -/
example : ∀ᶠ t in atTop, 1 / 2 * (2 : ℝ) ^ t ≤
    |deriv (deriv fun w => unitStates id w 0 1 t) 2| := by
  refine eventually_le_abs_deriv_deriv ?_
  filter_upwards [eventually_ge_atTop 1] with t ht
  obtain ⟨s, rfl⟩ : ∃ s, t = s + 1 := ⟨t - 1, by omega⟩
  rw [deriv_unitStates_id, Nat.add_sub_cancel, pow_succ,
    abs_of_nonneg (by positivity)]
  push_cast
  nlinarith [pow_pos (two_pos : (0 : ℝ) < 2) s, (Nat.cast_nonneg s : (0 : ℝ) ≤ s)]

end Transformer.RecurrentGradients

/-
# A training safeguard for Muon and DASH

User-requested convergence extension of arXiv:2502.16982, §2.1–2.2,
and arXiv:2602.02016v2, §2–4. Neither paper states a general training
convergence theorem. The added safeguard checks the complete proposed
direction against the current full gradient. It keeps that direction
when it has sufficient alignment and controlled length, and otherwise
uses the gradient. Momentum and preconditioner states are still updated.

The loss assumptions below describe the objective, not its iterates or
their convergence. They are explicit additional conditions for a fixed,
deterministic differentiable loss with a quadratic smooth upper model.
-/

import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Tactic

open scoped InnerProductSpace

noncomputable section

namespace Transformer.Optimization

variable {E S : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Differentiability and the usual quadratic smooth upper model for the
actual gradient. Additional loss assumption for the training extension of
arXiv:2502.16982, §2.1, and arXiv:2602.02016v2, §2, Algorithm 1. -/
def SmoothObjective [CompleteSpace E] (f : E → ℝ) (L : ℝ) : Prop :=
  Differentiable ℝ f ∧ ∀ x y,
    f y ≤ f x + ⟪gradient f x, y - x⟫_ℝ + L / 2 * ‖y - x‖ ^ 2

/-- Keep a direction only after checking alignment and length. Otherwise
use the current gradient. This is an explicit algorithm change, not a
hypothesis that the original momentum or approximate roots are reliable.
Source: training correction to arXiv:2502.16982, §2.1–2.2, and
arXiv:2602.02016v2, §2.3 and §3. -/
def descentGuard (σ : ℝ) (g d : E) : E :=
  if σ * ‖g‖ ^ 2 ≤ ⟪g, d⟫_ℝ ∧ ‖d‖ ≤ ‖g‖ then d else g

/-- The actual selected direction has the required alignment and length
for every candidate, including inaccurate roots or stale momentum.
Source: training correction to arXiv:2502.16982, §2.1, and
arXiv:2602.02016v2, §2–3. -/
theorem descentGuard_certificate (σ : ℝ) (g d : E) (hσ : σ ≤ 1) :
    σ * ‖g‖ ^ 2 ≤ ⟪g, descentGuard σ g d⟫_ℝ ∧ ‖descentGuard σ g d‖ ≤ ‖g‖ := by
  unfold descentGuard
  split
  · assumption
  · refine ⟨?_, le_rfl⟩
    rw [real_inner_self_eq_norm_sq]
    nlinarith [sq_nonneg ‖g‖]

/-- The safeguard's parameter domain is nonempty,
arXiv:2502.16982, §2.1, and arXiv:2602.02016v2, §2–3, training extension. -/
example : (1 / 2 : ℝ) ≤ 1 := by norm_num

/-- Candidates meeting the check are used exactly, including their
grafting scale or weight decay. Source: training correction to
arXiv:2502.16982, §2.2, and arXiv:2602.02016v2, §2.3. -/
theorem descentGuard_accepts (σ : ℝ) (g d : E)
    (h : σ * ‖g‖ ^ 2 ≤ ⟪g, d⟫_ℝ ∧ ‖d‖ ≤ ‖g‖) : descentGuard σ g d = d := by
  simp only [descentGuard, ite_eq_left h]

/-- A nonzero candidate passes the check,
arXiv:2502.16982, §2.2, and arXiv:2602.02016v2, §2.3, training extension. -/
example : (1 / 2 : ℝ) * ‖(1 : ℝ)‖ ^ 2 ≤ ⟪(1 : ℝ), (1 : ℝ)⟫_ℝ ∧
    ‖(1 : ℝ)‖ ≤ ‖(1 : ℝ)‖ := by norm_num

/-- Rejected candidates are replaced by the actual gradient.
Source: training correction to arXiv:2502.16982, §2.1, and
arXiv:2602.02016v2, §2.3 and §3. -/
theorem descentGuard_rejects (σ : ℝ) (g d : E)
    (h : ¬ (σ * ‖g‖ ^ 2 ≤ ⟪g, d⟫_ℝ ∧ ‖d‖ ≤ ‖g‖)) : descentGuard σ g d = g := by
  simp only [descentGuard, ite_eq_right h]

/-- An uphill candidate is rejected,
arXiv:2502.16982, §2.1, and arXiv:2602.02016v2, §2.3, training extension. -/
example : ¬ ((1 / 2 : ℝ) * ‖(1 : ℝ)‖ ^ 2 ≤ ⟪(1 : ℝ), (-1 : ℝ)⟫_ℝ ∧
    ‖(-1 : ℝ)‖ ≤ ‖(1 : ℝ)‖) := by norm_num

/-- The corrected parameter step uses `η=σ/L` and the checked candidate.
Source: training correction to arXiv:2502.16982, §2.1–2.2, and
arXiv:2602.02016v2, §2, Algorithm 1. -/
def safeguardedStep (σ L : ℝ) (x g d : E) : E :=
  x - (σ / L) • descentGuard σ g d

/-- Full stateful training recurrence. `propose` updates the optimizer's
actual auxiliary state and computes its candidate from the current weights
and full gradient; its direction alone is checked before updating weights.
Source: training correction to arXiv:2502.16982, §2.1–2.2, and
arXiv:2602.02016v2, §2–4. -/
def safeguardedRun [CompleteSpace E] (σ L : ℝ) (f : E → ℝ) (propose : S → E → E → S × E)
    (initialState : S) (initial : E) : ℕ → S × E
  | 0 => (initialState, initial)
  | t + 1 =>
    let state := safeguardedRun σ L f propose initialState initial t
    let g := gradient f state.2
    let candidate := propose state.1 state.2 g
    (candidate.1, safeguardedStep σ L state.2 g candidate.2)

/-- A nonconstant smooth objective witnessing the convergence assumptions.
Source: training extension of arXiv:2502.16982, §2.1, and
arXiv:2602.02016v2, §2, Algorithm 1. -/
def quadratic (x : ℝ) : ℝ := x ^ 2 / 2

/-- The witness has its actual gradient, rather than an arbitrary vector
field called a gradient. Source: training extension of arXiv:2502.16982,
§2.1, and arXiv:2602.02016v2, §2, Algorithm 1. -/
theorem quadratic_hasGradientAt (x : ℝ) : HasGradientAt quadratic x x := by
  apply HasDerivAt.hasGradientAt'
  convert ((hasDerivAt_id x).pow 2).div_const 2 using 1 <;> first | rfl | simp

/-- Actual gradient of the nonconstant witness,
arXiv:2502.16982, §2.1, and arXiv:2602.02016v2, §2, training extension. -/
theorem quadratic_gradient : gradient quadratic = id :=
  gradient_eq quadratic_hasGradientAt

/-- The usual quadratic satisfies the objective assumptions,
arXiv:2502.16982, §2.1, and arXiv:2602.02016v2, §2, training extension. -/
theorem quadratic_smooth : SmoothObjective quadratic 1 := by
  refine ⟨fun x => (quadratic_hasGradientAt x).differentiableAt, ?_⟩
  intro x y
  rw [quadratic_gradient]
  simp only [quadratic, id_eq, RCLike.inner_apply, conj_trivial, Real.norm_eq_abs, sq_abs]
  nlinarith

end Transformer.Optimization

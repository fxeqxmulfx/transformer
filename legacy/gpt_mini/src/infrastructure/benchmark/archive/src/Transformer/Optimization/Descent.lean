/-
# Sufficient decrease of the actual safeguarded training recurrence

Training extension of arXiv:2502.16982, §2.1–2.2, and
arXiv:2602.02016v2, §2–4. Alignment and length are derived from
the executable guard; neither is an assumption about the original
momentum, grafting direction or finite inverse-root approximation.
-/

import Transformer.Optimization.Basic

open scoped InnerProductSpace BigOperators

noncomputable section

namespace Transformer.Optimization

variable {E S : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- Every corrected step decreases a fixed smooth loss by at least
`σ²/(2L)` times the actual squared gradient norm. This supplies the missing
learning-rate and direction conditions in a training interpretation of
arXiv:2502.16982, §2.1–2.2, and arXiv:2602.02016v2, §2–3. -/
theorem safeguardedStep_descent (f : E → ℝ) (σ L : ℝ) (x d : E)
    (hf : SmoothObjective f L) (hL : 0 < L) (hσ : 0 < σ) (hσ' : σ ≤ 1) :
    f (safeguardedStep σ L x (gradient f x) d) ≤
      f x - σ ^ 2 / (2 * L) * ‖gradient f x‖ ^ 2 := by
  let D := descentGuard σ (gradient f x) d
  let η := σ / L
  have hη : 0 ≤ η := (div_pos hσ hL).le
  obtain ⟨halign, hnorm⟩ := descentGuard_certificate σ (gradient f x) d hσ'
  have hnormsq : ‖D‖ ^ 2 ≤ ‖gradient f x‖ ^ 2 :=
    pow_le_pow_left₀ (norm_nonneg _) hnorm 2
  have hdiff : safeguardedStep σ L x (gradient f x) d - x = -η • D := by
    dsimp only [safeguardedStep, η, D]
    module
  have hupper := hf.2 x (safeguardedStep σ L x (gradient f x) d)
  rw [hdiff, real_inner_smul_right, norm_smul, Real.norm_eq_abs, mul_pow, sq_abs,
    neg_sq] at hupper
  have ha := mul_le_mul_of_nonneg_left halign hη
  have hn := mul_le_mul_of_nonneg_left hnormsq
    (show 0 ≤ L / 2 * η ^ 2 by positivity)
  calc
    _ ≤ f x - η * ⟪gradient f x, D⟫_ℝ + L / 2 * η ^ 2 * ‖D‖ ^ 2 := by
      nlinarith [hupper]
    _ ≤ f x - η * (σ * ‖gradient f x‖ ^ 2) + L / 2 * η ^ 2 * ‖gradient f x‖ ^ 2 := by
      linarith
    _ = _ := by dsimp only [η]; field_simp; ring

/-- The descent assumptions hold for a nonconstant objective,
arXiv:2502.16982, §2.1, and arXiv:2602.02016v2, §2, training extension. -/
example : SmoothObjective quadratic 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 :=
  ⟨quadratic_smooth, by norm_num, by norm_num, by norm_num⟩

/-- Sufficient decrease holds along the full stateful optimizer, with
the actual candidate computed at the current weights and gradient.
Source: training correction to arXiv:2502.16982, §2.1–2.2, and
arXiv:2602.02016v2, §2–4. -/
theorem safeguardedRun_descent (f : E → ℝ) (σ L : ℝ)
    (propose : S → E → E → S × E) (initialState : S) (initial : E)
    (hf : SmoothObjective f L) (hL : 0 < L) (hσ : 0 < σ) (hσ' : σ ≤ 1) (t : ℕ) :
    f (safeguardedRun σ L f propose initialState initial (t + 1)).2 ≤
      f (safeguardedRun σ L f propose initialState initial t).2 -
        σ ^ 2 / (2 * L) *
          ‖gradient f (safeguardedRun σ L f propose initialState initial t).2‖ ^ 2 := by
  exact safeguardedStep_descent f σ L _ _ hf hL hσ hσ'

/-- The full-run descent assumptions are satisfiable,
arXiv:2502.16982, §2.1, and arXiv:2602.02016v2, §2–4, training extension. -/
example : SmoothObjective quadratic 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 :=
  ⟨quadratic_smooth, by norm_num, by norm_num, by norm_num⟩

/-- Telescoping the actual loss decreases controls every finite sum of
squared full gradients, independently of momentum and solver errors.
Source: training correction to arXiv:2502.16982, §2.1–2.2, and
arXiv:2602.02016v2, §2–4. -/
theorem safeguardedRun_gradient_budget (f : E → ℝ) (σ L : ℝ)
    (propose : S → E → E → S × E) (initialState : S) (initial : E)
    (hf : SmoothObjective f L) (hL : 0 < L) (hσ : 0 < σ) (hσ' : σ ≤ 1) (T : ℕ) :
    σ ^ 2 / (2 * L) * ∑ t ∈ Finset.range T,
        ‖gradient f (safeguardedRun σ L f propose initialState initial t).2‖ ^ 2 ≤
      f initial - f (safeguardedRun σ L f propose initialState initial T).2 := by
  induction T with
  | zero => simp [safeguardedRun]
  | succ T ih =>
    rw [Finset.sum_range_succ, mul_add]
    have hstep := safeguardedRun_descent f σ L propose initialState initial hf hL hσ hσ' T
    linarith

/-- The finite-budget assumptions are satisfiable,
arXiv:2502.16982, §2.1, and arXiv:2602.02016v2, §2–4, training extension. -/
example : SmoothObjective quadratic 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 :=
  ⟨quadratic_smooth, by norm_num, by norm_num, by norm_num⟩

end Transformer.Optimization

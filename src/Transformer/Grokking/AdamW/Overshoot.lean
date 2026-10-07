import Transformer.Grokking.AdamW.FiniteThreshold

/-!
# Centered overshoot and the absence of a uniform first-step rate

Source comparison: Liu et al., arXiv:2205.10343v2, section 3.2,
eq:l_eff, versus native PyTorch 2.14.1 AdamW at lab commit 43d4d66.
The effective source assumes Euclidean flow. Neither a negative native
learning-rate derivative nor scale invariance of the objective supplies
a scale-independent positive finite rate.

For every fixed positive rate and epsilon, and nonnegative decay, an
explicit positive centered scale makes the actual first step increase
the same initial loss 3/2. A concrete example retains experimental
betas, epsilon, decay and rate. Its infinitesimal derivative is negative.
Both the initial and updated family remain centered and nonzero, so
neither a singular loss nor lack of centering explains the overshoot.
These are exact-real effective-model counterexamples, not a measured
failure of GPTMini or an all-time assertion about momentum dynamics.
-/

namespace Transformer.Grokking.AdamW

open Transformer.Grokking.EffectiveTheory

/-- The exact positive-loss-change criterion complements the verified
descent threshold. Source: arXiv:2205.10343v2, eq:l_eff, evaluated under
native first-step AdamW at 43d4d66; no decay sign is needed here. -/
theorem centered_family_finite_increase_iff (b1 b2 eps decay eta t : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (ht : 0 < t) (heta : 0 < eta) :
    (normalizedLoss t (-t / 2) (-t / 2) <
      lossAfterFirstStep b1 b2 eps decay eta t (-t / 2) (-t / 2)) ↔
      3 * ((1 - eta * decay) * t) < 2 * (eta * familyDirection eps t) := by
  let L := (1 - eta * decay) * t
  let b := eta * familyDirection eps t
  have hb : 0 < b := by dsimp [b]; exact mul_pos heta (family_direction_pos eps t he ht)
  have hp : 0 < 3 * b := by positivity
  have hden : 0 < 3 * L ^ 2 / 2 + 2 * b ^ 2 := by positivity
  have heq := centered_family_loss_difference b1 b2 eps decay eta t h1 h2 he ht heta
  have hsub : (normalizedLoss t (-t / 2) (-t / 2) <
      lossAfterFirstStep b1 b2 eps decay eta t (-t / 2) (-t / 2)) ↔
      0 < lossAfterFirstStep b1 b2 eps decay eta t (-t / 2) (-t / 2) -
        normalizedLoss t (-t / 2) (-t / 2) := by constructor <;> intro h <;> linarith
  rw [hsub, heq, lt_div_iff₀ hden]
  rw [MulZeroClass.zero_mul]
  change 0 < 3 * b * (2 * b - 3 * L) ↔ 3 * L < 2 * b
  rw [mul_pos_iff]
  constructor
  · rintro (⟨_, h⟩ | ⟨h, _⟩) <;> linarith
  · intro h
    exact Or.inl ⟨hp, by linarith⟩

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 / 100000000 : ℝ) ∧ 0 < (1 / 1000000 : ℝ) ∧
    0 < (1 / 1000 : ℝ) := by norm_num

/-- Every fixed positive rate has a centered state on which the actual
first update raises the loss. Source: counterexample to transferring
arXiv:2205.10343v2's effective-flow decrease to a uniform native AdamW
finite rate at 43d4d66. The initial loss remains 3/2 at every chosen scale. -/
theorem every_positive_rate_has_centered_overshoot (b1 b2 eps decay eta : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (hd : 0 ≤ decay) (heta : 0 < eta) :
    ∃ t : ℝ, 0 < t ∧ normalizedLoss t (-t / 2) (-t / 2) = 3 / 2 ∧
      normalizedLoss t (-t / 2) (-t / 2) <
        lossAfterFirstStep b1 b2 eps decay eta t (-t / 2) (-t / 2) := by
  let t := eta / (4 + eps * eta)
  have hden : 0 < 4 + eps * eta := by positivity
  have ht : 0 < t := by dsimp [t]; exact div_pos heta hden
  have hid : (4 + eps * eta) * t = eta := by dsimp [t]; field_simp
  have hprod : 0 < eps * eta * t := by positivity
  have hsmall : 4 * t < eta := by nlinarith
  have heps : eps * t < 1 := by
    apply (mul_lt_mul_iff_right₀ heta).mp
    nlinarith
  have hA : (3 / 4 : ℝ) < familyDirection eps t := by
    unfold familyDirection
    apply (lt_div_iff₀ (by positivity : 0 < 3 + eps * t)).mpr
    nlinarith
  refine ⟨t, ht, (centered_family_loss t ht).2, ?_⟩
  apply (centered_family_finite_increase_iff b1 b2 eps decay eta t h1 h2 he ht heta).mpr
  have hdecay : 0 ≤ eta * decay * t := by positivity
  have hright : (3 / 2 : ℝ) * eta < 2 * (eta * familyDirection eps t) := by
    nlinarith
  nlinarith

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 / 100000000 : ℝ) ∧ 0 ≤ (1 / 10 : ℝ) ∧
    0 < (1 / 1000 : ℝ) := by norm_num

/-- A fixed rate cannot guarantee nonincrease on every positive centered
scale. Source: the proposed unrestricted finite-update transfer of
arXiv:2205.10343v2's effective-flow loss descent to AdamW at 43d4d66. -/
theorem no_uniform_positive_rate (b1 b2 eps decay : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (hd : 0 ≤ decay) :
    ¬∃ eta : ℝ, 0 < eta ∧ ∀ t : ℝ, 0 < t →
      lossAfterFirstStep b1 b2 eps decay eta t (-t / 2) (-t / 2) ≤
        normalizedLoss t (-t / 2) (-t / 2) := by
  rintro ⟨eta, heta, hrate⟩
  obtain ⟨t, ht, _, hworse⟩ :=
    every_positive_rate_has_centered_overshoot b1 b2 eps decay eta h1 h2 he hd heta
  have hbetter := hrate t ht
  linarith

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 : ℝ) ∧ 0 ≤ (0 : ℝ) := by norm_num

/-- Every positive family scale has strictly negative local derivative,
including the states chosen by the universal overshoot construction.
Source: arXiv:2205.10343v2, eq:l_eff, under native AdamW at 43d4d66;
the nonstationary gradient is derived from the actual family. -/
theorem centered_family_first_loss_derivative_negative (b1 b2 eps decay t : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (ht : 0 < t) :
    ∃ a : ℝ, a < 0 ∧
      HasDerivAt (fun eta => lossAfterFirstStep b1 b2 eps decay eta t (-t / 2) (-t / 2)) a 0 := by
  have hg : (quotientGradient t (-t / 2) (-t / 2)).1 ≠ 0 ∨
      (quotientGradient t (-t / 2) (-t / 2)).2.1 ≠ 0 ∨
      (quotientGradient t (-t / 2) (-t / 2)).2.2 ≠ 0 := by
    rw [centered_family_gradient t ht]
    right
    right
    dsimp
    positivity
  obtain ⟨hd, hneg⟩ := first_step_loss_negative_derivative
    b1 b2 eps decay t (-t / 2) (-t / 2) h1 h2 he (centered_family_loss t ht).1 hg
  exact ⟨_, hneg, hd⟩

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 / 100000000 : ℝ) ∧ 0 < (1 / 1000000 : ℝ) := by norm_num

/-- Concrete overshoot with all experimental optimizer hyperparameters
retained. Source: AdamW at 43d4d66, beta `(0.9, 0.98)`, epsilon `1e-8`,
decay `0.1`, rate `0.001`; the effective embedding scale is `1e-6`. -/
theorem finite_native_centered_overshoot :
    normalizedLoss (1 / 1000000) (-(1 / 1000000) / 2) (-(1 / 1000000) / 2) <
      lossAfterFirstStep (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000)
        (1 / 1000000) (-(1 / 1000000) / 2) (-(1 / 1000000) / 2) := by
  apply (centered_family_finite_increase_iff _ _ _ _ _ _
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)).mpr
  norm_num [familyDirection]

/-- The same finite overshoot occurs despite a strictly negative
learning-rate derivative at zero. Source: the limits of transferring
arXiv:2205.10343v2's effective-flow descent to native AdamW at 43d4d66;
this is a regular centered state, not a zero-norm quotient convention. -/
theorem finite_native_centered_step_reverses_local_descent :
    ∃ a : ℝ, a < 0 ∧
      HasDerivAt (fun eta => lossAfterFirstStep (9 / 10) (49 / 50)
        (1 / 100000000) (1 / 10) eta (1 / 1000000)
        (-(1 / 1000000) / 2) (-(1 / 1000000) / 2)) a 0 ∧
      normalizedLoss (1 / 1000000) (-(1 / 1000000) / 2) (-(1 / 1000000) / 2) <
        lossAfterFirstStep (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000)
          (1 / 1000000) (-(1 / 1000000) / 2) (-(1 / 1000000) / 2) := by
  have ht : 0 < (1 / 1000000 : ℝ) := by norm_num
  obtain ⟨hd, hneg⟩ := first_step_loss_negative_derivative
    (9 / 10) (49 / 50) (1 / 100000000) (1 / 10)
    (1 / 1000000) (-(1 / 1000000) / 2) (-(1 / 1000000) / 2)
    (by norm_num) (by norm_num) (by norm_num) (centered_family_loss _ ht).1
    (by rw [centered_family_gradient _ ht]; norm_num)
  exact ⟨_, hneg, hd, finite_native_centered_overshoot⟩

end Transformer.Grokking.AdamW

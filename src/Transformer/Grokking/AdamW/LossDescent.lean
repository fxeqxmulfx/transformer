import Transformer.Grokking.AdamW.LossDirection
import Mathlib.Analysis.Calculus.Deriv.Slope

/-!
# Local descent of the actual effective loss under native AdamW

Source: Liu et al., arXiv:2205.10343v2, section 3.2, eq:l_eff, and
PyTorch 2.14.1's first-step AdamW equations used at lab commit 43d4d66.
The existing one-constraint effective quotient and zero moment buffers
are retained. This is exact-real analysis of the finite native update;
it does not replace AdamW by a Euclidean gradient flow.

Positive epsilon makes its weighted gradient energy positive precisely
away from stationary points. The verified learning-rate derivative then
gives an actual positive interval of strictly improving finite steps.
The interval depends on the initial state and optimizer parameters.
No prescribed numerical rate, repeated-step convergence, rule selection
or identification with GPTMini cross-entropy is asserted.
-/

namespace Transformer.Grokking.AdamW

open Transformer.Grokking.EffectiveTheory
open Filter
open scoped Topology

/-- Every coordinate contribution to the first-step energy is
nonnegative with positive epsilon. Source: native first-step normalization
at 43d4d66, PyTorch 2.14.1 AdamW equations. -/
theorem first_gradient_energy_nonneg (eps : ℝ) (g : ℝ × ℝ × ℝ)
    (he : 0 < eps) : 0 ≤ firstGradientEnergy eps g := by
  unfold firstGradientEnergy
  positivity

example : 0 < (1 / 100000000 : ℝ) := by norm_num

/-- A nonzero coordinate gives strictly positive weighted energy.
Source: the native bias-corrected first-step equation at 43d4d66;
the nonstationary premise is required for strict descent. -/
theorem first_gradient_energy_pos (eps : ℝ) (g : ℝ × ℝ × ℝ)
    (he : 0 < eps) (hg : g.1 ≠ 0 ∨ g.2.1 ≠ 0 ∨ g.2.2 ≠ 0) :
    0 < firstGradientEnergy eps g := by
  rcases hg with h | h | h
  all_goals
    unfold firstGradientEnergy
    positivity

example : 0 < (1 / 100000000 : ℝ) ∧
    ((1 : ℝ) ≠ 0 ∨ (-1 : ℝ) ≠ 0 ∨ (0 : ℝ) ≠ 0) := by norm_num

/-- Zero weighted energy characterizes actual stationary gradients;
it is not an assumed loss plateau. Source: PyTorch 2.14.1 first-step
normalization and the effective quotient of arXiv:2205.10343v2, eq:l_eff. -/
theorem first_gradient_energy_eq_zero_iff (eps : ℝ) (g : ℝ × ℝ × ℝ)
    (he : 0 < eps) :
    firstGradientEnergy eps g = 0 ↔ g.1 = 0 ∧ g.2.1 = 0 ∧ g.2.2 = 0 := by
  constructor
  · intro hz
    by_contra hn
    have hg : g.1 ≠ 0 ∨ g.2.1 ≠ 0 ∨ g.2.2 ≠ 0 := by tauto
    have hp := first_gradient_energy_pos eps g he hg
    linarith
  · rintro ⟨h0, h1, h2⟩
    simp [firstGradientEnergy, h0, h1, h2]

example : 0 < (1 : ℝ) := by norm_num

/-- A strictly negative derivative at zero implies improvement for all
sufficiently small positive arguments. Source: the one-sided derivative
limit in Mathlib.Analysis.Calculus.Deriv.Slope, used to bridge the
effective-loss derivative in arXiv:2205.10343v2 to a finite learning rate. -/
theorem locally_decreases_of_negative_derivative (f : ℝ → ℝ) (a : ℝ)
    (hd : HasDerivAt f a 0) (ha : a < 0) :
    ∃ delta : ℝ, 0 < delta ∧ ∀ eta : ℝ, 0 < eta → eta < delta → f eta < f 0 := by
  have hlim := hd.tendsto_slope_zero_right
  have hneg := Filter.Tendsto.eventually_lt_const ha hlim
  have hev := eventually_nhdsWithin_iff.mp hneg
  obtain ⟨delta, hdelta, hs⟩ := Metric.eventually_nhds_iff.mp hev
  refine ⟨delta, hdelta, ?_⟩
  intro eta heta hsmall
  have hdist : dist eta 0 < delta := by
    simpa [Real.dist_eq, abs_of_pos heta] using hsmall
  have hquot : (f eta - f 0) / eta < 0 := by
    simpa [smul_eq_mul, div_eq_inv_mul] using hs hdist heta
  rcases div_neg_iff.mp hquot with h | h
  · linarith
  · linarith

example : HasDerivAt (fun eta : ℝ => -eta) (-1) 0 ∧ (-1 : ℝ) < 0 := by
  constructor
  · simpa using (hasDerivAt_id (0 : ℝ)).mul_const (-1)
  · norm_num

/-- The actual first-update loss has a strictly negative learning-rate
derivative at every nonstationary nonzero representation. Source:
arXiv:2205.10343v2, eq:l_eff, and native AdamW at 43d4d66.
No sign condition on decay is needed for this derivative at zero. -/
theorem first_step_loss_negative_derivative (b1 b2 eps decay x y z : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps)
    (hZ : squaredNorm x y z ≠ 0)
    (hg : (quotientGradient x y z).1 ≠ 0 ∨
      (quotientGradient x y z).2.1 ≠ 0 ∨ (quotientGradient x y z).2.2 ≠ 0) :
    HasDerivAt (fun eta => lossAfterFirstStep b1 b2 eps decay eta x y z)
      (-firstGradientEnergy eps (quotientGradient x y z)) 0 ∧
      -firstGradientEnergy eps (quotientGradient x y z) < 0 := by
  refine ⟨first_step_loss_deriv_eq_energy b1 b2 eps decay x y z h1 h2 hZ, ?_⟩
  have hp := first_gradient_energy_pos eps (quotientGradient x y z) he hg
  linarith

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 / 100000000 : ℝ) ∧ squaredNorm 1 2 0 ≠ 0 ∧
    ((quotientGradient 1 2 0).1 ≠ 0 ∨
      (quotientGradient 1 2 0).2.1 ≠ 0 ∨ (quotientGradient 1 2 0).2.2 ≠ 0) := by
  have hZ : squaredNorm 1 2 0 ≠ 0 := by norm_num [squaredNorm]
  rw [quotientGradient_eq 1 2 0 hZ]
  norm_num [numerator, EffectiveTheory.residual, squaredNorm]

/-- The finite native first update strictly lowers the effective loss
on an initial-state-dependent positive learning-rate interval. Source:
arXiv:2205.10343v2, section 3.2, eq:l_eff, and PyTorch 2.14.1 at
43d4d66. The source assumes a gradient flow; this is a separately proved
first-update consequence, not its conservation or all-time convergence. -/
theorem first_step_locally_lowers_effective_loss (b1 b2 eps decay x y z : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps)
    (hZ : squaredNorm x y z ≠ 0)
    (hg : (quotientGradient x y z).1 ≠ 0 ∨
      (quotientGradient x y z).2.1 ≠ 0 ∨ (quotientGradient x y z).2.2 ≠ 0) :
    ∃ delta : ℝ, 0 < delta ∧ ∀ eta : ℝ, 0 < eta → eta < delta →
      lossAfterFirstStep b1 b2 eps decay eta x y z < normalizedLoss x y z := by
  obtain ⟨hd, hneg⟩ := first_step_loss_negative_derivative
    b1 b2 eps decay x y z h1 h2 he hZ hg
  obtain ⟨delta, hp, hl⟩ := locally_decreases_of_negative_derivative _ _ hd hneg
  refine ⟨delta, hp, ?_⟩
  intro eta heta hsmall
  simpa [loss_after_zero_rate] using hl eta heta hsmall

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧ 0 < (1 : ℝ) ∧
    squaredNorm 1 2 0 ≠ 0 ∧ ((quotientGradient 1 2 0).1 ≠ 0 ∨
      (quotientGradient 1 2 0).2.1 ≠ 0 ∨ (quotientGradient 1 2 0).2.2 ≠ 0) := by
  have hZ : squaredNorm 1 2 0 ≠ 0 := by norm_num [squaredNorm]
  rw [quotientGradient_eq 1 2 0 hZ]
  norm_num [numerator, EffectiveTheory.residual, squaredNorm]

/-- The actual experimental hyperparameters instantiate the existence
result, without asserting that their fixed learning rate lies in the interval. -/
example : ∃ delta : ℝ, 0 < delta ∧ ∀ eta : ℝ, 0 < eta → eta < delta →
    lossAfterFirstStep (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) eta 1 2 0 <
      normalizedLoss 1 2 0 := by
  have hZ : squaredNorm 1 2 0 ≠ 0 := by norm_num [squaredNorm]
  apply first_step_locally_lowers_effective_loss _ _ _ _ _ _ _
    (by norm_num) (by norm_num) (by norm_num) hZ
  rw [quotientGradient_eq 1 2 0 hZ]
  norm_num [numerator, EffectiveTheory.residual, squaredNorm]

end Transformer.Grokking.AdamW

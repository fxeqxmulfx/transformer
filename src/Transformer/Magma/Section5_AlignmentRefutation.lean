/-
# The momentum-alignment lower bound is false as stated

Formalization and refutation of arXiv:2602.15322v1, Section 5,
sup_lem:descent_lower_bound, eq:lb_descent, and Appendix A.3.
The proof's event measures alignment with the true gradient, whereas the
algorithm measures alignment with momentum and then smooths the score.
Even an exact unbiased gradient with zero noise does not repair this gap.
-/

import Transformer.Magma.Section3_Algorithm
import Transformer.Optimization.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic

noncomputable section

namespace Transformer.Magma

open Transformer.Optimization

/-- Exact sigmoid endpoints for a positive admissible temperature.
Source: arXiv:2602.15322v1, Section 3, eq:masking_prob; Appendix A.3. -/
theorem sigmoid_log_three :
    Real.sigmoid (Real.log 3) = 3 / 4 ∧ Real.sigmoid (-Real.log 3) = 1 / 4 := by
  norm_num [Real.sigmoid, Real.exp_neg, Real.exp_log (by norm_num : (0 : ℝ) < 3)]

/-- The counterexample temperature is strictly positive.
Source: arXiv:2602.15322v1, Section 3, tau>0. -/
theorem alignment_counterexample_temperature : (0 : ℝ) < 1 / Real.log 3 := by
  exact div_pos one_pos (Real.log_pos (by norm_num))

/-- An admissible dense EMA produces opposing current momentum from
old=-11/9, beta=0.9, and the exact current gradient g=1. Source:
arXiv:2602.15322v1, Section 2, first-moment estimates, and Section 3. -/
theorem opposing_dense_momentum :
    (9 / 10 : ℝ) * (-11 / 9) + (1 - 9 / 10) * 1 = -1 := by norm_num

/-- Opposing momentum gives damping 1/4 at the chosen temperature and
previous scale 1/4. Both are admissible sigmoid/EMA values.
Source: arXiv:2602.15322v1, Section 3, eq:masking_prob and Algorithm 1. -/
theorem opposing_momentum_damping :
    damping (1 / Real.log 3) (1 / 4) (-1 : ℝ) 1 = 1 / 4 := by
  norm_num [damping, cosine, div_div, Real.sigmoid,
    Real.exp_log (by norm_num : (0 : ℝ) < 3)]

/-- Counterexample to the paper's claimed descent-efficiency bound, including
its squared-alpha version at the current point. Take a single block,
objective x^2/2, theta_t=1, exact g=1 and sigma=0. The proof's true-gradient event
at gamma=1 has probability e=1, hence alpha=sigmoid(1/tau)=3/4 and c=0.
Actual momentum-based Magma damping is 1/4. Thus the second-moment condition
holds exactly, but 1/4 < (3/4)^2, refuting the intended current-gradient
version of eq:lb_descent. The source writes theta_T instead of theta_t;
correcting that index and adding unbiasedness do not suffice. This is
a pointwise counterexample, not an assertion about a terminal iterate.
Source: arXiv:2602.15322v1, Section 5 and Appendix A.3. -/
theorem paper_alignment_counterexample :
    let τ := 1 / Real.log 3
    let a := gradient quadratic (1 : ℝ)
    (1 : ℝ) ^ 2 ≤ a ^ 2 + 0 ^ 2 ∧
      damping τ (1 / 4) (-1 : ℝ) 1 * 1 * a <
        (Real.sigmoid (1 / τ) * a) ^ 2 - 0 := by
  dsimp only
  rw [quadratic_gradient, opposing_momentum_damping]
  norm_num [div_div, Real.sigmoid, Real.exp_neg,
    Real.exp_log (by norm_num : (0 : ℝ) < 3)]

end Transformer.Magma

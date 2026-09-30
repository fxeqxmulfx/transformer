/-
# DASH — EVD dampening and correction of the double-shift description

arXiv:2602.02016v2, Appendix B. Every eigenvalue receives the
same shift, although the shift changes with the smallest eigenvalue.
-/

import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic

noncomputable section

namespace Transformer.DASH

/-- The stated EVD regularization from Appendix B's four-step algorithm,
arXiv:2602.02016v2: `λ_i-min(λ_min,0)+ε`. -/
def dampen (value minimum ε : ℝ) : ℝ := value - min minimum 0 + ε

/-- Applying that algorithm after `eigh(L+εI)` without subtracting `ε`,
arXiv:2602.02016v2, Appendix B, the double-regularization observation. -/
def doubleDampen (value minimum ε : ℝ) : ℝ :=
  dampen (value + ε) (minimum + ε) ε

/-- The intended regularization makes every eigenvalue at least `ε`.
Source: arXiv:2602.02016v2, Appendix B, “Regularization (dampening)”. -/
theorem dampen_lower_bound (value minimum ε : ℝ) (hmin : minimum ≤ value) :
    ε ≤ dampen value minimum ε := by
  unfold dampen
  linarith [min_le_left minimum 0]

/-- The minimum bound is satisfiable, arXiv:2602.02016v2, Appendix B. -/
example : (-1 : ℝ) ≤ 2 := by norm_num

/-- A positive regularizer gives strictly positive regularized eigenvalues.
Source: arXiv:2602.02016v2, Appendix B, the inverse-power requirement. -/
theorem dampen_pos (value minimum ε : ℝ) (hmin : minimum ≤ value) (hε : 0 < ε) :
    0 < dampen value minimum ε := hε.trans_le (dampen_lower_bound value minimum ε hmin)

/-- Positivity hypotheses are satisfiable, arXiv:2602.02016v2, Appendix B. -/
example : (-1 : ℝ) ≤ 2 ∧ (0 : ℝ) < 1 := by norm_num

/-- The double-regularized implementation adds the uniform shift
`2ε-min(λ_min+ε,0)` to every eigenvalue. The source's final summary instead
assigns different shifts to the minimum and the other eigenvalues.
Source: arXiv:2602.02016v2, Appendix B, “Regularization (dampening)”. -/
theorem doubleDampen_uniform_shift (value minimum ε : ℝ) :
    doubleDampen value minimum ε - value = 2 * ε - min (minimum + ε) 0 := by
  unfold doubleDampen dampen
  ring

/-- If the shifted minimum is nonnegative, the extra shift is `2ε` for
every eigenvalue, not only for the smallest one.
Source: arXiv:2602.02016v2, Appendix B, the positive-minimum example. -/
theorem doubleDampen_nonnegative_minimum (value minimum ε : ℝ) (hmin : 0 ≤ minimum + ε) :
    doubleDampen value minimum ε = value + 2 * ε := by
  simp only [doubleDampen, dampen, min_eq_right hmin]
  ring

/-- The nonnegative branch is inhabited, arXiv:2602.02016v2, Appendix B. -/
example : (0 : ℝ) ≤ 1 + 1 := by norm_num

/-- If the shifted minimum is negative, the uniform shift is `ε-λ_min`,
where `λ_min` here denotes the original, unshifted eigenvalue.
Source: arXiv:2602.02016v2, Appendix B, the negative-minimum branch. -/
theorem doubleDampen_negative_minimum (value minimum ε : ℝ) (hmin : minimum + ε ≤ 0) :
    doubleDampen value minimum ε = value - minimum + ε := by
  simp only [doubleDampen, dampen, min_eq_left hmin]
  ring

/-- The negative branch is inhabited, arXiv:2602.02016v2, Appendix B. -/
example : (-2 : ℝ) + 1 ≤ 0 := by norm_num

/-- Concrete refutation of the source's unequal-shift summary: original
eigenvalues 1 and 2 both increase by 2 when `ε=1`.
Source: arXiv:2602.02016v2, Appendix B, the paragraph beginning “In summary”. -/
theorem unequal_shift_summary_counterexample :
    doubleDampen 1 1 1 = 3 ∧ doubleDampen 2 1 1 = 4 ∧
      doubleDampen 2 1 1 ≠ (2 : ℝ) + (1 - 1) := by
  norm_num [doubleDampen, dampen]

/-- Subtracting the initial regularizer recovers the unshifted spectrum.
Source: arXiv:2602.02016v2, Appendix B, “corrected spectrum”. -/
theorem corrected_spectrum (value ε : ℝ) : (value + ε) - ε = value := by ring

/-- Absolute value gives nonnegative eigenvalues, not strict positivity.
Source: arXiv:2602.02016v2, Appendix B, “ABS-based heuristic”. -/
theorem abs_spectrum_nonneg (value : ℝ) : 0 ≤ |value| := abs_nonneg value

/-- A zero eigenvalue refutes strict positivity from ABS alone.
Source: arXiv:2602.02016v2, Appendix B, “make sure all eigenvalues are positive”. -/
theorem abs_strict_positivity_counterexample : |(0 : ℝ)| = 0 ∧ ¬ 0 < |(0 : ℝ)| := by
  norm_num

/-- ABS followed by positive `ε` does give a positive lower bound.
Source: arXiv:2602.02016v2, Appendix B, the optional ABS-ADD regularization. -/
theorem abs_add_lower_bound (value ε : ℝ) (hε : 0 < ε) :
    ε ≤ |value| + ε ∧ 0 < |value| + ε := by
  constructor <;> linarith [abs_nonneg value]

/-- ABS-ADD hypotheses are satisfiable, arXiv:2602.02016v2, Appendix B. -/
example : (0 : ℝ) < 1 := by norm_num

end Transformer.DASH

/-
# Smooth positive gains

arXiv:2606.25971v2, §4.1.3, the displayed softplus map; Appendix A,
Algorithm 2, backpropagation through the gain map.
-/

import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Tactic

noncomputable section

namespace Transformer.MagnitudeDirection

/-- The effective softplus gain, arXiv:2606.25971v2, §4.1.3. -/
def softplus (x : ℝ) : ℝ := Real.log (1 + Real.exp x)

/-- Derivative of the softplus map used for raw-gain backpropagation,
arXiv:2606.25971v2, Appendix A, Algorithm 2, line 5. -/
def softplusDerivative (x : ℝ) : ℝ := Real.exp x / (1 + Real.exp x)

/-- Softplus keeps every effective gain strictly positive, arXiv:2606.25971v2,
§4.1.3. Consequently division by gains is valid at every finite raw gain. -/
theorem softplus_pos (x : ℝ) : 0 < softplus x := by
  apply Real.log_pos
  linarith [Real.exp_pos x]

/-- Actual derivative of the printed gain map, arXiv:2606.25971v2, §4.1.3
and Appendix A, Algorithm 2, line 5. -/
theorem softplus_hasDerivAt (x : ℝ) :
    HasDerivAt softplus (softplusDerivative x) x := by
  change HasDerivAt (fun y => Real.log (1 + Real.exp y))
    (Real.exp x / (1 + Real.exp x)) x
  simpa using
    ((Real.hasDerivAt_exp x).const_add 1).log (by positivity : 1 + Real.exp x ≠ 0)

/-- The smooth gain derivative lies strictly between zero and one,
arXiv:2606.25971v2, §4.1.3 and Appendix A. -/
theorem softplusDerivative_bounds (x : ℝ) :
    0 < softplusDerivative x ∧ softplusDerivative x < 1 := by
  unfold softplusDerivative
  have hp := Real.exp_pos x
  constructor
  · positivity
  · exact (div_lt_one (by positivity)).mpr (by linarith)

/-- Every positive gain has a finite raw softplus parameter, arXiv:2606.25971v2,
§3.1 and §4.1.3. The inverse is `log(exp(gamma)-1)`. -/
theorem softplus_inverse (gamma : ℝ) (hg : 0 < gamma) :
    softplus (Real.log (Real.exp gamma - 1)) = gamma := by
  have hp : 0 < Real.exp gamma - 1 := by
    linarith [Real.one_lt_exp_iff.mpr hg]
  rw [softplus, Real.exp_log hp]
  convert Real.log_exp gamma using 1
  congr 1
  ring

/-- The inverse's domain is inhabited, arXiv:2606.25971v2, §4.1.3. -/
example : (0 : ℝ) < 1 := by norm_num

/-- Raw initialization for an effective gain of one, arXiv:2606.25971v2,
§4.1.3, “All of them are initialized at 1”. -/
theorem softplus_unit_initialization :
    softplus (Real.log (Real.exp 1 - 1)) = 1 :=
  softplus_inverse 1 (by norm_num)

/-- Exponential parametrization is positive and has the stated derivative,
arXiv:2606.25971v2, §4.1.3, `gamma = exp(raw)`. -/
theorem exponential_gain (x : ℝ) :
    0 < Real.exp x ∧ HasDerivAt Real.exp (Real.exp x) x :=
  ⟨Real.exp_pos x, Real.hasDerivAt_exp x⟩

end Transformer.MagnitudeDirection

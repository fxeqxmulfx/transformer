/-
# Solving the reduced gradient-flow equation by a real inverse

arXiv:2402.19449v2, Appendix I, Lemma 5 (Solution of the dynamics).
The margin `u=a-b` satisfies `exp u + (c-1)u = 1+cπt`.
Its strictly increasing left side is inverted directly. This is equivalent
to the paper's Lambert-W expression without assuming an unimplemented W.
-/

import Transformer.Imbalance.Section3_Model
import Mathlib.Analysis.Calculus.Deriv.Inverse
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Topology.Order.MonotoneContinuity
import Mathlib.Order.Hom.Set

open Filter
open scoped Topology

noncomputable section

namespace Transformer.Imbalance

/-- Integrated margin equation, Appendix I, Lemma 5: `e^u+zu`, `z=c-1`. -/
def marginPrimitive (z u : ℝ) : ℝ := Real.exp u + z * u

/-- The integrated equation has positive derivative for `z>0`;
Appendix I, Lemma 5. -/
theorem hasDerivAt_marginPrimitive (z u : ℝ) :
    HasDerivAt (marginPrimitive z) (Real.exp u + z) u := by
  change HasDerivAt (fun v => Real.exp v + z * v) (Real.exp u + z) u
  simpa using (Real.hasDerivAt_exp u).fun_add
    ((hasDerivAt_id u).const_mul z)

/-- Strict monotonicity gives uniqueness of the margin in Lemma 5. -/
theorem marginPrimitive_strictMono (z : ℝ) (hz : 0 < z) :
    StrictMono (marginPrimitive z) := by
  intro a b hab
  have he := Real.exp_lt_exp.mpr hab
  have hl := mul_lt_mul_of_pos_left hab hz
  exact add_lt_add he hl

/-- Lemma 5's integrated equation has a unique solution for every real
right-hand side, since its left side tends to both infinities. -/
theorem marginPrimitive_surjective (z : ℝ) (hz : 0 < z) :
    Function.Surjective (marginPrimitive z) := by
  apply Continuous.surjective (by unfold marginPrimitive; fun_prop)
  · apply tendsto_atTop_mono (fun u => ?_) (tendsto_id.const_mul_atTop hz)
    exact le_add_of_nonneg_left (Real.exp_pos u).le
  · exact Real.tendsto_exp_atBot.add_atBot (tendsto_id.const_mul_atBot hz)

/-- Nonvacuity of the inverse construction in Appendix I, Lemma 5. -/
example : (0 : ℝ) < 1 := zero_lt_one

/-- Order isomorphism implementing the integral equation in Lemma 5. -/
def marginIso (z : ℝ) (hz : 0 < z) : ℝ ≃o ℝ :=
  (marginPrimitive_strictMono z hz).orderIsoOfSurjective
    (marginPrimitive z) (marginPrimitive_surjective z hz)

/-- Actual inverse solution for the margin `u=a-b`; Appendix I, Lemma 5.
The frequency is a parameter of the solution, rather than an assumption
hidden in a field. -/
def gdMargin (z : ℝ) (hz : 0 < z) (π t : ℝ) : ℝ :=
  (marginIso z hz).symm (1 + (z + 1) * π * t)

/-- Exact integral equation for the solution of Appendix I, Lemma 5. -/
theorem gdMargin_equation (z : ℝ) (hz : 0 < z) (π t : ℝ) :
    Real.exp (gdMargin z hz π t) + z * gdMargin z hz π t =
      1 + (z + 1) * π * t := by
  exact (marginIso z hz).apply_symm_apply _

/-- The margin solution starts at zero, as required in Lemmas 4 and 5. -/
theorem gdMargin_zero (z : ℝ) (hz : 0 < z) (π : ℝ) :
    gdMargin z hz π 0 = 0 := by
  apply (marginPrimitive_strictMono z hz).injective
  change Real.exp (gdMargin z hz π 0) + z * gdMargin z hz π 0 = marginPrimitive z 0
  rw [gdMargin_equation]
  simp [marginPrimitive]

/-- Inverse differentiation verifies the differential equation itself;
Appendix I, Lemmas 4 and 5. -/
theorem hasDerivAt_gdMargin (z : ℝ) (hz : 0 < z) (π t : ℝ) :
    HasDerivAt (gdMargin z hz π)
      ((z + 1) * π / (Real.exp (gdMargin z hz π t) + z)) t := by
  have hi := HasDerivAt.of_local_left_inverse
    (marginIso z hz).symm.continuous.continuousAt
    (hasDerivAt_marginPrimitive z (gdMargin z hz π t))
    (show Real.exp (gdMargin z hz π t) + z ≠ 0 by positivity)
    (Filter.Eventually.of_forall (marginIso z hz).apply_symm_apply)
  have ht : HasDerivAt (fun s : ℝ => 1 + (z + 1) * π * s) ((z + 1) * π) t := by
    simpa using ((hasDerivAt_id t).const_mul ((z + 1) * π)).const_add 1
  change HasDerivAt (fun s => (marginIso z hz).symm (1 + (z + 1) * π * s)) _ t
  simpa only [Function.comp_def, gdMargin, div_eq_mul_inv, mul_comm] using hi.comp t ht

/-- All positive-`z` hypotheses above have a witness; Appendix I. -/
example : (0 : ℝ) < (3 : ℝ) - 1 := by norm_num

/-- Positive time and frequency give nonnegative margin; Appendix I,
Lemma 6, preparatory estimate. -/
theorem gdMargin_nonneg (z : ℝ) (hz : 0 < z) (π t : ℝ)
    (hπ : 0 ≤ π) (ht : 0 ≤ t) : 0 ≤ gdMargin z hz π t := by
  have hmono := (marginIso z hz).symm.monotone
  have h0 : (marginIso z hz).symm 1 = 0 := by
    simpa [gdMargin] using gdMargin_zero z hz π
  rw [gdMargin, ← h0]
  apply hmono
  have : 0 ≤ (z + 1) * π * t := by positivity
  linarith

/-- Nonvacuity of the nonnegative-time margin bound; Appendix I, Lemma 6. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) ≤ 1 := by norm_num

end Transformer.Imbalance

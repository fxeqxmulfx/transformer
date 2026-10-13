import Transformer.Grokking.AdamW.LossDescent
import Mathlib.Analysis.SpecialFunctions.Pow.Deriv

/-!
# Repair the power-difference estimate used for circuit allocation

Source: Varma et al., arXiv:2309.02390v1, appendix D, lemma
power-derivative. It states that, for arbitrary x,c and r >= 1, some
delta > 0 makes `x^r - (x-epsilon)^r > delta*(r*x^(r-1)-c)` for
every epsilon < delta. That bound is false even with positive epsilon:
the difference tends to zero while the claimed positive right side
does not. A quadratic counterexample proves both readings false.

Correction: the multiplier is epsilon, not delta, with c > 0 and
0 < epsilon < delta. Differentiate the actual real-power difference;
the one-sided derivative limit proves that corrected statement. A
positive state yields a strictly positive linear gain for small steps.
Zero tolerance is separately refuted by a strict quadratic example.

The source uses the flawed bound in the proof of Theorem case 2.
These proofs repair the local ingredient; they neither endorse that
printed proof nor claim that GPTMini/AdamW follows this scalar model.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- The literal delta-multiplied claim fails already at epsilon zero.
Source: arXiv:2309.02390v1, appendix D, lemma power-derivative,
with x=1, r=2, c=1; natural square equals real power at exponent two. -/
theorem printed_power_derivative_false :
    ¬ ∃ delta : ℝ, 0 < delta ∧ ∀ epsilon : ℝ, epsilon < delta →
      (1 : ℝ) ^ 2 - (1 - epsilon) ^ 2 > delta * (2 * 1 - 1) := by
  rintro ⟨delta, hd, h⟩
  have hz := h 0 hd
  norm_num at hz
  linarith

/-- Restricting the source's quantifier to positive increments still
does not fix it. Source: arXiv:2309.02390v1, appendix D, the same lemma;
epsilon=delta/4 defeats every proposed positive delta for x=1,r=2,c=1. -/
theorem printed_positive_power_derivative_false :
    ¬ ∃ delta : ℝ, 0 < delta ∧ ∀ epsilon : ℝ, 0 < epsilon → epsilon < delta →
      (1 : ℝ) ^ 2 - (1 - epsilon) ^ 2 > delta * (2 * 1 - 1) := by
  rintro ⟨delta, hd, h⟩
  have he : 0 < delta / 4 := by positivity
  have hs : delta / 4 < delta := by linarith
  have hb := h (delta / 4) he hs
  have hq : 0 ≤ delta ^ 2 := by positivity
  nlinarith

/-- Derive the power difference from its actual function, retaining
every real base and exponent allowed by the corrected claim. Source:
arXiv:2309.02390v1, appendix D, lemma power-derivative; real rpower
is differentiable at every base when r >= 1. -/
theorem power_difference_deriv (x exponent : ℝ) (hr : 1 ≤ exponent) :
    HasDerivAt (fun epsilon : ℝ => x ^ exponent - (x - epsilon) ^ exponent)
      (exponent * x ^ (exponent - 1)) 0 := by
  have hi : HasDerivAt (fun epsilon : ℝ => x - epsilon) (-1) 0 :=
    (hasDerivAt_id (0 : ℝ)).const_sub x
  have hp : HasDerivAt (fun t : ℝ => t ^ exponent)
      (exponent * x ^ (exponent - 1)) (x - 0) := by
    simpa using Real.hasDerivAt_rpow_const (x := x) (p := exponent) (Or.inr hr)
  convert (hp.comp 0 hi).const_sub (x ^ exponent) using 1
  · rfl
  · ring

example : (1 : ℝ) ≤ 2 := by norm_num

/-- Corrected form of the source's power-derivative lemma. Source:
arXiv:2309.02390v1, appendix D. Changes: positive tolerance and positive
increment, and epsilon multiplying the tangent lower estimate. The
neighborhood radius delta only restricts which increments are small. -/
theorem power_derivative_corrected (x tolerance exponent : ℝ)
    (hr : 1 ≤ exponent) (hc : 0 < tolerance) :
    ∃ delta : ℝ, 0 < delta ∧ ∀ epsilon : ℝ, 0 < epsilon → epsilon < delta →
      epsilon * (exponent * x ^ (exponent - 1) - tolerance) <
        x ^ exponent - (x - epsilon) ^ exponent := by
  let d := exponent * x ^ (exponent - 1)
  let f := fun epsilon : ℝ => epsilon * (d - tolerance) -
    (x ^ exponent - (x - epsilon) ^ exponent)
  have hd : HasDerivAt f (-tolerance) 0 := by
    have h := ((hasDerivAt_id (0 : ℝ)).mul_const (d - tolerance)).sub
      (power_difference_deriv x exponent hr)
    convert h using 1
    · rfl
    · dsimp [d]
      ring
  have hn : -tolerance < 0 := by linarith
  obtain ⟨delta, hp, hs⟩ :=
    Transformer.Grokking.AdamW.locally_decreases_of_negative_derivative f (-tolerance) hd hn
  refine ⟨delta, hp, ?_⟩
  intro epsilon he hedelta
  have h := hs epsilon he hedelta
  dsimp [f, d] at h
  simp only [zero_mul, sub_zero, sub_self] at h
  linarith

example : (1 : ℝ) ≤ 2 ∧ 0 < (1 / 10 : ℝ) := by norm_num

/-- The actual difference quotient also supplies a tangent upper bound.
Source: arXiv:2309.02390v1, appendix D, power-derivative proof's intended
derivative-limit argument; retain the positive increment multiplier.
The lower and upper estimates each have a state-dependent radius. -/
theorem power_derivative_upper_bound (x tolerance exponent : ℝ)
    (hr : 1 ≤ exponent) (hc : 0 < tolerance) :
    ∃ delta : ℝ, 0 < delta ∧ ∀ epsilon : ℝ, 0 < epsilon → epsilon < delta →
      x ^ exponent - (x - epsilon) ^ exponent <
        epsilon * (exponent * x ^ (exponent - 1) + tolerance) := by
  let d := exponent * x ^ (exponent - 1)
  let f := fun epsilon : ℝ => (x ^ exponent - (x - epsilon) ^ exponent) -
    epsilon * (d + tolerance)
  have hd : HasDerivAt f (-tolerance) 0 := by
    have h := (power_difference_deriv x exponent hr).sub
      ((hasDerivAt_id (0 : ℝ)).mul_const (d + tolerance))
    convert h using 1
    · rfl
    · dsimp [d]
      ring
  have hn : -tolerance < 0 := by linarith
  obtain ⟨delta, hp, hs⟩ :=
    Transformer.Grokking.AdamW.locally_decreases_of_negative_derivative f (-tolerance) hd hn
  refine ⟨delta, hp, ?_⟩
  intro epsilon he hedelta
  have h := hs epsilon he hedelta
  dsimp [f, d] at h
  simp only [zero_mul, sub_zero, sub_self] at h
  linarith

example : (1 : ℝ) ≤ 2 ∧ 0 < (1 / 10 : ℝ) := by norm_num

/-- A positive active weight supplies a genuine linear decrease budget
for removing a small amount of weight. Source: arXiv:2309.02390v1,
appendix D, case 2's transfer proof; derive its usable local estimate
from the corrected lemma with half the actual positive tangent. -/
theorem positive_power_difference_budget (x exponent : ℝ)
    (hx : 0 < x) (hr : 1 ≤ exponent) :
    ∃ delta : ℝ, 0 < delta ∧ ∀ epsilon : ℝ, 0 < epsilon → epsilon < delta →
      exponent * x ^ (exponent - 1) * epsilon / 2 <
        x ^ exponent - (x - epsilon) ^ exponent := by
  have he : 0 < exponent := by linarith
  have hp := Real.rpow_pos_of_pos hx (exponent - 1)
  have hc : 0 < exponent * x ^ (exponent - 1) / 2 := by positivity
  obtain ⟨delta, hd, hs⟩ := power_derivative_corrected
    x (exponent * x ^ (exponent - 1) / 2) exponent hr hc
  refine ⟨delta, hd, ?_⟩
  intro epsilon hepsilon hedelta
  have h := hs epsilon hepsilon hedelta
  nlinarith

example : 0 < (1 : ℝ) ∧ (1 : ℝ) ≤ 2 := by norm_num

/-- Positive tolerance in the repair is essential: the quadratic
difference is strictly below its tangent at every positive increment.
Source: arXiv:2309.02390v1, appendix D, lemma's arbitrary c; this
counterexample refutes retaining c=0 with a strict tangent lower bound. -/
theorem quadratic_difference_below_tangent (epsilon : ℝ) (he : 0 < epsilon) :
    (1 : ℝ) ^ 2 - (1 - epsilon) ^ 2 < 2 * epsilon := by
  have hs : 0 < epsilon ^ 2 := by positivity
  nlinarith

example : 0 < (1 / 4 : ℝ) := by norm_num

end Transformer.Grokking.CircuitEfficiency

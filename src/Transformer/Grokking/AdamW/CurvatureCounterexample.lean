import Transformer.Grokking.AdamW.CurvatureBound
import Transformer.Grokking.Composition.Basic

/-!
# Initial CE slope and Hessian cannot certify a finite rate

Source comparison: Prieto et al., arXiv:2501.04697v1 §4.2, actual
softmax CE and parameter-update directions; Mathlib's Taylor/mean-value
interval bound, and the archived finite-step observations at 444b4ad.
Explicit deviation: one binary score with a quartic parameter dependence,
not GPTMini's learned forward map. Every member uses ordinary positive
finite-class CE and has the same initial loss, derivative and curvature.

For every prescribed positive rate a positive quartic coefficient
nevertheless makes that finite step increase CE. The initial Hessian is
exactly one-quarter and the initial slope minus one-half for every member;
the higher-order term changes the interval. This disproves transferring
an initial quadratic crossing rate to arbitrary CE curves. It is not a
grokking impossibility or a statement about the measured transformer.

The coefficient varies within the family when the proposed rate changes.
This refutes a uniform certificate based on identical initial data across
that family; it does not say that one fixed curve has no improving rate.
Controlling the interval remainder remains a sufficient remedy.
-/

namespace Transformer.Grokking.AdamW

open Transformer.Grokking.GradientEvidence Transformer.Grokking.Composition

/-- Concrete score with an independently varying higher-order term.
Source: the explicit Taylor diagnostic counterexample above, retaining
the ordinary binary CE used in arXiv:2501.04697v1 §4.2. -/
def quarticScore (coefficient t : ℝ) : ℝ := t - coefficient * t ^ 4

/-- Actual binary softmax CE of that stated score and fixed correct class.
Source: BinaryObjectives' verified finite loss; only the score family is
specialized. Neither its derivatives nor its finite decrease are defined. -/
noncomputable def quarticCELoss (coefficient t : ℝ) : ℝ :=
  binaryCE true (quarticScore coefficient t)

/-- The actual quartic score derivative. Source: polynomial calculus
of the explicitly stated parameter dependence in this diagnostic. -/
theorem quarticScore_deriv (coefficient t : ℝ) :
    HasDerivAt (quarticScore coefficient) (1 - 4 * coefficient * t ^ 3) t := by
  have h := (hasDerivAt_id t).sub (((hasDerivAt_id t).pow 4).const_mul coefficient)
  convert h using 1
  · rfl
  · dsimp
    ring

/-- Retain the exact positive binary CE expression, not a polynomial
surrogate loss. Source: arXiv:2501.04697v1 §4.2, ordinary softmax CE;
the actual target subtraction is already verified in BinaryObjectives. -/
theorem quarticCELoss_eq_exp (coefficient t : ℝ) :
    quarticCELoss coefficient t = Real.log (1 + Real.exp (-quarticScore coefficient t)) := by
  simpa only [quarticCELoss, coupledLoss, mul_one] using
    coupledLoss_eq_exp (quarticScore coefficient t) 1

/-- Derive the actual CE slope at every parameter state. Source:
arXiv:2501.04697v1 §4.2's loss, specialized to the explicit quartic score;
the nonlinear denominator is retained. -/
theorem quarticCELoss_deriv (coefficient t : ℝ) :
    HasDerivAt (quarticCELoss coefficient)
      (-(1 - 4 * coefficient * t ^ 3) / (Real.exp (quarticScore coefficient t) + 1)) t := by
  have h := (binaryCE_deriv true (quarticScore coefficient t)).comp t
    (quarticScore_deriv coefficient t)
  convert h using 1
  · rfl
  · dsimp
    have hd : Real.exp (quarticScore coefficient t) + 1 ≠ 0 := by positivity
    field_simp
    ring

/-- The first derivative function is computed from that actual CE.
Source: the verified derivative above, used rather than imposing a
convenient slope as a premise or assigning it in a definition. -/
theorem quarticCELoss_deriv_eq (coefficient t : ℝ) :
    deriv (quarticCELoss coefficient) t =
      -(1 - 4 * coefficient * t ^ 3) / (Real.exp (quarticScore coefficient t) + 1) :=
  (quarticCELoss_deriv coefficient t).deriv

/-- The actual initial second derivative is independent of the quartic
coefficient. Source: real CE differentiation of the explicit score family;
the initial Hessian therefore does not reveal its interval remainder. -/
theorem quarticCELoss_second_deriv_at_origin (coefficient : ℝ) :
    HasDerivAt (deriv (quarticCELoss coefficient)) (1 / 4) 0 := by
  have hn : HasDerivAt (fun t : ℝ => 4 * coefficient * t ^ 3 - 1) 0 0 := by
    convert ((((hasDerivAt_id (0 : ℝ)).pow 3).const_mul (4 * coefficient)).sub_const 1) using 1 <;>
      norm_num [id]
  have hd : HasDerivAt (fun t : ℝ => Real.exp (quarticScore coefficient t) + 1) 1 0 := by
    simpa [quarticScore] using ((quarticScore_deriv coefficient 0).exp).add_const 1
  have hz : Real.exp (quarticScore coefficient 0) + 1 ≠ 0 := by norm_num [quarticScore]
  convert hn.div hd hz using 1
  · funext t
    rw [quarticCELoss_deriv_eq]
    change -(1 - 4 * coefficient * t ^ 3) / (Real.exp (quarticScore coefficient t) + 1) =
      (4 * coefficient * t ^ 3 - 1) / (Real.exp (quarticScore coefficient t) + 1)
    ring
  · norm_num [quarticScore]

/-- All members share the actual initial loss, negative gradient and
positive Hessian. Source: the explicit CE diagnostic above; these are
proved observations, not shape conditions supplied by a definition. -/
theorem quarticCE_same_initial_data (coefficient : ℝ) :
    quarticCELoss coefficient 0 = Real.log 2 ∧
      deriv (quarticCELoss coefficient) 0 = -1 / 2 ∧
      deriv (deriv (quarticCELoss coefficient)) 0 = 1 / 4 := by
  refine ⟨?_, ?_, (quarticCELoss_second_deriv_at_origin coefficient).deriv⟩
  · rw [quarticCELoss_eq_exp]
    norm_num [quarticScore]
  · rw [quarticCELoss_deriv_eq]
    norm_num [quarticScore]

/-- Every loss value remains positive, including the counterexample's
finite endpoint. Source: arXiv:2501.04697v1's ordinary CE, with two finite
class logits; failure is not caused by a negative surrogate objective. -/
theorem quarticCELoss_pos (coefficient t : ℝ) : 0 < quarticCELoss coefficient t := by
  rw [quarticCELoss_eq_exp]
  apply Real.log_pos
  have he := Real.exp_pos (-quarticScore coefficient t)
  linarith

/-- No prescribed positive rate is certified by the same initial CE,
slope and Hessian over this explicit family. Source: the Taylor interval
condition versus initial-point measurements in this diagnostic; the
quartic coefficient varies with the proposed rate. -/
theorem quarticCE_every_rate_can_overshoot (eta : ℝ) (heta : 0 < eta) :
    ∃ coefficient : ℝ, 0 < coefficient ∧
      quarticCELoss coefficient 0 = Real.log 2 ∧
      deriv (quarticCELoss coefficient) 0 = -1 / 2 ∧
      deriv (deriv (quarticCELoss coefficient)) 0 = 1 / 4 ∧
      quarticCELoss coefficient 0 < quarticCELoss coefficient eta := by
  let coefficient := 2 / eta ^ 3
  have hc : 0 < coefficient := by dsimp [coefficient]; positivity
  have hs : quarticScore coefficient eta = -eta := by
    have hn : eta ≠ 0 := by positivity
    unfold quarticScore
    dsimp [coefficient]
    field_simp
    ring
  obtain ⟨h0, hd, hh⟩ := quarticCE_same_initial_data coefficient
  refine ⟨coefficient, hc, h0, hd, hh, ?_⟩
  rw [h0, quarticCELoss_eq_exp, hs]
  simp only [neg_neg]
  apply Real.log_lt_log (by norm_num)
  have he := Real.exp_lt_exp.mpr heta
  rw [Real.exp_zero] at he
  linarith

example : 0 < (1 / 1000 : ℝ) := by norm_num

end Transformer.Grokking.AdamW

import Transformer.Modes.Section3_CumulantExpressions

/-!
# Actual derivatives of the moment expressions

The cumulants in arXiv:2412.09080v3, §3.1, are derivatives of the logarithm
of the actual moment generating function. The proof of `lem:eta` in §5.3
uses their expansions in moments. This module proves that the formal
product and inverse rules from `Section3_CumulantExpressions` give the
actual derivatives on the open square where exponential moments exist.

`evalAlong` fixes one coordinate and varies the other. Differentiating once
uses the proved differentiation under the integral sign for `expMoment`.
An induction, with equality on a neighbourhood at each step, gives every
iterated derivative. The same argument starts a logarithm with its ordinary
derivative `M'/M`. Thus the algebraic expressions are consequences of the
derivative definition, not a replacement for it.
All derivative identities are local to the open exponential-moment square;
no differentiability at its boundary is assumed.

Source: arXiv:2412.09080v3, §3.1, `lem:eta`, and its proof in §5.3.
-/

open Real MeasureTheory Filter
open scoped Topology

namespace Transformer.Modes
namespace MomentExpression

/-- Evaluate the moments while varying the selected MGF coordinate. -/
noncomputable def evalAlong (e : MomentExpression) (μ : Measure (ℝ × ℝ))
    (first : Bool) (fixed t : ℝ) : ℝ :=
  e.eval (fun i j => if first then expMoment μ i j t fixed else expMoment μ i j fixed t)

/-- At the common origin either coordinate evaluation uses the same actual
moment array. Source: arXiv:2412.09080v3, §3.1, definition of `κ^α`. -/
theorem evalAlong_origin (e : MomentExpression) (μ : Measure (ℝ × ℝ)) (first : Bool) :
    e.evalAlong μ first 0 0 = e.eval (fun i j => expMoment μ i j 0 0) := by
  cases first <;> rfl

/-- The expression `M'/M` for the first logarithmic derivative. -/
def logDerivative (first : Bool) : MomentExpression :=
  mul (if first then moment 1 0 else moment 0 1) inverseMGF

/-- Actual differentiation agrees with the formal moment derivative.
Source: arXiv:2412.09080v3, §3.1, definition of the cumulants. -/
theorem hasDerivAt_evalAlong {μ : Measure (ℝ × ℝ)} [IsProbabilityMeasure μ]
    {ε fixed t : ℝ} (hE : HasExpMomentsOn μ ε) (hf : |fixed| < ε) (ht : |t| < ε)
    (e : MomentExpression) (first : Bool) :
    HasDerivAt (e.evalAlong μ first fixed) ((e.diff first).evalAlong μ first fixed t) t := by
  apply e.hasDerivAt_eval first
    (fun i j r => if first then expMoment μ i j r fixed else expMoment μ i j fixed r) t
  · intro i j
    cases first with
    | false => exact hasDerivAt_expMoment_snd hE hf ht i j
    | true => exact hasDerivAt_expMoment_fst hE ht hf i j
  · cases first with
    | false => exact (expMoment_zero_zero_pos hE hf ht).ne'
    | true => exact (expMoment_zero_zero_pos hE ht hf).ne'

example : IsProbabilityMeasure stdGauss2 ∧ HasExpMomentsOn stdGauss2 1 ∧
    |(0 : ℝ)| < 1 ∧ |(0 : ℝ)| < 1 :=
  ⟨inferInstance, hasExpMomentsOn_stdGauss2, by norm_num, by norm_num⟩

/-- Every iterated derivative is computed by repeated formal derivatives.
Source: arXiv:2412.09080v3, §3.1 and the proof of `lem:eta`. -/
theorem iteratedDeriv_evalAlong {μ : Measure (ℝ × ℝ)} [IsProbabilityMeasure μ]
    {ε fixed t : ℝ} (hE : HasExpMomentsOn μ ε) (hf : |fixed| < ε) (ht : |t| < ε)
    (e : MomentExpression) (first : Bool) (n : ℕ) :
    iteratedDeriv n (e.evalAlong μ first fixed) t =
      (((diff first)^[n]) e).evalAlong μ first fixed t := by
  induction n generalizing t with
  | zero => rfl
  | succ n ih =>
      have heq : iteratedDeriv n (e.evalAlong μ first fixed) =ᶠ[𝓝 t]
          fun r => (((diff first)^[n]) e).evalAlong μ first fixed r := by
        filter_upwards [continuous_abs.continuousAt.eventually_lt continuousAt_const ht]
          with r hr
        exact ih hr
      rw [iteratedDeriv_succ, heq.deriv_eq, Function.iterate_succ_apply']
      exact (hasDerivAt_evalAlong hE hf ht _ first).deriv

example : IsProbabilityMeasure stdGauss2 ∧ HasExpMomentsOn stdGauss2 1 ∧
    |(0 : ℝ)| < 1 ∧ |(0 : ℝ)| < 1 :=
  ⟨inferInstance, hasExpMomentsOn_stdGauss2, by norm_num, by norm_num⟩

/-- The actual logarithmic derivative is the moment quotient `M'/M`.
Source: arXiv:2412.09080v3, §3.1, after `eq:psi`. -/
theorem hasDerivAt_logAlong {μ : Measure (ℝ × ℝ)} [IsProbabilityMeasure μ]
    {ε fixed t : ℝ} (hE : HasExpMomentsOn μ ε) (hf : |fixed| < ε) (ht : |t| < ε)
    (first : Bool) :
    HasDerivAt (fun r => log (if first then expMoment μ 0 0 r fixed
      else expMoment μ 0 0 fixed r)) ((logDerivative first).evalAlong μ first fixed t) t := by
  cases first with
  | false =>
      simpa [logDerivative, evalAlong, eval, div_eq_mul_inv] using
        (hasDerivAt_expMoment_snd hE hf ht 0 0).log
          (expMoment_zero_zero_pos hE hf ht).ne'
  | true =>
      simpa [logDerivative, evalAlong, eval, div_eq_mul_inv] using
        (hasDerivAt_expMoment_fst hE ht hf 0 0).log
          (expMoment_zero_zero_pos hE ht hf).ne'

example : IsProbabilityMeasure stdGauss2 ∧ HasExpMomentsOn stdGauss2 1 ∧
    |(0 : ℝ)| < 1 ∧ |(0 : ℝ)| < 1 :=
  ⟨inferInstance, hasExpMomentsOn_stdGauss2, by norm_num, by norm_num⟩

/-- All logarithmic derivatives follow from the quotient and moment rules.
Source: arXiv:2412.09080v3, §3.1 and the proof of `lem:eta`. -/
theorem iteratedDeriv_logAlong {μ : Measure (ℝ × ℝ)} [IsProbabilityMeasure μ]
    {ε fixed t : ℝ} (hE : HasExpMomentsOn μ ε) (hf : |fixed| < ε) (ht : |t| < ε)
    (first : Bool) (n : ℕ) :
    iteratedDeriv (n + 1) (fun r => log (if first then expMoment μ 0 0 r fixed
      else expMoment μ 0 0 fixed r)) t =
      (((diff first)^[n]) (logDerivative first)).evalAlong μ first fixed t := by
  have heq : deriv (fun r => log (if first then expMoment μ 0 0 r fixed
      else expMoment μ 0 0 fixed r)) =ᶠ[𝓝 t] (logDerivative first).evalAlong μ first fixed := by
    filter_upwards [continuous_abs.continuousAt.eventually_lt continuousAt_const ht]
      with r hr
    exact (hasDerivAt_logAlong hE hf hr first).deriv
  rw [iteratedDeriv_succ', heq.iteratedDeriv_eq n]
  exact iteratedDeriv_evalAlong hE hf ht _ first n

example : IsProbabilityMeasure stdGauss2 ∧ HasExpMomentsOn stdGauss2 1 ∧
    |(0 : ℝ)| < 1 ∧ |(0 : ℝ)| < 1 :=
  ⟨inferInstance, hasExpMomentsOn_stdGauss2, by norm_num, by norm_num⟩

/-- Each further derivative raises the homogeneous order by one.
Source: arXiv:2412.09080v3, proof of `lem:eta` in §5.3. -/
theorem Degree.iterate_diff {e : MomentExpression} {s : ℕ} (hd : Degree e s)
    (first : Bool) (n : ℕ) : Degree (((MomentExpression.diff first)^[n]) e) (s + n) := by
  induction n with
  | zero => exact hd
  | succ n ih =>
      rw [Function.iterate_succ_apply']
      simpa [Nat.add_assoc] using ih.diff first

example : Degree (moment 1 1) 2 := Degree.moment 1 1

/-- The first logarithmic derivative has homogeneous order one.
Source: arXiv:2412.09080v3, §3.1 and `lem:eta`. -/
theorem degree_logDerivative (first : Bool) : Degree (logDerivative first) 1 := by
  cases first with
  | false => simpa [logDerivative] using Degree.mul (Degree.moment 0 1) Degree.inverseMGF
  | true => simpa [logDerivative] using Degree.mul (Degree.moment 1 0) Degree.inverseMGF


end MomentExpression

end Transformer.Modes

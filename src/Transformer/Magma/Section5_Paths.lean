/-
# Finite stochastic trajectories

Formalization of total expectations over the iterates in
arXiv:2602.15322v1, Appendix A.4. The expectation below is constructed
from the transition function and the actual sampling law. It does not
postulate objective descent or stationarity.
-/

import Transformer.Magma.Section5_FiniteLaw

open scoped BigOperators

noncomputable section

namespace Transformer.Magma

variable {K State : Type*} [Fintype K]

/-- Expected observable after t sampled transitions from initial.
Time dependence can be stored in State together with the dense moments.
Source: arXiv:2602.15322v1, Appendix A.4, total expectation over iterates. -/
def pathExpectation (w : K → ℝ) (next : State → K → State) (initial : State) :
    ℕ → (State → ℝ) → ℝ
  | 0, f => f initial
  | t + 1, f => pathExpectation w next initial t
      (fun state => finiteExpectation w (fun k => f (next state k)))

/-- Constructed trajectory expectation preserves addition.
Source: arXiv:2602.15322v1, Appendix A.4. -/
theorem pathExpectation_add (w : K → ℝ) (next : State → K → State) (initial : State)
    (t : ℕ) (f g : State → ℝ) :
    pathExpectation w next initial t (fun state => f state + g state) =
      pathExpectation w next initial t f + pathExpectation w next initial t g := by
  induction t generalizing f g with
  | zero => rfl
  | succ t ih =>
    simp only [pathExpectation, finiteExpectation_add]
    exact ih _ _

/-- Constructed trajectory expectation preserves subtraction.
Source: arXiv:2602.15322v1, Appendix A.4. -/
theorem pathExpectation_sub (w : K → ℝ) (next : State → K → State) (initial : State)
    (t : ℕ) (f g : State → ℝ) :
    pathExpectation w next initial t (fun state => f state - g state) =
      pathExpectation w next initial t f - pathExpectation w next initial t g := by
  induction t generalizing f g with
  | zero => rfl
  | succ t ih =>
    simp only [pathExpectation, finiteExpectation_sub]
    exact ih _ _

/-- A scalar factors out of trajectory expectation.
Source: arXiv:2602.15322v1, Appendix A.4. -/
theorem pathExpectation_mul (w : K → ℝ) (next : State → K → State) (initial : State)
    (t : ℕ) (c : ℝ) (f : State → ℝ) :
    pathExpectation w next initial t (fun state => c * f state) =
      c * pathExpectation w next initial t f := by
  induction t generalizing f with
  | zero => rfl
  | succ t ih =>
    simp only [pathExpectation, finiteExpectation_mul]
    exact ih _

/-- A probability law preserves constants at every time.
Source: arXiv:2602.15322v1, Appendix A.4. -/
theorem pathExpectation_const (w : K → ℝ) (hw : ∑ k, w k = 1)
    (next : State → K → State) (initial : State) (t : ℕ) (c : ℝ) :
    pathExpectation w next initial t (fun _ => c) = c := by
  induction t with
  | zero => rfl
  | succ t ih =>
    simpa only [pathExpectation, finiteExpectation_const w hw] using ih

/-- The normalization hypothesis holds for a nonconstant scalar
transition with a singleton sampling law. Source:
arXiv:2602.15322v1, Appendix A.4. -/
example : (∑ _ : Unit, (1 : ℝ)) = 1 := by simp

/-- Nonnegative sampling weights make trajectory expectation monotone.
Source: arXiv:2602.15322v1, Appendix A.4, lower-bounded objective step. -/
theorem pathExpectation_mono (w : K → ℝ) (hw0 : ∀ k, 0 ≤ w k)
    (next : State → K → State) (initial : State) (t : ℕ)
    (f g : State → ℝ) (hfg : ∀ state, f state ≤ g state) :
    pathExpectation w next initial t f ≤ pathExpectation w next initial t g := by
  induction t generalizing f g with
  | zero => exact hfg initial
  | succ t ih =>
    apply ih
    intro state
    exact finiteExpectation_mono w hw0 _ _ (fun k => hfg (next state k))

/-- Monotonicity hypotheses hold on two nonconstant observables of a
scalar state. Source: arXiv:2602.15322v1, Appendix A.4. -/
example : (∀ _ : Unit, (0 : ℝ) ≤ 1) ∧
    (∀ state : ℝ, state ^ 2 ≤ state ^ 2 + 1) := by
  constructor
  · simp
  · intro state
    linarith

end Transformer.Magma

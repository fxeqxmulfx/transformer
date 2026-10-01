/-
# A nonempty model of every corrected convergence hypothesis

Witness for arXiv:2602.15322v1, Section 5, corrected theorem:main_result.
The objective is an actual nonconstant quadratic, gradients are its
derivatives, and surviving updates have nonzero scalar damping.
-/

import Transformer.Magma.Section5_Convergence

open scoped BigOperators InnerProductSpace

noncomputable section

namespace Transformer.Magma

open Transformer.Optimization

/-- An actual one-block, exact-gradient proposal. The auxiliary state is
trivial here; the convergence theorem permits dense nontrivial moment
states too. Source: arXiv:2602.15322v1, Section 5, corrected SGD domain. -/
def quadraticCandidate (state : ℝ × Unit) (_sample : Unit) : Unit × (Unit → ℝ) :=
  ((), fun _ => (1 / 2) * gradient quadratic state.1)

/-- The corrected bound applies to a genuine sampled quadratic trajectory
for every positive horizon. This instantiates all its analytic, sampling,
unbiasedness, operator, block, and lower-bound hypotheses. Source:
arXiv:2602.15322v1, Section 5, corrected theorem:main_result. -/
theorem quadratic_stationarity_witness (T : ℕ) (hT : 0 < T) :
    (∑ t ∈ Finset.range T,
      pathExpectation (jointMass (fun _ : Unit => (1 : ℝ)) (1 / 2))
        (normalizedTransition (1 / 100) (1 / 2) quadraticCandidate) ((1 : ℝ), ()) t
          (fun state => ‖gradient quadratic state.1‖ ^ 2)) / T ≤ 800 / T := by
  have h := finite_horizon_stationarity_corrected quadratic 1 (1 / 100) (1 / 2)
    (1 / 4) 1 0 0 quadratic_smooth (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
    (by intro x; unfold quadratic; positivity)
    (fun _ : Unit => (1 : ℝ)) (by simp) (by simp)
    (fun state _ => gradient quadratic state.1) (by simp)
    (by intro state; simp [finiteExpectation])
    (fun _ _ => (1 / 2 : ℝ) • ContinuousLinearMap.id ℝ ℝ)
    (fun _ _ => scalar_half_damping_bounds) quadraticCandidate
    (by simp [quadraticCandidate])
    (by
      intro state sample j k hjk
      exact (hjk (Subsingleton.elim j k)).elim)
    ((1 : ℝ), ()) T hT
  norm_num [quadratic] at h ⊢
  convert h using 1
  field_simp
  ring

/-- A positive horizon satisfies the witness theorem's only hypothesis.
Source: arXiv:2602.15322v1, Section 5, corrected finite horizon. -/
example : 0 < (1 : ℕ) := by decide

end Transformer.Magma

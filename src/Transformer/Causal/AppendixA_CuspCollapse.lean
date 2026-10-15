import Transformer.Causal.AppendixA_CuspClock
import Transformer.Causal.AppendixA_CuspHyperbolicity
import Transformer.Perceptron.Analytic

/-!
# Actual cusp trajectories with finite-time collapse

ArXiv:2411.04990v2, Appendix A, `lemma:convergence`, C¹ regularity
audit. The original cusp-energy velocities are `sin ψ₀` and
`sin ψ₁ - sin² ψ₀ · cuspForce ψ₁`, with normalization one.
For the actual first angle and integral clock already constructed,
put `z(t) = exp(t) max(c - clock(κ,t),0)²`.
The actual second angle is `2 arctan z(t)`.

The squared positive part is differentiable at zero. Thus these
formulas solve the original ODE at every real time, including the
hitting time. The proof uses the elementary squared-ReLU derivative
from `Transformer.Perceptron.Analytic`; it uses no convergence claim.
The square root of z has an exact formula, giving the actual
second-angle equation from the previously proved cusp-force identity.

Whenever `c < clock(κ,1)`, the original second angle is identically
zero after time one. For positive κ, the entire actual trajectory
therefore converges to `(π,0)`. That actual critical point has a
positive Jacobian eigenvalue, so uniqueness of limits excludes a
strongly stable limit. The initial parameters used here are genuine
real parameters; positivity and an actual initial-value formula are
proved explicitly for the subsequent positive-measure construction.
-/

open Real Filter Topology
namespace Transformer.Causal

/-- The actual squared positive-part solution in the half-angle coordinate.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, trajectory audit. -/
noncomputable def cuspCollapsedSize (κ c t : ℝ) : ℝ :=
  Real.exp t * max (c - cuspCollapseClock κ t) 0 ^ 2

/-- The actual half-angle coordinate is nonnegative at every real time.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, trajectory audit. -/
theorem cuspCollapsedSize_nonneg (κ c t : ℝ) : 0 ≤ cuspCollapsedSize κ c t := by
  unfold cuspCollapsedSize
  positivity

/-- The actual square root has this exact expression, also at a hitting time.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, trajectory audit. -/
theorem sqrt_cuspCollapsedSize (κ c t : ℝ) :
    Real.sqrt (cuspCollapsedSize κ c t) =
      Real.exp (t / 2) * max (c - cuspCollapseClock κ t) 0 := by
  have heq : cuspCollapsedSize κ c t =
      (Real.exp (t / 2) * max (c - cuspCollapseClock κ t) 0) ^ 2 := by
    unfold cuspCollapsedSize
    rw [mul_pow, pow_two (Real.exp (t / 2)), ← Real.exp_add, add_halves]
  rw [heq, Real.sqrt_sq (by positivity)]

/-- The actual half-angle coordinate solves its differential equation globally.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, trajectory audit. -/
theorem hasDerivAt_cuspCollapsedSize (κ c t : ℝ) :
    HasDerivAt (cuspCollapsedSize κ c)
      (cuspCollapsedSize κ c t - Real.sin (cuspFirstAngle κ t) ^ 2 *
        Real.sqrt (cuspCollapsedSize κ c t)) t := by
  have hv := (hasDerivAt_const t c).sub (hasDerivAt_cuspCollapseClock κ t)
  have hu := (Transformer.Perceptron.hasDerivAt_reluSq (c - cuspCollapseClock κ t)).comp t hv
  have h := (Real.hasDerivAt_exp t).mul hu
  apply h.congr_deriv
  simp only [Function.comp_apply, Pi.sub_apply]
  rw [sqrt_cuspCollapsedSize, cuspCollapsedSize]
  have he : Real.exp t * Real.exp (-t / 2) = Real.exp (t / 2) := by
    rw [← Real.exp_add]
    congr 1
    ring
  linear_combination -(max (c - cuspCollapseClock κ t) 0 * Real.sin (cuspFirstAngle κ t) ^ 2) * he

/-- The actual second angle solves the original angular equation globally.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, trajectory audit. -/
theorem hasDerivAt_cuspSecondAngle (κ c t : ℝ) :
    HasDerivAt (fun s => 2 * Real.arctan (cuspCollapsedSize κ c s))
      (Real.sin (2 * Real.arctan (cuspCollapsedSize κ c t)) -
        Real.sin (cuspFirstAngle κ t) ^ 2 *
          cuspForce (2 * Real.arctan (cuspCollapsedSize κ c t))) t := by
  have h := ((hasDerivAt_cuspCollapsedSize κ c t).arctan).const_mul 2
  apply h.congr_deriv
  rw [sin_double_arctan_cusp, cuspForce_double_arctan (cuspCollapsedSize_nonneg κ c t)]
  ring

/-- The two actual angular coordinates of the original sequential system.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, trajectory audit. -/
noncomputable def cuspCollapsedTrajectory (κ c t : ℝ) : Idx 2 → ℝ :=
  ![cuspFirstAngle κ t, 2 * Real.arctan (cuspCollapsedSize κ c t)]

/-- The explicit trajectory is a genuine solution of the original system.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, trajectory audit. -/
theorem sequentialFlow_cuspCollapsedTrajectory (κ c : ℝ) :
    SequentialFlow 2 cuspEnergy (fun _ _ => 1) (cuspCollapsedTrajectory κ c) := by
  intro t k
  fin_cases k
  · change HasDerivAt (cuspFirstAngle κ)
      (seqVelocity 2 cuspEnergy (fun _ _ => 1) 0 (cuspCollapsedTrajectory κ c t)) t
    rw [seqVelocity_cuspEnergy_zero]
    exact hasDerivAt_cuspFirstAngle κ t
  · change HasDerivAt (fun s => 2 * Real.arctan (cuspCollapsedSize κ c s))
      (seqVelocity 2 cuspEnergy (fun _ _ => 1) 1 (cuspCollapsedTrajectory κ c t)) t
    rw [seqVelocity_cuspEnergy_one]
    exact hasDerivAt_cuspSecondAngle κ c t

/-- For nonnegative c the trajectory starts at the displayed original angles.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, trajectory audit. -/
theorem cuspCollapsedTrajectory_zero (κ c : ℝ) (hc : 0 ≤ c) :
    cuspCollapsedTrajectory κ c 0 = ![2 * Real.arctan κ, 2 * Real.arctan (c ^ 2)] := by
  ext i
  fin_cases i <;> simp [cuspCollapsedTrajectory, cuspCollapsedSize, max_eq_left hc]

/-- The actual second angle has reached zero by time one and stays there.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, trajectory audit. -/
theorem cuspCollapsedTrajectory_second_eq_zero (κ c : ℝ) (hc : c < cuspCollapseClock κ 1)
    (t : ℝ) (ht : 1 ≤ t) : cuspCollapsedTrajectory κ c t 1 = 0 := by
  have hclock := monotone_cuspCollapseClock κ ht
  have hu : c - cuspCollapseClock κ t ≤ 0 := by linarith
  simp [cuspCollapsedTrajectory, cuspCollapsedSize, max_eq_right hu]

/-- The actual trajectory converges to the unstable critical tuple `(π,0)`.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, trajectory audit. -/
theorem tendsto_cuspCollapsedTrajectory {κ c : ℝ} (hκ : 0 < κ) (hc : c < cuspCollapseClock κ 1) :
    Tendsto (cuspCollapsedTrajectory κ c) atTop (nhds (![π, 0] : Idx 2 → ℝ)) := by
  apply tendsto_pi_nhds.mpr
  intro i
  fin_cases i
  · change Tendsto (cuspFirstAngle κ) atTop (nhds π)
    exact tendsto_cuspFirstAngle hκ
  · change Tendsto (fun t => cuspCollapsedTrajectory κ c t 1) atTop (nhds 0)
    have heq : (fun t => cuspCollapsedTrajectory κ c t 1) =ᶠ[atTop] (fun _ : ℝ => (0 : ℝ)) := by
      filter_upwards [eventually_ge_atTop (1 : ℝ)] with t ht
      exact cuspCollapsedTrajectory_second_eq_zero κ c hc t ht
    exact tendsto_const_nhds.congr' heq.symm

/-- The actual solution has no strongly stable limit of the printed kind.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, trajectory audit. -/
theorem cuspCollapsedTrajectory_no_stable_limit {κ c : ℝ} (hκ : 0 < κ) (hc : c < cuspCollapseClock κ 1) :
    ¬ ∃ ψstar : Idx 2 → ℝ, Tendsto (cuspCollapsedTrajectory κ c) atTop (nhds ψstar) ∧
      (∀ k : Idx 2, IsCriticalPrefix 2 cuspEnergy (fun _ _ => 1) k ψstar ∧
        StronglyStablePrefix 2 cuspEnergy (fun _ _ => 1) k ψstar) := by
  rintro ⟨ψstar, hlim, hstable⟩
  have heq := tendsto_nhds_unique hlim (tendsto_cuspCollapsedTrajectory hκ hc)
  subst ψstar
  exact not_stronglyStablePrefix_cuspEnergy_target (hstable 1).2

/-- A positive actual κ and a nonnegative actual c jointly realize the
initial-value, finite-collapse and original sequential-flow hypotheses. -/
example : ∃ c : ℝ, (0 : ℝ) < 1 ∧ 0 ≤ c ∧ c < cuspCollapseClock 1 1 ∧
    SequentialFlow 2 cuspEnergy (fun _ _ => 1) (cuspCollapsedTrajectory 1 c) := by
  refine ⟨cuspCollapseClock 1 1 / 2, zero_lt_one, ?_, ?_, sequentialFlow_cuspCollapsedTrajectory _ _⟩
  · exact (div_pos (cuspCollapseClock_one_pos zero_lt_one) (by norm_num)).le
  · linarith [cuspCollapseClock_one_pos zero_lt_one]

end Transformer.Causal

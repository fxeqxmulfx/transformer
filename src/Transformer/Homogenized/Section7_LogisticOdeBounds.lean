/-
# Range, monotonicity, and relative growth of the original logistic reference

ArXiv:2604.01978v1, §7, `eq:logistic_u_smallbeta`, `eq: logistic.sol`,
and the proof of `thm:clustering_small_beta`.
The source reference solves `u'=u(1-u)` with positive initial mean at most one.
Strict scalar barriers prove positivity, the upper bound one, and growth
at most `u(0) exp(t)` directly from that original equation.
The invariant range then proves monotonicity and the initial lower bound.

All derivatives remain right derivatives at time zero, matching the
source Cauchy problem on nonnegative times. No closed formula,
solution uniqueness, invariant interval, or monotonicity is assumed.
-/

import Transformer.Homogenized.Section7_GaussianMeanLogProbability
import Mathlib.Analysis.Calculus.MeanValue

open Set

namespace Transformer.Homogenized

/-- The original logistic right derivative supplies continuous paths on every finite horizon.
Source: arXiv:2604.01978v1, §7, `eq:logistic_u_smallbeta`. -/
theorem logistic_ode_continuousOn (u : ℝ → ℝ)
    (hu : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt u (u t * (1 - u t)) (Ici 0) t) (T : ℝ) :
    ContinuousOn u (Icc (0 : ℝ) T) := fun t ht =>
  (hu t ht.1).continuousWithinAt.mono (fun _ hs => hs.1)

/-- A strict constant barrier proves the original logistic reference stays below one.
Source: arXiv:2604.01978v1, §7, `eq:logistic_u_smallbeta`. -/
theorem logistic_ode_le_one (u : ℝ → ℝ) (hinit : u 0 ≤ 1)
    (hu : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt u (u t * (1 - u t)) (Ici 0) t)
    (T : ℝ) (hT : 0 ≤ T) : u T ≤ 1 := by
  apply le_of_forall_pos_le_add
  intro ε hε
  apply image_le_of_deriv_right_lt_deriv_boundary (logistic_ode_continuousOn u hu T)
    (fun t ht => (hu t ht.1).mono (fun s hs => ht.1.trans hs))
    (B := fun _ => 1 + ε) (B' := fun _ => 0) (by linarith) (fun t => hasDerivAt_const t _)
    (fun t ht heq => ?_) ⟨hT, le_rfl⟩
  rw [heq]
  nlinarith

/-- A positive exponentially decreasing barrier proves positivity without assuming the range.
Source: arXiv:2604.01978v1, §7, `eq:logistic_u_smallbeta`. -/
theorem logistic_ode_exp_lower (u : ℝ → ℝ) (hpos : 0 < u 0) (hinit : u 0 ≤ 1)
    (hu : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt u (u t * (1 - u t)) (Ici 0) t)
    (T : ℝ) (hT : 0 ≤ T) : u 0 * Real.exp (-T) ≤ u T := by
  have hb (t : ℝ) : HasDerivAt (fun s : ℝ => -(u 0 * Real.exp (-s))) (u 0 * Real.exp (-t)) t := by
    convert (((hasDerivAt_neg t).exp).const_mul (u 0)).neg using 1
    ring
  have hbound : -u T ≤ -(u 0 * Real.exp (-T)) := by
    apply image_le_of_deriv_right_lt_deriv_boundary (logistic_ode_continuousOn u hu T).neg
      (fun t ht => ((hu t ht.1).mono (fun s hs => ht.1.trans hs)).neg)
      (B := fun t => -(u 0 * Real.exp (-t))) (B' := fun t => u 0 * Real.exp (-t))
      (by simp) hb (fun t ht heq => ?_) ⟨hT, le_rfl⟩
    have hp : 0 < u 0 * Real.exp (-t) := mul_pos hpos (Real.exp_pos _)
    have he : Real.exp (-t) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith [ht.1])
    have hy : u 0 * Real.exp (-t) ≤ 1 := by
      have hm := mul_le_mul_of_nonneg_left he hpos.le
      nlinarith
    change -(u t) = -(u 0 * Real.exp (-t)) at heq
    have heq' : u t = u 0 * Real.exp (-t) := by linarith
    rw [heq']
    nlinarith
  linarith

/-- The genuine original logistic reference stays positive and at most one.
Source: arXiv:2604.01978v1, §7, `eq:logistic_u_smallbeta` and `eq: logistic.sol`. -/
theorem logistic_ode_mem_Ioc (u : ℝ → ℝ) (hpos : 0 < u 0) (hinit : u 0 ≤ 1)
    (hu : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt u (u t * (1 - u t)) (Ici 0) t)
    (t : ℝ) (ht : 0 ≤ t) : u t ∈ Ioc (0 : ℝ) 1 :=
  ⟨(mul_pos hpos (Real.exp_pos _)).trans_le (logistic_ode_exp_lower u hpos hinit hu t ht),
    logistic_ode_le_one u hinit hu t ht⟩

/-- The original logistic Cauchy problem itself bounds its relative growth by `exp(t)`.
Source: arXiv:2604.01978v1, §7, `eq:logistic_u_smallbeta` and `eq: logistic.sol`. -/
theorem logistic_ode_exp_upper (u : ℝ → ℝ) (hpos : 0 < u 0)
    (hu : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt u (u t * (1 - u t)) (Ici 0) t)
    (T : ℝ) (hT : 0 ≤ T) : u T ≤ u 0 * Real.exp T := by
  have hb (t : ℝ) : HasDerivAt (fun s : ℝ => u 0 * Real.exp s) (u 0 * Real.exp t) t :=
    (Real.hasDerivAt_exp t).const_mul (u 0)
  apply image_le_of_deriv_right_lt_deriv_boundary (logistic_ode_continuousOn u hu T)
    (fun t ht => (hu t ht.1).mono (fun s hs => ht.1.trans hs))
    (B := fun t => u 0 * Real.exp t) (B' := fun t => u 0 * Real.exp t) (by simp) hb
    (fun t ht heq => ?_) ⟨hT, le_rfl⟩
  have hp : 0 < u 0 * Real.exp t := mul_pos hpos (Real.exp_pos _)
  rw [heq]
  nlinarith

/-- The proved original range makes the logistic reference nondecreasing.
Source: arXiv:2604.01978v1, §7, `eq:logistic_u_smallbeta`. -/
theorem logistic_ode_monotoneOn (u : ℝ → ℝ) (hpos : 0 < u 0) (hinit : u 0 ≤ 1)
    (hu : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt u (u t * (1 - u t)) (Ici 0) t) :
    MonotoneOn u (Ici (0 : ℝ)) := by
  apply monotoneOn_of_hasDerivWithinAt_nonneg (convex_Ici 0)
    (fun t ht => (hu t ht).continuousWithinAt)
  · intro t ht
    rw [interior_Ici] at ht ⊢
    have ht0 : 0 < t := ht
    exact (hu t ht0.le).mono (fun s hs => (show 0 < s from hs).le)
  · intro t ht
    rw [interior_Ici] at ht
    have ht0 : 0 < t := ht
    have hr := logistic_ode_mem_Ioc u hpos hinit hu t ht0.le
    exact mul_nonneg hr.1.le (sub_nonneg.mpr hr.2)

/-- Every original reference value is at least its positive initial mean.
Source: arXiv:2604.01978v1, §7, `eq:logistic_u_smallbeta`. -/
theorem logistic_ode_initial_le (u : ℝ → ℝ) (hpos : 0 < u 0) (hinit : u 0 ≤ 1)
    (hu : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt u (u t * (1 - u t)) (Ici 0) t)
    (t : ℝ) (ht : 0 ≤ t) : u 0 ≤ u t :=
  logistic_ode_monotoneOn u hpos hinit hu (by simp) ht ht

/-- The original positive reference has logarithmic derivative `1-u`.
Source: arXiv:2604.01978v1, §7, `eq:logistic_u_smallbeta`,
the logarithmic comparison used for `thm:clustering_small_beta`. -/
theorem hasDerivWithinAt_log_logistic_ode (u : ℝ → ℝ) (hpos : 0 < u 0) (hinit : u 0 ≤ 1)
    (hu : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt u (u t * (1 - u t)) (Ici 0) t)
    (t : ℝ) (ht : 0 ≤ t) :
    HasDerivWithinAt (fun s => Real.log (u s)) (1 - u t) (Ici 0) t := by
  have hp := (logistic_ode_mem_Ioc u hpos hinit hu t ht).1
  simpa only [mul_div_cancel_left₀ _ hp.ne'] using (hu t ht).log hp.ne'

/-- Its logarithm is continuous throughout every original finite source horizon.
Source: arXiv:2604.01978v1, §7, the original logistic comparison. -/
theorem continuousOn_log_logistic_ode (u : ℝ → ℝ) (hpos : 0 < u 0) (hinit : u 0 ≤ 1)
    (hu : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt u (u t * (1 - u t)) (Ici 0) t) (T : ℝ) :
    ContinuousOn (fun s => Real.log (u s)) (Icc (0 : ℝ) T) :=
  (logistic_ode_continuousOn u hu T).log
    (fun t ht => (logistic_ode_mem_Ioc u hpos hinit hu t ht.1).1.ne')

/-- Dividing by the original positive initial mean retains the relative exponential bound.
Source: arXiv:2604.01978v1, §7, `eq: logistic.sol`. -/
theorem logistic_ode_relative_growth (u : ℝ → ℝ) (hpos : 0 < u 0)
    (hu : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt u (u t * (1 - u t)) (Ici 0) t)
    (t : ℝ) (ht : 0 ≤ t) : u t / u 0 ≤ Real.exp t := by
  apply (div_le_iff₀ hpos).2
  exact (logistic_ode_exp_upper u hpos hu t ht).trans_eq (mul_comm _ _)

/-- The original consensus reference realizes every Cauchy-problem premise,
including the endpoint initial mean one and the right derivative at zero. -/
example : ∃ u : ℝ → ℝ, 0 < u 0 ∧ u 0 ≤ 1 ∧
    (∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt u (u t * (1 - u t)) (Ici 0) t) ∧
    (0 : ℝ) ≤ 1 ∧ MonotoneOn u (Ici (0 : ℝ)) ∧
    (∀ t ∈ Ici (0 : ℝ), u t ∈ Ioc (0 : ℝ) 1) := by
  let u : ℝ → ℝ := fun _ => 1
  have hu : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt u (u t * (1 - u t)) (Ici 0) t := by
    intro t ht
    simpa only [u, sub_self, mul_zero] using (hasDerivAt_const t (1 : ℝ)).hasDerivWithinAt
  exact ⟨u, one_pos, le_rfl, hu, zero_le_one,
    logistic_ode_monotoneOn u one_pos le_rfl hu,
    fun t ht => logistic_ode_mem_Ioc u one_pos le_rfl hu t ht⟩

end Transformer.Homogenized

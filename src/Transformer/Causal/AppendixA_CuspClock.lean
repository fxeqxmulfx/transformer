module

public import Transformer.Causal.AppendixA_CuspJacobian
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.ArctanDeriv
public import Mathlib.MeasureTheory.Integral.DominatedConvergence
public import Mathlib.Analysis.Calculus.Deriv.MeanValue

/-!
# Actual first-angle trajectory and accumulated cusp forcing

ArXiv:2411.04990v2, Appendix A, `lemma:convergence`, C¹ regularity
audit. The actual first velocity of the constructed energies is
`sin ψ₀`. Its explicit global solution is `2 arctan(κ exp t)`.
For positive κ this angle stays between zero and π and converges
to π. Its genuine derivative is proved from the original sine law.

The accumulated forcing is the actual interval integral
`(1/2) ∫₀ᵗ exp(-s/2) sin²(firstAngle(κ,s)) ds`.
It is defined for every real time, is monotone, and has the displayed
actual derivative by the fundamental theorem of calculus.
At time one it is strictly positive for every positive κ and depends
continuously on κ. These facts supply an actual open range of initial
second angles for the subsequent finite-time collapse construction.

The original energy and normalization are unchanged. This is an
explicit trajectory and forcing calculation, independent of the
printed convergence lemma and its unstable-manifold step.
-/

@[expose] public section

open Real Filter MeasureTheory Topology
namespace Transformer.Causal

/-- The explicit global first angle of the actual cusp-energy system.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, actual ODE audit. -/
noncomputable def cuspFirstAngle (κ t : ℝ) : ℝ := 2 * Real.arctan (Real.exp t * κ)

/-- The actual doubled-arctangent sine has this exact rational expression.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, trajectory construction. -/
theorem sin_double_arctan_cusp (z : ℝ) :
    Real.sin (2 * Real.arctan z) = 2 * z / (1 + z ^ 2) := by
  rw [Real.sin_two_mul, Real.sin_arctan, Real.cos_arctan]
  have hsq : Real.sqrt (1 + z ^ 2) ^ 2 = 1 + z ^ 2 := Real.sq_sqrt (by positivity)
  have hpos : 0 < Real.sqrt (1 + z ^ 2) := Real.sqrt_pos.mpr (by positivity)
  field_simp
  rw [hsq]

/-- The actual first angle solves its original sine equation at every real time.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, original ODE. -/
theorem hasDerivAt_cuspFirstAngle (κ t : ℝ) :
    HasDerivAt (cuspFirstAngle κ) (Real.sin (cuspFirstAngle κ t)) t := by
  have h := (((Real.hasDerivAt_exp t).mul_const κ).arctan).const_mul 2
  apply h.congr_deriv
  rw [cuspFirstAngle, sin_double_arctan_cusp]
  ring

/-- The actual first angle depends continuously on both original parameters.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, trajectory construction. -/
@[fun_prop] theorem continuous_cuspFirstAngle_uncurry :
    Continuous (fun p : ℝ × ℝ => cuspFirstAngle p.1 p.2) := by
  unfold cuspFirstAngle
  fun_prop

/-- Each actual first-angle trajectory is continuous for all real times.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, actual trajectories. -/
@[fun_prop] theorem continuous_cuspFirstAngle (κ : ℝ) : Continuous (cuspFirstAngle κ) :=
  continuous_iff_continuousAt.mpr (fun t => (hasDerivAt_cuspFirstAngle κ t).continuousAt)

/-- The actual trajectory starts at its displayed angular initial value.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, initial conditions. -/
@[simp] theorem cuspFirstAngle_zero (κ : ℝ) : cuspFirstAngle κ 0 = 2 * Real.arctan κ := by
  simp [cuspFirstAngle]

/-- Positive κ keeps the actual first angle strictly above the angular origin.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, trajectory construction. -/
theorem cuspFirstAngle_pos {κ : ℝ} (hκ : 0 < κ) (t : ℝ) : 0 < cuspFirstAngle κ t := by
  unfold cuspFirstAngle
  exact mul_pos (by norm_num) (Real.arctan_pos.mpr (mul_pos (Real.exp_pos _) hκ))

/-- The actual first angle remains strictly below the opposite angle.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, trajectory construction. -/
theorem cuspFirstAngle_lt_pi (κ t : ℝ) : cuspFirstAngle κ t < π := by
  unfold cuspFirstAngle
  have h := Real.arctan_lt_pi_div_two (Real.exp t * κ)
  linarith

/-- The actual first angle tends to its stable sine equilibrium.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, actual trajectory limit. -/
theorem tendsto_cuspFirstAngle {κ : ℝ} (hκ : 0 < κ) :
    Tendsto (cuspFirstAngle κ) atTop (nhds π) := by
  change Tendsto (fun t : ℝ => 2 * Real.arctan (Real.exp t * κ)) atTop (nhds π)
  have hz := Filter.Tendsto.atTop_mul_const hκ Real.tendsto_exp_atTop
  have h := (Real.tendsto_arctan_atTop.mono_right nhdsWithin_le_nhds).comp hz
  have htwo := h.const_mul (2 : ℝ)
  simpa only [Function.comp_def, mul_div_cancel₀ _ (by norm_num : (2 : ℝ) ≠ 0)] using htwo

/-- The actual accumulated forcing for the original second angular equation.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, trajectory construction. -/
noncomputable def cuspCollapseClock (κ t : ℝ) : ℝ :=
  (2 : ℝ)⁻¹ * ∫ s in (0 : ℝ)..t, Real.exp (-s / 2) * Real.sin (cuspFirstAngle κ s) ^ 2

/-- The actual forcing integrand is jointly continuous in κ and real time.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, trajectory construction. -/
@[fun_prop] theorem continuous_cuspClockIntegrand_uncurry :
    Continuous (fun p : ℝ × ℝ => Real.exp (-p.2 / 2) * Real.sin (cuspFirstAngle p.1 p.2) ^ 2) := by
  unfold cuspFirstAngle
  fun_prop

/-- The actual forcing integrand is continuous along each trajectory.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, trajectory construction. -/
@[fun_prop] theorem continuous_cuspClockIntegrand (κ : ℝ) :
    Continuous (fun s : ℝ => Real.exp (-s / 2) * Real.sin (cuspFirstAngle κ s) ^ 2) := by
  unfold cuspFirstAngle
  fun_prop

/-- The actual integral has the displayed derivative at every original time.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, actual ODE audit. -/
theorem hasDerivAt_cuspCollapseClock (κ t : ℝ) :
    HasDerivAt (cuspCollapseClock κ)
      ((2 : ℝ)⁻¹ * (Real.exp (-t / 2) * Real.sin (cuspFirstAngle κ t) ^ 2)) t := by
  have hc := continuous_cuspClockIntegrand κ
  exact (intervalIntegral.integral_hasDerivAt_right (hc.intervalIntegrable 0 t)
    hc.stronglyMeasurable.stronglyMeasurableAtFilter hc.continuousAt).const_mul _

/-- The actual accumulated forcing starts at zero.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, initial conditions. -/
@[simp] theorem cuspCollapseClock_zero (κ : ℝ) : cuspCollapseClock κ 0 = 0 := by
  simp [cuspCollapseClock]

/-- Nonnegative original forcing makes the actual accumulated integral monotone.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, collapse construction. -/
theorem monotone_cuspCollapseClock (κ : ℝ) : Monotone (cuspCollapseClock κ) :=
  monotone_of_hasDerivAt_nonneg (hasDerivAt_cuspCollapseClock κ) (fun _ => by positivity)

/-- The actual time-one forcing is continuous in the actual initial parameter.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, initial-set construction. -/
@[fun_prop] theorem continuous_cuspCollapseClock_one : Continuous (fun κ : ℝ => cuspCollapseClock κ 1) := by
  have h := intervalIntegral.continuous_parametric_intervalIntegral_of_continuous'
    (μ := volume) (f := fun κ s : ℝ => Real.exp (-s / 2) * Real.sin (cuspFirstAngle κ s) ^ 2)
    continuous_cuspClockIntegrand_uncurry 0 1
  exact h.const_mul _

/-- Every positive original κ supplies strictly positive actual forcing by time one.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, collapse construction. -/
theorem cuspCollapseClock_one_pos {κ : ℝ} (hκ : 0 < κ) : 0 < cuspCollapseClock κ 1 := by
  have hs : 0 < Real.sin (cuspFirstAngle κ 0) :=
    Real.sin_pos_of_pos_of_lt_pi (cuspFirstAngle_pos hκ 0) (cuspFirstAngle_lt_pi κ 0)
  have hI : 0 < ∫ s in (0 : ℝ)..1, Real.exp (-s / 2) * Real.sin (cuspFirstAngle κ s) ^ 2 :=
    intervalIntegral.integral_pos zero_lt_one (continuous_cuspClockIntegrand κ).continuousOn
      (fun _ _ => by positivity) ⟨0, by constructor <;> norm_num, by simpa using sq_pos_of_pos hs⟩
  exact mul_pos (by norm_num) hI

/-- The actual parameter κ equal to one jointly realizes the positivity,
angular interval and accumulated-forcing inputs of this trajectory calculation. -/
example : (0 : ℝ) < 1 ∧ 0 < cuspFirstAngle 1 0 ∧ cuspFirstAngle 1 0 < π ∧
    0 < cuspCollapseClock 1 1 :=
  ⟨zero_lt_one, cuspFirstAngle_pos zero_lt_one _, cuspFirstAngle_lt_pi _ _,
    cuspCollapseClock_one_pos zero_lt_one⟩

end Transformer.Causal

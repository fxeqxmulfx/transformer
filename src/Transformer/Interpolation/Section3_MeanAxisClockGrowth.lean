module

public import Transformer.Interpolation.Section3_MeanAxisLaw

/-!
# Sign preservation and growth of the actual mean-attention clock

ArXiv:2411.04551v3, §3.2, `eq: cauchy.pb.reloaded`, Step 1 of
`lem: induction.barycenter`. The actual axial mean satisfies a scalar
homogeneous ODE with a continuous coefficient. A nonzero mean keeps
its sign for every real time, by ODE uniqueness and the intermediate
value theorem. For a nonnegative control factor, positive means grow
and negative means decrease. Integrating these inequalities gives
linear lower or upper bounds for the original intrinsic clock.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- A positive initial scalar stays positive in a continuous homogeneous
ODE. Auxiliary for arXiv:2411.04551v3, §3.2, the sign argument of Step 1. -/
theorem scalar_pos_of_linearODE (a u : ℝ → ℝ) (ha : Continuous a)
    (hu : ∀ t, HasDerivAt u (a t * u t) t) (h0 : 0 < u 0) (t : ℝ) : 0 < u t := by
  have hn (s : ℝ) : u s ≠ 0 := by
    intro hs
    have hd (r : ℝ) : HasDerivAt (fun r => u (s + r))
        (a (s + r) * u (s + r)) r := by
      simpa only [Function.comp_def, one_mul, mul_one] using
        (hu (s + r)).comp r ((hasDerivAt_id r).const_add s)
    have he := Perspective.scalar_eq_zero_of_linearODE (fun r => a (s + r))
      (fun r => u (s + r)) (ha.comp (continuous_const.add continuous_id)) hd
      (by simpa only [add_zero] using hs) (-s)
    have he0 : u 0 = 0 := by simpa only [add_neg_cancel] using he
    exact (ne_of_gt h0) he0
  have hc : Continuous u := continuous_iff_continuousAt.mpr fun s => (hu s).continuousAt
  by_contra hh
  have hut : u t ≤ 0 := le_of_not_gt hh
  rcases le_total 0 t with ht | ht
  · obtain ⟨s, _, he⟩ := intermediate_value_Icc' ht hc.continuousOn
      (show (0 : ℝ) ∈ Icc (u t) (u 0) from ⟨hut, h0.le⟩)
    exact hn s he
  · obtain ⟨s, _, he⟩ := intermediate_value_Icc ht hc.continuousOn
      (show (0 : ℝ) ∈ Icc (u t) (u 0) from ⟨hut, h0.le⟩)
    exact hn s he

/-- A negative initial scalar stays negative in the same actual ODE.
Auxiliary for arXiv:2411.04551v3, §3.2, Step 1. -/
theorem scalar_neg_of_linearODE (a u : ℝ → ℝ) (ha : Continuous a)
    (hu : ∀ t, HasDerivAt u (a t * u t) t) (h0 : u 0 < 0) (t : ℝ) : u t < 0 := by
  have hd (s : ℝ) : HasDerivAt (fun r => -u r) (a s * (-u s)) s := by
    simpa only [mul_neg] using (hu s).fun_neg
  have hh := scalar_pos_of_linearODE a (fun s => -u s) ha hd (neg_pos.mpr h0) t
  exact neg_pos.mp hh

/-- The original transported mean obeys its actual homogeneous ODE.
Source: arXiv:2411.04551v3, §3.2, `eq: cauchy.pb.reloaded`. -/
theorem hasDerivAt_meanAxisLaw_moment (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) (t : ℝ) :
    HasDerivAt (fun s => meanAxisClockSpeed w ν (meanAxisClock w κ ν s))
      ((κ * deriv (meanAxisClockSpeed w ν) (meanAxisClock w κ ν t)) *
        meanAxisClockSpeed w ν (meanAxisClock w κ ν t)) t := by
  have hd := (hasDerivAt_meanAxisClockSpeed w ν (meanAxisClock w κ ν t)).comp t
    (hasDerivAt_meanAxisClock w κ ν t)
  rw [← (hasDerivAt_meanAxisClockSpeed w ν (meanAxisClock w κ ν t)).deriv] at hd
  convert hd using 1 <;> first | rfl | ring

/-- A positive initial mean remains positive, including for reversed
controls. Source: arXiv:2411.04551v3, §3.2, Step 1. -/
theorem meanAxisLaw_moment_pos (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (h0 : 0 < meanAxisMoment (w : EucSpace d) ν) (t : ℝ) :
    0 < meanAxisMoment (w : EucSpace d) (meanAxisLaw w κ ν t) := by
  have ha : Continuous (fun s => κ * deriv (meanAxisClockSpeed w ν) (meanAxisClock w κ ν s)) :=
    continuous_const.mul (((contDiff_one_iff_deriv.mp (contDiff_meanAxisClockSpeed w ν)).2).comp
      (continuous_meanAxisClock w κ ν))
  exact scalar_pos_of_linearODE _ _ ha (hasDerivAt_meanAxisLaw_moment w κ ν)
    (by rw [meanAxisClock_zero, meanAxisClockSpeed_zero]; exact h0) t

/-- A negative initial mean remains negative at every real time.
Source: arXiv:2411.04551v3, §3.2, Step 1. -/
theorem meanAxisLaw_moment_neg (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (h0 : meanAxisMoment (w : EucSpace d) ν < 0) (t : ℝ) :
    meanAxisMoment (w : EucSpace d) (meanAxisLaw w κ ν t) < 0 := by
  have ha : Continuous (fun s => κ * deriv (meanAxisClockSpeed w ν) (meanAxisClock w κ ν s)) :=
    continuous_const.mul (((contDiff_one_iff_deriv.mp (contDiff_meanAxisClockSpeed w ν)).2).comp
      (continuous_meanAxisClock w κ ν))
  exact scalar_neg_of_linearODE _ _ ha (hasDerivAt_meanAxisLaw_moment w κ ν)
    (by rw [meanAxisClock_zero, meanAxisClockSpeed_zero]; exact h0) t

/-- A nonnegative control increases a positive actual axial mean.
Source: arXiv:2411.04551v3, §3.2, Step 1. -/
theorem monotone_meanAxisLaw_moment (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (hκ : 0 ≤ κ) (h0 : 0 < meanAxisMoment (w : EucSpace d) ν) :
    Monotone (fun t => meanAxisMoment (w : EucSpace d) (meanAxisLaw w κ ν t)) := by
  apply monotone_of_hasDerivAt_nonneg (hasDerivAt_meanAxisLaw_moment w κ ν)
  intro t
  exact mul_nonneg (mul_nonneg hκ (meanAxisClockSpeed_deriv_bounds w ν _).1)
    (meanAxisLaw_moment_pos w κ ν h0 t).le

/-- A nonnegative control decreases a negative actual axial mean.
Source: arXiv:2411.04551v3, §3.2, Step 1. -/
theorem antitone_meanAxisLaw_moment (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (hκ : 0 ≤ κ) (h0 : meanAxisMoment (w : EucSpace d) ν < 0) :
    Antitone (fun t => meanAxisMoment (w : EucSpace d) (meanAxisLaw w κ ν t)) := by
  apply antitone_of_hasDerivAt_nonpos (hasDerivAt_meanAxisLaw_moment w κ ν)
  intro t
  exact mul_nonpos_of_nonneg_of_nonpos (mul_nonneg hκ (meanAxisClockSpeed_deriv_bounds w ν _).1)
    (meanAxisLaw_moment_neg w κ ν h0 t).le

/-- A positive initial mean gives a linear lower bound for intrinsic
time. Source: arXiv:2411.04551v3, §3.2, the attraction argument of Step 1. -/
theorem meanAxisClock_lower_bound (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (hκ : 0 ≤ κ) (h0 : 0 < meanAxisMoment (w : EucSpace d) ν) (t : ℝ) (ht : 0 ≤ t) :
    κ * meanAxisMoment (w : EucSpace d) ν * t ≤ meanAxisClock w κ ν t := by
  let c := κ * meanAxisMoment (w : EucSpace d) ν
  let f (s : ℝ) := κ * meanAxisClockSpeed w ν (meanAxisClock w κ ν s)
  have hf : Continuous f := continuous_const.mul
    ((continuous_meanAxisClockSpeed w ν).comp (continuous_meanAxisClock w κ ν))
  have hbound (s : ℝ) (hs : s ∈ Icc 0 t) : c ≤ f s := by
    have hc := monotone_meanAxisLaw_moment w κ ν hκ h0 hs.1
    change meanAxisMoment (w : EucSpace d) (meanAxisLaw w κ ν 0) ≤
      meanAxisMoment (w : EucSpace d) (meanAxisLaw w κ ν s) at hc
    rw [meanAxisLaw_zero] at hc
    exact mul_le_mul_of_nonneg_left hc hκ
  have hi := intervalIntegral.integral_mono_on ht (intervalIntegrable_const : IntervalIntegrable (fun _ : ℝ => c) volume 0 t)
    (hf.intervalIntegrable 0 t) hbound
  have he := intervalIntegral.integral_eq_sub_of_hasDerivAt
    (fun s _ => hasDerivAt_meanAxisClock w κ ν s) (hf.intervalIntegrable 0 t)
  rw [meanAxisClock_zero, sub_zero] at he
  rw [he] at hi
  simpa only [intervalIntegral.integral_const, sub_zero, smul_eq_mul, mul_comm] using hi

/-- A negative initial mean gives a linear upper bound for intrinsic
time. Source: arXiv:2411.04551v3, §3.2, the attraction argument of Step 1. -/
theorem meanAxisClock_upper_bound (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (hκ : 0 ≤ κ) (h0 : meanAxisMoment (w : EucSpace d) ν < 0) (t : ℝ) (ht : 0 ≤ t) :
    meanAxisClock w κ ν t ≤ κ * meanAxisMoment (w : EucSpace d) ν * t := by
  let c := κ * meanAxisMoment (w : EucSpace d) ν
  let f (s : ℝ) := κ * meanAxisClockSpeed w ν (meanAxisClock w κ ν s)
  have hf : Continuous f := continuous_const.mul
    ((continuous_meanAxisClockSpeed w ν).comp (continuous_meanAxisClock w κ ν))
  have hbound (s : ℝ) (hs : s ∈ Icc 0 t) : f s ≤ c := by
    have hc := antitone_meanAxisLaw_moment w κ ν hκ h0 hs.1
    change meanAxisMoment (w : EucSpace d) (meanAxisLaw w κ ν s) ≤
      meanAxisMoment (w : EucSpace d) (meanAxisLaw w κ ν 0) at hc
    rw [meanAxisLaw_zero] at hc
    exact mul_le_mul_of_nonneg_left hc hκ
  have hi := intervalIntegral.integral_mono_on ht (hf.intervalIntegrable 0 t)
    (intervalIntegrable_const : IntervalIntegrable (fun _ : ℝ => c) volume 0 t) hbound
  have he := intervalIntegral.integral_eq_sub_of_hasDerivAt
    (fun s _ => hasDerivAt_meanAxisClock w κ ν s) (hf.intervalIntegrable 0 t)
  rw [meanAxisClock_zero, sub_zero] at he
  rw [he] at hi
  simpa only [intervalIntegral.integral_const, sub_zero, smul_eq_mul, mul_comm] using hi

/-- Nonconstant positive and negative scalar solutions realize both
homogeneous ODE hypotheses with an actual continuous coefficient. -/
example : Continuous (fun _ : ℝ => (1 : ℝ)) ∧
    (∀ t : ℝ, HasDerivAt exp (1 * exp t) t) ∧ 0 < exp 0 ∧
    (∀ t : ℝ, HasDerivAt (fun s => -exp s) (1 * (-exp t)) t) ∧ -exp 0 < 0 := by
  exact ⟨continuous_const, fun t => by simpa using hasDerivAt_exp t, exp_pos 0,
    fun t => by simpa using (hasDerivAt_exp t).fun_neg, neg_neg_of_pos (exp_pos 0)⟩

/-- Opposite actual Dirac laws realize the positive and negative mean
hypotheses with a positive control factor and positive observation time. -/
example : ∃ (w : SSphere 2) (νPos νNeg : Perspective.ProbSphere 2) (κ t : ℝ),
    0 ≤ κ ∧ 0 < t ∧ 0 < meanAxisMoment (w : EucSpace 2) νPos ∧
      meanAxisMoment (w : EucSpace 2) νNeg < 0 := by
  let w : SSphere 2 := basePoint 1
  refine ⟨w, Perspective.diracProb 2 w, Perspective.diracProb 2 (antipode 2 w), 1, 1,
    zero_le_one, zero_lt_one, ?_, ?_⟩
  · rw [meanAxisMoment_eq_integral]
    change (∫ y : SSphere 2, inner ℝ (w : EucSpace 2) (y : EucSpace 2) ∂Measure.dirac w) > 0
    rw [integral_dirac, real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp w.2]
    norm_num
  · rw [meanAxisMoment_eq_integral]
    change (∫ y : SSphere 2, inner ℝ (w : EucSpace 2) (y : EucSpace 2)
      ∂Measure.dirac (antipode 2 w)) < 0
    rw [integral_dirac]
    change inner ℝ (w : EucSpace 2) (-(w : EucSpace 2)) < 0
    rw [inner_neg_right, real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp w.2]
    norm_num

end Transformer.Interpolation

/-
# Complete scalar clocks for original rank-one mean attention

ArXiv:2411.04551v3, §3.2, `eq: cauchy.pb.reloaded`, proof of
`lem: induction.barycenter`, Steps 1 and 3. The scalar equation
`q' = κ a(q)`, `q(0)=0`, has a bounded globally Lipschitz right-hand
side, where `a` is the actual transported mean component.

The complete clock is chosen from the proved global ODE construction.
Its original derivative, initial value and uniqueness are proved. The
zero initial mean gives the zero clock, and a Dirac law on the axis
gives the actual clock `q(t)=κ t`. Negative factors are retained for
the source's reversal of the attention phase.
-/

import Transformer.Interpolation.Section3_MeanAxisClockSpeed
import Transformer.Landscape.GlobalFlow.TimeDependent

open Real MeasureTheory Set
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The actual scalar clock field is globally Lipschitz.
Auxiliary for arXiv:2411.04551v3, §3.2, `eq: cauchy.pb.reloaded`. -/
theorem lipschitzWith_meanAxis_clock_field (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) :
    LipschitzWith ‖κ‖₊ (fun q => κ * meanAxisClockSpeed w ν q) := by
  apply LipschitzWith.of_dist_le_mul
  intro p q
  rw [Real.dist_eq, Real.dist_eq, ← mul_sub, abs_mul]
  change |κ| * |meanAxisClockSpeed w ν p - meanAxisClockSpeed w ν q| ≤ |κ| * |p - q|
  have hh := (lipschitzWith_meanAxisClockSpeed w ν).dist_le_mul p q
  rw [Real.dist_eq, Real.dist_eq] at hh
  simp only [NNReal.coe_one, one_mul] at hh
  exact mul_le_mul_of_nonneg_left hh (abs_nonneg κ)

/-- A complete actual scalar clock exists for every original initial
probability law. Auxiliary for arXiv:2411.04551v3, §3.2, Step 1. -/
theorem exists_meanAxisClock (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) :
    ∃ q : ℝ → ℝ, q 0 = 0 ∧ ∀ t, HasDerivAt q (κ * meanAxisClockSpeed w ν (q t)) t := by
  apply Landscape.GlobalFlow.exists_timeDependent_global (fun _ q => κ * meanAxisClockSpeed w ν q)
    ‖κ‖₊ |κ| (abs_nonneg κ) (fun _ => lipschitzWith_meanAxis_clock_field w κ ν)
    (fun _ => continuous_const) ?_ 0
  intro t q
  rw [Real.norm_eq_abs, abs_mul]
  exact (mul_le_mul_of_nonneg_left (abs_meanAxisClockSpeed_le_one w ν q) (abs_nonneg κ)).trans_eq (mul_one _)

/-- The actual complete solution of the rank-one attention clock ODE.
Source: arXiv:2411.04551v3, §3.2, characteristics of Step 1. -/
noncomputable def meanAxisClock (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) : ℝ → ℝ :=
  (exists_meanAxisClock w κ ν).choose

/-- The clock starts at intrinsic time zero. Auxiliary for
arXiv:2411.04551v3, §3.2, `eq: cauchy.pb.reloaded`. -/
theorem meanAxisClock_zero (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) :
    meanAxisClock w κ ν 0 = 0 := (exists_meanAxisClock w κ ν).choose_spec.1

/-- The chosen clock satisfies the actual nonlinear scalar equation.
Source: arXiv:2411.04551v3, §3.2, characteristics of Step 1. -/
theorem hasDerivAt_meanAxisClock (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) (t : ℝ) :
    HasDerivAt (meanAxisClock w κ ν) (κ * meanAxisClockSpeed w ν (meanAxisClock w κ ν t)) t :=
  (exists_meanAxisClock w κ ν).choose_spec.2 t

/-- Every complete clock solution with the same initial value agrees.
Auxiliary for arXiv:2411.04551v3, §3.2, flow reversal. -/
theorem meanAxisClock_eq_of_solution (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (q : ℝ → ℝ) (h0 : q 0 = 0) (hq : ∀ t, HasDerivAt q (κ * meanAxisClockSpeed w ν (q t)) t) :
    meanAxisClock w κ ν = q :=
  Landscape.GlobalFlow.timeDependent_unique (fun _ q => κ * meanAxisClockSpeed w ν q) ‖κ‖₊
    (fun _ => lipschitzWith_meanAxis_clock_field w κ ν) _ q (hasDerivAt_meanAxisClock w κ ν)
    hq ((meanAxisClock_zero w κ ν).trans h0.symm)

/-- The complete clock is continuous. Auxiliary for
arXiv:2411.04551v3, §3.2, the original characteristic construction. -/
theorem continuous_meanAxisClock (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) :
    Continuous (meanAxisClock w κ ν) :=
  continuous_iff_continuousAt.mpr fun t => (hasDerivAt_meanAxisClock w κ ν t).continuousAt

/-- The original clock is C1. Auxiliary for arXiv:2411.04551v3,
§3.2, differentiating the actual transported weak law. -/
theorem contDiff_meanAxisClock (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) :
    ContDiff ℝ 1 (meanAxisClock w κ ν) := by
  refine contDiff_one_iff_deriv.mpr ⟨fun t => (hasDerivAt_meanAxisClock w κ ν t).differentiableAt, ?_⟩
  have he : deriv (meanAxisClock w κ ν) = fun t => κ * meanAxisClockSpeed w ν (meanAxisClock w κ ν t) :=
    funext fun t => (hasDerivAt_meanAxisClock w κ ν t).deriv
  rw [he]
  exact continuous_const.mul ((continuous_meanAxisClockSpeed w ν).comp (continuous_meanAxisClock w κ ν))

/-- The factor `|κ|` bounds the clock's actual speed for all times.
Auxiliary for arXiv:2411.04551v3, §3.2, time rescaling. -/
theorem lipschitzWith_meanAxisClock (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) :
    LipschitzWith ‖κ‖₊ (meanAxisClock w κ ν) := by
  apply lipschitzWith_of_nnnorm_deriv_le (𝕜 := ℝ)
    (fun t => (hasDerivAt_meanAxisClock w κ ν t).differentiableAt)
  intro t
  apply NNReal.coe_le_coe.mp
  rw [coe_nnnorm, coe_nnnorm, (hasDerivAt_meanAxisClock w κ ν t).deriv, Real.norm_eq_abs, Real.norm_eq_abs, abs_mul]
  exact (mul_le_mul_of_nonneg_left (abs_meanAxisClockSpeed_le_one w ν _) (abs_nonneg κ)).trans_eq (mul_one _)

/-- A vanishing initial mean component gives the zero intrinsic clock.
Source: arXiv:2411.04551v3, §3.2, the invariant laws of Step 1. -/
theorem meanAxisClock_of_moment_eq_zero (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (h0 : meanAxisMoment (w : EucSpace d) ν = 0) : meanAxisClock w κ ν = fun _ => 0 := by
  apply meanAxisClock_eq_of_solution w κ ν _ rfl
  intro t
  rw [meanAxisClockSpeed_zero, h0, mul_zero]
  exact hasDerivAt_const t 0

/-- The zero control leaves the intrinsic clock at zero.
Auxiliary for arXiv:2411.04551v3, §3.2, time rescaling. -/
theorem meanAxisClock_of_zero_factor (w : SSphere d) (ν : Perspective.ProbSphere d) :
    meanAxisClock w 0 ν = fun _ => 0 := by
  apply meanAxisClock_eq_of_solution w 0 ν _ rfl
  intro t
  rw [zero_mul]
  exact hasDerivAt_const t 0

/-- The mean coefficient of an axis Dirac law is exactly one.
Auxiliary for arXiv:2411.04551v3, §3.2, `eq: cauchy.pb.reloaded`. -/
theorem meanAxisClockSpeed_dirac_axis (w : SSphere d) (q : ℝ) :
    meanAxisClockSpeed w (Perspective.diracProb d w) q = 1 := by
  rw [meanAxisClockSpeed, projectionLaw_dirac_axis, meanAxisMoment_eq_integral]
  change (∫ y : SSphere d, inner ℝ (w : EucSpace d) (y : EucSpace d) ∂Measure.dirac w) = 1
  rw [integral_dirac, real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp w.2]
  norm_num

/-- An axis Dirac law gives the nonconstant clock `κ t`.
Auxiliary for arXiv:2411.04551v3, §3.2, characteristics of Step 1. -/
theorem meanAxisClock_dirac_axis (w : SSphere d) (κ : ℝ) :
    meanAxisClock w κ (Perspective.diracProb d w) = fun t => κ * t := by
  apply meanAxisClock_eq_of_solution w κ _ _ (mul_zero κ)
  intro t
  rw [meanAxisClockSpeed_dirac_axis, mul_one]
  convert (hasDerivAt_id t).const_mul κ using 1 <;> simp

/-- A transverse Dirac law realizes the zero-mean hypothesis. An axis
Dirac law and the actual nonconstant clock `q(t)=t` jointly realize the
clock-equation and initial-value hypotheses. -/
example : ∃ (w : SSphere 2) (ν₀ ν₁ : Perspective.ProbSphere 2) (q : ℝ → ℝ),
    meanAxisMoment (w : EucSpace 2) ν₀ = 0 ∧ q 0 = 0 ∧
    (∀ t, HasDerivAt q ((1 : ℝ) * meanAxisClockSpeed w ν₁ (q t)) t) := by
  let w : SSphere 2 := basePoint 1
  let z : SSphere 2 := ⟨EuclideanSpace.single 1 1, by simp⟩
  refine ⟨w, Perspective.diracProb 2 z, Perspective.diracProb 2 w, id, ?_, rfl, ?_⟩
  · rw [meanAxisMoment_eq_integral]
    change (∫ y : SSphere 2, inner ℝ (w : EucSpace 2) (y : EucSpace 2) ∂Measure.dirac z) = 0
    rw [integral_dirac]
    simp [w, z, basePoint, EuclideanSpace.inner_single_left]
  · intro t
    rw [meanAxisClockSpeed_dirac_axis, mul_one]
    exact hasDerivAt_id t

end Transformer.Interpolation

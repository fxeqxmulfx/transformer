/-
# Continuity and quantiles of the actual ball mass

ArXiv:2411.04551v3, §5, proof of `cl: balls`, the function
`f(s,r) = ν(B(γ(s),r))`. The manuscript uses absolute continuity to
make every ball boundary null, then chooses radii and centers by the
intermediate value theorem. These statements prove that measure step
for actual metric balls, with the weaker null-sphere hypothesis.

The center and radius vary jointly. At a fixed parameter pair the ball
indicators are locally constant away from its boundary; dominated
convergence therefore gives continuity of their actual integrals.
The quantile retains the original probability and the prescribed mass.
No nesting or separation property of the later construction is assumed
or encoded in the definition of that mass.
-/

import Transformer.Basic
import Mathlib.MeasureTheory.Integral.DominatedConvergence
import Mathlib.MeasureTheory.Integral.Bochner.Set
import Mathlib.MeasureTheory.Constructions.BorelSpace.Real
import Mathlib.MeasureTheory.Measure.Typeclasses.NullSingletonClass
import Mathlib.Topology.Order.IntermediateValue

open Real MeasureTheory Metric Set Filter

namespace Transformer.Interpolation

variable {X : Type*} [MetricSpace X] [MeasurableSpace X] [BorelSpace X]

/-- Null mass on the actual boundary gives joint continuity of the
actual open-ball mass at its center and radius.
Source: arXiv:2411.04551v3, §5, `cl: balls`, continuity of `f(s,r)`. -/
theorem continuousAt_ball_mass (ν : Measure X) [IsFiniteMeasure ν]
    (y : X) (r : ℝ) (hboundary : ν (sphere y r) = 0) :
    ContinuousAt (fun p : X × ℝ => ν.real (ball p.1 p.2)) (y, r) := by
  let F : X × ℝ → X → ℝ := fun p => (ball p.1 p.2).indicator 1
  have hF : ∀ p, AEStronglyMeasurable (F p) ν := fun p =>
    (measurable_const.indicator measurableSet_ball).aestronglyMeasurable
  have hbound : ∀ p a, ‖F p a‖ ≤ 1 := by
    intro p a
    by_cases ha : a ∈ ball p.1 p.2 <;> simp [F, ha]
  have hne : ∀ᵐ a ∂ν, dist a y ≠ r := by
    simpa only [ae_iff, not_not, sphere, mem_ofPred_eq] using hboundary
  have hcont : ∀ᵐ a ∂ν, ContinuousAt (fun p => F p a) (y, r) := by
    filter_upwards [hne] with a ha
    have hc : Continuous (fun p : X × ℝ => dist a p.1) := continuous_const.dist continuous_fst
    by_cases hlt : dist a y < r
    · have hev := (hc.continuousAt (x := (y, r))).eventually_lt
        continuous_snd.continuousAt hlt
      apply (continuousAt_const : ContinuousAt (fun _ : X × ℝ => (1 : ℝ)) (y, r)).congr_of_eventuallyEq
      exact hev.mono fun p hp => by simp [F, mem_ball, hp]
    · have hgt : r < dist a y := lt_of_le_of_ne (le_of_not_gt hlt) ha.symm
      have hev := (continuous_snd.continuousAt (x := (y, r))).eventually_lt
        hc.continuousAt hgt
      apply (continuousAt_const : ContinuousAt (fun _ : X × ℝ => (0 : ℝ)) (y, r)).congr_of_eventuallyEq
      exact hev.mono fun p hp => by simp [F, mem_ball, not_lt_of_gt hp]
  have h := continuousAt_of_dominated (Eventually.of_forall hF)
    (Eventually.of_forall fun p => ae_of_all ν (hbound p)) (integrable_const (1 : ℝ)) hcont
  simpa only [F, integral_indicator_one measurableSet_ball] using h

/-- Null spheres at every center and radius give global joint
continuity of the original ball mass. Source: arXiv:2411.04551v3,
§5, `cl: balls`, first property of `f(s,r)`. -/
theorem continuous_ball_mass (ν : Measure X) [IsFiniteMeasure ν]
    (hboundary : ∀ y : X, ∀ r : ℝ, ν (sphere y r) = 0) :
    Continuous (fun p : X × ℝ => ν.real (ball p.1 p.2)) :=
  continuous_iff_continuousAt.mpr fun p => continuousAt_ball_mass ν p.1 p.2 (hboundary p.1 p.2)

/-- Along an actual continuous center path, ball mass varies
continuously with both path time and radius. Source:
arXiv:2411.04551v3, §5, the geodesic `γ` in the proof of `cl: balls`. -/
theorem continuous_ball_mass_along {Y : Type*} [TopologicalSpace Y]
    (ν : Measure X) [IsFiniteMeasure ν]
    (hboundary : ∀ y : X, ∀ r : ℝ, ν (sphere y r) = 0)
    (γ : Y → X) (hγ : Continuous γ) :
    Continuous (fun p : Y × ℝ => ν.real (ball (γ p.1) p.2)) :=
  (continuous_ball_mass ν hboundary).comp ((hγ.comp continuous_fst).prodMk continuous_snd)

/-- Every intermediate original mass is attained by an original ball
at a strictly positive radius when a larger ball carries full mass.
Source: arXiv:2411.04551v3, §5, `cl: balls`, choosing `rⁱ` by continuity.
Only null spheres at this fixed center are needed. -/
theorem exists_ball_of_prescribed_mass (ν : Measure X) [IsProbabilityMeasure ν]
    (y : X) (hboundary : ∀ r : ℝ, ν (sphere y r) = 0)
    (R : ℝ) (hR : 0 < R) (hfull : ν (ball y R) = 1)
    (q : ℝ) (hq : 0 < q) (hq1 : q ≤ 1) :
    ∃ r : ℝ, 0 < r ∧ r ≤ R ∧ ν (ball y r) = ENNReal.ofReal q := by
  have hc : Continuous (fun r : ℝ => ν.real (ball y r)) := by
    apply continuous_iff_continuousAt.mpr
    intro r
    exact (continuousAt_ball_mass ν y r (hboundary r)).comp
      (continuous_const.prodMk continuous_id).continuousAt
  have h0 : ν.real (ball y 0) = 0 := by simp
  have h1 : ν.real (ball y R) = 1 := by rw [measureReal_def, hfull, ENNReal.toReal_one]
  obtain ⟨r, hr, heq⟩ := intermediate_value_Icc hR.le hc.continuousOn
    (show q ∈ Icc (ν.real (ball y 0)) (ν.real (ball y R)) by rw [h0, h1]; exact ⟨hq.le, hq1⟩)
  change ν.real (ball y r) = q at heq
  have hrpos : 0 < r := by
    by_contra h
    have hr0 : r = 0 := le_antisymm (le_of_not_gt h) hr.1
    rw [hr0, h0] at heq
    exact hq.ne' heq.symm
  refine ⟨r, hrpos, hr.2, ?_⟩
  rw [← heq, measureReal_def, ENNReal.ofReal_toReal (measure_ne_top _ _)]

/-- The actual quantile mass `1/M` is positive for every positive token
count. Source: arXiv:2411.04551v3, §5, first ball in `cl: balls`.
This includes `M = 1`, where the carrying ball itself is admissible. -/
theorem exists_ball_of_mass_inv_nat (ν : Measure X) [IsProbabilityMeasure ν]
    (y : X) (hboundary : ∀ r : ℝ, ν (sphere y r) = 0)
    (R : ℝ) (hR : 0 < R) (hfull : ν (ball y R) = 1)
    (M : ℕ) (hM : 0 < M) :
    ∃ r : ℝ, 0 < r ∧ r ≤ R ∧ ν (ball y r) = ENNReal.ofReal ((M : ℝ)⁻¹) := by
  have hMreal : (0 : ℝ) < M := by exact_mod_cast hM
  exact exists_ball_of_prescribed_mass ν y hboundary R hR hfull _ (by positivity)
    (inv_le_one_of_one_le₀ (by exact_mod_cast hM))

/-- An atomless finite real measure has null actual spheres, since
each sphere has at most its two genuine endpoints. Source:
arXiv:2411.04551v3, §5, the null-boundary condition behind `cl: balls`. -/
theorem real_sphere_null (ν : Measure ℝ) [NullSingletonClass ν] (y r : ℝ) : ν (sphere y r) = 0 := by
  rcases le_or_gt 0 r with hr | hr
  · rw [Real.sphere_eq_pair y hr]
    exact ((finite_singleton (y + r)).insert (y - r)).measure_zero ν
  · rw [sphere_eq_empty_of_neg hr, measure_empty]

/-- The continuity and quantile hypotheses hold jointly for genuine
unit-interval probability, with every original real center and radius.
Source: arXiv:2411.04551v3, §5, the continuous diffuse mass function. -/
example : let ν := (volume : Measure ℝ).restrict (Icc (0 : ℝ) 1)
    IsProbabilityMeasure ν ∧
      (∀ y r : ℝ, ν (sphere y r) = 0) ∧ (0 : ℝ) < 2 ∧
      ν (ball (1 / 2 : ℝ) 2) = 1 ∧ (0 : ℝ) < 1 / 3 ∧ (1 / 3 : ℝ) ≤ 1 ∧
      ∃ r : ℝ, 0 < r ∧ r ≤ 2 ∧
        ν (ball (1 / 2 : ℝ) r) = ENNReal.ofReal ((3 : ℝ)⁻¹) := by
  intro ν
  have hprob : IsProbabilityMeasure ν := ⟨by simp [ν]⟩
  have hsub : Icc (0 : ℝ) 1 ⊆ ball (1 / 2 : ℝ) 2 := by
    intro x hx
    rw [mem_ball, Real.dist_eq, abs_lt]
    constructor <;> linarith [hx.1, hx.2]
  have hfull : ν (ball (1 / 2 : ℝ) 2) = 1 := by
    rw [Measure.restrict_apply measurableSet_ball, inter_eq_right.mpr hsub]
    simp
  refine ⟨hprob, real_sphere_null ν, by norm_num, hfull, by norm_num, by norm_num, ?_⟩
  exact exists_ball_of_mass_inv_nat ν (1 / 2) (real_sphere_null ν (1 / 2))
    2 (by norm_num) hfull 3 (by decide)

end Transformer.Interpolation

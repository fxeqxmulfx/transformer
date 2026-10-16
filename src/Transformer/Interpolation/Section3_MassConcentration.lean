/-
# Mass concentration under actual spherical attraction

Auxiliaries for arXiv:2411.04551v3, Appendix, `lem: two.balls`,
`eq: claim.statement.lemmaA1` and the Wasserstein concentration conclusion.

Almost-everywhere attraction inside a measurable active set, together with
exact fixation of its complement, makes the transported mass of any open
target neighborhood inside that set converge to its original active mass.
The proof applies dominated convergence to genuine indicator functions.

The resulting finite-time mass bound holds even when the active mass is
zero. It can be achieved simultaneously with the quadratic Wasserstein
bound, with one strictly positive terminal time for the given initial law.
-/

import Transformer.Interpolation.Section3_CollapseLaws
import Mathlib.MeasureTheory.Integral.DominatedConvergence

open Real Set MeasureTheory Filter
open scoped Topology ENNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Transported mass in an open target neighborhood converges to the
original active mass. Auxiliary for arXiv:2411.04551v3, Appendix,
`eq: claim.statement.lemmaA1`. -/
theorem tendsto_map_mass_of_attraction (ν : Perspective.ProbSphere d)
    (Φ : ℝ → SSphere d → SSphere d)
    (hΦ : ∀ t : ℝ, Measurable (Φ t)) (S A : Set (SSphere d)) (hS : MeasurableSet S)
    (hA : IsOpen A) (hAS : A ⊆ S) (ω : SSphere d) (hω : ω ∈ A)
    (hactive : ∀ᵐ x ∂(ν : Measure (SSphere d)),
      x ∈ S → Tendsto (fun t => Φ t x) atTop (nhds ω))
    (hinactive : ∀ x ∉ S, ∀ t : ℝ, Φ t x = x) :
    Tendsto (fun t : ℝ => (Measure.map (Φ t) (ν : Measure (SSphere d))).real A)
      atTop (nhds ((ν : Measure (SSphere d)).real S)) := by
  classical
  let F (t : ℝ) (x : SSphere d) : ℝ := A.indicator 1 (Φ t x)
  have hlim : ∀ᵐ x ∂(ν : Measure (SSphere d)),
      Tendsto (fun t => F t x) atTop (nhds (S.indicator (1 : SSphere d → ℝ) x)) :=
    hactive.mono fun x hxlim => by
      by_cases hx : x ∈ S
      · have he := (hxlim hx).eventually (hA.mem_nhds hω)
        have heq : (fun _ : ℝ => (1 : ℝ)) =ᶠ[atTop] (fun t => F t x) :=
          he.mono fun t ht => by simp only [F, indicator_of_mem ht, Pi.one_apply]
        simpa only [indicator_of_mem hx, Pi.one_apply] using tendsto_const_nhds.congr' heq
      · have hxA : x ∉ A := fun h => hx (hAS h)
        simpa only [F, hinactive x hx, indicator_of_notMem hxA, indicator_of_notMem hx] using
          (tendsto_const_nhds : Tendsto (fun _ : ℝ => (0 : ℝ)) atTop (nhds 0))
  have hbound (t : ℝ) (x : SSphere d) : ‖F t x‖ ≤ (1 : ℝ) := by
    by_cases hx : Φ t x ∈ A <;> simp [F, hx]
  have hl := tendsto_integral_filter_of_dominated_convergence (μ := (ν : Measure (SSphere d)))
    (F := F) (f := S.indicator (1 : SSphere d → ℝ)) (fun _ => (1 : ℝ))
    (Eventually.of_forall fun t =>
      (((measurable_const : Measurable (1 : SSphere d → ℝ)).indicator hA.measurableSet).comp (hΦ t)).aestronglyMeasurable)
    (Eventually.of_forall fun t => Eventually.of_forall (hbound t)) (integrable_const 1) hlim
  have heq (t : ℝ) : (∫ x, F t x ∂(ν : Measure (SSphere d))) =
      (Measure.map (Φ t) (ν : Measure (SSphere d))).real A := by
    rw [Measure.real, Measure.map_apply (hΦ t) hA.measurableSet]
    change (∫ x, F t x ∂(ν : Measure (SSphere d))) = (ν : Measure (SSphere d)).real (Φ t ⁻¹' A)
    rw [← integral_indicator_one (μ := (ν : Measure (SSphere d))) (hA.measurableSet.preimage (hΦ t))]
    apply integral_congr_ae
    exact Eventually.of_forall fun x => by
      by_cases hx : Φ t x ∈ A <;> simp [F, hx]
  simpa only [heq, integral_indicator_one hS] using hl

/-- For every positive accuracy at most one, all sufficiently late times
retain the requested fraction of active mass in the open target region.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem eventually_map_mass_ge (ν : Perspective.ProbSphere d)
    (Φ : ℝ → SSphere d → SSphere d)
    (hΦ : ∀ t : ℝ, Measurable (Φ t)) (S A : Set (SSphere d)) (hS : MeasurableSet S)
    (hA : IsOpen A) (hAS : A ⊆ S) (ω : SSphere d) (hω : ω ∈ A)
    (hactive : ∀ᵐ x ∂(ν : Measure (SSphere d)),
      x ∈ S → Tendsto (fun t => Φ t x) atTop (nhds ω))
    (hinactive : ∀ x ∉ S, ∀ t : ℝ, Φ t x = x) (ε : ℝ) (hε : 0 < ε) (hε1 : ε ≤ 1) :
    ∀ᶠ t : ℝ in atTop, ENNReal.ofReal (1 - ε) * (ν : Measure (SSphere d)) S ≤
      Measure.map (Φ t) (ν : Measure (SSphere d)) A := by
  have hν : ENNReal.ofReal ((ν : Measure (SSphere d)).real S) = (ν : Measure (SSphere d)) S :=
    ENNReal.ofReal_toReal (measure_ne_top _ _)
  by_cases hzero : (ν : Measure (SSphere d)).real S = 0
  · apply Eventually.of_forall
    intro t
    rw [← hν, hzero, ENNReal.ofReal_zero, mul_zero]
    positivity
  have hpos : 0 < (ν : Measure (SSphere d)).real S :=
    lt_of_le_of_ne ENNReal.toReal_nonneg (Ne.symm hzero)
  have hlt : (1 - ε) * (ν : Measure (SSphere d)).real S < (ν : Measure (SSphere d)).real S := by
    nlinarith
  have hl := (tendsto_map_mass_of_attraction ν Φ hΦ S A hS hA hAS ω hω hactive hinactive).eventually
    (lt_mem_nhds hlt)
  filter_upwards [hl] with t ht
  calc
    _ = ENNReal.ofReal ((1 - ε) * (ν : Measure (SSphere d)).real S) := by
      rw [ENNReal.ofReal_mul (by linarith : 0 ≤ 1 - ε), hν]
    _ ≤ ENNReal.ofReal ((Measure.map (Φ t) (ν : Measure (SSphere d))).real A) :=
      ENNReal.ofReal_le_ofReal ht.le
    _ = _ := ENNReal.ofReal_toReal (measure_ne_top _ _)

/-- One positive time gives both the source's mass bound and its original
quadratic Wasserstein bound for the specified initial probability.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem exists_time_mass_and_W2 (ν : Perspective.ProbSphere d)
    (Φ : ℝ → SSphere d → SSphere d)
    (hΦ : ∀ t : ℝ, Measurable (Φ t)) (S A : Set (SSphere d)) (hS : MeasurableSet S)
    (hA : IsOpen A) (hAS : A ⊆ S) (ω : SSphere d) (hω : ω ∈ A)
    (hactive : ∀ᵐ x ∂(ν : Measure (SSphere d)),
      x ∈ S → Tendsto (fun t => Φ t x) atTop (nhds ω))
    (hinactive : ∀ x ∉ S, ∀ t : ℝ, Φ t x = x) (ε : ℝ) (hε : 0 < ε) (hε1 : ε ≤ 1) :
    ∃ t : ℝ, 0 < t ∧
      ENNReal.ofReal (1 - ε) * (ν : Measure (SSphere d)) S ≤
        Measure.map (Φ t) (ν : Measure (SSphere d)) A ∧
      W2 d (Measure.map (Φ t) (ν : Measure (SSphere d))) (collapseInto d ν S ω) ≤ ε := by
  have hmass := eventually_map_mass_ge ν Φ hΦ S A hS hA hAS ω hω hactive hinactive ε hε hε1
  have hw := (tendsto_W2_map_collapse ν Φ hΦ S hS ω hactive hinactive).eventually (gt_mem_nhds hε)
  obtain ⟨t, hm, hwt, ht⟩ := (hmass.and (hw.and (eventually_gt_atTop (0 : ℝ)))).exists
  exact ⟨t, ht, hm, hwt.le⟩

/-- An actual inward ReLU flow, an open nonempty hemisphere, a target
inside it and a two-atom initial probability jointly satisfy the mass and
Wasserstein hypotheses, with positive accuracy at most one. -/
example : ∃ (ν : Perspective.ProbSphere 2) (Φ : ℝ → SSphere 2 → SSphere 2)
    (S A : Set (SSphere 2)) (ω : SSphere 2) (ε : ℝ),
    (∀ t : ℝ, Measurable (Φ t)) ∧ MeasurableSet S ∧ IsOpen A ∧ A ⊆ S ∧ ω ∈ A ∧
    (∀ᵐ x ∂(ν : Measure (SSphere 2)), x ∈ S → Tendsto (fun t => Φ t x) atTop (nhds ω)) ∧
    (∀ x ∉ S, ∀ t : ℝ, Φ t x = x) ∧ 0 < ε ∧ ε ≤ 1 := by
  let ω : SSphere 2 := basePoint 1
  let S := {x : SSphere 2 | 0 < inner ℝ (x : EucSpace 2) (ω : EucSpace 2)}
  let Φ (t : ℝ) (x : SSphere 2) :=
    ridgeFlow (ω : EucSpace 2) (-(ω : EucSpace 2)) 0 x t
  have ho : IsOpen S := isOpen_lt continuous_const (continuous_subtype_val.inner continuous_const)
  refine ⟨halfDirac 2 ω (antipode 2 ω), Φ, S, S, ω, 1 / 2,
    fun t => measurable_ridgeFlow_initial _ _ _ t, ho.measurableSet, ho,
    Subset.rfl, ?_, ?_, ?_, by norm_num, by norm_num⟩
  · change 0 < inner ℝ (ω : EucSpace 2) (ω : EucSpace 2)
    norm_num [ω, basePoint]
  · apply Eventually.of_forall
    intro x hx
    change 0 < inner ℝ (x : EucSpace 2) (ω : EucSpace 2) at hx
    simpa only [Φ, neg_zero] using tendsto_ridge_cap_center ω x 0 (by norm_num)
      (by simpa only [sub_zero] using hx)
  · intro x hx t
    apply ridgeFlow_inactive
    rw [inner_neg_right]
    change ¬0 < inner ℝ (x : EucSpace 2) (ω : EucSpace 2) at hx
    linarith

end Transformer.Interpolation

/-
# From almost-everywhere spherical attraction to a collapsed probability

Auxiliaries for arXiv:2411.04551v3, Appendix, `lem: two.balls`,
the final mass-collapse and Wasserstein concentration argument.

The collapse map sends the active set to the prescribed target and fixes
its complement. Its exact pushforward is the source's measure `α`.
Pairing that map with a measurable transported map gives a genuine Monge
coupling and bounds the original quadratic Wasserstein distance.

If almost every active particle tends to the target and every inactive
particle is fixed, dominated convergence makes the coupling cost tend to
zero. The sphere supplies the uniform squared-distance bound `4`.
No atomicity assumption on the initial probability is made.
-/

import Transformer.Interpolation.Section3_BallFlow
import Transformer.Interpolation.Monge
import Transformer.Interpolation.Section3_CapCenter
import Mathlib.MeasureTheory.Integral.DominatedConvergence

open Real Set MeasureTheory Filter
open scoped Topology ENNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Unit vectors have chordal distance at most `2`. Auxiliary for
arXiv:2411.04551v3, Appendix, the quadratic cost in `lem: two.balls`. -/
theorem norm_spherical_sub_le_two (x y : SSphere d) :
    ‖(x : EucSpace d) - (y : EucSpace d)‖ ≤ 2 := by
  have hn := norm_sub_le (x : EucSpace d) (y : EucSpace d)
  rwa [mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp y.2,
    show (1 : ℝ) + 1 = 2 by norm_num] at hn

/-- The genuine squared chordal cost is bounded by `4` on the sphere.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem spherical_squared_distance_le_four (x y : SSphere d) :
    ‖(x : EucSpace d) - (y : EucSpace d)‖ ^ 2 ≤ 4 := by
  have hn := norm_spherical_sub_le_two x y
  nlinarith [norm_nonneg ((x : EucSpace d) - (y : EucSpace d))]

/-- Antipodal unit vectors attain the quadratic bound exactly. -/
example : ‖(basePoint 1 : EucSpace 2) - (antipode 2 (basePoint 1) : EucSpace 2)‖ ^ 2 = 4 := by
  let z : SSphere 2 := basePoint 1
  have he : (z : EucSpace 2) - (antipode 2 z : EucSpace 2) = (2 : ℝ) • (z : EucSpace 2) := by
    change (z : EucSpace 2) - (-(z : EucSpace 2)) = (2 : ℝ) • (z : EucSpace 2)
    module
  change ‖(z : EucSpace 2) - (antipode 2 z : EucSpace 2)‖ ^ 2 = 4
  rw [he, norm_smul, mem_sphere_zero_iff_norm.mp z.2]
  norm_num

/-- A genuine Monge coupling bounds the original distance to `α`.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem W2_map_le_collapse_cost (ν : Perspective.ProbSphere d) (Φ : SSphere d → SSphere d)
    (hΦ : Measurable Φ) (S : Set (SSphere d)) (hS : MeasurableSet S) (ω : SSphere d) :
    W2 d (Measure.map Φ (ν : Measure (SSphere d))) (collapseInto d ν S ω) ≤
      sqrt (∫ x : SSphere d, ‖(Φ x : EucSpace d) - (collapseMap S ω x : EucSpace d)‖ ^ 2
        ∂(ν : Measure (SSphere d))) := by
  rw [← map_collapseMap (ν : Measure (SSphere d)) S hS ω]
  exact monge d ν Φ (collapseMap S ω) hΦ (measurable_collapseMap S hS ω)

/-- Almost-everywhere spherical map convergence passes to the genuine
quadratic cost against any measurable target map. Auxiliary for
arXiv:2411.04551v3, Appendix, the Wasserstein step in `lem: two.balls`. -/
theorem tendsto_integral_spherical_distance_sq (ν : Perspective.ProbSphere d)
    (Φ : ℝ → SSphere d → SSphere d) (hΦ : ∀ t : ℝ, Measurable (Φ t))
    (Ψ Λ : SSphere d → SSphere d) (hΨ : Measurable Ψ)
    (hlim : ∀ᵐ x ∂(ν : Measure (SSphere d)), Tendsto (fun t => Φ t x) atTop (nhds (Λ x))) :
    Tendsto (fun t : ℝ => ∫ x : SSphere d,
      ‖(Φ t x : EucSpace d) - (Ψ x : EucSpace d)‖ ^ 2 ∂(ν : Measure (SSphere d))) atTop
      (nhds (∫ x : SSphere d, ‖(Λ x : EucSpace d) - (Ψ x : EucSpace d)‖ ^ 2
        ∂(ν : Measure (SSphere d)))) := by
  have hbound (t : ℝ) (x : SSphere d) :
      ‖‖(Φ t x : EucSpace d) - (Ψ x : EucSpace d)‖ ^ 2‖ ≤ (4 : ℝ) := by
    rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
    exact spherical_squared_distance_le_four (Φ t x) (Ψ x)
  exact tendsto_integral_filter_of_dominated_convergence (μ := (ν : Measure (SSphere d)))
    (fun _ => (4 : ℝ))
    (Eventually.of_forall fun t => (((continuous_subtype_val.measurable.comp (hΦ t)).sub
      (continuous_subtype_val.measurable.comp hΨ)).norm.pow_const 2).aestronglyMeasurable)
    (Eventually.of_forall fun t => Eventually.of_forall (hbound t)) (integrable_const 4)
    (hlim.mono fun x hx => by
      simpa only [Subtype.dist_eq, dist_eq_norm] using
        (hx.dist (tendsto_const_nhds (x := Ψ x))).pow 2)

/-- Almost-everywhere particle attraction and exact fixation outside the
active set force the genuine coupling cost to zero, for any initial law.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem tendsto_collapse_cost (ν : Perspective.ProbSphere d) (Φ : ℝ → SSphere d → SSphere d)
    (hΦ : ∀ t : ℝ, Measurable (Φ t)) (S : Set (SSphere d)) (hS : MeasurableSet S) (ω : SSphere d)
    (hactive : ∀ᵐ x ∂(ν : Measure (SSphere d)), x ∈ S → Tendsto (fun t => Φ t x) atTop (nhds ω))
    (hinactive : ∀ x ∉ S, ∀ t : ℝ, Φ t x = x) :
    Tendsto (fun t : ℝ => ∫ x : SSphere d,
      ‖(Φ t x : EucSpace d) - (collapseMap S ω x : EucSpace d)‖ ^ 2 ∂(ν : Measure (SSphere d)))
      atTop (nhds 0) := by
  classical
  have hG :=
    measurable_collapseMap S hS ω
  have hlim : ∀ᵐ x ∂(ν : Measure (SSphere d)),
      Tendsto (fun t => Φ t x) atTop (nhds (collapseMap S ω x)) := hactive.mono fun x hxlim => by
    by_cases hx : x ∈ S
    · simpa only [collapseMap, ite_eq_left hx] using hxlim hx
    · simpa only [hinactive x hx, collapseMap, ite_eq_right hx] using
        (tendsto_const_nhds : Tendsto (fun _ : ℝ => x) atTop (nhds x))
  have hl := tendsto_integral_spherical_distance_sq ν Φ hΦ (collapseMap S ω) (collapseMap S ω) hG hlim
  simpa only [sub_self, norm_zero, zero_pow (by decide : 2 ≠ 0), integral_zero] using hl

/-- The original Wasserstein distance to the exact collapsed measure
tends to zero. Auxiliary for arXiv:2411.04551v3, Appendix,
the Wasserstein conclusion in `lem: two.balls`. -/
theorem tendsto_W2_map_collapse (ν : Perspective.ProbSphere d) (Φ : ℝ → SSphere d → SSphere d)
    (hΦ : ∀ t : ℝ, Measurable (Φ t)) (S : Set (SSphere d)) (hS : MeasurableSet S) (ω : SSphere d)
    (hactive : ∀ᵐ x ∂(ν : Measure (SSphere d)), x ∈ S → Tendsto (fun t => Φ t x) atTop (nhds ω))
    (hinactive : ∀ x ∉ S, ∀ t : ℝ, Φ t x = x) :
    Tendsto (fun t : ℝ => W2 d (Measure.map (Φ t) (ν : Measure (SSphere d))) (collapseInto d ν S ω))
      atTop (nhds 0) := by
  have hc := tendsto_collapse_cost ν Φ hΦ S hS ω hactive hinactive
  have hs := (continuous_sqrt.tendsto (0 : ℝ)).comp hc
  simp only [sqrt_zero] at hs
  exact squeeze_zero
    (fun t => W2_nonneg d _ _)
    (fun t => W2_map_le_collapse_cost ν (Φ t) (hΦ t) S hS ω) hs

/-- The identity gives genuine measurable target and limit maps for
the general quadratic cost-convergence hypotheses. -/
example (ω : SSphere 2) :
    (ℝ → Measurable (id : SSphere 2 → SSphere 2)) ∧
    Measurable (id : SSphere 2 → SSphere 2) ∧
    (∀ᵐ x ∂(Perspective.diracProb 2 ω : Measure (SSphere 2)),
      Tendsto (fun _ : ℝ => x) atTop (nhds x)) :=
  ⟨fun _ => measurable_id, measurable_id, Eventually.of_forall fun _ => tendsto_const_nhds⟩

/-- An actual nonzero inward ReLU flow and a genuine open hemisphere
jointly realize the measurability, attraction and inactive-fixation premises.
The initial probability has an active atom and an inactive atom. -/
example : ∃ (ν : Perspective.ProbSphere 2) (Φ : ℝ → SSphere 2 → SSphere 2)
    (S : Set (SSphere 2)) (ω : SSphere 2),
    MeasurableSet S ∧ (∀ t : ℝ, Measurable (Φ t)) ∧
    (∀ᵐ x ∂(ν : Measure (SSphere 2)), x ∈ S → Tendsto (fun t => Φ t x) atTop (nhds ω)) ∧
    (∀ x ∉ S, ∀ t : ℝ, Φ t x = x) := by
  let ω : SSphere 2 := basePoint 1
  let S := {x : SSphere 2 | 0 < inner ℝ (x : EucSpace 2) (ω : EucSpace 2)}
  let Φ (t : ℝ) (x : SSphere 2) := ridgeFlow (ω : EucSpace 2) (-(ω : EucSpace 2)) 0 x t
  refine ⟨halfDirac 2 ω (antipode 2 ω), Φ, S, ω, ?_, ?_, ?_, ?_⟩
  · exact (isOpen_lt continuous_const (continuous_subtype_val.inner continuous_const)).measurableSet
  · exact fun t => measurable_ridgeFlow_initial _ _ _ t
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

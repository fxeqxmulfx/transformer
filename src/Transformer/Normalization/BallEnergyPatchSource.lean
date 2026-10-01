/-
# Energy-graph patches retaining the closed-ball source constraint

Each patch lies in the original ball and covers all nearby ball points.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Normalization.EnergyChartCoordinates

noncomputable section
open Filter Set Topology

namespace Transformer.Normalization

open AnalyticPreparation

/-- The exact analytic squared-radius constraint on a spatial chart.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def energyChartBallConstraint {n : ℕ} (w center : EucSpace (n + 1))
    (L : Ambient n ≃L[ℝ] EucSpace (n + 1)) (R : ℝ) : Ambient (n + 1) → ℝ :=
  fun x => R ^ 2 - ‖energyChartMap w L x - center‖ ^ 2

/-- The actual energy-graph image of the constrained coordinate box.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def ballEnergyPatchSource {n : ℕ} (E : EucSpace (n + 1) → ℝ)
    (w center : EucSpace (n + 1)) (L : Ambient n ≃L[ℝ] EucSpace (n + 1))
    (R r : ℝ) : Set (EucSpace (n + 1)) :=
  energyChartMap w L '' {x | ‖x.1‖ < r ∧ |x.2| < r ∧
    0 ≤ energyChartBallConstraint w center L R x ∧
      E (energyChartMap w L x) = energyChartLevel E w x.1}

/-- The radius constraint is analytic in every chart, including at
boundary centers. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem energyChartBallConstraint_analytic {n : ℕ} (w center : EucSpace (n + 1))
    (L : Ambient n ≃L[ℝ] EucSpace (n + 1)) (R : ℝ) :
    AnalyticAt ℝ (energyChartBallConstraint w center L R) 0 := by
  have hdiff : AnalyticAt ℝ (fun x => energyChartMap w L x - center) 0 :=
    (energyChartMap_analytic w L).1.fun_sub analyticAt_const
  exact analyticAt_const.fun_sub
    ((quadratic_energy_analytic (energyChartMap w L 0 - center) (mem_univ _)).comp
      (f := fun x : Ambient (n + 1) => energyChartMap w L x - center) (x := 0) hdiff)

/-- Nonnegativity of the actual squared-radius constraint is exactly
membership in the original closed ball. The nonnegative radius is
essential for the reverse square comparison. Auxiliary for Appendix
D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem energyChartBallConstraint_iff {n : ℕ} (w center : EucSpace (n + 1))
    (L : Ambient n ≃L[ℝ] EucSpace (n + 1)) {R : ℝ} (hR : 0 ≤ R)
    (x : Ambient (n + 1)) :
    0 ≤ energyChartBallConstraint w center L R x ↔
      energyChartMap w L x ∈ Metric.closedBall center R := by
  simp only [energyChartBallConstraint, sub_nonneg, Metric.mem_closedBall, dist_eq_norm]
  exact sq_le_sq₀ (norm_nonneg _) hR

/-- Every point retained by the patch belongs to the original source
ball. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem ballEnergyPatchSource_subset {n : ℕ} (E : EucSpace (n + 1) → ℝ)
    (w center : EucSpace (n + 1)) (L : Ambient n ≃L[ℝ] EucSpace (n + 1))
    {R : ℝ} (hR : 0 ≤ R) (r : ℝ) :
    ballEnergyPatchSource E w center L R r ⊆ Metric.closedBall center R := by
  rintro x ⟨y, hy, rfl⟩
  exact (energyChartBallConstraint_iff w center L hR y).mp hy.2.2.1

/-- For every positive coordinate radius, the energy-graph patch
contains all sufficiently nearby points of the original closed ball.
The actual inverse coordinates and the actual energy are constructed;
no pointwise or analytic selection is assumed. Auxiliary for Appendix
D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem ballEnergyPatchSource_contains_neighborhood {n : ℕ}
    (E : EucSpace (n + 1) → ℝ) (w center : EucSpace (n + 1))
    (L : Ambient n ≃L[ℝ] EucSpace (n + 1)) (hE : AnalyticAt ℝ E w)
    {R r : ℝ} (hR : 0 ≤ R) (hr : 0 < r) :
    ∃ delta > 0, ∀ x ∈ Metric.closedBall center R,
      x ∈ Metric.ball w delta → x ∈ ballEnergyPatchSource E w center L R r := by
  have hsmall : ∀ᶠ x in nhds w, ‖inverseEnergyChart E w L x‖ < r :=
    (inverseEnergyChart_analytic E w L hE).continuousAt.norm.eventually
      (Iio_mem_nhds (by simpa only [inverseEnergyChart_center, norm_zero] using hr))
  obtain ⟨delta, hd, hball⟩ := Metric.eventually_nhds_iff.mp hsmall
  refine ⟨delta, hd, ?_⟩
  intro x hxK hxball
  have hcoords := hball (Metric.mem_ball.mp hxball)
  rw [Prod.norm_def, Real.norm_eq_abs, max_lt_iff] at hcoords
  refine ⟨inverseEnergyChart E w L x, ⟨hcoords.1, hcoords.2, ?_, ?_⟩,
    energyChartMap_inverse E w x L⟩
  · apply (energyChartBallConstraint_iff w center L hR _).mpr
    rwa [energyChartMap_inverse]
  · rw [energyChartMap_inverse, energyChartLevel_inverse]

/-- A genuine nonconstant quadratic center and positive radii witness
all patch-subset and neighborhood inputs jointly. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let E : EucSpace 1 → ℝ := fun x => ‖x‖ ^ 2
    ∃ (L : Ambient 0 ≃L[ℝ] EucSpace 1) (delta : ℝ), 0 < delta ∧
      ∀ x ∈ Metric.closedBall (0 : EucSpace 1) 1,
        x ∈ Metric.ball 0 delta → x ∈ ballEnergyPatchSource E 0 0 L 1 1 := by
  intro E
  have hE : AnalyticAt ℝ E 0 := quadratic_energy_analytic 0 (mem_univ _)
  obtain ⟨L, _, _, _, _⟩ := exists_regular_energy_coordinates E 0 hE
    (nonflatEnergyLevel_nonconstant E (Metric.closedBall 0 1) 0 zero_mem_nonflat_quadratic_level)
  obtain ⟨delta, hd, hcover⟩ := ballEnergyPatchSource_contains_neighborhood E 0 0 L hE
    (by norm_num : (0 : ℝ) ≤ 1) (by norm_num : (0 : ℝ) < 1)
  exact ⟨L, delta, hd, hcover⟩

end Transformer.Normalization

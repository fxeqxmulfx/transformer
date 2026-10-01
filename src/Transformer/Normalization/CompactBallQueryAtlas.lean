/-
# Scalar query elimination on the entire compact analytic ball

Preparation at the nonflat central fiber and compactness construct a
finite atlas covering every point on nearby noncentral energy levels.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Normalization.BallEnergyQueryPreparation

noncomputable section
open Set

namespace Transformer.Normalization

open AnalyticPreparation

/-- Analyticity on a compact ball constructs a finite family of scalar
queries for comparisons with every point of that ball on all sufficiently
nearby noncentral energy levels. The directional orders, coordinate maps,
analytic remainder coefficients, finite cover, and energy radius are all
constructed. No pre-existing atlas, analytic parameter curve, or minimum
is assumed. The threshold is arbitrary and common to every chart. Only
the base-coordinate quantifiers remain. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem analytic_compact_ball_lower_bounds_arithmetic {n : ℕ}
    (E : EucSpace (n + 1) → ℝ) (center : EucSpace (n + 1)) (R c : ℝ)
    (hR : 0 ≤ R) (hE : AnalyticOnNhd ℝ E (Metric.closedBall center R)) :
    ∃ (t : Finset (nonflatEnergyLevel E (Metric.closedBall center R) c))
      (L : nonflatEnergyLevel E (Metric.closedBall center R) c →
        Ambient n ≃L[ℝ] EucSpace (n + 1))
      (d : nonflatEnergyLevel E (Metric.closedBall center R) c → ℕ)
      (a b v : ∀ j : nonflatEnergyLevel E (Metric.closedBall center R) c,
        Fin (d j) → Base (n + 1) → ℝ)
      (r : nonflatEnergyLevel E (Metric.closedBall center R) c → ℝ) (epsilon : ℝ),
      0 < epsilon ∧ (∀ j, 0 < d j) ∧ (∀ j, 0 < r j) ∧
      (∀ j i, AnalyticAt ℝ (a j i) 0) ∧ (∀ j i, a j i 0 = 0) ∧
      (∀ j i, AnalyticAt ℝ (b j i) 0) ∧ (∀ j i, AnalyticAt ℝ (v j i) 0) ∧
      (∀ j : nonflatEnergyLevel E (Metric.closedBall center R) c,
        BallEnergyQueryValues E j center (L j) R (r j) (a j) (b j) (v j)) ∧
      ∀ e threshold : ℝ, |e - c| < epsilon → e ≠ c →
        ((∀ x ∈ Metric.closedBall center R, E x = e → threshold ≤ squaredGradientNorm E x) ↔
          ∀ j ∈ t, ∀ z : Base (n + 1), ‖z‖ < r j → energyChartLevel E j z = e →
            ballEnergyLowerBoundQuery (a j) (b j) (v j) z threshold (r j) = 0) := by
  let K : Set (EucSpace (n + 1)) := Metric.closedBall center R
  let A : Set (EucSpace (n + 1)) := nonflatEnergyLevel E K c
  have hlocal (j : A) := exists_ball_energy_query_preparation E j center R hR
    (hE j j.property.1.1) (nonflatEnergyLevel_nonconstant E K c j.property)
  choose L d a b v r delta hd hr hdelta ha ha0 hb hv hvalues hcover using hlocal
  obtain ⟨t, epsilon, hepsilon, hcovered⟩ := finite_cover_near_energy_level E K
    (isCompact_closedBall _ _) hE.continuousOn c
    (fun j => Metric.ball (j : EucSpace (n + 1)) (delta j)) (fun _ => Metric.isOpen_ball)
    (fun j => Metric.mem_ball_self (hdelta j))
  refine ⟨t, L, d, a, b, v, r, epsilon, hepsilon, hd, hr, ha, ha0, hb, hv, hvalues, ?_⟩
  intro e threshold he hne
  constructor
  · intro hbound j hj
    apply (ball_energy_patch_lower_bound_iff E j center (L j) R (r j)
      (a j) (b j) (v j) (hvalues j) e threshold).mp
    intro x hx hxe
    exact hbound x (ballEnergyPatchSource_subset E j center (L j) hR (r j) hx) hxe
  · intro hqueries x hx hxe
    have hxcovered := hcovered x hx (by rwa [hxe]) (by rwa [hxe])
    obtain ⟨j, hxj⟩ := mem_iUnion.mp hxcovered
    obtain ⟨hj, hxball⟩ := mem_iUnion.mp hxj
    have hxpatch := hcover j x hx hxball
    exact (ball_energy_patch_lower_bound_iff E j center (L j) R (r j)
      (a j) (b j) (v j) (hvalues j) e threshold).mpr (hqueries j hj) x hxpatch hxe

/-- A quadratic energy on a genuine compact ball witnesses the radius
and analytic hypotheses of the automatic finite-atlas theorem. The
central nonflat fiber is inhabited by the origin. Auxiliary for Appendix
D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : (0 : ℝ) ≤ 1 ∧
    AnalyticOnNhd ℝ (fun x : EucSpace 1 => ‖x‖ ^ 2) (Metric.closedBall 0 1) ∧
    (0 : EucSpace 1) ∈ nonflatEnergyLevel (fun x => ‖x‖ ^ 2) (Metric.closedBall 0 1) 0 :=
  ⟨by norm_num, quadratic_energy_analytic.mono (subset_univ _), zero_mem_nonflat_quadratic_level⟩

end Transformer.Normalization

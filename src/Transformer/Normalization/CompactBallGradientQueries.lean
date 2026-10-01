/-
# Exact scalar query tests for compact-ball gradient minimizers

The finite atlas is constructed from analyticity, and comparisons retain
every point of the original ball on the candidate's energy level.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Normalization.CompactBallQueryAtlas
import Transformer.Normalization.GradientMinimumCluster

noncomputable section

namespace Transformer.Normalization

open AnalyticPreparation

/-- On every sufficiently nearby noncentral energy level, the actual
gradient-minimizer set in the original compact ball has exact scalar-query
tests on a finite family of prepared charts. All charts and coefficients
are constructed from analyticity, including at degenerate critical points.
Comparisons retain the entire ball, not just a coordinate neighborhood.
The source's analytic ball constraint is retained, and no analytic curve
or selected minimum is assumed. The universal base quantifiers remain
explicit; their geometric projection and curve selection are further
steps. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analytic_compact_ball_gradient_minima_arithmetic {n : ℕ}
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
      ∀ S : Set ℝ, ∀ w ∈ Metric.closedBall center R,
        |E w - c| < epsilon → E w ≠ c →
        (w ∈ energyFiberGradientMinima E (Metric.closedBall center R) S ↔
          E w ∈ S ∧ ∀ j ∈ t, ∀ z : Base (n + 1), ‖z‖ < r j →
            energyChartLevel E j z = E w →
              ballEnergyLowerBoundQuery (a j) (b j) (v j)
                z (squaredGradientNorm E w) (r j) = 0) := by
  obtain ⟨t, L, d, a, b, v, r, epsilon, hepsilon, hd, hr, ha, ha0, hb, hv, hvalues, heq⟩ :=
    analytic_compact_ball_lower_bounds_arithmetic E center R c hR hE
  refine ⟨t, L, d, a, b, v, r, epsilon, hepsilon, hd, hr, ha, ha0, hb, hv, hvalues, ?_⟩
  intro S w hw hsmall hne
  constructor
  · rintro ⟨_, hwE, hmin⟩
    refine ⟨hwE, (heq (E w) (squaredGradientNorm E w) hsmall hne).mp ?_⟩
    intro x hx hxE
    exact pow_le_pow_left₀ (norm_nonneg _) (hmin x hx hxE) 2
  · rintro ⟨hwE, hqueries⟩
    refine ⟨hw, hwE, ?_⟩
    intro x hx hxE
    exact (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mp
      ((heq (E w) (squaredGradientNorm E w) hsmall hne).mpr hqueries x hx hxE)

/-- A degenerate two-dimensional energy and a genuine closed unit ball
satisfy all hypotheses of the automatic compact-ball query construction.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : (0 : ℝ) ≤ 1 ∧
    AnalyticOnNhd ℝ (fun x : EucSpace 2 => (x 0) ^ 2 + (x 1) ^ 4)
      (Metric.closedBall 0 1) := by
  refine ⟨by norm_num, ?_⟩
  intro x hx
  exact (((EuclideanSpace.proj 0 : EucSpace 2 →L[ℝ] ℝ).analyticAt x).fun_pow 2).add
    (((EuclideanSpace.proj 1 : EucSpace 2 →L[ℝ] ℝ).analyticAt x).fun_pow 4)

end Transformer.Normalization

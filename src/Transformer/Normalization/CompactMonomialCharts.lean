/-
# Compact monomial charts and the analytic gradient inequality

A finite family of analytic maps, with compact source sets and local
monomial-times-unit expressions above the base point, suffices for the
gradient inequality. Existence of such charts is a separate geometric input
in this route to Appendix D.1 of arXiv:2510.22026v2.
-/

import Transformer.Normalization.NormalCrossingCharts
import Transformer.Normalization.CompactFiberGradient

open Filter Set
open scoped BigOperators

namespace Transformer.Normalization

/-- Explicit geometric preparation data for an energy germ at `z`:
finitely many analytic maps with compact source sets cover a neighborhood
of `z`, and the pulled-back energy is a positive-degree monomial times an
analytic unit near each point above `z`. No injectivity or inverse is
required. This is an auxiliary predicate for the analytic step of
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2; it does not assert that
arbitrary analytic germs admit such preparation. -/
def HasCompactMonomialCharts {N : ℕ} (E : EucSpace N → ℝ) (z : EucSpace N) : Prop :=
  ∃ (D J : ℕ) (phi : Fin J → EucSpace D → EucSpace N)
    (K : Fin J → Set (EucSpace D)),
    (∀ j, IsCompact (K j)) ∧ (∀ j, AnalyticOnNhd ℝ (phi j) (K j)) ∧
    (∀ j w, w ∈ K j → phi j w = z →
      ∃ (p : Fin D → ℕ) (u : EucSpace D → ℝ),
        0 < ∑ i, p i ∧ AnalyticAt ℝ u w ∧ u w ≠ 0 ∧
          ∀ᶠ y in nhds w, E (phi j y) - E z = u y * coordinateMonomial p (y - w)) ∧
    ∀ᶠ y in nhds z, ∃ j w, w ∈ K j ∧ phi j w = y

/-- Compact monomial charts imply the local analytic gradient inequality.
Local bounds on each compact fiber uniformize, and the finitely many
exponents and constants have common maxima. This proves the geometric-to-
analytic implication used in this route to Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. Chart existence remains an explicit hypothesis. -/
theorem analytic_gradient_inequality_of_compact_monomial_charts {N : ℕ}
    (E : EucSpace N → ℝ) (z : EucSpace N) (hE : AnalyticAt ℝ E z)
    (hcharts : HasCompactMonomialCharts E z) :
    ∃ alpha k : ℝ, ∃ V : Set (EucSpace N),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ z ∈ V ∧
        ∀ y ∈ V, |E y - E z| ^ alpha ≤ k * ‖gradient E y‖ := by
  classical
  obtain ⟨D, J, phi, K, hK, hphi, hmono, hlift⟩ := hcharts
  have hchart (j : Fin J) := uniform_gradient_inequality_at_compact_fiber
    E z hE.continuousAt (phi j) (K j) (hK j) (hphi j).continuousOn (by
      intro w hw hwz
      obtain ⟨p, u, hp, hu, hu0, heq⟩ := hmono j w hw hwz
      have hEw : AnalyticAt ℝ E (phi j w) := by rw [hwz]; exact hE
      simpa only [hwz] using analytic_gradient_inequality_on_monomial_image
        E (phi j) w hEw (hphi j w hw) p hp u hu hu0 (by simpa only [hwz] using heq))
  choose alpha k V ha ha1 hk hV hzV hineq using hchart
  obtain ⟨j0, _, _, _⟩ := hlift.self_of_nhds
  have : Nonempty (Fin J) := ⟨j0⟩
  have hs : (Finset.univ : Finset (Fin J)).Nonempty := Finset.univ_nonempty
  let A := Finset.univ.sup' hs alpha
  let B := Finset.univ.sup' hs k
  have hA : 0 < A := (ha j0).trans_le (Finset.le_sup' alpha (Finset.mem_univ j0))
  have hA1 : A < 1 := (Finset.sup'_lt_iff hs).mpr (fun j _ => ha1 j)
  have hB : 0 < B := (hk j0).trans_le (Finset.le_sup' k (Finset.mem_univ j0))
  have hVs : ∀ᶠ y in nhds z, ∀ j, y ∈ V j :=
    Filter.eventually_all.mpr (fun j => (hV j).mem_nhds (hzV j))
  have hsmall : ∀ᶠ y in nhds z, |E y - E z| < 1 := by
    simpa only [Real.dist_eq] using
      (Metric.tendsto_nhds.mp hE.continuousAt.tendsto) 1 zero_lt_one
  have hevent : ∀ᶠ y in nhds z,
      |E y - E z| ^ A ≤ B * ‖gradient E y‖ := by
    filter_upwards [hlift, hVs, hsmall] with y hy hyV hygap
    obtain ⟨j, w, hw, hwphi⟩ := hy
    have hbound : |E y - E z| ^ alpha j ≤ k j * ‖gradient E y‖ := by
      simpa only [hwphi] using hineq j w hw (by simpa only [hwphi] using hyV j)
    by_cases hzero : E y - E z = 0
    · rw [hzero, abs_zero, Real.zero_rpow hA.ne']
      exact mul_nonneg hB.le (norm_nonneg _)
    · calc
        |E y - E z| ^ A ≤ |E y - E z| ^ alpha j :=
          Real.rpow_le_rpow_of_exponent_ge (abs_pos.mpr hzero) hygap.le
            (Finset.le_sup' alpha (Finset.mem_univ j))
        _ ≤ k j * ‖gradient E y‖ := hbound
        _ ≤ B * ‖gradient E y‖ := mul_le_mul_of_nonneg_right
          (Finset.le_sup' k (Finset.mem_univ j)) (norm_nonneg _)
  obtain ⟨W, hW, hWo, hzW⟩ := eventually_nhds_iff.mp hevent
  exact ⟨A, B, W, hA, hA1, hB, hWo, hzW, hW⟩

/-- The square monomial admits one identity chart on a closed unit ball,
so all compact-chart hypotheses are jointly satisfiable and yield a
nonconstant instance of the gradient inequality. Auxiliary example for
Appendix D.1 of arXiv:2510.22026v2. -/
example : ∃ alpha k : ℝ, ∃ V : Set (EucSpace 1),
    0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ (0 : EucSpace 1) ∈ V ∧
      ∀ y ∈ V, |coordinateMonomial (fun _ : Fin 1 => 2) y| ^ alpha ≤ k *
        ‖gradient (coordinateMonomial (fun _ : Fin 1 => 2)) y‖ := by
  have hcharts : HasCompactMonomialCharts
      (coordinateMonomial (fun _ : Fin 1 => 2)) (0 : EucSpace 1) := by
    refine ⟨1, 1, (fun _ => id), (fun _ => Metric.closedBall 0 1),
      (fun _ => isCompact_closedBall _ _), (fun _ _ _ => analyticAt_id), ?_, ?_⟩
    · intro j w hw hw0
      have hwzero : w = 0 := hw0
      subst w
      refine ⟨(fun _ : Fin 1 => 2), (fun _ => 1), by decide,
        analyticAt_const, by norm_num, ?_⟩
      filter_upwards [] with y
      simp [coordinateMonomial]
    · filter_upwards [Metric.ball_mem_nhds (0 : EucSpace 1)
        (by norm_num : (0 : ℝ) < 1)] with y hy
      exact ⟨0, y, Metric.mem_closedBall.mpr (Metric.mem_ball.mp hy).le, rfl⟩
  simpa only [coordinateMonomial, Fin.prod_univ_one, PiLp.zero_apply,
    zero_pow (by norm_num : (2 : ℕ) ≠ 0), sub_zero] using
      analytic_gradient_inequality_of_compact_monomial_charts
        (coordinateMonomial (fun _ : Fin 1 => 2)) 0
        (coordinateMonomial_analytic _ _) hcharts

end Transformer.Normalization

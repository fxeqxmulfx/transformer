/-
# A degenerate isolated critical point prepared by two blowup charts

The quartic `(x²+y²)²` becomes `r⁴(1+s²)²` in either of the maps
`(r,s) ↦ (r,rs)` and `(r,s) ↦ (rs,r)`. Compact source sets cover a
neighborhood of the origin, proving a concrete instance of the compact
parametrization route to Appendix D.1 of arXiv:2510.22026v2.
-/

import Transformer.Normalization.CompactMonomialCharts

open Filter Set
open scoped BigOperators

namespace Transformer.Normalization

/-- The two elementary real blowup maps used to prepare the quartic
energy. Auxiliary definition for Appendix D.1 of arXiv:2510.22026v2. -/
def radialBlowupChart (j : Fin 2) (y : EucSpace 2) : EucSpace 2 :=
  (y 0) • (PiLp.single 2 j 1 : EucSpace 2) +
    (y 0 * y 1) • (PiLp.single 2 j.rev 1 : EucSpace 2)

/-- Both elementary blowup maps are analytic at every source point;
auxiliary chart fact for Appendix D.1 of arXiv:2510.22026v2. -/
theorem radialBlowupChart_analytic (j : Fin 2) (w : EucSpace 2) :
    AnalyticAt ℝ (radialBlowupChart j) w := by
  have h0 := (EuclideanSpace.proj 0 : EucSpace 2 →L[ℝ] ℝ).analyticAt w
  have h1 := (EuclideanSpace.proj 1 : EucSpace 2 →L[ℝ] ℝ).analyticAt w
  exact (h0.fun_smul analyticAt_const).add ((h0.mul h1).fun_smul analyticAt_const)

/-- The quartic energy pulls back to the same monomial times a positive
unit in both blowup charts. This is an exact polynomial identity,
auxiliary to Appendix D.1 of arXiv:2510.22026v2. -/
theorem radialBlowupChart_quartic (j : Fin 2) (y : EucSpace 2) :
    ((radialBlowupChart j y 0) ^ 2 + (radialBlowupChart j y 1) ^ 2) ^ 2 =
      (1 + (y 1) ^ 2) ^ 2 * (y 0) ^ 4 := by
  fin_cases j <;> simp [radialBlowupChart] <;> ring

/-- The quartic `(x²+y²)²` admits two compact monomial charts at zero.
The slope is chosen in the chart of a coordinate with maximal absolute
value; it lies in `[-1,1]`, and the source point lies in a closed radius-two
ball. Auxiliary prepared-germ instance for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem radial_quartic_has_compact_monomial_charts :
    HasCompactMonomialCharts
      (fun y : EucSpace 2 => ((y 0) ^ 2 + (y 1) ^ 2) ^ 2) 0 := by
  classical
  refine ⟨2, 2, radialBlowupChart, (fun _ => Metric.closedBall 0 2),
    (fun _ => isCompact_closedBall _ _),
    (fun j w _ => radialBlowupChart_analytic j w), ?_, ?_⟩
  · intro j w hw hw0
    have hwzero : w 0 = 0 := by
      have h := congrArg (fun x : EucSpace 2 => x j) hw0
      fin_cases j <;> simpa [radialBlowupChart] using h
    refine ⟨(fun i : Fin 2 => if i = 0 then 4 else 0),
      (fun y : EucSpace 2 => (1 + (y 1) ^ 2) ^ 2), by decide, ?_, ?_, ?_⟩
    · exact (analyticAt_const.add
        (((EuclideanSpace.proj 1 : EucSpace 2 →L[ℝ] ℝ).analyticAt w).fun_pow 2)).fun_pow 2
    · exact ne_of_gt (sq_pos_of_pos (by positivity))
    · filter_upwards [] with y
      rw [radialBlowupChart_quartic]
      simp [coordinateMonomial, hwzero]
  · filter_upwards [Metric.ball_mem_nhds (0 : EucSpace 2)
      (by norm_num : (0 : ℝ) < 1)] with x hx
    have hxnorm : ‖x‖ < 1 := by simpa only [Metric.mem_ball, dist_zero_right] using hx
    by_cases hxzero : x = 0
    · exact ⟨0, 0, Metric.mem_closedBall.mpr (by simp), by simp [radialBlowupChart, hxzero]⟩
    obtain ⟨j, hj, hmax⟩ := Finset.exists_max_image Finset.univ
      (fun i : Fin 2 => |x i|) Finset.univ_nonempty
    have hmax' (i : Fin 2) : |x i| ≤ |x j| := hmax i (Finset.mem_univ i)
    have hxj : x j ≠ 0 := by
      intro hxj
      apply hxzero
      ext i
      have hi : |x i| = 0 := le_antisymm
        (by simpa only [hxj, abs_zero] using hmax' i) (abs_nonneg _)
      simpa only [PiLp.zero_apply] using abs_eq_zero.mp hi
    let r : ℝ := x j
    let s : ℝ := x j.rev / x j
    let w : EucSpace 2 := PiLp.single 2 (0 : Fin 2) r + PiLp.single 2 (1 : Fin 2) s
    have hr : |r| < 1 := (PiLp.norm_apply_le x j).trans_lt hxnorm
    have hs : |s| ≤ 1 := by
      dsimp [s]
      rw [abs_div]
      exact (div_le_one (abs_pos.mpr hxj)).mpr (hmax' j.rev)
    have hw : w ∈ Metric.closedBall 0 2 := by
      rw [Metric.mem_closedBall, dist_zero_right]
      have hnorm : ‖w‖ ^ 2 = r ^ 2 + s ^ 2 := by
        simp [EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_two, w]
      have hr2 : r ^ 2 ≤ 1 := by
        simpa only [sq_abs, one_pow] using pow_le_pow_left₀ (abs_nonneg r) hr.le 2
      have hs2 : s ^ 2 ≤ 1 := by
        simpa only [sq_abs, one_pow] using pow_le_pow_left₀ (abs_nonneg s) hs 2
      nlinarith [norm_nonneg w]
    refine ⟨j, w, hw, ?_⟩
    have hcancel : r * s = x j.rev := by
      dsimp [r, s]
      field_simp
    ext i
    fin_cases j <;> fin_cases i <;> simp [radialBlowupChart, w, r, s] at hcancel ⊢ <;>
      exact hcancel

/-- The two compact blowup charts prove the analytic gradient inequality
at the degenerate critical point of `(x²+y²)²`. This is a proved case of
the analytic input in Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem radial_quartic_gradient_inequality :
    ∃ alpha k : ℝ, ∃ V : Set (EucSpace 2),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ (0 : EucSpace 2) ∈ V ∧
        ∀ y ∈ V, |((y 0) ^ 2 + (y 1) ^ 2) ^ 2| ^ alpha ≤ k *
          ‖gradient (fun t : EucSpace 2 => ((t 0) ^ 2 + (t 1) ^ 2) ^ 2) y‖ := by
  have h0 := (EuclideanSpace.proj 0 : EucSpace 2 →L[ℝ] ℝ).analyticAt 0
  have h1 := (EuclideanSpace.proj 1 : EucSpace 2 →L[ℝ] ℝ).analyticAt 0
  simpa only [PiLp.zero_apply, zero_pow (by norm_num : (2 : ℕ) ≠ 0),
    add_zero, sub_zero] using analytic_gradient_inequality_of_compact_monomial_charts
      (fun y : EucSpace 2 => ((y 0) ^ 2 + (y 1) ^ 2) ^ 2) 0
      (((h0.fun_pow 2).add (h1.fun_pow 2)).fun_pow 2)
      radial_quartic_has_compact_monomial_charts

end Transformer.Normalization

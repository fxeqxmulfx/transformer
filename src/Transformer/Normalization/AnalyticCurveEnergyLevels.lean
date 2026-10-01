/-
# Energy levels covered by a one-sided analytic curve

An analytic scalar curve which starts at zero and has positive values
for positive parameters covers all sufficiently small positive levels,
with arbitrarily small parameters. This replaces an explicit energy-level
lifting assumption by the usual one-sided conclusion of analytic curve
selection in the route to Appendix D.1 of arXiv:2510.22026v2.
-/

import Transformer.Normalization.AnalyticCurveGradient
import Mathlib.Topology.Order.IntermediateValue

open Set

namespace Transformer.Normalization

/-- A one-sided positive analytic curve covers every sufficiently small
positive scalar level with arbitrarily small positive parameters. This
uses continuity and the intermediate value theorem, without requiring a
nonzero derivative or an analytic inverse. Auxiliary energy-level lifting
fact for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analytic_curve_covers_positive_levels (f : ℝ → ℝ)
    (hf : AnalyticAt ℝ f 0) (hzero : f 0 = 0) (epsilon : ℝ)
    (he : 0 < epsilon) (hpositive : ∀ t ∈ Ioo 0 epsilon, 0 < f t) :
    ∀ delta : ℝ, 0 < delta → ∃ eta : ℝ, 0 < eta ∧
      ∀ c ∈ Ioo 0 eta, ∃ t ∈ Ioo 0 delta, t < epsilon ∧ f t = c := by
  intro delta hd
  obtain ⟨r, hr, hfa⟩ := hf.exists_ball_analyticOnNhd
  let T : ℝ := min (min (epsilon / 2) (delta / 2)) (r / 2)
  have hT : 0 < T := lt_min (lt_min (half_pos he) (half_pos hd)) (half_pos hr)
  have hTe : T < epsilon :=
    ((min_le_left _ _).trans (min_le_left _ _)).trans_lt (half_lt_self he)
  have hTd : T < delta :=
    ((min_le_left _ _).trans (min_le_right _ _)).trans_lt (half_lt_self hd)
  have hTr : T < r := (min_le_right _ _).trans_lt (half_lt_self hr)
  have hcont : ContinuousOn f (Icc 0 T) := hfa.continuousOn.mono (by
    intro t ht
    rw [Metric.mem_ball, Real.dist_eq, sub_zero, abs_of_nonneg ht.1]
    exact ht.2.trans_lt hTr)
  refine ⟨f T, hpositive T ⟨hT, hTe⟩, ?_⟩
  intro c hc
  obtain ⟨t, ht, htc⟩ := intermediate_value_Icc hT.le hcont
    (show c ∈ Icc (f 0) (f T) from ⟨by rw [hzero]; exact hc.1.le, hc.2.le⟩)
  have ht0 : 0 < t := lt_of_le_of_ne ht.1 (by
    intro h
    have hc0 : c = 0 := by simpa only [← h, hzero] using htc.symm
    exact hc.1.ne' hc0)
  exact ⟨t, ⟨ht0, ht.2.trans_lt hTd⟩, ht.2.trans_lt hTe, htc⟩

/-- The squared parameter is analytic and positive on the positive
unit interval, while its derivative at zero vanishes. Thus all lifting
hypotheses are jointly satisfiable even for a degenerate parameter.
Auxiliary example for Appendix D.1 of arXiv:2510.22026v2. -/
example : ∀ delta : ℝ, 0 < delta → ∃ eta : ℝ, 0 < eta ∧
    ∀ c ∈ Ioo 0 eta, ∃ t ∈ Ioo 0 delta, t < 1 ∧ t ^ 2 = c := by
  exact analytic_curve_covers_positive_levels (fun t => t ^ 2)
    (analyticAt_id.fun_pow 2) (by norm_num) 1 zero_lt_one
    (fun t ht => sq_pos_of_pos ht.1)

end Transformer.Normalization

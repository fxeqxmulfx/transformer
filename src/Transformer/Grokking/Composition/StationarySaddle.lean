import Transformer.Grokking.Composition.Curvature
import Mathlib.Analysis.Convex.Function
import Mathlib.Topology.Order.LocalExtr
import Mathlib.Topology.MetricSpace.Pseudo.Constructions

/-!
# Flat coordinate restrictions do not trap the joint CE objective

Source: Nanda et al., arXiv:2301.05217v1, appendix section Further
speculations on grokking, subsection Hypothesis: Phase Transitions are
inherent to composition. This tests the source's multi-part-circuit
intuition in the explicitly stated bilinear binary-CE specialization.

Prove actual better and worse states arbitrarily close to the stationary
origin, with the ordinary product metric. The origin is not a local
minimum, and the objective is not jointly convex. Both flat coordinate
restrictions and the zero full gradient nevertheless hold there.

These are loss-landscape statements. They do not prove escape from an
exact zero deterministic initialization, a probabilistic initialization
claim, a learned attention circuit or grokking on held-out tokens.
-/

namespace Transformer.Grokking.Composition

open Filter
open scoped Topology

/-- Simultaneously reversing both component signs preserves the actual
loss. Source: the bilinear specialization of arXiv:2301.05217v1's
appendix composition hypothesis; the target class is unchanged. -/
theorem coupledLoss_negate_both (x y : ℝ) :
    coupledLoss (-x) (-y) = coupledLoss x y := by
  rw [coupledLoss_eq_exp, coupledLoss_eq_exp]
  have hs : -((-x) * (-y)) = -(x * y) := by ring
  rw [hs]

/-- Opposed components increase CE above the axis value. Source:
arXiv:2301.05217v1, appendix compositional hypothesis, specialized
to the actual binary loss rather than an assumed saddle function. -/
theorem coupledLoss_above_axes (x y : ℝ) (hxy : x * y < 0) :
    Real.log 2 < coupledLoss x y := by
  rw [coupledLoss_eq_exp]
  have he : 1 < Real.exp (-(x * y)) := by
    have h := Real.exp_lt_exp.mpr (show (0 : ℝ) < -(x * y) by linarith)
    simpa using h
  apply Real.log_lt_log (by norm_num)
  linarith

example : (1 : ℝ) * (-1) < 0 := by norm_num

/-- Every positive radius contains a jointly improving state. Source:
arXiv:2301.05217v1, appendix compositional hypothesis; the distance
bound is an ordinary product-metric neighborhood, not a defined success set. -/
theorem arbitrarily_small_joint_improvement (radius : ℝ) (hr : 0 < radius) :
    ∃ p : ℝ × ℝ, dist p (0, 0) < radius ∧ coupledLoss p.1 p.2 < coupledLoss 0 0 := by
  have ht : 0 < radius / 2 := by positivity
  have hs : radius / 2 < radius := by linarith
  refine ⟨(radius / 2, radius / 2), ?_, ?_⟩
  · simpa [Prod.dist_eq, Real.dist_eq, abs_of_pos hr] using hs
  · rw [(coupledLoss_flat_axes 0 0).1]
    exact coupledLoss_below_axes _ _ (by positivity)

example : 0 < (1 / 1000000 : ℝ) := by norm_num

/-- Every positive radius also contains a jointly worsening state.
Source: arXiv:2301.05217v1, appendix compositional hypothesis, explicit
bilinear CE specialization. Nearby loss variation has both signs. -/
theorem arbitrarily_small_joint_increase (radius : ℝ) (hr : 0 < radius) :
    ∃ p : ℝ × ℝ, dist p (0, 0) < radius ∧ coupledLoss 0 0 < coupledLoss p.1 p.2 := by
  have ht : 0 < radius / 2 := by positivity
  have hs : radius / 2 < radius := by linarith
  refine ⟨(radius / 2, -(radius / 2)), ?_, ?_⟩
  · simpa [Prod.dist_eq, Real.dist_eq, abs_neg, abs_of_pos hr] using hs
  · rw [(coupledLoss_flat_axes 0 0).1]
    apply coupledLoss_above_axes
    nlinarith [sq_pos_of_pos ht]

example : 0 < (1 : ℝ) := by norm_num

/-- The actual stationary origin fails the usual topological local-minimum
property. Source: arXiv:2301.05217v1, appendix composition hypothesis;
flat coordinate losses are insufficient to infer a joint local trap. -/
theorem coupled_origin_not_local_min :
    ¬IsLocalMin (fun p : ℝ × ℝ => coupledLoss p.1 p.2) (0, 0) := by
  intro h
  unfold IsLocalMin IsMinFilter at h
  obtain ⟨radius, hr, hall⟩ := Metric.eventually_nhds_iff.mp h
  obtain ⟨p, hp, hbetter⟩ := arbitrarily_small_joint_improvement radius hr
  have hmin := hall hp
  linarith

/-- Actual jointly trained component parameters need not form a convex
objective even though ordinary CE is convex in its logits. Source:
arXiv:2301.05217v1, appendix compositional hypothesis; equal-loss
opposite parameter pairs have a strictly worse midpoint. -/
theorem coupledLoss_not_jointly_convex :
    ¬ConvexOn ℝ Set.univ (fun p : ℝ × ℝ => coupledLoss p.1 p.2) := by
  intro h
  have hconv := h.2
  have hm := hconv (x := ((1, 1) : ℝ × ℝ)) (by simp)
    (y := ((-1, -1) : ℝ × ℝ)) (by simp)
    (a := (1 / 2 : ℝ)) (b := (1 / 2 : ℝ)) (by norm_num) (by norm_num) (by norm_num)
  norm_num at hm
  rw [coupledLoss_negate_both, (coupledLoss_flat_axes 0 0).1] at hm
  have hb := coupledLoss_below_axes 1 1 (by norm_num)
  linarith

/-- A concrete counterexample to inferring a joint minimum from flat
coordinate restrictions and even a zero full gradient. Source comparison:
arXiv:2301.05217v1, appendix composition argument. The loss is genuine
finite-class CE with a bilinear logit, not a designed flatness predicate. -/
theorem flat_axes_and_stationary_do_not_give_minimum :
    coupledGradient 0 0 = (0, 0) ∧
      (∀ t : ℝ, coupledLoss t 0 = coupledLoss 0 0 ∧ coupledLoss 0 t = coupledLoss 0 0) ∧
      ¬IsLocalMin (fun p : ℝ × ℝ => coupledLoss p.1 p.2) (0, 0) := by
  refine ⟨coupled_origin_stationary, ?_, coupled_origin_not_local_min⟩
  intro t
  rw [(coupledLoss_flat_axes t t).1, (coupledLoss_flat_axes t t).2,
    (coupledLoss_flat_axes 0 0).1]
  exact ⟨rfl, rfl⟩

/-- Four actual losses in a two-component removal contrast. Source:
arXiv:2301.05217v1, appendix composition hypothesis; explicit deviation:
scalar removal sets a component to zero. Transformer head removal is
a separate intervention and is not identified with this parameterization. -/
noncomputable def coupledInteraction (x y : ℝ) : ℝ :=
  coupledLoss x y - coupledLoss x 0 - coupledLoss 0 y + coupledLoss 0 0

/-- The four-corner contrast in this specialization reduces to improvement
over the axes. Source: arXiv:2301.05217v1, appendix compositional hypothesis.
The contrast is computed from actual CE, not stipulated as a circuit score. -/
theorem coupledInteraction_eq (x y : ℝ) :
    coupledInteraction x y = coupledLoss x y - Real.log 2 := by
  unfold coupledInteraction
  rw [(coupledLoss_flat_axes x y).1, (coupledLoss_flat_axes x y).2,
    (coupledLoss_flat_axes 0 0).1]
  ring

/-- Aligned components have a negative loss-removal contrast. Source:
arXiv:2301.05217v1, appendix composition hypothesis, scalar specialization;
the sign alone does not identify a learned transformer algorithm. -/
theorem coupledInteraction_neg (x y : ℝ) (hxy : 0 < x * y) :
    coupledInteraction x y < 0 := by
  rw [coupledInteraction_eq]
  have h := coupledLoss_below_axes x y hxy
  linarith

example : 0 < (1 : ℝ) * 1 := by norm_num

/-- Opposed components give the opposite contrast sign. Source:
arXiv:2301.05217v1, appendix composition hypothesis, explicit bilinear CE
specialization. Different signs can occur in one fixed supervised objective. -/
theorem coupledInteraction_pos (x y : ℝ) (hxy : x * y < 0) :
    0 < coupledInteraction x y := by
  rw [coupledInteraction_eq]
  have h := coupledLoss_above_axes x y hxy
  linarith

example : (1 : ℝ) * (-1) < 0 := by norm_num

end Transformer.Grokking.Composition

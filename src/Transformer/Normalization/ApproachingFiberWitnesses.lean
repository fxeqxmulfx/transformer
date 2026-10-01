/-
# Joint accumulation from approaching witnesses in parameter fibers

If every small positive parameter has a witness arbitrarily close to the
fiber origin, the parameter-witness pairs accumulate jointly at zero.
-/

import Mathlib.Analysis.Normed.Group.Constructions
import Mathlib.Analysis.Normed.Group.Real
import Mathlib.Topology.Instances.Real.Lemmas

open Filter Set

namespace Transformer.Normalization

/-- Pointwise fiber witnesses approaching zero uniformly in the positive
parameter produce joint positive accumulation. The predicate may include
global objective comparisons and need not be analytic. Auxiliary
topological step for lifting projected gradient-minimum curves in
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem frequent_joint_of_approaching_fiber_witnesses (P : ℝ × ℝ → Prop)
    (h : ∀ delta : ℝ, 0 < delta → ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0),
      ∃ y : ℝ, |y| < delta ∧ P (t, y)) :
    ∃ᶠ x in nhds (0 : ℝ × ℝ), 0 < x.1 ∧ P x := by
  by_contra hno
  obtain ⟨r, hr, hball⟩ := Metric.eventually_nhds_iff.mp (not_frequently.mp hno)
  have : NeBot (nhdsWithin (0 : ℝ) (Ioi 0)) := nhdsWithin_Ioi_neBot le_rfl
  have hp : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < t := self_mem_nhdsWithin
  have hsmall : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), |t| < r :=
    (continuousAt_id.abs.eventually (Iio_mem_nhds (by simpa using hr))).filter_mono
      nhdsWithin_le_nhds
  obtain ⟨t, ht, hy⟩ := ((hp.and hsmall).and (h r hr)).exists
  obtain ⟨y, hy, hP⟩ := hy
  apply hball (y := (t, y))
    (by simpa only [dist_zero_right, Prod.norm_def, Real.norm_eq_abs] using max_lt ht.2 hy)
  exact ⟨ht.1, hP⟩

/-- Diagonal witnesses `y=t` approach the origin and give genuine
positive-side joint accumulation. This satisfies the witness condition
for every positive radius. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
example : ∃ᶠ x in nhds (0 : ℝ × ℝ), 0 < x.1 ∧ x.2 = x.1 := by
  apply frequent_joint_of_approaching_fiber_witnesses (fun x => x.2 = x.1)
  intro delta hd
  have hsmall : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), |t| < delta :=
    (continuousAt_id.abs.eventually (Iio_mem_nhds (by simpa using hd))).filter_mono
      nhdsWithin_le_nhds
  exact hsmall.mono (fun t ht => ⟨t, ht, rfl⟩)

end Transformer.Normalization

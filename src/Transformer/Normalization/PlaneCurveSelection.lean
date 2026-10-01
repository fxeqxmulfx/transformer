/-
# Analytic curve selection in prepared real plane zero sets

A prepared analytic zero set in the plane admits curve selection under
arbitrary finite analytic sign constraints. Axis isolation, orientation,
and constrained ramified branch lifting provide all geometric steps.
-/

import Transformer.Normalization.PlaneCurveOrientation

open Filter Set

namespace Transformer.Normalization

open AnalyticPreparation

/-- Nonzero zeros of a real plane analytic equation satisfying finite sign
requirements. Auxiliary set for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
def preparedPlaneSignSet {l : ℕ} (F : Ambient 1 → ℝ)
    (C : Fin l → Ambient 1 → ℝ) (requirement : Fin l → AnalyticSignRequirement) :
    Set (Ambient 1) :=
  {x | F x = 0 ∧ x ≠ 0 ∧ ∀ i, (requirement i).Holds (C i x)}

/-- Every accumulation point at the origin of a prepared real plane zero
set with finite analytic sign requirements is reached by a nonconstant
real analytic curve staying in that set for positive parameters.
The distinguished-variable order is arbitrary and roots may be multiple.
This is genuine curve selection, with the base-curve orientation derived
from accumulation rather than assumed. Auxiliary geometric input for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. The quantified minimum
sets in the general higher-dimensional theorem require a further
projection argument. -/
theorem prepared_plane_curve_selection {d l : ℕ} (F : Ambient 1 → ℝ)
    (hF : AnalyticAt ℝ F 0) (horder : ExactOrderInLastVariable F d)
    (C : Fin l → Ambient 1 → ℝ) (hC : ∀ i, AnalyticAt ℝ (C i) 0)
    (requirement : Fin l → AnalyticSignRequirement)
    (hacc : (0 : Ambient 1) ∈ closure (preparedPlaneSignSet F C requirement)) :
    ∃ (curve : ℝ → Ambient 1) (epsilon : ℝ), 0 < epsilon ∧
      AnalyticAt ℝ curve 0 ∧ curve 0 = 0 ∧
        ∀ t ∈ Ioo 0 epsilon, curve t ∈ preparedPlaneSignSet F C requirement := by
  obtain ⟨sign, hsign, hside⟩ := prepared_plane_positive_accumulation F hF horder C requirement hacc
  have hsign0 : sign ≠ 0 := by
    rcases hsign with h | h <;> rw [h] <;> norm_num
  let gamma : ℝ → Base 1 := fun t _ => sign * t
  have hgamma : AnalyticAt ℝ gamma 0 :=
    AnalyticAt.pi (fun _ => analyticAt_const.fun_mul analyticAt_id)
  have hgamma0 : gamma 0 = 0 := by ext i; simp [gamma]
  obtain ⟨q, g, hq, hg, hg0, hroot, hconditions⟩ :=
    analytic_equation_curve_lifting F hF horder gamma hgamma hgamma0 C hC requirement hside
  let curve : ℝ → Ambient 1 := fun t => (gamma (t ^ q), g t)
  have hgammap : AnalyticAt ℝ gamma ((0 : ℝ) ^ q) := by simpa [hq.ne'] using hgamma
  have hcurve : AnalyticAt ℝ curve 0 :=
    (hgammap.comp (f := fun t : ℝ => t ^ q) (x := 0) (analyticAt_id.fun_pow q)).prod hg
  have hcurve0 : curve 0 = 0 := by simp [curve, hq.ne', hgamma0, hg0, Prod.mk_zero_zero]
  have hpositive : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < t := self_mem_nhdsWithin
  have hpoints : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0),
      curve t ∈ preparedPlaneSignSet F C requirement := by
    filter_upwards [hroot.filter_mono nhdsWithin_le_nhds, hconditions, hpositive] with t ht hc htp
    refine ⟨ht, ?_, hc⟩
    intro heq
    have hz : sign * t ^ q = 0 := by
      simpa only [curve, gamma, Prod.fst_zero, Pi.zero_apply] using
        congrArg (fun x : Ambient 1 => x.1 0) heq
    exact mul_ne_zero hsign0 (pow_ne_zero _ htp.ne') hz
  obtain ⟨epsilon, he, hball⟩ :=
    Metric.eventually_nhds_iff.mp (eventually_nhdsWithin_iff.mp hpoints)
  refine ⟨curve, epsilon, he, hcurve, hcurve0, ?_⟩
  intro t ht
  apply hball
  · simpa only [Real.dist_eq, sub_zero, abs_of_pos ht.1] using ht.2
  · exact ht.1

end Transformer.Normalization

/-
# Positive-side sign stability of real analytic germs

Finite analytic order makes a scalar germ either zero, positive, or negative
on a sufficiently small positive interval. Accumulating sign constraints
therefore hold eventually on that interval.
-/

import Mathlib.Analysis.Analytic.IsolatedZeros

open Filter Set

namespace Transformer.Normalization

/-- The three basic sign requirements used in local semianalytic
conditions. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
inductive AnalyticSignRequirement
  | zero
  | positive
  | nonnegative

/-- Actual equality, strict positivity, or nonnegativity of a real value.
Auxiliary predicate for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def AnalyticSignRequirement.Holds (requirement : AnalyticSignRequirement) (v : ℝ) : Prop :=
  match requirement with
  | .zero => v = 0
  | .positive => 0 < v
  | .nonnegative => 0 ≤ v

/-- A real analytic germ has a constant zero, positive, or negative sign
on a sufficiently small positive interval. This includes germs of any
finite order and identically zero germs. Auxiliary for constrained curve
selection in Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem scalar_analytic_sign_trichotomy (f : ℝ → ℝ) (hf : AnalyticAt ℝ f 0) :
    (∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), f t = 0) ∨
      (∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < f t) ∨
        (∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), f t < 0) := by
  by_cases hzero : ∀ᶠ t in nhds (0 : ℝ), f t = 0
  · exact Or.inl (hzero.filter_mono nhdsWithin_le_nhds)
  obtain ⟨m, g, hg, hg0, hfactor⟩ := hf.exists_eventuallyEq_pow_smul_nonzero_iff.mpr hzero
  have hpositive : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < t := self_mem_nhdsWithin
  rcases lt_or_gt_of_ne hg0 with hnegative | hunit
  · right; right
    have hnear : ∀ᶠ t in nhds (0 : ℝ), g t < 0 := hg.continuousAt.eventually (Iio_mem_nhds hnegative)
    filter_upwards [hfactor.filter_mono nhdsWithin_le_nhds,
      hnear.filter_mono nhdsWithin_le_nhds, hpositive] with t ht hgt htp
    simpa only [ht, sub_zero, smul_eq_mul] using mul_neg_of_pos_of_neg (pow_pos htp m) hgt
  · right; left
    have hnear : ∀ᶠ t in nhds (0 : ℝ), 0 < g t := hg.continuousAt.eventually (Ioi_mem_nhds hunit)
    filter_upwards [hfactor.filter_mono nhdsWithin_le_nhds,
      hnear.filter_mono nhdsWithin_le_nhds, hpositive] with t ht hgt htp
    simpa only [ht, sub_zero, smul_eq_mul] using mul_pos (pow_pos htp m) hgt

/-- An analytic equality or sign constraint holding frequently from the
positive side holds on an entire sufficiently small positive interval.
Auxiliary for constrained real curve lifting in Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. -/
theorem analytic_sign_requirement_of_frequent (f : ℝ → ℝ) (hf : AnalyticAt ℝ f 0)
    (requirement : AnalyticSignRequirement)
    (h : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), requirement.Holds (f t)) :
    ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), requirement.Holds (f t) := by
  have hsign := scalar_analytic_sign_trichotomy f hf
  cases requirement with
  | zero =>
    rcases hsign with hz | hp | hn
    · exact hz
    · obtain ⟨t, ht, htp⟩ := (h.and_eventually hp).exists
      exact (htp.ne' ht).elim
    · obtain ⟨t, ht, htn⟩ := (h.and_eventually hn).exists
      exact (htn.ne ht).elim
  | positive =>
    rcases hsign with hz | hp | hn
    · obtain ⟨t, ht, htz⟩ := (h.and_eventually hz).exists
      exact (ht.ne' htz).elim
    · exact hp
    · obtain ⟨t, ht, htn⟩ := (h.and_eventually hn).exists
      exact (not_lt_of_gt htn ht).elim
  | nonnegative =>
    rcases hsign with hz | hp | hn
    · exact hz.mono (fun t ht => by change 0 ≤ f t; rw [ht])
    · exact hp.mono (fun t ht => ht.le)
    · obtain ⟨t, ht, htn⟩ := (h.and_eventually hn).exists
      exact (not_lt_of_ge ht htn).elim

/-- The odd-order germ `t³` satisfies a positive accumulating sign
requirement and is eventually positive on the positive side. Both sign
stability hypotheses are simultaneously exercised. Auxiliary example
for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0),
    AnalyticSignRequirement.positive.Holds (t ^ 3) := by
  have : NeBot (nhdsWithin (0 : ℝ) (Ioi 0)) := nhdsWithin_Ioi_neBot le_rfl
  apply analytic_sign_requirement_of_frequent (fun t : ℝ => t ^ 3)
    (analyticAt_id.fun_pow 3) .positive
  have hpositive : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < t := self_mem_nhdsWithin
  exact hpositive.frequently.mono (fun t ht => pow_pos ht 3)

end Transformer.Normalization

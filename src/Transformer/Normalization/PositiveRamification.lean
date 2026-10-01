/-
# Power substitutions preserve positive-side accumulation

Every positive parameter has a unique positive real power root. The inverse
power tends to zero on the positive side, so frequently occurring properties
remain frequent after a positive integral power substitution.
-/

import Mathlib.Analysis.SpecialFunctions.Pow.Continuity

open Filter Set

namespace Transformer.Normalization

/-- A positive integral power substitution preserves accumulation of any
property on the positive side of zero. No regularity of the property is
assumed. Auxiliary for ramified real curve lifting in Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem frequently_positive_power_iff (q : ℕ) (hq : 0 < q) (P : ℝ → Prop) :
    (∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), P (t ^ q)) ↔
      ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), P t := by
  have hpositive : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < t := self_mem_nhdsWithin
  have hpow : Tendsto (fun t : ℝ => t ^ q)
      (nhdsWithin 0 (Ioi 0)) (nhdsWithin 0 (Ioi 0)) := by
    apply tendsto_nhdsWithin_iff.mpr
    constructor
    · have h : Tendsto (fun t : ℝ => t ^ q) (nhds 0) (nhds 0) := by
        have hc : ContinuousAt (fun t : ℝ => t ^ q) 0 := by fun_prop
        simpa only [zero_pow hq.ne'] using hc.tendsto
      exact h.mono_left nhdsWithin_le_nhds
    · exact hpositive.mono (fun t ht => pow_pos ht q)
  have hqR : (0 : ℝ) < q := by exact_mod_cast hq
  let root : ℝ → ℝ := fun t => t ^ ((q : ℝ)⁻¹)
  have hroot : Tendsto root (nhdsWithin 0 (Ioi 0)) (nhdsWithin 0 (Ioi 0)) := by
    apply tendsto_nhdsWithin_iff.mpr
    constructor
    · exact (tendsto_id.mono_right nhdsWithin_le_nhds).rpow_const_nhds_zero (inv_pos.mpr hqR)
    · exact hpositive.mono (fun t ht => Real.rpow_pos_of_pos ht _)
  constructor
  · exact hpow.frequently
  · intro h
    apply hroot.frequently
    apply (h.and_eventually hpositive).mono
    intro t ht
    change P ((t ^ ((q : ℝ)⁻¹)) ^ q)
    rw [Real.rpow_inv_natCast_pow ht.2.le hq.ne']
    exact ht.1

/-- The positive parameters themselves accumulate at zero, and the
ramification of degree three preserves this property. This satisfies
the positive-degree hypothesis and exercises both implications. Auxiliary
example for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : (∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < t ^ 3) ↔
    ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < t :=
  frequently_positive_power_iff 3 (by omega) (fun t => 0 < t)

end Transformer.Normalization

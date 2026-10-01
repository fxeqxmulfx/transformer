/-
# Complete real root coverage after a common ramification

Removing analytic root factors one at a time gives finitely many analytic
branches covering every real root at all sufficiently small positive
parameters. The number of branches is bounded by the polynomial degree.
-/

import Transformer.Normalization.RamifiedPolynomialRoot
import Transformer.Normalization.PolynomialRootDivision
import Mathlib.Data.Fin.Tuple.Basic

open Filter Set

namespace Transformer.Normalization

/-- All real roots of an analytic monic polynomial family with central
polynomial `y^d` are covered by finitely many analytic branches after one
common positive integral ramification. Coverage holds at every sufficiently
small positive new parameter, and there are at most `d` branches. The
statement includes families with no real roots and repeated roots.
Auxiliary for constrained real curve selection in Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem analytic_ramified_polynomial_branches (d : ℕ) (a : Fin d → ℝ → ℝ)
    (ha : ∀ i, AnalyticAt ℝ (a i) 0) (ha0 : ∀ i, a i 0 = 0) :
    ∃ (q k : ℕ) (g : Fin k → ℝ → ℝ), 0 < q ∧ k ≤ d ∧
      (∀ i, AnalyticAt ℝ (g i) 0 ∧ g i 0 = 0) ∧
      ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), ∀ y : ℝ,
        polynomialFamily d a (s ^ q, y) = 0 ↔ ∃ i, y = g i s := by
  classical
  induction d with
  | zero =>
    refine ⟨1, 0, Fin.elim0, by omega, le_rfl, fun i => Fin.elim0 i, ?_⟩
    exact Eventually.of_forall (fun s y => by simp [polynomialFamily])
  | succ n ih =>
    by_cases hroots : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0),
        ∃ y : ℝ, polynomialFamily (n + 1) a (t, y) = 0
    · obtain ⟨q, g, hq, hg, hg0, hgroot⟩ := analytic_ramified_polynomial_root _ a ha ha0 hroots
      obtain ⟨b, hb, hb0, hfactor⟩ :=
        analytic_polynomial_division_by_root a ha ha0 hq g hg hg0 hgroot
      obtain ⟨r, k, beta, hr, hk, hbeta, hcover⟩ := ih b hb hb0
      have hQ : 0 < r * q := Nat.mul_pos hr hq
      have hpower : AnalyticAt ℝ (fun s : ℝ => s ^ r) 0 := analyticAt_id.fun_pow r
      have hgat : AnalyticAt ℝ g ((0 : ℝ) ^ r) := by simpa [hr.ne'] using hg
      let first : ℝ → ℝ := fun s => g (s ^ r)
      have hfirst : AnalyticAt ℝ first 0 :=
        hgat.comp (f := fun s : ℝ => s ^ r) (x := 0) hpower
      have hfirst0 : first 0 = 0 := by simp [first, hr.ne', hg0]
      let branches : Fin (k + 1) → ℝ → ℝ := Fin.cons first beta
      have hbranches (i : Fin (k + 1)) : AnalyticAt ℝ (branches i) 0 ∧ branches i 0 = 0 := by
        refine Fin.cases ?_ (fun j => ?_) i
        · exact ⟨hfirst, hfirst0⟩
        · exact hbeta j
      refine ⟨r * q, k + 1, branches, hQ, by omega, hbranches, ?_⟩
      have ht : Tendsto (fun s : ℝ => s ^ r)
          (nhdsWithin 0 (Ioi 0)) (nhds 0) := by
        have ht0 : Tendsto (fun s : ℝ => s ^ r) (nhds 0) (nhds 0) := by
          simpa [hr.ne'] using hpower.continuousAt.tendsto
        exact ht0.mono_left nhdsWithin_le_nhds
      filter_upwards [hcover, ht.eventually hfactor] with s hs hfac
      intro y
      have heq : polynomialFamily (n + 1) a (s ^ (r * q), y) =
          (y - first s) * polynomialFamily n b (s ^ r, y) := by
        simpa only [← pow_mul, first] using hfac y
      rw [heq, mul_eq_zero, sub_eq_zero, hs y, Fin.exists_fin_succ]
      rfl
    · refine ⟨1, 0, Fin.elim0, by omega, by omega, fun i => Fin.elim0 i, ?_⟩
      filter_upwards [not_frequently.mp hroots] with s hs
      intro y
      constructor
      · intro hy
        exact (hs ⟨y, by simpa only [pow_one] using hy⟩).elim
      · rintro ⟨i, hi⟩
        exact Fin.elim0 i

/-- The cubic `(y² - t)(y - t)` has three real branches for small positive
parameters, with different orders at zero. Its analytic coefficients all
vanish centrally and exercise complete coverage by the arbitrary-degree
theorem. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let a : Fin 3 → ℝ → ℝ := fun i t => if i = 0 then t ^ 2 else -t
    ∃ (q k : ℕ) (g : Fin k → ℝ → ℝ), 0 < q ∧ k ≤ 3 ∧
      (∀ i, AnalyticAt ℝ (g i) 0 ∧ g i 0 = 0) ∧
      ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), ∀ y : ℝ,
        polynomialFamily 3 a (s ^ q, y) = 0 ↔ ∃ i, y = g i s := by
  apply analytic_ramified_polynomial_branches
  · intro i
    split_ifs
    · exact analyticAt_id.fun_pow 2
    · exact analyticAt_id.fun_neg
  · intro i
    simp

end Transformer.Normalization

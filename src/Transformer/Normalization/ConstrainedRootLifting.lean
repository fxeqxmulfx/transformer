/-
# Ramified root branches retain finite analytic sign conditions

Complete finite branch coverage and one-variable sign stability select a
branch satisfying all accumulating analytic equality and inequality
conditions on a positive interval.
-/

import Transformer.Normalization.RamifiedBranchIdentity
import Transformer.Normalization.AnalyticSignStability

open Filter Set

namespace Transformer.Normalization

/-- A ramified analytic root branch can be selected subject to any finite
family of analytic zero, positive, and nonnegative sign requirements which
hold along accumulating positive-parameter roots. The root identity holds
on a full neighborhood, and every sign requirement holds on a sufficiently
small positive interval. Auxiliary constrained lifting step in Appendix
D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analytic_root_with_sign_constraints {d l : ℕ} (a : Fin d → ℝ → ℝ)
    (ha : ∀ i, AnalyticAt ℝ (a i) 0) (ha0 : ∀ i, a i 0 = 0)
    (F : Fin l → ℝ × ℝ → ℝ) (hF : ∀ i, AnalyticAt ℝ (F i) 0)
    (requirement : Fin l → AnalyticSignRequirement)
    (hroots : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), ∃ y : ℝ,
      polynomialFamily d a (t, y) = 0 ∧ ∀ i, (requirement i).Holds (F i (t, y))) :
    ∃ (q : ℕ) (g : ℝ → ℝ), 0 < q ∧ AnalyticAt ℝ g 0 ∧ g 0 = 0 ∧
      (∀ᶠ s in nhds (0 : ℝ), polynomialFamily d a (s ^ q, g s) = 0) ∧
      ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), ∀ i,
        (requirement i).Holds (F i (s ^ q, g s)) := by
  obtain ⟨q, k, branches, hq, _, hbranches, hcover⟩ :=
    analytic_ramified_polynomial_branches d a ha ha0
  have hramified : ∃ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), ∃ y : ℝ,
      polynomialFamily d a (s ^ q, y) = 0 ∧ ∀ i,
        (requirement i).Holds (F i (s ^ q, y)) :=
    (frequently_positive_power_iff q hq (fun t => ∃ y : ℝ,
      polynomialFamily d a (t, y) = 0 ∧ ∀ i, (requirement i).Holds (F i (t, y)))).mpr hroots
  have hfrequent : ∃ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), ∃ j : Fin k,
      ∀ i, (requirement i).Holds (F i (s ^ q, branches j s)) := by
    apply (hramified.and_eventually hcover).mono
    rintro s ⟨⟨y, hy, hconditions⟩, hs⟩
    obtain ⟨j, hj⟩ := (hs y).mp hy
    refine ⟨j, ?_⟩
    simpa only [← hj] using hconditions
  obtain ⟨j, hj⟩ := frequently_exists.mp hfrequent
  have hg : AnalyticAt ℝ (branches j) 0 := (hbranches j).1
  have hg0 : branches j 0 = 0 := (hbranches j).2
  let path : ℝ → ℝ × ℝ := fun s => (s ^ q, branches j s)
  have hpath : AnalyticAt ℝ path 0 := (analyticAt_id.fun_pow q).prod hg
  have hpath0 : path 0 = 0 := by simp [path, hq.ne', hg0, Prod.mk_zero_zero]
  have hconstraints (i : Fin l) : ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0),
      (requirement i).Holds (F i (s ^ q, branches j s)) := by
    have hFi : AnalyticAt ℝ (F i) (path 0) := by simpa only [hpath0] using hF i
    exact analytic_sign_requirement_of_frequent (fun s => F i (path s))
      (hFi.comp (f := path) (x := 0) hpath) (requirement i) (hj.mono (fun s hs => hs i))
  have hrootPositive : ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0),
      polynomialFamily d a (s ^ q, branches j s) = 0 :=
    hcover.mono (fun s hs => (hs _).mpr ⟨j, rfl⟩)
  exact ⟨q, branches j, hq, hg, hg0,
    analytic_ramified_branch_identity a ha hq (branches j) hg hg0 hrootPositive,
    eventually_all.mpr hconstraints⟩

/-- The square-root family `y² = t` satisfies equality of the equation,
strict positivity of `y`, and nonnegativity of `t` simultaneously at every
positive parameter. The selected branch retains all three distinct kinds
of sign requirement. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
example : let a : Fin 2 → ℝ → ℝ := fun i t => if i = 0 then -t else 0
    let F : Fin 3 → ℝ × ℝ → ℝ := fun i x => if i = 0 then x.2 ^ 2 - x.1 else
      if i = 1 then x.2 else x.1
    let requirement : Fin 3 → AnalyticSignRequirement := fun i =>
      if i = 0 then .zero else if i = 1 then .positive else .nonnegative
    ∃ (q : ℕ) (g : ℝ → ℝ), 0 < q ∧ AnalyticAt ℝ g 0 ∧ g 0 = 0 ∧
      (∀ᶠ s in nhds (0 : ℝ), polynomialFamily 2 a (s ^ q, g s) = 0) ∧
      ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), ∀ i,
        (requirement i).Holds (F i (s ^ q, g s)) := by
  intro a F requirement
  apply analytic_root_with_sign_constraints (a := a) (F := F) (requirement := requirement)
  · intro i
    fin_cases i
    · exact analyticAt_id.fun_neg
    · exact analyticAt_const
  · intro i
    simp [a]
  · intro i
    fin_cases i
    · exact (analyticAt_snd.fun_pow 2).fun_sub analyticAt_fst
    · exact analyticAt_snd
    · exact analyticAt_fst
  · have : NeBot (nhdsWithin (0 : ℝ) (Ioi 0)) := nhdsWithin_Ioi_neBot le_rfl
    have hpositive : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < t := self_mem_nhdsWithin
    apply hpositive.frequently.mono
    intro t ht
    refine ⟨Real.sqrt t, ?_, ?_⟩
    · simpa [a, polynomialFamily, sub_eq_add_neg] using sub_eq_zero.mpr (Real.sq_sqrt ht.le)
    · intro i
      fin_cases i
      · simpa [F, requirement, AnalyticSignRequirement.Holds] using
          sub_eq_zero.mpr (Real.sq_sqrt ht.le)
      · exact Real.sqrt_pos.mpr ht
      · exact ht.le

end Transformer.Normalization

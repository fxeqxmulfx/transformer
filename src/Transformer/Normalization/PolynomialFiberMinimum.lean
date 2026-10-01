/-
# Analytic minimizers over every real root in a prepared fiber

Complete ramified root coverage and stable finite minima eliminate the
universal objective comparison over the distinguished-variable roots.
Finite analytic equality and sign constraints are retained.
-/

import Transformer.Normalization.FiniteAnalyticMinima
import Transformer.Normalization.RamifiedBranchIdentity

open Filter Set

namespace Transformer.Normalization

/-- An analytic monic family whose central polynomial is a pure power
admits an analytic constrained minimum over every real root after positive
integral ramification. Root existence is only assumed at accumulating
positive parameters. The selected root minimizes an arbitrary analytic
objective against all admissible real roots, including repeated roots;
there is no assumed minimizer or branch-selection premise. Auxiliary for
the quantified fiber step of Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem analytic_polynomial_fiber_minimum {d l : ℕ} (a : Fin d → ℝ → ℝ)
    (ha : ∀ i, AnalyticAt ℝ (a i) 0) (ha0 : ∀ i, a i 0 = 0)
    (V : ℝ × ℝ → ℝ) (hV : AnalyticAt ℝ V 0)
    (C : Fin l → ℝ × ℝ → ℝ) (hC : ∀ i, AnalyticAt ℝ (C i) 0)
    (requirement : Fin l → AnalyticSignRequirement)
    (hroots : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), ∃ y : ℝ,
      polynomialFamily d a (t, y) = 0 ∧ ∀ i, (requirement i).Holds (C i (t, y))) :
    ∃ (q : ℕ) (g : ℝ → ℝ), 0 < q ∧ AnalyticAt ℝ g 0 ∧ g 0 = 0 ∧
      (∀ᶠ s in nhds (0 : ℝ), polynomialFamily d a (s ^ q, g s) = 0) ∧
      ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0),
        (∀ i, (requirement i).Holds (C i (s ^ q, g s))) ∧
          ∀ y : ℝ, polynomialFamily d a (s ^ q, y) = 0 →
            (∀ i, (requirement i).Holds (C i (s ^ q, y))) →
              V (s ^ q, g s) ≤ V (s ^ q, y) := by
  obtain ⟨q, k, branches, hq, _, hbranches, hcover⟩ :=
    analytic_ramified_polynomial_branches d a ha ha0
  let path : Fin k → ℝ → ℝ × ℝ := fun j s => (s ^ q, branches j s)
  have hpath (j : Fin k) : AnalyticAt ℝ (path j) 0 :=
    (analyticAt_id.fun_pow q).prod (hbranches j).1
  have hpath0 (j : Fin k) : path j 0 = 0 := by
    simp [path, hq.ne', (hbranches j).2, Prod.mk_zero_zero]
  have hvalues (j : Fin k) : AnalyticAt ℝ (fun s => V (path j s)) 0 :=
    (by simpa only [hpath0 j] using hV : AnalyticAt ℝ V (path j 0)).comp
      (f := path j) (x := 0) (hpath j)
  have hconstraints (j : Fin k) (i : Fin l) :
      AnalyticAt ℝ (fun s => C i (path j s)) 0 :=
    (by simpa only [hpath0 j] using hC i : AnalyticAt ℝ (C i) (path j 0)).comp
      (f := path j) (x := 0) (hpath j)
  have hramified : ∃ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), ∃ y : ℝ,
      polynomialFamily d a (s ^ q, y) = 0 ∧ ∀ i,
        (requirement i).Holds (C i (s ^ q, y)) :=
    (frequently_positive_power_iff q hq (fun t => ∃ y : ℝ,
      polynomialFamily d a (t, y) = 0 ∧ ∀ i, (requirement i).Holds (C i (t, y)))).mpr hroots
  have hfeasible : ∃ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), ∃ j : Fin k,
      ∀ i, (requirement i).Holds (C i (path j s)) := by
    apply (hramified.and_eventually hcover).mono
    rintro s ⟨⟨y, hy, hc⟩, hs⟩
    obtain ⟨j, hj⟩ := (hs y).mp hy
    exact ⟨j, by simpa only [path, ← hj] using hc⟩
  obtain ⟨j, hj⟩ := analytic_finite_minimum_branch (fun b s => V (path b s)) hvalues
    (fun b i s => C i (path b s)) hconstraints requirement hfeasible
  have hroot : ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0),
      polynomialFamily d a (s ^ q, branches j s) = 0 :=
    hcover.mono (fun s hs => (hs _).mpr ⟨j, rfl⟩)
  refine ⟨q, branches j, hq, (hbranches j).1, (hbranches j).2,
    analytic_ramified_branch_identity a ha hq (branches j)
      (hbranches j).1 (hbranches j).2 hroot, ?_⟩
  filter_upwards [hj, hcover] with s hs hcov
  refine ⟨hs.1, ?_⟩
  intro y hy hc
  obtain ⟨b, hb⟩ := (hcov y).mp hy
  simpa only [path, ← hb] using hs.2 b (by simpa only [path, ← hb] using hc)

/-- The repeated-root quartic `(y²-t)²` has two admissible real roots for
positive `t`; the objective `y` requires selecting the lower one. Thus
root multiplicity, genuine minimization and analytic constraints are
simultaneously exercised. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
example : let a : Fin 4 → ℝ → ℝ := fun i t =>
      if i = 0 then t ^ 2 else if i = 2 then -2 * t else 0
    ∃ (q : ℕ) (g : ℝ → ℝ), 0 < q ∧ AnalyticAt ℝ g 0 ∧ g 0 = 0 ∧
      (∀ᶠ s in nhds (0 : ℝ), polynomialFamily 4 a (s ^ q, g s) = 0) ∧
        ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), (0 ≤ s ^ q) ∧
          ∀ y : ℝ, polynomialFamily 4 a (s ^ q, y) = 0 → 0 ≤ s ^ q → g s ≤ y := by
  intro a
  have ha (i : Fin 4) : AnalyticAt ℝ (a i) 0 := by
    dsimp only [a]
    split_ifs
    · exact analyticAt_id.fun_pow 2
    · exact analyticAt_const.fun_mul analyticAt_id
    · exact analyticAt_const
  have ha0 (i : Fin 4) : a i 0 = 0 := by simp [a]
  have hroots : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), ∃ y : ℝ,
      polynomialFamily 4 a (t, y) = 0 ∧ ∀ i : Fin 1,
        AnalyticSignRequirement.nonnegative.Holds t := by
    have : NeBot (nhdsWithin (0 : ℝ) (Ioi 0)) := nhdsWithin_Ioi_neBot le_rfl
    have hp : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < t := self_mem_nhdsWithin
    apply hp.frequently.mono
    intro t ht
    refine ⟨Real.sqrt t, ?_, fun i => ht.le⟩
    have heq : polynomialFamily 4 a (t, Real.sqrt t) = ((Real.sqrt t) ^ 2 - t) ^ 2 := by
      simp [a, polynomialFamily, Fin.sum_univ_succ]
      ring
    rw [heq, Real.sq_sqrt ht.le, sub_self, zero_pow (by omega : (2 : ℕ) ≠ 0)]
  simpa only [AnalyticSignRequirement.Holds, forall_const] using
    analytic_polynomial_fiber_minimum a ha ha0 (fun x => x.2) analyticAt_snd
      (fun (_ : Fin 1) x => x.1) (fun _ => analyticAt_fst) (fun _ => .nonnegative) hroots

end Transformer.Normalization

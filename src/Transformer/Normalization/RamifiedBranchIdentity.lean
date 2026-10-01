/-
# Full analytic identities from positive ramified root branches

Analytic isolated zeros upgrade a root identity holding on a positive
interval to an identity of the full germ at zero.
-/

import Transformer.Normalization.RamifiedPolynomialBranches
import Mathlib.Analysis.Analytic.IsolatedZeros

open Filter Set

namespace Transformer.Normalization

/-- A positive-side analytic polynomial root branch satisfies the equation
on a full neighborhood of zero, including the negative side. This uses
analytic isolated zeros rather than a continuity argument. Auxiliary for
analytic fiber minima in Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem analytic_ramified_branch_identity {d q : ℕ} (a : Fin d → ℝ → ℝ)
    (ha : ∀ i, AnalyticAt ℝ (a i) 0) (hq : 0 < q)
    (g : ℝ → ℝ) (hg : AnalyticAt ℝ g 0) (hg0 : g 0 = 0)
    (hroot : ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0),
      polynomialFamily d a (s ^ q, g s) = 0) :
    ∀ᶠ s in nhds (0 : ℝ), polynomialFamily d a (s ^ q, g s) = 0 := by
  let path : ℝ → ℝ × ℝ := fun s => (s ^ q, g s)
  have hpath : AnalyticAt ℝ path 0 := (analyticAt_id.fun_pow q).prod hg
  have hpath0 : path 0 = 0 := by simp [path, hq.ne', hg0, Prod.mk_zero_zero]
  have hPa : AnalyticAt ℝ (fun x : ℝ × ℝ => polynomialFamily d a x) 0 := by
    simpa only [zero_add] using polynomialFamily_analyticAt a ha 0
  have hPas : AnalyticAt ℝ (fun s => polynomialFamily d a (path s)) 0 :=
    (by simpa only [hpath0] using hPa : AnalyticAt ℝ
      (fun x : ℝ × ℝ => polynomialFamily d a x) (path 0)).comp
        (f := path) (x := 0) hpath
  have : NeBot (nhdsWithin (0 : ℝ) (Ioi 0)) := nhdsWithin_Ioi_neBot le_rfl
  have hpuncture : nhdsWithin (0 : ℝ) (Ioi 0) ≤ nhdsWithin (0 : ℝ) {0}ᶜ :=
    nhdsWithin_mono _ (fun t ht => by simpa only [mem_compl_iff, mem_singleton_iff] using ht.ne')
  exact hPas.frequently_zero_iff_eventually_zero.mp
    ((tendsto_id.mono_right hpuncture).frequently hroot.frequently)

/-- For `y² = t`, the ramified branch `t=s²`, `y=s` satisfies the
positive-side equation and is an analytic root on the full germ. All
identity-extension hypotheses are satisfied by nonconstant data.
Auxiliary example for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let a : Fin 2 → ℝ → ℝ := fun i t => if i = 0 then -t else 0
    ∀ᶠ s in nhds (0 : ℝ), polynomialFamily 2 a (s ^ 2, s) = 0 := by
  intro a
  apply analytic_ramified_branch_identity a (q := 2) (g := fun s => s)
  · intro i
    fin_cases i
    · exact analyticAt_id.fun_neg
    · exact analyticAt_const
  · omega
  · exact analyticAt_id
  · rfl
  · exact Eventually.of_forall (fun s => by
      simp [a, polynomialFamily])

end Transformer.Normalization

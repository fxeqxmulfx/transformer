/-
# Finite root orders and strict multiplicity reduction

Polynomial root multiplicity is the analytic order of the shifted polynomial
germ. For a centered monic polynomial which is not a pure power, every real
root has multiplicity strictly smaller than its degree. These facts give
the decreasing induction parameter in ramified analytic root lifting.
-/

import Transformer.AnalyticPreparation.Basic
import Mathlib.Analysis.Analytic.Polynomial
import Mathlib.Algebra.Polynomial.RingDivision

open Filter Polynomial

namespace Transformer.Normalization

open AnalyticPreparation

/-- The analytic order of a nonzero real polynomial shifted to a base point
is exactly its algebraic root multiplicity. The quotient by the maximal
power of `X - r` is a nonvanishing analytic unit at `r`. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem polynomial_analytic_order_at_root (P : Polynomial ℝ) (hP : P ≠ 0) (r : ℝ) :
    analyticOrderAt (fun t : ℝ => P.eval (r + t)) 0 = (P.rootMultiplicity r : ℕ) := by
  let m := P.rootMultiplicity r
  let Q := P /ₘ (X - C r) ^ m
  let g : ℝ → ℝ := fun t => Q.eval (r + t)
  have hshift : AnalyticAt ℝ (fun t : ℝ => r + t) 0 := analyticAt_const.fun_add analyticAt_id
  have hQ : AnalyticAt ℝ (fun y : ℝ => Q.eval y) (r + 0) :=
    AnalyticOnNhd.eval_polynomial Q _ (Set.mem_univ _)
  have hg : AnalyticAt ℝ g 0 :=
    hQ.comp (f := fun t : ℝ => r + t) (x := 0) hshift
  have hg0 : g 0 ≠ 0 := by
    simpa only [g, add_zero, Q, m] using eval_divByMonic_pow_rootMultiplicity_ne_zero r hP
  have hPa : AnalyticAt ℝ (fun t : ℝ => P.eval (r + t)) 0 :=
    (AnalyticOnNhd.eval_polynomial P _ (Set.mem_univ _)).comp
      (f := fun t : ℝ => r + t) (x := 0) hshift
  apply hPa.analyticOrderAt_eq_natCast.mpr
  refine ⟨g, hg, hg0, Eventually.of_forall (fun t => ?_)⟩
  have hfactor := congrArg (fun q : Polynomial ℝ => q.eval (r + t))
    (pow_mul_divByMonic_rootMultiplicity_eq P r)
  simpa only [eval_mul, eval_pow, eval_sub, eval_X, eval_C,
    add_sub_cancel_left, sub_zero, smul_eq_mul, Q, g, m] using hfactor.symm

/-- A nonzero polynomial gives the exact distinguished-variable order
needed by real Weierstrass preparation, even at a multiple root. Auxiliary
for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem polynomial_exact_order_at_root {n : ℕ} (P : Polynomial ℝ)
    (hP : P ≠ 0) (r : ℝ) :
    ExactOrderInLastVariable (fun x : Ambient n => P.eval (r + x.2))
      (P.rootMultiplicity r) := by
  have ha : AnalyticAt ℝ (fun t : ℝ => P.eval (r + t)) 0 :=
    (AnalyticOnNhd.eval_polynomial P _ (Set.mem_univ _)).comp
      (f := fun t : ℝ => r + t) (x := 0) (analyticAt_const.fun_add analyticAt_id)
  exact (analyticOrderAt_eq_nat_iff_iteratedDeriv_eq_zero ha).mp
    (polynomial_analytic_order_at_root P hP r)

/-- Every root of a centered monic real polynomial which is not a pure
power has positive multiplicity strictly less than the degree. If the
entire degree were concentrated at one root, the next coefficient would
force that root to be zero, contradicting the nontrivial central
polynomial produced by Newton scaling. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem centered_polynomial_root_multiplicity_lt (P : Polynomial ℝ)
    (hmonic : P.Monic) (hcenter : P.nextCoeff = 0)
    (hnontrivial : P ≠ X ^ P.natDegree) (r : ℝ) (hroot : P.IsRoot r) :
    0 < P.rootMultiplicity r ∧ P.rootMultiplicity r < P.natDegree := by
  have hmpos : 0 < P.rootMultiplicity r := (rootMultiplicity_pos hmonic.ne_zero).mpr hroot
  have hdvd := pow_rootMultiplicity_dvd P r
  have hmle : P.rootMultiplicity r ≤ P.natDegree := by
    simpa only [natDegree_pow, natDegree_X_sub_C, mul_one] using
      natDegree_le_of_dvd hdvd hmonic.ne_zero
  refine ⟨hmpos, lt_of_le_of_ne hmle ?_⟩
  intro heq
  have hpow : P = (X - C r) ^ P.natDegree := by
    apply eq_of_monic_of_dvd_of_natDegree_le ((monic_X_sub_C r).pow _) hmonic
    · simpa only [heq] using hdvd
    · simp
  have hdpos : 0 < P.natDegree := by omega
  have hr : r = 0 := by
    rw [hpow, (monic_X_sub_C r).nextCoeff_pow, nextCoeff_X_sub_C, nsmul_eq_mul] at hcenter
    exact neg_eq_zero.mp ((mul_eq_zero.mp hcenter).resolve_left
      (by exact_mod_cast hdpos.ne'))
  apply hnontrivial
  simpa only [hr, C_0, sub_zero] using hpow

/-- The centered cubic `(X - 1)² (X + 2)` has a multiple real root at one.
It is monic, is not a pure power, and satisfies all the multiplicity and
analytic-order hypotheses together. Auxiliary example for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
example : let P : Polynomial ℝ := (X - C 1) ^ 2 * (X + C 2)
    (0 < P.rootMultiplicity 1 ∧ P.rootMultiplicity 1 < P.natDegree) ∧
      ExactOrderInLastVariable (fun x : Ambient 1 => P.eval (1 + x.2))
        (P.rootMultiplicity 1) := by
  let P : Polynomial ℝ := (X - C 1) ^ 2 * (X + C 2)
  have hmonic : P.Monic := ((monic_X_sub_C 1).pow 2).mul (monic_X_add_C 2)
  have hroot : P.IsRoot 1 := by simp [P, IsRoot]
  have hcenter : P.nextCoeff = 0 := by
    change (((X - C 1) ^ 2 * (X + C 2) : Polynomial ℝ).nextCoeff) = 0
    rw [((monic_X_sub_C (1 : ℝ)).pow 2).nextCoeff_mul (monic_X_add_C 2),
      (monic_X_sub_C (1 : ℝ)).nextCoeff_pow, nextCoeff_X_sub_C, nextCoeff_X_add_C]
    norm_num
  have hnontrivial : P ≠ X ^ P.natDegree := by
    intro h
    have heq := congrArg (fun q : Polynomial ℝ => q.eval 1) h
    simp [P] at heq
  exact ⟨centered_polynomial_root_multiplicity_lt P hmonic hcenter hnontrivial 1 hroot,
    polynomial_exact_order_at_root P hmonic.ne_zero 1⟩

end Transformer.Normalization

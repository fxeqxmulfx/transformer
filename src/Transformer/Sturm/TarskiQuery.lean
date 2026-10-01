/-
# Sturm--Tarski queries with arbitrary multiplicities and common roots

Only derivative, gcd, field division, and sign variations are used to
compute the query. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2.
-/

import Transformer.Sturm.GlobalQuery
import Transformer.Sturm.CoprimePart

noncomputable section
open Polynomial
open scoped BigOperators

namespace Transformer.Sturm

/-- The arithmetic Sturm--Tarski query: first remove multiplicities and
the roots where the query vanishes, then count signed crossings for
the derivative times the query. No root is used in the construction.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def realRootQuery (p q : Polynomial ℝ) : ℤ :=
  let s := coprimePart (squarefreePart p) q
  signedRootQueryValue s (s.derivative * q)

/-- The arithmetic query equals the sum of query signs over every
distinct real root of a nonzero polynomial. The original polynomial
may have arbitrary degree and repeated roots, and the query may vanish
at any or all roots. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem realRootQuery_eq {p : Polynomial ℝ} (hp : p ≠ 0) (q : Polynomial ℝ) :
    realRootQuery p q = ∑ r ∈ p.roots.toFinset, (SignType.sign (q.eval r) : ℤ) := by
  let s := squarefreePart p
  let c := coprimePart s q
  have hs : s ≠ 0 := squarefreePart_ne_zero hp
  have hsd : ∀ x : ℝ, s.eval x = 0 → s.derivative.eval x ≠ 0 :=
    squarefreePart_derivative_roots hp
  have hc : c ≠ 0 := coprimePart_ne_zero hs
  have hcd : ∀ x : ℝ, c.eval x = 0 → c.derivative.eval x ≠ 0 :=
    coprimePart_derivative_roots hs hsd
  have hcommon : ∀ x : ℝ, c.eval x = 0 → (c.derivative * q).eval x ≠ 0 := by
    intro x hx
    rw [eval_mul]
    exact mul_ne_zero (hcd x hx) ((coprimePart_real_roots hs hsd x).mp hx).2
  have hquery := signedRootQueryValue_eq hc hcommon hcd
  have hvalues : (∑ r ∈ c.roots.toFinset,
      (SignType.sign (c.derivative.eval r * (c.derivative * q).eval r) : ℤ)) =
      ∑ r ∈ c.roots.toFinset, (SignType.sign (q.eval r) : ℤ) := by
    apply Finset.sum_congr rfl
    intro r hr
    have hdr := hcd r ((mem_roots hc).mp (Multiset.mem_toFinset.mp hr))
    have heq : c.derivative.eval r * (c.derivative.eval r * q.eval r) =
        c.derivative.eval r ^ 2 * q.eval r := by ring
    rw [eval_mul, heq, sign_mul, sign_pos (sq_pos_of_ne_zero hdr), one_mul]
  have hroots : c.roots.toFinset = s.roots.toFinset.filter (fun r => q.eval r ≠ 0) := by
    ext r
    simp only [Finset.mem_filter, Multiset.mem_toFinset, mem_roots hc, mem_roots hs,
      Polynomial.IsRoot]
    exact coprimePart_real_roots hs hsd r
  have hsum : (∑ r ∈ c.roots.toFinset, (SignType.sign (q.eval r) : ℤ)) =
      ∑ r ∈ p.roots.toFinset, (SignType.sign (q.eval r) : ℤ) := by
    rw [hroots, Finset.sum_filter]
    have hzero : (∑ r ∈ s.roots.toFinset,
        if q.eval r ≠ 0 then (SignType.sign (q.eval r) : ℤ) else 0) =
        ∑ r ∈ s.roots.toFinset, (SignType.sign (q.eval r) : ℤ) := by
      apply Finset.sum_congr rfl
      intro r _
      by_cases hq : q.eval r = 0
      · simp only [hq, ne_self_iff_false, ite_false, sign_zero, SignType.coe_zero]
      · rw [ite_eq_left hq]
    rw [hzero, squarefreePart_roots_toFinset hp]
  exact hquery.trans (hvalues.trans hsum)

/-- A repeated-root polynomial and a query vanishing at its central
root are legitimate inputs. Auxiliary for Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. -/
example : (X ^ 4 : Polynomial ℝ) ≠ 0 ∧ (X : Polynomial ℝ).eval 0 = 0 := by
  exact ⟨pow_ne_zero _ X_ne_zero, by simp⟩

end Transformer.Sturm

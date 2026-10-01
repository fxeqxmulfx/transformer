/-
# Signed Euclidean chains over the real numbers

The chain uses actual field division of real polynomials and terminates
by strict decrease of the remainder degree. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Sturm.Chains
import Mathlib.Algebra.Polynomial.FieldDivision

noncomputable section
open Polynomial

namespace Transformer.Sturm

/-- Successive negative remainders, with no terminal zero entry.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def signedRemainderChain (p q : Polynomial ℝ) : List (Polynomial ℝ) :=
  if q = 0 then [p]
  else if p % q = 0 then [p, q]
  else p :: signedRemainderChain q (-(p % q))
termination_by q.natDegree
decreasing_by
  simpa only [natDegree_neg] using
    natDegree_lt_natDegree (by assumption) (degree_mod_lt p (by assumption))

/-- The head is the first polynomial, including constant chains.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem signedRemainderChain_head (p q : Polynomial ℝ) :
    (signedRemainderChain p q).head? = some p := by
  rw [signedRemainderChain]
  split_ifs <;> rfl

/-- A nonzero second polynomial is the second chain entry.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem signedRemainderChain_second {p q : Polynomial ℝ} (hq : q ≠ 0) :
    (signedRemainderChain p q)[1]? = some q := by
  rw [signedRemainderChain, ite_eq_right hq]
  split
  · rfl
  · simpa only [List.getElem?_cons_succ, ← List.head?_eq_getElem?] using
      signedRemainderChain_head q (-(p % q))

/-- Evaluating division at a zero of the divisor evaluates the remainder
to the dividend. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem eval_mod_of_eval_eq_zero (p q : Polynomial ℝ) {x : ℝ}
    (hq : q.eval x = 0) : (p % q).eval x = p.eval x := by
  have h := congrArg (fun f : Polynomial ℝ => f.eval x)
    (EuclideanDomain.mod_add_div p q)
  simpa only [eval_add, eval_mul, hq, zero_mul, add_zero] using h

/-- The absence of common real zeros passes to the next remainder pair.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem no_common_zero_neg_mod {p q : Polynomial ℝ}
    (h : ∀ x : ℝ, p.eval x = 0 → q.eval x ≠ 0) :
    ∀ x : ℝ, q.eval x = 0 → (-(p % q)).eval x ≠ 0 := by
  intro x hqx hrem
  have hp : p.eval x = 0 := by
    simpa only [eval_neg, eval_mod_of_eval_eq_zero p q hqx, neg_eq_zero] using hrem
  exact h x hp hqx

/-- A genuine two-step polynomial example satisfies the second-entry,
division-evaluation, and no-common-zero hypotheses jointly. Auxiliary
for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : (1 : Polynomial ℝ) ≠ 0 ∧ (1 : Polynomial ℝ).eval 0 = 1 ∧
    (∀ x : ℝ, (X : Polynomial ℝ).eval x = 0 → (1 : Polynomial ℝ).eval x ≠ 0) ∧
    (X : Polynomial ℝ).eval 0 = 0 := by
  simp

end Transformer.Sturm

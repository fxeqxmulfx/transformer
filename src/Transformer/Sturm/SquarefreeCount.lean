/-
# Root counts for arbitrary nonzero real polynomials

Every distinct root is counted once, including repeated roots of the
original polynomial. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2.
-/

import Transformer.Sturm.Construction
import Transformer.Sturm.SquarefreePart

noncomputable section
open Polynomial

namespace Transformer.Sturm

/-- All real roots of the multiplicity-free quotient are simple.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem squarefreePart_derivative_roots {p : Polynomial ℝ} (hp : p ≠ 0) :
    ∀ x : ℝ, (squarefreePart p).eval x = 0 →
      (squarefreePart p).derivative.eval x ≠ 0 := by
  intro x hx hd
  have hr := (squarefreePart_real_roots hp x).mp hx
  have hm := squarefreePart_rootMultiplicity hp hr
  have hg := (one_lt_rootMultiplicity_iff_isRoot (squarefreePart_ne_zero hp)).mpr ⟨hx, hd⟩
  rw [hm] at hg
  exact (lt_irrefl 1) hg

/-- The root multiset of the quotient contains no duplicate real roots.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem squarefreePart_roots_nodup {p : Polynomial ℝ} (hp : p ≠ 0) :
    (squarefreePart p).roots.Nodup :=
  real_roots_nodup_of_derivative (squarefreePart_ne_zero hp)
    (squarefreePart_derivative_roots hp)

/-- The finite sets of roots coincide exactly before and after
multiplicity removal. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem squarefreePart_roots_toFinset {p : Polynomial ℝ} (hp : p ≠ 0) :
    (squarefreePart p).roots.toFinset = p.roots.toFinset := by
  ext x
  simp only [Multiset.mem_toFinset, mem_roots (squarefreePart_ne_zero hp), mem_roots hp,
    Polynomial.IsRoot, squarefreePart_real_roots hp]

/-- A Sturm chain constructed solely from field divisions and the
derivative, valid for nonzero input of any degree or multiplicity.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def sturmChain (p : Polynomial ℝ) : List (Polynomial ℝ) :=
  signedRemainderChain (squarefreePart p) (squarefreePart p).derivative

/-- The constructed chain satisfies the full geometric Sturm predicate
without a simplicity assumption on the original polynomial. Auxiliary
for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem sturmChain_isSturm {p : Polynomial ℝ} (hp : p ≠ 0) :
    IsSturmChain (squarefreePart p) (sturmChain p) :=
  signedRemainderChain_isSturm (squarefreePart_ne_zero hp) (squarefreePart_derivative_roots hp)

/-- The variation drop counts all distinct real roots in `(a, b]`.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem sturmChain_count_Ioc {p : Polynomial ℝ} (hp : p ≠ 0)
    {a b : ℝ} (hab : a ≤ b) :
    sturmVar (sturmChain p) b + (p.roots.toFinset.filter (fun r => r ∈ Set.Ioc a b)).card =
      sturmVar (sturmChain p) a := by
  have hcount := (sturmChain_isSturm hp).sturm_Ioc (squarefreePart_roots_nodup hp) hab
  have hcard : ((squarefreePart p).roots.filter (fun r => r ∈ Set.Ioc a b)).card =
      (p.roots.toFinset.filter (fun r => r ∈ Set.Ioc a b)).card := by
    rw [← Multiset.toFinset_card_of_nodup ((squarefreePart_roots_nodup hp).filter _),
      Multiset.toFinset_filter, squarefreePart_roots_toFinset hp]
  simpa only [hcard] using hcount

/-- The variations at infinity count every distinct real root.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem sturmChain_count {p : Polynomial ℝ} (hp : p ≠ 0) :
    sturmVarPosInf (sturmChain p) + p.roots.toFinset.card =
      sturmVarNegInf (sturmChain p) := by
  have hcount := (sturmChain_isSturm hp).sturm (squarefreePart_roots_nodup hp)
  have hcard : (squarefreePart p).roots.card = p.roots.toFinset.card := by
    rw [← Multiset.toFinset_card_of_nodup (squarefreePart_roots_nodup hp),
      squarefreePart_roots_toFinset hp]
  simpa only [hcard] using hcount

/-- A repeated-root polynomial and a nontrivial interval jointly
witness all count hypotheses. Auxiliary for Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. -/
example : ((X ^ 2 - 1) ^ 3 : Polynomial ℝ) ≠ 0 ∧ (-2 : ℝ) ≤ 2 := by
  refine ⟨pow_ne_zero _ ?_, by norm_num⟩
  intro hp
  have h := congrArg (fun f : Polynomial ℝ => f.eval 0) hp
  norm_num at h

end Transformer.Sturm

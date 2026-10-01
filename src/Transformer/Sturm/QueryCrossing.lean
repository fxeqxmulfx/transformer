/-
# Signed Euclidean-chain crossings for real-root queries

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Sturm.SignedRootCrossing
import Transformer.Sturm.PolynomialFlanks

noncomputable section
open Polynomial

namespace Transformer.Sturm

/-- At an isolated simple root, the variation change of the actual
signed remainder chain is the sign of `p'(r) q(r)`. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem signedRemainderChain_root_query_cross {p q : Polynomial ℝ} (hp : p ≠ 0)
    (hcommon : ∀ x : ℝ, p.eval x = 0 → q.eval x ≠ 0)
    (hsimple : ∀ x : ℝ, p.eval x = 0 → p.derivative.eval x ≠ 0)
    {r a b : ℝ} (hr : p.eval r = 0) (har : a < r) (hrb : r < b)
    (hz : ∀ f ∈ signedRemainderChain p q, ∀ x ∈ Set.Icc a b,
      x ≠ r → f.eval x ≠ 0) :
    (sturmVar (signedRemainderChain p q) a : ℤ) -
      sturmVar (signedRemainderChain p q) b =
        (SignType.sign (p.derivative.eval r * q.eval r) : ℤ) := by
  have hq : q ≠ 0 := by
    intro hq
    exact hcommon r hr (by simp only [hq, eval_zero])
  have hsecond := signedRemainderChain_second (p := p) hq
  have hqmem := List.mem_of_getElem? hsecond
  have hc := signedRemainderChain_isRootAlternating hp hcommon
  have hpair := sturmVar_root_pair_cross hc r hsecond (hcommon r hr) a b har hrb hz
  have hprod : (p * q).eval r = 0 := by simp only [eval_mul, hr, zero_mul]
  have hderiv : (p * q).derivative.eval r = p.derivative.eval r * q.eval r := by
    simp only [derivative_mul, eval_add, eval_mul, hr, zero_mul, add_zero]
  have hne : (p * q).derivative.eval r ≠ 0 := by
    rw [hderiv]
    exact mul_ne_zero (hsimple r hr) (hcommon r hr)
  have hnonzero : ∀ x ∈ Set.Icc a b, x ≠ r → (p * q).eval x ≠ 0 := by
    intro x hx hxr
    rw [eval_mul]
    exact mul_ne_zero (hz p hc.head_mem x hx hxr) (hz q hqmem x hx hxr)
  rw [hpair, polynomial_zero_indicator_jump hprod hne har hrb hnonzero, hderiv]

/-- A nonzero constant query and the polynomial `X` witness the
simple-root and no-common-zero requirements. Auxiliary for Appendix
D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : (X : Polynomial ℝ) ≠ 0 ∧
    (∀ x : ℝ, (X : Polynomial ℝ).eval x = 0 → (-1 : Polynomial ℝ).eval x ≠ 0) ∧
    (∀ x : ℝ, (X : Polynomial ℝ).eval x = 0 → (X : Polynomial ℝ).derivative.eval x ≠ 0) ∧
    (X : Polynomial ℝ).eval 0 = 0 ∧ (-1 : ℝ) < 0 ∧ (0 : ℝ) < 1 := by
  refine ⟨X_ne_zero, fun _ _ => by simp, fun _ _ => by simp, by simp,
    by norm_num, by norm_num⟩

end Transformer.Sturm

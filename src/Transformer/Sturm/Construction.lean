/-
# Constructing Sturm chains for polynomials with simple real roots

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Sturm.RootFlanks
import Transformer.Sturm.Infinity

noncomputable section
open Polynomial

namespace Transformer.Sturm

/-- Nonvanishing of the derivative at each real root rules out repeated
entries in the actual real root multiset. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem real_roots_nodup_of_derivative {p : Polynomial ℝ} (hp : p ≠ 0)
    (h : ∀ x : ℝ, p.eval x = 0 → p.derivative.eval x ≠ 0) : p.roots.Nodup := by
  apply Multiset.nodup_iff_count_le_one.mpr
  intro x
  rw [count_roots]
  by_contra hx
  have hm : 1 < p.rootMultiplicity x := lt_of_not_ge hx
  obtain ⟨hr, hd⟩ := (one_lt_rootMultiplicity_iff_isRoot hp).mp hm
  exact h x hr hd

/-- The signed Euclidean chain starting with the derivative satisfies
every geometric Sturm condition. No chain is assumed as an input.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem signedRemainderChain_isSturm {p : Polynomial ℝ} (hp : p ≠ 0)
    (h : ∀ x : ℝ, p.eval x = 0 → p.derivative.eval x ≠ 0) :
    IsSturmChain p (signedRemainderChain p p.derivative) := by
  refine ⟨signedRemainderChain_head p p.derivative, ?_,
    signedRemainderChain_nonzero hp, signedRemainderChain_interior h,
    signedRemainderChain_last_no_root h⟩
  intro r hr
  have hd := h r hr
  have hq : p.derivative ≠ 0 := by
    intro hq
    exact hd (by simp only [hq, eval_zero])
  exact ⟨p.derivative, signedRemainderChain_second hq, hd,
    polynomial_derivative_root_flanks hr hd⟩

/-- Field division computes the root count on a half-open interval for
every nonzero polynomial with simple real roots. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem signedRemainderChain_count_Ioc {p : Polynomial ℝ} (hp : p ≠ 0)
    (h : ∀ x : ℝ, p.eval x = 0 → p.derivative.eval x ≠ 0)
    {a b : ℝ} (hab : a ≤ b) :
    sturmVar (signedRemainderChain p p.derivative) b +
      (p.roots.filter (fun r => r ∈ Set.Ioc a b)).card =
        sturmVar (signedRemainderChain p p.derivative) a :=
  (signedRemainderChain_isSturm hp h).sturm_Ioc (real_roots_nodup_of_derivative hp h) hab

/-- Field division also computes the total real root count.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem signedRemainderChain_count {p : Polynomial ℝ} (hp : p ≠ 0)
    (h : ∀ x : ℝ, p.eval x = 0 → p.derivative.eval x ≠ 0) :
    sturmVarPosInf (signedRemainderChain p p.derivative) + p.roots.card =
      sturmVarNegInf (signedRemainderChain p p.derivative) :=
  (signedRemainderChain_isSturm hp h).sturm (real_roots_nodup_of_derivative hp h)

/-- The polynomial `X² - 1`, with its two real roots, satisfies the
nonzero and simple-root inputs on the nontrivial interval `[-2, 2]`.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : (X ^ 2 - 1 : Polynomial ℝ) ≠ 0 ∧
    (∀ x : ℝ, (X ^ 2 - 1 : Polynomial ℝ).eval x = 0 →
      (X ^ 2 - 1 : Polynomial ℝ).derivative.eval x ≠ 0) ∧ (-2 : ℝ) ≤ 2 := by
  refine ⟨?_, ?_, by norm_num⟩
  · intro hz
    have h := congrArg (fun f : Polynomial ℝ => f.eval 0) hz
    norm_num at h
  · intro x hx hd
    simp only [eval_sub, eval_pow, eval_X, eval_one] at hx
    simp only [derivative_sub, derivative_pow, derivative_X, derivative_one,
      sub_zero, eval_mul, eval_C, eval_pow, eval_X, mul_one] at hd
    norm_num at hd
    nlinarith

end Transformer.Sturm

/-
# Real sign properties of signed remainder chains

No coprimality over a larger field is assumed: absence of common real
zeros suffices. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2.
-/

import Transformer.Sturm.EuclideanChain

noncomputable section
open Polynomial

namespace Transformer.Sturm

/-- All entries are nonzero when the initial dividend is nonzero.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem signedRemainderChain_nonzero {p q : Polynomial ℝ} (hp : p ≠ 0) :
    ∀ f ∈ signedRemainderChain p q, f ≠ 0 := by
  intro f hf
  rw [signedRemainderChain] at hf
  split_ifs at hf with hq hr
  · simp only [List.mem_singleton] at hf
    exact hf ▸ hp
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
    rcases hf with rfl | rfl
    · exact hp
    · exact hq
  · simp only [List.mem_cons] at hf
    rcases hf with rfl | hf
    · exact hp
    · exact signedRemainderChain_nonzero hq f hf
termination_by q.natDegree
decreasing_by
  simpa only [natDegree_neg] using
    natDegree_lt_natDegree (by assumption) (degree_mod_lt p (by assumption))

/-- The terminal entry has no real root if the initial pair has no
common real zero. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem signedRemainderChain_last_no_root {p q : Polynomial ℝ}
    (h : ∀ x : ℝ, p.eval x = 0 → q.eval x ≠ 0) :
    ∀ f : Polynomial ℝ, (signedRemainderChain p q).getLast? = some f →
      ∀ x : ℝ, f.eval x ≠ 0 := by
  intro f hf x
  rw [signedRemainderChain] at hf
  split_ifs at hf with hq hr
  · have hfp : f = p := by simpa using hf.symm
    subst f
    intro hp
    exact h x hp (by simp only [hq, eval_zero])
  · have hfq : f = q := by simpa using hf.symm
    subst f
    intro hqx
    have hp : p.eval x = 0 := by
      rw [← eval_mod_of_eval_eq_zero p q hqx, hr, eval_zero]
    exact h x hp hqx
  · have hlast : (signedRemainderChain q (-(p % q))).getLast? = some f := by
      cases ht : signedRemainderChain q (-(p % q)) with
      | nil =>
          have hh := signedRemainderChain_head q (-(p % q))
          rw [ht] at hh
          contradiction
      | cons g rest =>
          simpa only [ht, List.getLast?_cons_cons] using hf
    exact signedRemainderChain_last_no_root (no_common_zero_neg_mod h) f hlast x
termination_by q.natDegree
decreasing_by
  simpa only [natDegree_neg] using
    natDegree_lt_natDegree (by assumption) (degree_mod_lt p (by assumption))

/-- At a zero of an interior entry, its two neighbors are nonzero and
have opposite signs. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem signedRemainderChain_interior {p q : Polynomial ℝ}
    (h : ∀ x : ℝ, p.eval x = 0 → q.eval x ≠ 0) :
    ∀ (i : ℕ) (x : ℝ) (a b c : Polynomial ℝ),
      (signedRemainderChain p q)[i]? = some a →
      (signedRemainderChain p q)[i + 1]? = some b →
      (signedRemainderChain p q)[i + 2]? = some c →
      b.eval x = 0 → a.eval x ≠ 0 ∧ c.eval x ≠ 0 ∧ a.eval x * c.eval x < 0 := by
  intro i x a b c ha hb hc hx
  rw [signedRemainderChain] at ha hb hc
  split_ifs at ha hb hc with hq hr
  · have : ([p] : List (Polynomial ℝ))[i + 2]? = none :=
      List.getElem?_eq_none (by simp)
    rw [this] at hc
    contradiction
  · have : ([p, q] : List (Polynomial ℝ))[i + 2]? = none :=
      List.getElem?_eq_none (by simp)
    rw [this] at hc
    contradiction
  · cases i with
    | zero =>
        have hap : a = p := by
          simpa only [List.getElem?_cons_zero, Option.some.injEq] using ha.symm
        have hbq : b = q := by
          simpa only [List.getElem?_cons_succ, ← List.head?_eq_getElem?,
            signedRemainderChain_head, Option.some.injEq] using hb.symm
        have hcr : c = -(p % q) := by
          simpa only [List.getElem?_cons_succ, signedRemainderChain_second
            (neg_ne_zero.mpr hr), Option.some.injEq] using hc.symm
        subst a
        subst b
        subst c
        have hp : p.eval x ≠ 0 := fun hp => h x hp hx
        have heq : (-(p % q)).eval x = -p.eval x := by
          rw [eval_neg, eval_mod_of_eval_eq_zero p q hx]
        refine ⟨hp, by simpa only [heq, neg_ne_zero], ?_⟩
        rw [heq]
        nlinarith [sq_pos_of_ne_zero hp]
    | succ i =>
        simp only [List.getElem?_cons_succ] at ha hb hc
        exact signedRemainderChain_interior (no_common_zero_neg_mod h) i x a b c ha hb hc hx
termination_by q.natDegree
decreasing_by
  simpa only [natDegree_neg] using
    natDegree_lt_natDegree (by assumption) (degree_mod_lt p (by assumption))

/-- `X` and the nonzero constant `1` satisfy all joint input hypotheses
for the remainder-chain sign proofs. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
example : (X : Polynomial ℝ) ≠ 0 ∧
    ∀ x : ℝ, (X : Polynomial ℝ).eval x = 0 → (1 : Polynomial ℝ).eval x ≠ 0 := by
  exact ⟨X_ne_zero, fun _ _ => by simp⟩

end Transformer.Sturm

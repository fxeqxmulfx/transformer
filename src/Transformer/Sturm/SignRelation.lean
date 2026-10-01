/-
Copyright (c) 2026 Lean FRO, LLC. All rights reserved.
Released under Apache 2.0 license as described in
third_party/hex-real-roots-mathlib/LICENSE.
Authors: Kim Morrison

Adapted from leanprover/hex-real-roots-mathlib,
revision 53ce31466dc5ff520463249d470119c1e0006e22.
-/

import Transformer.Sturm.Chains
import Transformer.Sturm.Sign
import Mathlib.Analysis.Polynomial.Order

/-!
# Variation-preserving local sign patterns

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

open Filter Topology

namespace Transformer.Sturm

/-- A local sign-pattern relation between two real lists: they agree entry by
entry except that a nonzero entry flanked by two opposite-sign neighbours may
collapse to `0`. Such a collapse is variation-neutral, so `signVariations` and
the leading sign are preserved (`SignRelation.signVariations_eq`).

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
inductive SignRelation : List ℝ → List ℝ → Prop
  | nil : SignRelation [] []
  | same {x y : ℝ} {l m : List ℝ} (hx : x ≠ 0) (hy : y ≠ 0)
      (hs : SignType.sign x = SignType.sign y) (h : SignRelation l m) :
      SignRelation (x :: l) (y :: m)
  | collapse {x X x' : ℝ} {l m : List ℝ} {y y' : ℝ}
      (hx : x ≠ 0) (hX : X ≠ 0) (hy' : y' ≠ 0)
      (hsx : SignType.sign x = SignType.sign x')
      (hsy : SignType.sign y = SignType.sign y')
      (hopp : SignType.sign x * SignType.sign y = -1)
      (h : SignRelation (y :: l) (y' :: m)) :
      SignRelation (x :: X :: y :: l) (x' :: 0 :: y' :: m)

/-- Opposite endpoint signs contribute exactly one variation across a nonzero entry.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem sign_changes_of_opposite (u v w : SignType) (huw : u * w = -1) (hv : v ≠ 0) :
    (if u * v = -1 then (1 : ℕ) else 0) + (if v * w = -1 then 1 else 0) = 1 := by
  revert huw hv; revert u v w; decide

/-- Lists related by `SignRelation` have equal sign variations and equal leading signs.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem SignRelation.signVariations_eq {L M : List ℝ} (h : SignRelation L M) :
    signVariations L = signVariations M ∧ firstSign L = firstSign M := by
  induction h with
  | nil => exact ⟨rfl, rfl⟩
  | @same x y l m hx hy hs h ih =>
    refine ⟨?_, ?_⟩
    · rw [signVariations_cons l hx, signVariations_cons m hy, ih.1, ih.2, hs]
    · rw [firstSign_cons_ne l hx, firstSign_cons_ne m hy, hs]
  | @collapse x X x' l m y y' hx hX hy' hsx hsy hopp h ih =>
    have hy : y ≠ 0 := by
      intro hy0; rw [hy0, sign_zero, mul_zero] at hopp; exact absurd hopp (by decide)
    have hx' : x' ≠ 0 := by
      intro hx0; rw [hx0, sign_zero] at hsx; exact hx (sign_eq_zero_iff.mp hsx)
    refine ⟨?_, ?_⟩
    · -- signVariations L
      rw [signVariations_cons (X :: y :: l) hx,
        firstSign_cons_ne (y :: l) hX, signVariations_cons (y :: l) hX,
        firstSign_cons_ne l hy]
      rw [signVariations_cons (0 :: y' :: m) hx',
        firstSign_cons_zero (y' :: m), firstSign_cons_ne m hy',
        signVariations_cons_zero]
      rw [← add_assoc, ih.1]
      congr 1
      rw [← hsx, ← hsy, ite_eq_left hopp]
      exact sign_changes_of_opposite _ _ _ hopp (fun h => hX (sign_eq_zero_iff.mp h))
    · rw [firstSign_cons_ne (X :: y :: l) hx, firstSign_cons_ne (0 :: y' :: m) hx', hsx]


/-- A real polynomial with no roots on an interval has equal signs at its endpoints.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem eval_sign_eq_of_no_zero {q : Polynomial ℝ} {a b : ℝ} (hab : a ≤ b)
    (hz : ∀ x ∈ Set.Icc a b, q.eval x ≠ 0) :
    SignType.sign (q.eval a) = SignType.sign (q.eval b) :=
  sign_eq_of_continuousOn isPreconnected_Icc q.continuousOn hz
    ⟨le_refl a, hab⟩ ⟨hab, le_refl b⟩

/-- Build the sign-pattern relation `SignRelation` between the evaluations of a
polynomial list at a "generic" point `a` (where every element is nonzero) and a
"special" point `r` (where some interior elements may vanish). The hypotheses
are exactly what an `IsSturmChain` supplies restricted to the relevant interval:
every element is nonzero at `a`; the head and last elements are nonzero at `r`;
whenever an interior element vanishes at `r` its neighbours are nonzero there
with opposite signs; and every element nonzero at `r` has the same sign at `a`
and `r`.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem signRelation_eval (a r : ℝ) :
    ∀ (cs : List (Polynomial ℝ)),
      (∀ q ∈ cs, q.eval a ≠ 0) →
      (∀ q, cs.head? = some q → q.eval r ≠ 0) →
      (∀ q, cs.getLast? = some q → q.eval r ≠ 0) →
      (∀ (i : ℕ) (q0 q1 q2 : Polynomial ℝ), cs[i]? = some q0 → cs[i + 1]? = some q1 →
        cs[i + 2]? = some q2 → q1.eval r = 0 →
        q0.eval r ≠ 0 ∧ q2.eval r ≠ 0 ∧ q0.eval r * q2.eval r < 0) →
      (∀ q ∈ cs, q.eval r ≠ 0 → SignType.sign (q.eval a) = SignType.sign (q.eval r)) →
      SignRelation (cs.map (Polynomial.eval a)) (cs.map (Polynomial.eval r))
  | [], _, _, _, _, _ => SignRelation.nil
  | [q0], hne0, hfront, _, _, hsame => by
      have hr : q0.eval r ≠ 0 := hfront q0 rfl
      exact SignRelation.same (hne0 q0 (by simp)) hr (hsame q0 (by simp) hr) SignRelation.nil
  | q0 :: q1 :: rest, hne0, hfront, hlast, halt, hsame => by
      have hr0 : q0.eval r ≠ 0 := hfront q0 rfl
      have ha0 : q0.eval a ≠ 0 := hne0 q0 (by simp)
      by_cases hq1 : q1.eval r = 0
      · cases rest with
        | nil => exact absurd hq1 (hlast q1 (by simp))
        | cons q2 rest' =>
            obtain ⟨hn0, hn2, hoppR⟩ := halt 0 q0 q1 q2 rfl rfl rfl hq1
            have hsx : SignType.sign (q0.eval a) = SignType.sign (q0.eval r) :=
              hsame q0 (by simp) hn0
            have hsy : SignType.sign (q2.eval a) = SignType.sign (q2.eval r) :=
              hsame q2 (by simp) hn2
            have hoppA : SignType.sign (q0.eval a) * SignType.sign (q2.eval a) = -1 := by
              rw [hsx, hsy, ← sign_mul, sign_eq_neg_one_iff]; exact hoppR
            have hne0' : ∀ q ∈ q2 :: rest', q.eval a ≠ 0 := fun q hq =>
              hne0 q (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ hq))
            have hfront' : ∀ q, (q2 :: rest').head? = some q → q.eval r ≠ 0 := by
              intro q hq; rw [List.head?_cons] at hq; cases hq; exact hn2
            have hlast' : ∀ q, (q2 :: rest').getLast? = some q → q.eval r ≠ 0 := by
              intro q hq
              exact hlast q (by rw [List.getLast?_cons_cons, List.getLast?_cons_cons]; exact hq)
            have halt' : ∀ (i : ℕ) (p0 p1 p2 : Polynomial ℝ), (q2 :: rest')[i]? = some p0 →
                (q2 :: rest')[i + 1]? = some p1 → (q2 :: rest')[i + 2]? = some p2 →
                p1.eval r = 0 → p0.eval r ≠ 0 ∧ p2.eval r ≠ 0 ∧ p0.eval r * p2.eval r < 0 := by
              intro i p0 p1 p2 h0 h1 h2 hz
              exact halt (i + 2) p0 p1 p2
                (by rw [List.getElem?_cons_succ, List.getElem?_cons_succ]; exact h0)
                (by rw [List.getElem?_cons_succ, List.getElem?_cons_succ]; exact h1)
                (by rw [List.getElem?_cons_succ, List.getElem?_cons_succ]; exact h2) hz
            have hsame' : ∀ q ∈ q2 :: rest', q.eval r ≠ 0 →
                SignType.sign (q.eval a) = SignType.sign (q.eval r) := fun q hq =>
              hsame q (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ hq))
            have IH := signRelation_eval a r (q2 :: rest') hne0' hfront' hlast' halt' hsame'
            simp only [List.map_cons] at IH ⊢
            rw [hq1]
            exact SignRelation.collapse ha0 (hne0 q1 (by simp)) hn2 hsx hsy hoppA IH
      · have hne0' : ∀ q ∈ q1 :: rest, q.eval a ≠ 0 := fun q hq =>
          hne0 q (List.mem_cons_of_mem _ hq)
        have hfront' : ∀ q, (q1 :: rest).head? = some q → q.eval r ≠ 0 := by
          intro q hq; rw [List.head?_cons] at hq; cases hq; exact hq1
        have hlast' : ∀ q, (q1 :: rest).getLast? = some q → q.eval r ≠ 0 := by
          intro q hq
          exact hlast q (by rw [List.getLast?_cons_cons]; exact hq)
        have halt' : ∀ (i : ℕ) (p0 p1 p2 : Polynomial ℝ), (q1 :: rest)[i]? = some p0 →
            (q1 :: rest)[i + 1]? = some p1 → (q1 :: rest)[i + 2]? = some p2 →
            p1.eval r = 0 → p0.eval r ≠ 0 ∧ p2.eval r ≠ 0 ∧ p0.eval r * p2.eval r < 0 := by
          intro i p0 p1 p2 h0 h1 h2 hz
          exact halt (i + 1) p0 p1 p2
            (by rw [List.getElem?_cons_succ]; exact h0)
            (by rw [List.getElem?_cons_succ]; exact h1)
            (by rw [List.getElem?_cons_succ]; exact h2) hz
        have hsame' : ∀ q ∈ q1 :: rest, q.eval r ≠ 0 →
            SignType.sign (q.eval a) = SignType.sign (q.eval r) := fun q hq =>
          hsame q (List.mem_cons_of_mem _ hq)
        have IH := signRelation_eval a r (q1 :: rest) hne0' hfront' hlast' halt' hsame'
        simp only [List.map_cons] at IH ⊢
        exact SignRelation.same ha0 hr0 (hsame q0 (by simp) hr0) IH

end Transformer.Sturm

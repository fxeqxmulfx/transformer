/-
Copyright (c) 2026 Lean FRO, LLC. All rights reserved.
Released under Apache 2.0 license as described in
third_party/hex-real-roots-mathlib/LICENSE.
Authors: Kim Morrison

Adapted from leanprover/hex-real-roots-mathlib,
revision 53ce31466dc5ff520463249d470119c1e0006e22.
-/

import Transformer.Sturm.LocalVariation

/-!
# Signed variation changes across a head root

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

noncomputable section
open Filter Topology Polynomial

namespace Transformer.Sturm

/-- Prepending a nonzero head contributes a variation exactly when its
product with the nonzero second entry is negative. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem sturmVar_head_pair {p q : Polynomial ℝ} {x : ℝ}
    (hp : p.eval x ≠ 0) (hq : q.eval x ≠ 0) (tail : List (Polynomial ℝ)) :
    sturmVar (p :: q :: tail) x =
      (if (p * q).eval x < 0 then 1 else 0) + sturmVar (q :: tail) x := by
  change signVariations (p.eval x :: (q :: tail).map (Polynomial.eval x)) = _
  rw [signVariations_cons _ hp]
  simp only [List.map_cons, firstSign_cons_ne _ hq, ← sign_mul,
    sign_eq_neg_one_iff, ← eval_mul]
  rfl

/-- If all chain zeros on an interval lie at one point and the second
entry is nonzero there, the signed variation change is the difference
between the two head-pair contributions. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem sturmVar_root_pair_cross {p q : Polynomial ℝ} {chain : List (Polynomial ℝ)}
    (hchain : IsRootAlternatingChain p chain) (r : ℝ)
    (hq1 : chain[1]? = some q) (hqr : q.eval r ≠ 0)
    (a b : ℝ) (har : a < r) (hrb : r < b)
    (hz : ∀ f ∈ chain, ∀ x ∈ Set.Icc a b, x ≠ r → f.eval x ≠ 0) :
    (sturmVar chain a : ℤ) - sturmVar chain b =
      (if (p * q).eval a < 0 then (1 : ℤ) else 0) -
        (if (p * q).eval b < 0 then (1 : ℤ) else 0) := by
  have hpz : ∀ x ∈ Set.Icc a b, x ≠ r → ¬ p.IsRoot x :=
    fun x hx hxr => hz p hchain.head_mem x hx hxr
  have hab : a ≤ b := (har.trans hrb).le
  -- Split the chain into its head `p` and second element `q`.
  rcases chain with _ | ⟨p0, _ | ⟨q0, tail⟩⟩
  · exact absurd hchain.head (by simp)
  · exact absurd hq1 (by simp)
  have hp0 : p0 = p := by simpa using hchain.head
  subst p0
  have hq0 : q0 = q := by simpa using hq1
  subst q0
  -- Basic nonvanishing facts.
  have hpa : p.eval a ≠ 0 := fun h => hpz a ⟨le_refl a, hab⟩ (ne_of_lt har) h
  have hpb : p.eval b ≠ 0 := fun h => hpz b ⟨hab, le_refl b⟩ (ne_of_lt hrb).symm h
  have hqa : q.eval a ≠ 0 := hz q (by simp) a ⟨le_refl a, hab⟩ (ne_of_lt har)
  have hqb : q.eval b ≠ 0 := hz q (by simp) b ⟨hab, le_refl b⟩ (ne_of_lt hrb).symm
  -- Point-`r` chain hypotheses for the tail `q :: tail`, shared by both calls.
  have hfront_rest : ∀ s, (q :: tail).head? = some s → s.eval r ≠ 0 := by
    intro s hs; rw [List.head?_cons] at hs; cases hs; exact hqr
  have hlast_rest : ∀ s, (q :: tail).getLast? = some s → s.eval r ≠ 0 := by
    intro s hs
    exact hchain.last_no_root s (by rw [List.getLast?_cons_cons]; exact hs) r
  have halt_rest : ∀ (i : ℕ) (s0 s1 s2 : Polynomial ℝ), (q :: tail)[i]? = some s0 →
      (q :: tail)[i + 1]? = some s1 → (q :: tail)[i + 2]? = some s2 → s1.eval r = 0 →
      s0.eval r ≠ 0 ∧ s2.eval r ≠ 0 ∧ s0.eval r * s2.eval r < 0 := by
    intro i s0 s1 s2 h0 h1 h2 hz0
    exact hchain.interior_alternates (i + 1) r s0 s1 s2
      (by rw [List.getElem?_cons_succ]; exact h0)
      (by rw [List.getElem?_cons_succ]; exact h1)
      (by rw [List.getElem?_cons_succ]; exact h2) hz0
  -- Sign persistence for the tail elements, at `a` and at `b`.
  have hsame_a : ∀ s ∈ q :: tail, s.eval r ≠ 0 →
      SignType.sign (s.eval a) = SignType.sign (s.eval r) := fun s hs hsr =>
    eval_sign_eq_of_no_zero har.le (fun x hx => by
      by_cases hxr : x = r
      · rw [hxr]; exact hsr
      · exact hz s (List.mem_cons_of_mem _ hs) x ⟨hx.1, hx.2.trans hrb.le⟩ hxr)
  have hsame_b : ∀ s ∈ q :: tail, s.eval r ≠ 0 →
      SignType.sign (s.eval b) = SignType.sign (s.eval r) := fun s hs hsr =>
    (eval_sign_eq_of_no_zero hrb.le (fun x hx => by
      by_cases hxr : x = r
      · rw [hxr]; exact hsr
      · exact hz s (List.mem_cons_of_mem _ hs) x ⟨har.le.trans hx.1, hx.2⟩ hxr)).symm
  -- The tail's `sturmVar` is the same at `a`, `r`, `b` (interior-crossing).
  have hEqA : sturmVar (q :: tail) a = sturmVar (q :: tail) r :=
    (signRelation_eval a r (q :: tail)
      (fun s hs => hz s (List.mem_cons_of_mem _ hs) a ⟨le_refl a, hab⟩ (ne_of_lt har))
      hfront_rest hlast_rest halt_rest hsame_a).signVariations_eq.1
  have hEqB : sturmVar (q :: tail) b = sturmVar (q :: tail) r :=
    (signRelation_eval b r (q :: tail)
      (fun s hs => hz s (List.mem_cons_of_mem _ hs) b ⟨hab, le_refl b⟩ (ne_of_lt hrb).symm)
      hfront_rest hlast_rest halt_rest hsame_b).signVariations_eq.1
  rw [sturmVar_head_pair hpa hqa, sturmVar_head_pair hpb hqb, hEqA, hEqB]
  by_cases ha : (p * q).eval a < 0 <;> by_cases hb : (p * q).eval b < 0 <;>
    simp only [ha, hb, ite_true, ite_false, Nat.zero_add, Nat.cast_add,
      Nat.cast_one] <;> omega

end Transformer.Sturm

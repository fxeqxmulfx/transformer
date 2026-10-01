/-
Copyright (c) 2026 Lean FRO, LLC. All rights reserved.
Released under Apache 2.0 license as described in
third_party/hex-real-roots-mathlib/LICENSE.
Authors: Kim Morrison

Adapted from leanprover/hex-real-roots-mathlib,
revision 53ce31466dc5ff520463249d470119c1e0006e22.
-/

import Transformer.Sturm.SignRelation
import Transformer.Sturm.AlternatingChains

/-!
# Constancy of variations across interior roots

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

open Filter Topology

namespace Transformer.Sturm

variable {p : Polynomial ℝ} {chain : List (Polynomial ℝ)}

/-- Sign variations are constant on an interval containing no zero of any chain entry.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem sturmVar_const_of_no_zero
    (a b : ℝ) (hab : a ≤ b)
    (hz : ∀ q ∈ chain, ∀ x ∈ Set.Icc a b, q.eval x ≠ 0) :
    sturmVar chain a = sturmVar chain b := by
  change signVariations (chain.map (Polynomial.eval a))
    = signVariations (chain.map (Polynomial.eval b))
  apply signVariations_congr
  rw [List.forall₂_map_left_iff, List.forall₂_map_right_iff, List.forall₂_same]
  intro q hq
  exact eval_sign_eq_of_no_zero hab (fun x hx => hz q hq x hx)

/-- Crossing a zero of an interior entry preserves the variation count.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem sturmVar_interior_cross_general (hchain : IsRootAlternatingChain p chain) (r : ℝ)
    (hpr : ¬ p.IsRoot r) (a b : ℝ) (har : a < r) (hrb : r < b)
    (hz : ∀ q ∈ chain, ∀ x ∈ Set.Icc a b, x ≠ r → q.eval x ≠ 0) :
    sturmVar chain a = sturmVar chain r ∧ sturmVar chain r = sturmVar chain b := by
  have hab : a ≤ b := (har.trans hrb).le
  have hpr' : p.eval r ≠ 0 := hpr
  -- Chain-structure hypotheses at the special point `r`, shared by both calls.
  have hfront : ∀ q, chain.head? = some q → q.eval r ≠ 0 := by
    intro q hq; rw [hchain.head] at hq; cases hq; exact hpr'
  have hlast : ∀ q, chain.getLast? = some q → q.eval r ≠ 0 :=
    fun q hq => hchain.last_no_root q hq r
  have halt : ∀ (i : ℕ) (q0 q1 q2 : Polynomial ℝ), chain[i]? = some q0 →
      chain[i + 1]? = some q1 → chain[i + 2]? = some q2 → q1.eval r = 0 →
      q0.eval r ≠ 0 ∧ q2.eval r ≠ 0 ∧ q0.eval r * q2.eval r < 0 :=
    fun i q0 q1 q2 h0 h1 h2 hz => hchain.interior_alternates i r q0 q1 q2 h0 h1 h2 hz
  constructor
  · change signVariations (chain.map (Polynomial.eval a))
      = signVariations (chain.map (Polynomial.eval r))
    refine (signRelation_eval a r chain (fun q hq => hz q hq a ⟨le_refl a, hab⟩ (ne_of_lt har))
      hfront hlast halt (fun q hq hqr => ?_)).signVariations_eq.1
    exact eval_sign_eq_of_no_zero har.le (fun x hx => by
      by_cases hxr : x = r
      · rw [hxr]; exact hqr
      · exact hz q hq x ⟨hx.1, hx.2.trans hrb.le⟩ hxr)
  · change signVariations (chain.map (Polynomial.eval r))
      = signVariations (chain.map (Polynomial.eval b))
    refine ((signRelation_eval b r chain
      (fun q hq => hz q hq b ⟨hab, le_refl b⟩ (ne_of_lt hrb).symm)
      hfront hlast halt (fun q hq hqr => ?_)).signVariations_eq.1).symm
    exact (eval_sign_eq_of_no_zero hrb.le (fun x hx => by
      by_cases hxr : x = r
      · rw [hxr]; exact hqr
      · exact hz q hq x ⟨har.le.trans hx.1, hx.2⟩ hxr)).symm

/-- Crossing an interior zero of a Sturm chain preserves its variation
count. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem sturmVar_interior_cross (hchain : IsSturmChain p chain) (r : ℝ)
    (hpr : ¬ p.IsRoot r) (a b : ℝ) (har : a < r) (hrb : r < b)
    (hz : ∀ q ∈ chain, ∀ x ∈ Set.Icc a b, x ≠ r → q.eval x ≠ 0) :
    sturmVar chain a = sturmVar chain r ∧ sturmVar chain r = sturmVar chain b :=
  sturmVar_interior_cross_general hchain.toRootAlternatingChain r hpr a b har hrb hz

end Transformer.Sturm

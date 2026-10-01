/-
Copyright (c) 2026 Lean FRO, LLC. All rights reserved.
Released under Apache 2.0 license as described in
third_party/hex-real-roots-mathlib/LICENSE.
Authors: Kim Morrison

Adapted from leanprover/hex-real-roots-mathlib,
revision 53ce31466dc5ff520463249d470119c1e0006e22.
-/

import Transformer.Sturm.ChainZeros

/-!
# Sturm root count on a half-open interval

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

open Filter Topology

namespace Transformer.Sturm

variable {p : Polynomial ℝ} {chain : List (Polynomial ℝ)}

/-- **Sturm's theorem** on a half-open interval.

The decrease in sign variations from `a` to `b` counts the roots in `(a, b]`.
The hypothesis on `p.roots` ensures each real root has multiplicity one.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem IsSturmChain.sturm_Ioc (hchain : IsSturmChain p chain) (hnod : p.roots.Nodup)
    {a b : ℝ} (hab : a ≤ b) :
    sturmVar chain b + (p.roots.filter (fun r => r ∈ Set.Ioc a b)).card =
      sturmVar chain a := by
  classical
  have hne := hchain.nonzero_mem
  suffices H : ∀ n : ℕ, ∀ a b : ℝ, a ≤ b →
      ((chainZeros chain).filter (fun x => a < x ∧ x ≤ b)).card = n →
      sturmVar chain b + (p.roots.filter (fun r => a < r ∧ r ≤ b)).card =
        sturmVar chain a by
    simpa only [Set.mem_Ioc] using H _ a b hab rfl
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro a b hab hcard
    set F := (chainZeros chain).filter (fun x => a < x ∧ x ≤ b) with hF
    by_cases hemp : F = ∅
    · -- No break point in `(a, b]`: `sturmVar` is constant and there are no roots.
      have hclear : ∀ x, a < x → x ≤ b → x ∉ chainZeros chain := by
        intro x hx1 hx2 hxZ
        have hxF : x ∈ F := by rw [hF, Finset.mem_filter]; exact ⟨hxZ, hx1, hx2⟩
        rw [hemp] at hxF; exact absurd hxF (Finset.notMem_empty x)
      have heqv : sturmVar chain a = sturmVar chain b :=
        sturmVar_eq_right hchain hab hclear
      have hroots0 : p.roots.filter (fun r => a < r ∧ r ≤ b) = 0 := by
        rw [Multiset.filter_eq_nil]
        rintro x hx ⟨h1, h2⟩
        exact hclear x h1 h2
          ((mem_chainZeros hne).mpr
                  ⟨p, hchain.head_mem, (Polynomial.mem_roots hchain.ne_zero).mp hx⟩)
      rw [heqv, hroots0]; simp
    · -- Peel off the largest break point `z` in `(a, b]`.
      have hFne : F.Nonempty := Finset.nonempty_iff_ne_empty.mpr hemp
      let z := F.max' hFne
      have hzmem : z ∈ F := F.max'_mem hFne
      have hzmax : ∀ x ∈ F, x ≤ z := fun x hx => F.le_max' x hx
      obtain ⟨hzS, haz, hzb⟩ : z ∈ chainZeros chain ∧ a < z ∧ z ≤ b := by
        have h := hzmem; rw [hF, Finset.mem_filter] at h; exact ⟨h.1, h.2.1, h.2.2⟩
      obtain ⟨a', ha_a', ha'z, ha'gap⟩ := exists_left_gap (chainZeros chain) z a haz
      obtain ⟨b', hzb', _, hb'gap⟩ := exists_right_gap (chainZeros chain) z (z + 1) (by linarith)
      -- Only break point in `(a', b]` is `z`.
      have honly : ∀ x, a' < x → x ≤ b → x ∈ chainZeros chain → x = z := by
        intro x hx1 hx2 hxZ
        rcases lt_trichotomy x z with hlt' | heq | hgt'
        · exact absurd (ha'gap x hxZ hlt') (not_lt.mpr hx1.le)
        · exact heq
        · have hxF : x ∈ F := by rw [hF, Finset.mem_filter]; exact ⟨hxZ, lt_trans ha_a' hx1, hx2⟩
          exact absurd (hzmax x hxF) (not_le.mpr hgt')
      -- `[a', b']` has no break point except `z`.
      have hz_ex : ∀ q ∈ chain, ∀ x ∈ Set.Icc a' b', x ≠ z → q.eval x ≠ 0 := by
        intro q hq x hx hxz hqx
        have hxZ : x ∈ chainZeros chain := (mem_chainZeros hne).mpr ⟨q, hq, hqx⟩
        rcases lt_trichotomy x z with hlt' | heq | hgt'
        · exact absurd (ha'gap x hxZ hlt') (not_lt.mpr hx.1)
        · exact hxz heq
        · exact absurd (hb'gap x hxZ hgt') (not_lt.mpr hx.2)
      -- Right registration: `sturmVar z = sturmVar b`.
      have hzeqb : sturmVar chain z = sturmVar chain b := by
        apply sturmVar_eq_right hchain hzb
        intro x hx1 hx2 hxZ
        have hxF : x ∈ F := by rw [hF, Finset.mem_filter]; exact ⟨hxZ, lt_trans haz hx1, hx2⟩
        exact absurd (hzmax x hxF) (not_le.mpr hx1)
      -- Inductive hypothesis on `(a, a']`.
      have hsub : (chainZeros chain).filter (fun x => a < x ∧ x ≤ a') ⊆ F := by
        rw [hF]; intro x hx; rw [Finset.mem_filter] at hx ⊢
        exact ⟨hx.1, hx.2.1, le_trans hx.2.2 (le_trans ha'z.le hzb)⟩
      have hznotin : z ∉ (chainZeros chain).filter (fun x => a < x ∧ x ≤ a') := by
        rw [Finset.mem_filter]; rintro ⟨_, _, hza'⟩; exact absurd hza' (not_le.mpr ha'z)
      have hlt_card : ((chainZeros chain).filter (fun x => a < x ∧ x ≤ a')).card < n := by
        rw [← hcard]
        exact Finset.card_lt_card ((Finset.ssubset_iff_of_subset hsub).mpr ⟨z, hzmem, hznotin⟩)
      have IHres := ih _ hlt_card a a' ha_a'.le rfl
      have hsplit := card_filter_Ioc_split p.roots ha_a'.le (ha'z.le.trans hzb)
      by_cases hzroot : p.IsRoot z
      · obtain ⟨hcrossL, hcrossR⟩ :=
          sturmVar_root_cross hchain z hzroot a' b' ha'z hzb' hz_ex
        have ha'b : sturmVar chain a' = sturmVar chain b + 1 := by
          rw [hcrossL, ← hcrossR, hzeqb]
        have hRZ : (p.roots.filter (fun r => a' < r ∧ r ≤ b)).card = 1 := by
          have hzrootmem : z ∈ p.roots := (Polynomial.mem_roots hchain.ne_zero).mpr hzroot
          have hfeq : p.roots.filter (fun r => a' < r ∧ r ≤ b)
              = p.roots.filter (fun r => r = z) := by
            apply Multiset.filter_congr
            intro x hx
            constructor
            · rintro ⟨h1, h2⟩
              exact honly x h1 h2
                ((mem_chainZeros hne).mpr
                  ⟨p, hchain.head_mem, (Polynomial.mem_roots hchain.ne_zero).mp hx⟩)
            · rintro rfl; exact ⟨ha'z, hzb⟩
          rw [hfeq, Multiset.filter_eq', Multiset.card_replicate,
            Multiset.count_eq_one_of_mem hnod hzrootmem]
        omega
      · obtain ⟨hcrossL, _⟩ :=
          sturmVar_interior_cross hchain z hzroot a' b' ha'z hzb' hz_ex
        have ha'b : sturmVar chain a' = sturmVar chain b := by
          rw [hcrossL, hzeqb]
        have hRZ : (p.roots.filter (fun r => a' < r ∧ r ≤ b)).card = 0 := by
          have hfeq : p.roots.filter (fun r => a' < r ∧ r ≤ b) = 0 := by
            rw [Multiset.filter_eq_nil]
            rintro x hx ⟨h1, h2⟩
            have hxz : x = z := honly x h1 h2
              ((mem_chainZeros hne).mpr
                  ⟨p, hchain.head_mem, (Polynomial.mem_roots hchain.ne_zero).mp hx⟩)
            rw [hxz] at hx
            exact hzroot ((Polynomial.mem_roots hchain.ne_zero).mp hx)
          rw [hfeq]; rfl
        omega

end Transformer.Sturm

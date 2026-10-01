/-
Copyright (c) 2026 Lean FRO, LLC. All rights reserved.
Released under Apache 2.0 license as described in
third_party/hex-real-roots-mathlib/LICENSE.
Authors: Kim Morrison

The finite-breakpoint induction is adapted from SturmTheorem.lean at
revision 53ce31466dc5ff520463249d470119c1e0006e22, with signed jumps.
-/

import Transformer.Sturm.QueryCrossing
import Transformer.Sturm.IntervalSums

/-!
# Signed Sturm queries on intervals with generic endpoints

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

noncomputable section
open Polynomial
open scoped BigOperators

namespace Transformer.Sturm

/-- The signed variation change of the Euclidean chain sums the signs
of `p'(r) q(r)` over all real roots in a bounded interval. Only the
endpoints avoid the finitely many chain zeros; the roots inside are
retained. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem signedRemainderChain_query_Ioc {p q : Polynomial ℝ} (hp : p ≠ 0)
    (hcommon : ∀ x : ℝ, p.eval x = 0 → q.eval x ≠ 0)
    (hsimple : ∀ x : ℝ, p.eval x = 0 → p.derivative.eval x ≠ 0)
    {a b : ℝ} (hab : a ≤ b)
    (ha : ∀ f ∈ signedRemainderChain p q, f.eval a ≠ 0)
    (hb : ∀ f ∈ signedRemainderChain p q, f.eval b ≠ 0) :
    (sturmVar (signedRemainderChain p q) a : ℤ) - sturmVar (signedRemainderChain p q) b =
      ∑ r ∈ p.roots.toFinset.filter (fun x => x ∈ Set.Ioc a b),
        (SignType.sign (p.derivative.eval r * q.eval r) : ℤ) := by
  let chain := signedRemainderChain p q
  have hc := signedRemainderChain_isRootAlternating hp hcommon
  let w : ℝ → ℤ := fun r => (SignType.sign (p.derivative.eval r * q.eval r) : ℤ)
  suffices H : ∀ n : ℕ, ∀ a b : ℝ, a ≤ b →
      (∀ f ∈ chain, f.eval a ≠ 0) → (∀ f ∈ chain, f.eval b ≠ 0) →
      ((chainZeros chain).filter (fun x => x ∈ Set.Ioc a b)).card = n →
      (sturmVar chain a : ℤ) - sturmVar chain b =
        ∑ r ∈ p.roots.toFinset.filter (fun x => x ∈ Set.Ioc a b), w r from
    H _ a b hab ha hb rfl
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro a b hab ha hb hcard
    let F := (chainZeros chain).filter (fun x => x ∈ Set.Ioc a b)
    by_cases hemp : F = ∅
    · have hzall : ∀ f ∈ chain, ∀ x ∈ Set.Icc a b, f.eval x ≠ 0 := by
        intro f hf x hx hfx
        rcases eq_or_lt_of_le hx.1 with heq | hax
        · exact ha f hf (heq ▸ hfx)
        · have hxF : x ∈ F := Finset.mem_filter.mpr
            ⟨(mem_chainZeros hc.nonzero_mem).mpr ⟨f, hf, hfx⟩, hax, hx.2⟩
          rw [hemp] at hxF
          exact Finset.notMem_empty x hxF
      have hv := sturmVar_const_of_no_zero a b hab hzall
      have hroots : p.roots.toFinset.filter (fun x => x ∈ Set.Ioc a b) = ∅ := by
        apply Finset.eq_empty_iff_forall_notMem.mpr
        intro x hx
        obtain ⟨hxr, hax, hxb⟩ := Finset.mem_filter.mp hx
        exact hzall p hc.head_mem x ⟨hax.le, hxb⟩ ((mem_roots hp).mp
          (Multiset.mem_toFinset.mp hxr))
      rw [hv, hroots]
      simp
    · have hFne : F.Nonempty := Finset.nonempty_iff_ne_empty.mpr hemp
      let z := F.max' hFne
      have hzmem : z ∈ F := F.max'_mem hFne
      obtain ⟨hzS, haz, hzb⟩ : z ∈ chainZeros chain ∧ a < z ∧ z ≤ b :=
        Finset.mem_filter.mp hzmem
      have hzmax : ∀ x ∈ F, x ≤ z := fun x hx => F.le_max' x hx
      have hzb' : z < b := by
        apply lt_of_le_of_ne hzb
        intro heq
        obtain ⟨f, hf, hfr⟩ := (mem_chainZeros hc.nonzero_mem).mp hzS
        exact hb f hf (heq ▸ hfr)
      obtain ⟨c, hac, hcz, hgap⟩ := exists_left_gap (chainZeros chain) z a haz
      have hcn : ∀ f ∈ chain, f.eval c ≠ 0 := by
        intro f hf hfc
        have hcS : c ∈ chainZeros chain := (mem_chainZeros hc.nonzero_mem).mpr ⟨f, hf, hfc⟩
        exact (lt_irrefl c) (hgap c hcS hcz)
      have honly : ∀ x, c ≤ x → x ≤ b → x ∈ chainZeros chain → x = z := by
        intro x hcx hxb hxS
        rcases lt_trichotomy x z with hxz | heq | hzx
        · exact (not_lt_of_ge hcx (hgap x hxS hxz)).elim
        · exact heq
        · have hxF : x ∈ F := Finset.mem_filter.mpr ⟨hxS, hac.trans_le hcx, hxb⟩
          exact (not_lt_of_ge (hzmax x hxF) hzx).elim
      have hzex : ∀ f ∈ chain, ∀ x ∈ Set.Icc c b, x ≠ z → f.eval x ≠ 0 := by
        intro f hf x hx hne hfx
        exact hne (honly x hx.1 hx.2 ((mem_chainZeros hc.nonzero_mem).mpr ⟨f, hf, hfx⟩))
      have hsub : (chainZeros chain).filter (fun x => x ∈ Set.Ioc a c) ⊆ F := by
        intro x hx
        obtain ⟨hxS, hax, hxc⟩ := Finset.mem_filter.mp hx
        exact Finset.mem_filter.mpr ⟨hxS, hax, hxc.trans (hcz.le.trans hzb)⟩
      have hznot : z ∉ (chainZeros chain).filter (fun x => x ∈ Set.Ioc a c) := by
        intro hz
        exact not_le_of_gt hcz (Finset.mem_filter.mp hz).2.2
      have hlt : ((chainZeros chain).filter (fun x => x ∈ Set.Ioc a c)).card < n := by
        rw [← hcard]
        exact Finset.card_lt_card ((Finset.ssubset_iff_of_subset hsub).mpr ⟨z, hzmem, hznot⟩)
      have hprefix := ih _ hlt a c hac.le ha hcn rfl
      have hsplit := sum_filter_Ioc_split p.roots.toFinset w hac.le (hcz.le.trans hzb)
      by_cases hzroot : p.IsRoot z
      · have hcross := signedRemainderChain_root_query_cross hp hcommon hsimple hzroot
          hcz hzb' hzex
        have htail : p.roots.toFinset.filter (fun x => x ∈ Set.Ioc c b) = {z} := by
          ext x
          simp only [Finset.mem_filter, Finset.mem_singleton]
          constructor
          · rintro ⟨hxr, hcx, hxb⟩
            exact honly x hcx.le hxb ((mem_chainZeros hc.nonzero_mem).mpr
              ⟨p, hc.head_mem, (mem_roots hp).mp (Multiset.mem_toFinset.mp hxr)⟩)
          · rintro rfl
            exact ⟨Multiset.mem_toFinset.mpr ((mem_roots hp).mpr hzroot), hcz, hzb⟩
        rw [htail, Finset.sum_singleton] at hsplit
        change (sturmVar chain c : ℤ) - sturmVar chain b = w z at hcross
        omega
      · have hcross := sturmVar_interior_cross_general hc z hzroot c b hcz hzb' hzex
        have hv : sturmVar chain c = sturmVar chain b := hcross.1.trans hcross.2
        have htail : p.roots.toFinset.filter (fun x => x ∈ Set.Ioc c b) = ∅ := by
          apply Finset.eq_empty_iff_forall_notMem.mpr
          intro x hx
          obtain ⟨hxr, hcx, hxb⟩ := Finset.mem_filter.mp hx
          have hpr := (mem_roots hp).mp (Multiset.mem_toFinset.mp hxr)
          have hxz := honly x hcx.le hxb ((mem_chainZeros hc.nonzero_mem).mpr
            ⟨p, hc.head_mem, hpr⟩)
          exact hzroot (hxz ▸ hpr)
        rw [htail, Finset.sum_empty, add_zero] at hsplit
        rw [hv] at hprefix
        exact hprefix.trans hsplit.symm

end Transformer.Sturm

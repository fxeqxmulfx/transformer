/-
Copyright (c) 2026 Lean FRO, LLC. All rights reserved.
Released under Apache 2.0 license as described in
third_party/hex-real-roots-mathlib/LICENSE.
Authors: Kim Morrison

Adapted from leanprover/hex-real-roots-mathlib,
revision 53ce31466dc5ff520463249d470119c1e0006e22.
-/

import Transformer.Sturm.RootCrossing

/-!
# Finite chain zeros and right continuity

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

open Filter Topology

namespace Transformer.Sturm

variable {p : Polynomial ℝ} {chain : List (Polynomial ℝ)}

/-- The union of the real root sets of the chain entries.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
noncomputable def chainZeros (cs : List (Polynomial ℝ)) : Finset ℝ :=
  cs.toFinset.biUnion (fun q => q.roots.toFinset)

/-- Membership in `chainZeros`: a point lies in it exactly when some chain
element vanishes there (using that every chain element is nonzero).

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem mem_chainZeros {cs : List (Polynomial ℝ)} (hne : ∀ q ∈ cs, q ≠ 0) {x : ℝ} :
    x ∈ chainZeros cs ↔ ∃ q ∈ cs, q.eval x = 0 := by
  simp only [chainZeros, Finset.mem_biUnion, List.mem_toFinset, Multiset.mem_toFinset]
  exact exists_congr fun q => and_congr_right fun hq => Polynomial.mem_roots (hne q hq)

/-- Choose a point between `lo` and `z` above every element of `S` below `z`.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_left_gap (S : Finset ℝ) (z lo : ℝ) (hlo : lo < z) :
    ∃ a, lo < a ∧ a < z ∧ ∀ x ∈ S, x < z → x < a := by
  have h : ∀ᶠ a in 𝓝[<] z, ∀ x ∈ S, x < z → x < a := by
    rw [S.eventually_all]
    intro x _
    by_cases hx : x < z
    · exact ((eventually_gt_nhds hx).filter_mono nhdsWithin_le_nhds).mono fun _ ha _ => ha
    · simp [hx]
  obtain ⟨a, ha, hla, haz⟩ := (h.and (Ioo_mem_nhdsLT hlo)).exists
  exact ⟨a, hla, haz, ha⟩

/-- Choose a point between `z` and `hi` below every element of `S` above `z`.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_right_gap (S : Finset ℝ) (z hi : ℝ) (hhi : z < hi) :
    ∃ b, z < b ∧ b < hi ∧ ∀ x ∈ S, z < x → b < x := by
  have h : ∀ᶠ b in 𝓝[>] z, ∀ x ∈ S, z < x → b < x := by
    rw [S.eventually_all]
    intro x _
    by_cases hx : z < x
    · exact ((eventually_lt_nhds hx).filter_mono nhdsWithin_le_nhds).mono fun _ hb _ => hb
    · simp [hx]
  obtain ⟨b, hb, hzb, hbh⟩ := (h.and (Ioo_mem_nhdsGT hhi)).exists
  exact ⟨b, hzb, hbh, hb⟩

/-- The variation count agrees with its value immediately to the right,
including at zeros of chain entries.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem sturmVar_eq_right (hchain : IsSturmChain p chain) {z c : ℝ} (hzc : z ≤ c)
    (hclear : ∀ x, z < x → x ≤ c → x ∉ chainZeros chain) :
    sturmVar chain z = sturmVar chain c := by
  rcases eq_or_lt_of_le hzc with rfl | hlt
  · rfl
  have hne := hchain.nonzero_mem
  by_cases hzZ : z ∈ chainZeros chain
  · obtain ⟨a₀, _, ha₀z, ha₀gap⟩ := exists_left_gap (chainZeros chain) z (z - 1) (by linarith)
    have hz_ex : ∀ q ∈ chain, ∀ x ∈ Set.Icc a₀ c, x ≠ z → q.eval x ≠ 0 := by
      intro q hq x hx hxz hqx
      have hxZ : x ∈ chainZeros chain := (mem_chainZeros hne).mpr ⟨q, hq, hqx⟩
      rcases lt_trichotomy x z with hlt' | heq | hgt'
      · exact absurd (ha₀gap x hxZ hlt') (not_lt.mpr hx.1)
      · exact hxz heq
      · exact hclear x hgt' hx.2 hxZ
    by_cases hroot : p.IsRoot z
    · exact (sturmVar_root_cross hchain z hroot a₀ c ha₀z hlt hz_ex).2
    · exact (sturmVar_interior_cross hchain z hroot a₀ c ha₀z hlt hz_ex).2
  · have hz_all : ∀ q ∈ chain, ∀ x ∈ Set.Icc z c, q.eval x ≠ 0 := by
      intro q hq x hx hqx
      have hxZ : x ∈ chainZeros chain := (mem_chainZeros hne).mpr ⟨q, hq, hqx⟩
      rcases eq_or_lt_of_le hx.1 with heq | hgt
      · exact hzZ (by rw [heq]; exact hxZ)
      · exact hclear x hgt hx.2 hxZ
    exact sturmVar_const_of_no_zero z c hzc hz_all

/-- Splitting a half-open interval count: for `a ≤ a' ≤ b`, the number of
multiset entries in `(a, b]` is the sum of those in `(a, a']` and `(a', b]`.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem card_filter_Ioc_split (s : Multiset ℝ) {a a' b : ℝ} (h1 : a ≤ a') (h2 : a' ≤ b) :
    (s.filter (fun r => a < r ∧ r ≤ b)).card
      = (s.filter (fun r => a < r ∧ r ≤ a')).card
        + (s.filter (fun r => a' < r ∧ r ≤ b)).card := by
  classical
  rw [← Multiset.card_add, ← Multiset.filter_add_not (fun r => r ≤ a')
    (s.filter (fun r => a < r ∧ r ≤ b)), Multiset.filter_filter, Multiset.filter_filter]
  congr 2 <;> apply Multiset.filter_congr <;> intro x _ <;> constructor
  · rintro ⟨h, hax, hxb⟩
    exact ⟨hax, h⟩
  · rintro ⟨hax, hxa⟩
    exact ⟨hxa, hax, hxa.trans h2⟩
  · rintro ⟨h, hax, hxb⟩
    exact ⟨lt_of_not_ge h, hxb⟩
  · rintro ⟨hax, hxb⟩
    exact ⟨hax.not_ge, h1.trans_lt hax, hxb⟩

end Transformer.Sturm

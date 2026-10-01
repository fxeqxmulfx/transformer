/-
Copyright (c) 2026 Lean FRO, LLC. All rights reserved.
Released under Apache 2.0 license as described in
third_party/hex-real-roots-mathlib/LICENSE.
Authors: Kim Morrison

The choice of bounding endpoints and infinity signs is adapted from
SturmTheorem.lean at revision 53ce31466dc5ff520463249d470119c1e0006e22.
-/

import Transformer.Sturm.BoundedQuery
import Transformer.Sturm.Infinity

/-!
# Signed real-root queries from variations at infinity

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

noncomputable section
open Filter Topology Polynomial
open scoped BigOperators

namespace Transformer.Sturm

/-- The integer difference of the variations at both infinities for an
actual signed Euclidean chain. Auxiliary for Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. -/
def signedRootQueryValue (p q : Polynomial ℝ) : ℤ :=
  (sturmVarNegInf (signedRemainderChain p q) : ℤ) -
    sturmVarPosInf (signedRemainderChain p q)

/-- For a simple-root polynomial and a pair without common real zeros,
the variation difference equals the sum of the signs of `p'(r) q(r)`
over all distinct real roots. Auxiliary for Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. -/
theorem signedRootQueryValue_eq {p q : Polynomial ℝ} (hp : p ≠ 0)
    (hcommon : ∀ x : ℝ, p.eval x = 0 → q.eval x ≠ 0)
    (hsimple : ∀ x : ℝ, p.eval x = 0 → p.derivative.eval x ≠ 0) :
    signedRootQueryValue p q = ∑ r ∈ p.roots.toFinset,
      (SignType.sign (p.derivative.eval r * q.eval r) : ℤ) := by
  let chain := signedRemainderChain p q
  have hc := signedRemainderChain_isRootAlternating hp hcommon
  obtain ⟨M, hMpos, hM⟩ : ∃ M : ℝ, 0 < M ∧ ∀ x ∈ chainZeros chain, |x| < M := by
    have h : ∀ᶠ M : ℝ in atTop, ∀ x ∈ chainZeros chain, |x| < M := by
      rw [Finset.eventually_all]
      exact fun x _ => eventually_gt_atTop |x|
    exact ((eventually_gt_atTop 0).and h).exists
  have hpos : ∀ f ∈ chain, SignType.sign (f.eval M) = SignType.sign f.leadingCoeff := by
    intro f hf
    apply eval_sign_pos_inf (hc.nonzero_mem f hf)
    intro x hx
    exact (abs_lt.mp (hM x ((mem_chainZeros hc.nonzero_mem).mpr ⟨f, hf, hx⟩))).2
  have hneg : ∀ f ∈ chain,
      SignType.sign (f.eval (-M)) = SignType.sign (f.leadingCoeff * (-1) ^ f.natDegree) := by
    intro f hf
    apply eval_sign_neg_inf (hc.nonzero_mem f hf)
    intro x hx
    exact (abs_lt.mp (hM x ((mem_chainZeros hc.nonzero_mem).mpr ⟨f, hf, hx⟩))).1
  have hvpos : sturmVar chain M = sturmVarPosInf chain := by
    apply signVariations_congr
    rw [List.forall₂_map_left_iff, List.forall₂_map_right_iff, List.forall₂_same]
    exact hpos
  have hvneg : sturmVar chain (-M) = sturmVarNegInf chain := by
    apply signVariations_congr
    rw [List.forall₂_map_left_iff, List.forall₂_map_right_iff, List.forall₂_same]
    exact hneg
  have hMn : ∀ f ∈ chain, f.eval M ≠ 0 := by
    intro f hf hfx
    have h := hM M ((mem_chainZeros hc.nonzero_mem).mpr ⟨f, hf, hfx⟩)
    rw [abs_of_pos hMpos] at h
    exact (lt_irrefl M) h
  have hnegMn : ∀ f ∈ chain, f.eval (-M) ≠ 0 := by
    intro f hf hfx
    have h := hM (-M) ((mem_chainZeros hc.nonzero_mem).mpr ⟨f, hf, hfx⟩)
    rw [abs_neg, abs_of_pos hMpos] at h
    exact (lt_irrefl M) h
  have hkey := signedRemainderChain_query_Ioc hp hcommon hsimple (a := -M) (b := M)
    (by linarith) hnegMn hMn
  have hfilter : p.roots.toFinset.filter (fun x => x ∈ Set.Ioc (-M) M) = p.roots.toFinset := by
    apply Finset.filter_eq_self.mpr
    intro r hr
    have hz : r ∈ chainZeros chain := (mem_chainZeros hc.nonzero_mem).mpr
      ⟨p, hc.head_mem, (mem_roots hp).mp (Multiset.mem_toFinset.mp hr)⟩
    have h := abs_lt.mp (hM r hz)
    exact ⟨h.1, h.2.le⟩
  rw [hvneg, hvpos, hfilter] at hkey
  exact hkey

/-- A polynomial and negative constant query have no common root and
simple actual real roots. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
example : (X : Polynomial ℝ) ≠ 0 ∧
    (∀ x : ℝ, (X : Polynomial ℝ).eval x = 0 → (-1 : Polynomial ℝ).eval x ≠ 0) ∧
    (∀ x : ℝ, (X : Polynomial ℝ).eval x = 0 → (X : Polynomial ℝ).derivative.eval x ≠ 0) := by
  exact ⟨X_ne_zero, fun _ _ => by simp, fun _ _ => by simp⟩

end Transformer.Sturm

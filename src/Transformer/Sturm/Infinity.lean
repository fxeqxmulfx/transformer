/-
Copyright (c) 2026 Lean FRO, LLC. All rights reserved.
Released under Apache 2.0 license as described in
third_party/hex-real-roots-mathlib/LICENSE.
Authors: Kim Morrison

Adapted from leanprover/hex-real-roots-mathlib,
revision 53ce31466dc5ff520463249d470119c1e0006e22.
-/

import Transformer.Sturm.RootCount

/-!
# Sturm root count on the real line

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

open Filter Topology

namespace Transformer.Sturm

variable {p : Polynomial ℝ} {chain : List (Polynomial ℝ)}

/-- **Sign at `+∞`.** Past all its real roots, a nonzero real polynomial has the
sign of its leading coefficient.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem eval_sign_pos_inf {q : Polynomial ℝ} (hq : q ≠ 0) {x : ℝ}
    (hbeyond : ∀ y, q.IsRoot y → y < x) :
    SignType.sign (q.eval x) = SignType.sign q.leadingCoeff := by
  have hlc : q.leadingCoeff ≠ 0 := Polynomial.leadingCoeff_ne_zero.mpr hq
  rcases lt_or_gt_of_ne hlc with h | h
  · rw [sign_neg (Polynomial.eval_lt_zero_of_roots_lt_of_leadingCoeff_nonpos hbeyond h.le),
      sign_neg h]
  · rw [sign_pos (Polynomial.zero_lt_eval_of_roots_lt_of_leadingCoeff_nonneg hbeyond h.le),
      sign_pos h]

/-- **Sign at `−∞`.** Below all its real roots, a nonzero real polynomial has the
sign of `leadingCoeff · (-1) ^ natDegree`.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem eval_sign_neg_inf {q : Polynomial ℝ} (hq : q ≠ 0) {x : ℝ}
    (hbeyond : ∀ y, q.IsRoot y → x < y) :
    SignType.sign (q.eval x) = SignType.sign (q.leadingCoeff * (-1) ^ q.natDegree) := by
  set r := q.comp (-Polynomial.X) with hr
  have hlcr : r.leadingCoeff = q.leadingCoeff * (-1) ^ q.natDegree := by
    rw [hr, Polynomial.leadingCoeff_comp (by simp)]; simp
  have hrne : r ≠ 0 := by
    intro h; rw [h, Polynomial.leadingCoeff_zero] at hlcr
    exact (mul_ne_zero (Polynomial.leadingCoeff_ne_zero.mpr hq)
      (pow_ne_zero _ (by norm_num))) hlcr.symm
  have heval : r.eval (-x) = q.eval x := by rw [hr, Polynomial.eval_comp]; simp
  have hbeyond' : ∀ y, r.IsRoot y → y < -x := by
    intro y hy
    have hqy : q.IsRoot (-y) := by
      rw [hr, Polynomial.IsRoot, Polynomial.eval_comp] at hy; simpa using hy
    have := hbeyond (-y) hqy
    linarith
  have hsign := eval_sign_pos_inf hrne hbeyond'
  rw [heval, hlcr] at hsign
  exact hsign

/-- **Sturm's theorem** on the real line: the decrease in sign variations from
`-∞` to `+∞` counts all real roots.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem IsSturmChain.sturm (hchain : IsSturmChain p chain) (hnod : p.roots.Nodup) :
    sturmVarPosInf chain + p.roots.card = sturmVarNegInf chain := by
  classical
  have hne := hchain.nonzero_mem
  -- A bound `M > 0` strictly beyond every chain zero (hence every root of every element).
  obtain ⟨M, hMpos, hM⟩ : ∃ M : ℝ, 0 < M ∧ ∀ x ∈ chainZeros chain, |x| < M := by
    have h : ∀ᶠ M : ℝ in atTop, ∀ x ∈ chainZeros chain, |x| < M := by
      rw [Finset.eventually_all]
      exact fun x _ => eventually_gt_atTop |x|
    exact ((eventually_gt_atTop 0).and h).exists
  -- Sign of each element at `±M` is its sign at the corresponding infinity.
  have hpos : ∀ q ∈ chain, SignType.sign (q.eval M) = SignType.sign q.leadingCoeff := by
    intro q hq
    apply eval_sign_pos_inf (hne q hq)
    intro y hy
    have hyz : y ∈ chainZeros chain := (mem_chainZeros hne).mpr ⟨q, hq, hy⟩
    have hya := hM y hyz; rw [abs_lt] at hya; exact hya.2
  have hneg : ∀ q ∈ chain,
      SignType.sign (q.eval (-M)) = SignType.sign (q.leadingCoeff * (-1) ^ q.natDegree) := by
    intro q hq
    apply eval_sign_neg_inf (hne q hq)
    intro y hy
    have hyz : y ∈ chainZeros chain := (mem_chainZeros hne).mpr ⟨q, hq, hy⟩
    have hya := hM y hyz; rw [abs_lt] at hya; exact hya.1
  -- Hence `sturmVar` at `±M` equals the `±∞` counts.
  have hMposEq : sturmVar chain M = sturmVarPosInf chain := by
    change signVariations (chain.map (Polynomial.eval M))
      = signVariations (chain.map Polynomial.leadingCoeff)
    apply signVariations_congr
    rw [List.forall₂_map_left_iff, List.forall₂_map_right_iff, List.forall₂_same]
    exact hpos
  have hMnegEq : sturmVar chain (-M) = sturmVarNegInf chain := by
    change signVariations (chain.map (Polynomial.eval (-M)))
      = signVariations (chain.map (fun q => q.leadingCoeff * (-1) ^ q.natDegree))
    apply signVariations_congr
    rw [List.forall₂_map_left_iff, List.forall₂_map_right_iff, List.forall₂_same]
    exact hneg
  -- Apply the half-open form on `(-M, M]`, which catches every root.
  have hkey := hchain.sturm_Ioc hnod (a := -M) (b := M) (by linarith)
  have hfilter : p.roots.filter (fun r => r ∈ Set.Ioc (-M) M) = p.roots := by
    rw [Multiset.filter_eq_self]
    intro r hr
    have hroot : p.eval r = 0 := (Polynomial.mem_roots hchain.ne_zero).mp hr
    have hrz : r ∈ chainZeros chain := (mem_chainZeros hne).mpr ⟨p, hchain.head_mem, hroot⟩
    have hra := hM r hrz; rw [abs_lt] at hra
    exact ⟨hra.1, hra.2.le⟩
  simpa only [hMnegEq, hMposEq, hfilter] using hkey

end Transformer.Sturm

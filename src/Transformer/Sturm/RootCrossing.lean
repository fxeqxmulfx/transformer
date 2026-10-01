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
# The variation jump at a simple head root

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

open Filter Topology

namespace Transformer.Sturm

variable {p : Polynomial ℝ} {chain : List (Polynomial ℝ)}

/-- Crossing a root of the first entry decreases the variation count by one.
The count at the root equals the count just to its right.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem sturmVar_root_cross (hchain : IsSturmChain p chain) (r : ℝ) (hr : p.IsRoot r)
    (a b : ℝ) (har : a < r) (hrb : r < b)
    (hz : ∀ q ∈ chain, ∀ x ∈ Set.Icc a b, x ≠ r → q.eval x ≠ 0) :
    sturmVar chain a = sturmVar chain b + 1 ∧ sturmVar chain r = sturmVar chain b := by
  have hpz : ∀ x ∈ Set.Icc a b, x ≠ r → ¬ p.IsRoot x :=
    fun x hx hxr => hz p hchain.head_mem x hx hxr
  have hab : a ≤ b := (har.trans hrb).le
  obtain ⟨q, hq1, hqr, hflL, hflR⟩ := hchain.root_flank r hr
  -- Split the chain into its head `p` and second element `q`.
  rcases chain with _ | ⟨p0, _ | ⟨q0, tail⟩⟩
  · exact absurd hchain.head (by simp)
  · exact absurd hq1 (by simp)
  have hp0 : p0 = p := by simpa using hchain.head
  subst p0
  have hq0 : q0 = q := by simpa using hq1
  subst q0
  -- Basic nonvanishing facts.
  have hpr0 : p.eval r = 0 := hr
  have hpa : p.eval a ≠ 0 := fun h => hpz a ⟨le_refl a, hab⟩ (ne_of_lt har) h
  have hpb : p.eval b ≠ 0 := fun h => hpz b ⟨hab, le_refl b⟩ (ne_of_lt hrb).symm h
  have hqa : q.eval a ≠ 0 := hz q (by simp) a ⟨le_refl a, hab⟩ (ne_of_lt har)
  have hqb : q.eval b ≠ 0 := hz q (by simp) b ⟨hab, le_refl b⟩ (ne_of_lt hrb).symm
  -- The head pair `p * q` is negative just left of `r` and positive just right,
  -- and (only zero at `r`) this persists to the endpoints.
  have hpqa : (p * q).eval a < 0 := by
    obtain ⟨c, hclt, hcmem⟩ := (hflL.and (Ioo_mem_nhdsLT har)).exists
    have hsg : SignType.sign ((p * q).eval a) = SignType.sign ((p * q).eval c) :=
      eval_sign_eq_of_no_zero hcmem.1.le (fun x hx => by
        have hxr : x ≠ r := ne_of_lt (lt_of_le_of_lt hx.2 hcmem.2)
        have hxab : x ∈ Set.Icc a b := ⟨hx.1, hx.2.trans (hcmem.2.le.trans hrb.le)⟩
        rw [Polynomial.eval_mul]
        exact mul_ne_zero (fun h => hpz x hxab hxr h) (hz q (by simp) x hxab hxr))
    rw [sign_neg hclt] at hsg
    exact sign_eq_neg_one_iff.mp hsg
  have hpqb : 0 < (p * q).eval b := by
    obtain ⟨c, hcgt, hcmem⟩ := (hflR.and (Ioo_mem_nhdsGT hrb)).exists
    have hsg : SignType.sign ((p * q).eval c) = SignType.sign ((p * q).eval b) :=
      eval_sign_eq_of_no_zero hcmem.2.le (fun x hx => by
        have hxr : x ≠ r := (ne_of_lt (lt_of_lt_of_le hcmem.1 hx.1)).symm
        have hxab : x ∈ Set.Icc a b :=
          ⟨har.le.trans (hcmem.1.le.trans hx.1), hx.2⟩
        rw [Polynomial.eval_mul]
        exact mul_ne_zero (fun h => hpz x hxab hxr h) (hz q (by simp) x hxab hxr))
    rw [sign_pos hcgt] at hsg
    exact sign_eq_one_iff.mp hsg.symm
  have hsignA : SignType.sign (p.eval a) * SignType.sign (q.eval a) = -1 := by
    rw [← sign_mul, sign_eq_neg_one_iff, ← Polynomial.eval_mul]; exact hpqa
  have hsignB : ¬ (SignType.sign (p.eval b) * SignType.sign (q.eval b) = -1) := by
    rw [← sign_mul, sign_eq_neg_one_iff, ← Polynomial.eval_mul]
    exact not_lt.mpr hpqb.le
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
  -- Head-pair bookkeeping at each point.
  have hSVa : sturmVar (p :: q :: tail) a = 1 + sturmVar (q :: tail) a := by
    change signVariations (p.eval a :: (q :: tail).map (Polynomial.eval a))
      = 1 + signVariations ((q :: tail).map (Polynomial.eval a))
    rw [signVariations_cons _ hpa]
    simp only [List.map_cons]
    rw [firstSign_cons_ne _ hqa, ite_eq_left hsignA]
  have hSVb : sturmVar (p :: q :: tail) b = sturmVar (q :: tail) b := by
    change signVariations (p.eval b :: (q :: tail).map (Polynomial.eval b))
      = signVariations ((q :: tail).map (Polynomial.eval b))
    rw [signVariations_cons _ hpb]
    simp only [List.map_cons]
    rw [firstSign_cons_ne _ hqb, ite_eq_right hsignB, zero_add]
  have hSVr : sturmVar (p :: q :: tail) r = sturmVar (q :: tail) r := by
    change signVariations (p.eval r :: (q :: tail).map (Polynomial.eval r))
      = signVariations ((q :: tail).map (Polynomial.eval r))
    rw [hpr0]; exact signVariations_cons_zero _
  refine ⟨?_, ?_⟩
  · rw [hSVa, hSVb, hEqA, hEqB]; omega
  · rw [hSVr, hSVb]; exact hEqB.symm

end Transformer.Sturm

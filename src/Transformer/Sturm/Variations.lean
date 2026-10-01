/-
Copyright (c) 2026 Lean FRO, LLC. All rights reserved.
Released under Apache 2.0 license as described in
third_party/hex-real-roots-mathlib/LICENSE.
Authors: Kim Morrison

Adapted from leanprover/hex-real-roots-mathlib,
revision 53ce31466dc5ff520463249d470119c1e0006e22.
-/

import Mathlib.Basic.Sign.Basic
import Mathlib.Algebra.Polynomial.Eval.Defs
import Mathlib.Algebra.Polynomial.Degree.Defs
import Mathlib.Topology.Instances.Real.Lemmas

/-!
# Zero-skipping sign variations

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

open Filter Topology

namespace Transformer.Sturm

/-- Count the sign changes of a real list: the number of adjacent pairs
whose product is negative. Callers first drop the zero entries (see
`Transformer.Sturm.signVariations`), so on a zero-free list this is exactly the number
of adjacent opposite-sign pairs.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
noncomputable def countSignChanges : List ℝ → ℕ
  | a :: b :: rest => (if a * b < 0 then 1 else 0) + countSignChanges (b :: rest)
  | _ => 0

/-- countSignChanges nil. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] theorem countSignChanges_nil : countSignChanges [] = 0 := rfl

/-- countSignChanges singleton. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] theorem countSignChanges_singleton (a : ℝ) : countSignChanges [a] = 0 := rfl

theorem countSignChanges_cons_cons (a b : ℝ) (rest : List ℝ) :
    countSignChanges (a :: b :: rest) =
      (if a * b < 0 then 1 else 0) + countSignChanges (b :: rest) := rfl

/-- Zero-skipping sign variations of a real list: drop the zeros, then count
the adjacent opposite-sign pairs. This is the variation count that both the
pointwise chain evaluations and the leading-coefficient signs at `±∞` feed
into.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
noncomputable def signVariations (l : List ℝ) : ℕ :=
  countSignChanges (l.filter (fun v => decide (v ≠ 0)))

/-- signVariations nil. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] theorem signVariations_nil : signVariations [] = 0 := rfl

/-- signVariations singleton. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] theorem signVariations_singleton (a : ℝ) : signVariations [a] = 0 := by
  by_cases ha : a = 0 <;> simp [signVariations, ha]

/-- Prepending a zero entry does not change the sign variations.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] theorem signVariations_cons_zero (l : List ℝ) :
    signVariations (0 :: l) = signVariations l := by
  simp [signVariations]

/-- A nonzero first entry survives removal of zero entries.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem signVariations_cons_ne (a : ℝ) (l : List ℝ) (ha : a ≠ 0) :
    signVariations (a :: l) =
      countSignChanges (a :: l.filter (fun v => decide (v ≠ 0))) := by
  simp [signVariations, ha]

/-- Zero-skipping sign variations of the chain `chain` evaluated at `x`:
the sign variations of the list of evaluations `chain.map (·.eval x)`.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
noncomputable def sturmVar (chain : List (Polynomial ℝ)) (x : ℝ) : ℕ :=
  signVariations (chain.map (Polynomial.eval x))

/-- sturmVar nil. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] theorem sturmVar_nil (x : ℝ) : sturmVar [] x = 0 := rfl

/-- A chain element that vanishes at `x` contributes no variation at `x`:
`sturmVar` ignores it.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem sturmVar_cons_zero {q : Polynomial ℝ} {x : ℝ} (h : q.eval x = 0)
    (chain : List (Polynomial ℝ)) :
    sturmVar (q :: chain) x = sturmVar chain x := by
  simp [sturmVar, List.map_cons, signVariations, h]

/-- Two real lists whose entries have pointwise equal signs have equal
`countSignChanges`: the sign-change count reads only the signs of the entries.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem countSignChanges_congr {l₁ l₂ : List ℝ}
    (h : List.Forall₂ (fun u v => SignType.sign u = SignType.sign v) l₁ l₂) :
    countSignChanges l₁ = countSignChanges l₂ := by
  induction h with
  | nil => rfl
  | @cons a b l₁' l₂' hab htail ih =>
    cases htail with
    | nil => rfl
    | @cons c d l₁'' l₂'' hcd _ =>
      rw [countSignChanges_cons_cons, countSignChanges_cons_cons]
      have hiff : (a * c < 0) ↔ (b * d < 0) := by
        rw [← sign_eq_neg_one_iff, ← sign_eq_neg_one_iff, sign_mul, sign_mul, hab, hcd]
      simp only [hiff, ih]

/-- Dropping the zero entries commutes with a pointwise sign-equal
correspondence: the filtered lists remain pointwise sign-equal.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem filter_ne_zero_congr {l₁ l₂ : List ℝ}
    (h : List.Forall₂ (fun u v => SignType.sign u = SignType.sign v) l₁ l₂) :
    List.Forall₂ (fun u v => SignType.sign u = SignType.sign v)
      (l₁.filter (fun v => decide (v ≠ 0))) (l₂.filter (fun v => decide (v ≠ 0))) := by
  induction h with
  | nil => exact List.Forall₂.nil
  | @cons a b l₁' l₂' hab htail ih =>
    have hzero : (a = 0) ↔ (b = 0) := by
      rw [← sign_eq_zero_iff (a := a), ← sign_eq_zero_iff (a := b), hab]
    by_cases ha : a = 0
    · simpa [ha, hzero.mp ha] using ih
    · simpa [ha, mt hzero.mpr ha] using
        List.Forall₂.cons (R := fun u v : ℝ => SignType.sign u = SignType.sign v) hab ih

/-- `signVariations` reads only the signs of the entries: two real lists whose
entries are pointwise sign-equal have equal sign variations.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem signVariations_congr {l₁ l₂ : List ℝ}
    (h : List.Forall₂ (fun u v => SignType.sign u = SignType.sign v) l₁ l₂) :
    signVariations l₁ = signVariations l₂ :=
  countSignChanges_congr (filter_ne_zero_congr h)

/-- The sign of the first nonzero entry of a real list, or `0` if every entry is zero.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
noncomputable def firstSign (l : List ℝ) : SignType :=
  ((l.filter (fun v => decide (v ≠ 0))).head?.map SignType.sign).getD 0

/-- firstSign nil. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] theorem firstSign_nil : firstSign [] = 0 := rfl

/-- firstSign cons zero. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] theorem firstSign_cons_zero (l : List ℝ) : firstSign (0 :: l) = firstSign l := by
  simp [firstSign]

/-- firstSign cons ne. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] theorem firstSign_cons_ne {a : ℝ} (l : List ℝ) (ha : a ≠ 0) :
    firstSign (a :: l) = SignType.sign a := by
  simp [firstSign, ha]

theorem sign_mul_eq_neg_one {a b : ℝ} :
    (SignType.sign a * SignType.sign b = -1) ↔ a * b < 0 := by
  rw [← sign_mul, sign_eq_neg_one_iff]

/-- Prepending a nonzero entry `a` adds one variation exactly when its sign is
opposite the sign of the next surviving entry.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem signVariations_cons {a : ℝ} (l : List ℝ) (ha : a ≠ 0) :
    signVariations (a :: l) =
      (if SignType.sign a * firstSign l = -1 then 1 else 0) + signVariations l := by
  induction l with
  | nil => rw [signVariations_cons_ne a [] ha]; simp [firstSign]
  | cons b l' ih =>
    by_cases hb : b = 0
    · subst hb
      rw [firstSign_cons_zero l', signVariations_cons_zero l',
        signVariations_cons_ne a (0 :: l') ha, List.filter_cons_of_neg (by simp),
        ← signVariations_cons_ne a l' ha]
      exact ih
    · rw [firstSign_cons_ne l' hb, signVariations_cons_ne a (b :: l') ha,
        List.filter_cons_of_pos (by simp [hb]), countSignChanges_cons_cons,
        ← signVariations_cons_ne b l' hb]
      congr 1
      simp only [sign_mul_eq_neg_one]

end Transformer.Sturm

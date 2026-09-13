/-
# The hull tests in finite precision

Every comparison the hull makes goes through `isect`, and `isect` is a
division (`Transformer.ALM.Envelope`, `interX`).  The implementation runs it
in `long double`, so the two breakpoints being compared are each off by a
little.  Nothing proved so far says the comparisons still come out right.

What is needed is a separation condition, and the paraboloid supplies one.
`interX_liftKey` says the breakpoint of two lifted keys is their midpoint, so
consecutive breakpoints differ by `(k₃ - k₁) / 2`.  For the integer keys the
`LookUp` primitive compiles to that is at least `1/2`, a gap that does not
shrink as the instance grows — the comparison is safe as soon as the rounding
error is below a quarter of it.

`FPArith` carries the standard model of floating-point arithmetic: the
computed value of a quantity differs from the exact one by at most a relative
`u` (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., §2.2,
eq. (2.4)).  `exactArith` is the `u = 0` instance, so nothing here is vacuous.

Two results follow.  `fp_no_spurious_erase` — rounding never fires
`add_line`'s erase loop, so `liftKey_not_dominated` survives finite
precision.  `fp_query_branch` — a query separated from the breakpoint takes
the same branch as exact arithmetic would, and `lineEval_eq_at_midpoint`
shows the excluded case is the one where the branch does not matter: at the
breakpoint the two keys score equally, so either is an argmax.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 60-70 (`isect`) and
95-98 (the comparison `add_line` loops on).
-/

import Transformer.ALM.Hull

namespace Transformer
namespace ALM

/-! ### A separated comparison is decided correctly -/

/-- **The sign of a comparison survives rounding when the operands are
separated.**  If `a'` and `b'` are within `δ₁` and `δ₂` of `a` and `b`, and
`a` and `b` are farther apart than `δ₁ + δ₂`, the approximate comparison has
the same truth value as the exact one. -/
theorem cmp_of_sep {a b a' b' δ₁ δ₂ : ℝ} (ha : |a' - a| ≤ δ₁) (hb : |b' - b| ≤ δ₂)
    (hsep : δ₁ + δ₂ < |a - b|) : a' ≤ b' ↔ a ≤ b := by
  have hδ₁ : 0 ≤ δ₁ := le_trans (abs_nonneg _) ha
  have hδ₂ : 0 ≤ δ₂ := le_trans (abs_nonneg _) hb
  rw [abs_le] at ha hb
  rcases le_total a b with h | h
  · have hab : δ₁ + δ₂ < b - a := by
      rwa [abs_of_nonpos (by linarith), neg_sub] at hsep
    exact ⟨fun _ => h, fun _ => by linarith [ha.2, hb.1]⟩
  · have hab : δ₁ + δ₂ < a - b := by rwa [abs_of_nonneg (by linarith)] at hsep
    exact ⟨fun _ => by linarith [ha.1, hb.2], fun _ => by linarith⟩

/-! ### The arithmetic the implementation runs on -/

/-- The breakpoint computation as it is actually performed, together with the
relative accuracy it guarantees.  This is the standard model of floating-point
arithmetic applied to `isect`: the computed result is the exact one perturbed
by a relative `u`. -/
structure FPArith where
  /-- The breakpoint as computed. -/
  isect : ℝ × ℝ → ℝ × ℝ → ℝ
  /-- The relative accuracy it achieves. -/
  u : ℝ
  /-- Accuracy is a bound, hence nonnegative. -/
  u_nonneg : 0 ≤ u
  /-- And the bound it satisfies, whenever the breakpoint is defined. -/
  isect_rel : ∀ l l' : ℝ × ℝ, l.1 ≠ l'.1 →
    |isect l l' - interX l l'| ≤ u * |interX l l'|

/-- Exact arithmetic is the `u = 0` instance: everything below is a statement
about a structure that exists. -/
noncomputable def exactArith : FPArith where
  isect := interX
  u := 0
  u_nonneg := le_rfl
  isect_rel := fun l l' _ => by simp

/-! ### The error on a lifted-key breakpoint -/

/-- On keys bounded by `M` the computed midpoint is off by at most `u · M`:
the relative bound becomes an absolute one, because the midpoint of two keys
in `[-M, M]` is itself in `[-M, M]`. -/
lemma isect_liftKey_error (F : FPArith) {k k' M : ℝ} (h : k ≠ k')
    (hk : |k| ≤ M) (hk' : |k'| ≤ M) :
    |F.isect (liftKey k) (liftKey k') - (k + k') / 2| ≤ F.u * M := by
  have hslope : (liftKey k).1 ≠ (liftKey k').1 := by
    unfold liftKey
    simp only []
    intro hz
    exact h (by linarith)
  have hrel := F.isect_rel (liftKey k) (liftKey k') hslope
  rw [interX_liftKey h] at hrel
  have hmid : |(k + k') / 2| ≤ M := by
    have habs : |(k + k') / 2| = |k + k'| / 2 := by rw [abs_div]; norm_num
    have := abs_add_le k k'
    rw [habs]
    linarith
  exact le_trans hrel (mul_le_mul_of_nonneg_left hmid F.u_nonneg)

/-! ### The erase loop never fires by accident -/

/-- **Rounding does not discard a key.**  For three keys in `[-M, M]` whose
outer two are at least `1` apart — which integer keys always are — the
computed comparison `isect(k₂,k₃) ≤ isect(k₁,k₂)` is still false, so
`add_line` keeps the middle line.  `liftKey_not_dominated` therefore holds of
the arithmetic that actually runs, not only of the reals. -/
theorem fp_no_spurious_erase (F : FPArith) {k₁ k₂ k₃ M : ℝ}
    (h₁₂ : k₁ < k₂) (h₂₃ : k₂ < k₃) (hgap : 1 ≤ k₃ - k₁)
    (hb₁ : |k₁| ≤ M) (hb₂ : |k₂| ≤ M) (hb₃ : |k₃| ≤ M)
    (hu : F.u * M < 1 / 4) :
    ¬ (F.isect (liftKey k₂) (liftKey k₃) ≤ F.isect (liftKey k₁) (liftKey k₂)) := by
  have e₂₃ := isect_liftKey_error F (ne_of_lt h₂₃) hb₂ hb₃
  have e₁₂ := isect_liftKey_error F (ne_of_lt h₁₂) hb₁ hb₂
  have hsep : F.u * M + F.u * M < |(k₂ + k₃) / 2 - (k₁ + k₂) / 2| := by
    rw [show (k₂ + k₃) / 2 - (k₁ + k₂) / 2 = (k₃ - k₁) / 2 by ring,
      abs_of_nonneg (by linarith)]
    linarith
  rw [cmp_of_sep e₂₃ e₁₂ hsep]
  intro hle
  linarith

/-! ### The query branch -/

/-- **A separated query takes the right branch.**  If the query is farther
from the midpoint than the rounding error, the test `q ≤ isect(k,k')` that
the binary search of `Transformer.ALM.BinSearch` performs agrees with the
exact one, so `hull_bsearch_isGreatest` applies unchanged. -/
theorem fp_query_branch (F : FPArith) {k k' q M : ℝ} (h : k ≠ k')
    (hk : |k| ≤ M) (hk' : |k'| ≤ M)
    (hsep : F.u * M < |q - (k + k') / 2|) :
    q ≤ F.isect (liftKey k) (liftKey k') ↔ q ≤ (k + k') / 2 := by
  refine cmp_of_sep (a := q) (b := (k + k') / 2) (δ₁ := 0) (by simp)
    (isect_liftKey_error F h hk hk') ?_
  rwa [zero_add]

/-- **And the excluded case does not matter.**  Exactly at the midpoint the
two keys score the same, so whichever branch a mis-rounded test takes, the
key it returns is still an exact argmax of the pair. -/
theorem lineEval_eq_at_midpoint {k k' : ℝ} (h : k ≠ k') :
    lineEval (liftKey k) ((k + k') / 2) = lineEval (liftKey k') ((k + k') / 2) := by
  have hslope : (liftKey k).1 ≠ (liftKey k').1 := by
    unfold liftKey
    simp only []
    intro hz
    exact h (by linarith)
  have := lineEval_interX (liftKey k) (liftKey k') hslope
  rwa [interX_liftKey h] at this

/-! ### The margin the implementation has -/

/-- **`long double` is far more than enough.**  Allowing a relative error of
`2^-60` — well above the `2^-64` unit roundoff of the 80-bit format, with room
to spare for the subtractions inside `isect` — integer keys of magnitude up to
`2^56` still never trigger a spurious erase.  The machine's keys are far
smaller than that. -/
theorem fp_longDouble (F : FPArith) (hu : F.u ≤ 1 / 2 ^ (60 : ℕ)) {k₁ k₂ k₃ : ℝ}
    (h₁₂ : k₁ < k₂) (h₂₃ : k₂ < k₃) (hgap : 1 ≤ k₃ - k₁)
    (hb₁ : |k₁| ≤ 2 ^ (56 : ℕ)) (hb₂ : |k₂| ≤ 2 ^ (56 : ℕ))
    (hb₃ : |k₃| ≤ 2 ^ (56 : ℕ)) :
    ¬ (F.isect (liftKey k₂) (liftKey k₃) ≤ F.isect (liftKey k₁) (liftKey k₂)) := by
  refine fp_no_spurious_erase F h₁₂ h₂₃ hgap hb₁ hb₂ hb₃ ?_
  have hpos : (0 : ℝ) < 2 ^ (56 : ℕ) := by positivity
  have := mul_le_mul_of_nonneg_right hu hpos.le
  have hval : (1 / 2 ^ (60 : ℕ)) * (2 : ℝ) ^ (56 : ℕ) = 1 / 16 := by norm_num
  rw [hval] at this
  linarith

/-- The hypotheses are satisfiable: exact arithmetic has `u = 0`, and the keys
`0 < 1 < 2` meet the gap and range conditions. -/
example : ¬ (exactArith.isect (liftKey 1) (liftKey 2)
    ≤ exactArith.isect (liftKey 0) (liftKey 1)) :=
  fp_longDouble exactArith (by norm_num [exactArith]) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end ALM
end Transformer

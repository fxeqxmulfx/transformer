/-
# The query need not land on the grid, only near it

`Transformer.ALM.LiftCompare` rebuilds the head's comparison from the integer
key, and asks the query to be an integer too: `upper_lt_iff` is stated at
`(q, ±1)` with `q : ℤ`.  The machine cannot always supply that.  The compiler
multiplies every hard-attention query by `HARD_K · √2`, and `QueryScale`'s
`onTheGrid` divides it back out; the division is a rounding, so what reaches
the head is `z₀ + ε` for an integer `z₀` and a residual `ε` of a few ulps —
one ulp, on the shipped programs, which at a query of `2^33` is `1.9 · 10⁻⁶`.

`QueryScale.rounded_normalization_keeps_the_winner` already pays for that
residual, and pays too much: its margin is `4Kε < 1`, so at keys of `3 · 10⁸`
it needs `ε < 10⁻⁹` and the shipped residual is three orders over.  The margin
is conservative because it bounds the *score* perturbation, and the score moves
by `2ε(k - k')`, which is large.  What does not move is the *order*: the keys
are integers, so a squared distance is an integer, and the residual reaches the
comparison only multiplied by the same `k - k'` that the integer gap carries.
`upper_near_lt_iff` and `lower_near_lt_iff` are that cancellation — the same
equivalences as `LiftCompare`'s, with `ε` in them, under the one hypothesis
`|ε| < 1/2 - A`, which the shipped offset spread leaves as `0.067`
(`the_shipped_window`) against a measured `1.9 · 10⁻⁶`.

The residual is not rounded away: it is carried into the tie, which is the one
place it decides anything.  Two keys equidistant from `z₀` are separated by
`2ε(k - k')` and by the offsets, and `upper_lt_iff_of_sq_eq` is that comparison
exactly, with no hypothesis on `ε` at all.

Source: `todo3.md` §0, §4a; `vm-rs/alm-hull/src/liftkey.rs` (`UnitQuery::of`,
`UnitQuery::cmp`), `vm-rs/alm-model/src/cache.rs` (`on_the_grid`).
-/

import Transformer.ALM.LiftCompare

namespace Transformer
namespace ALM
namespace LiftResidual

/-! ### The split -/

/-- **The score difference at a near-integer query.**  At `q = z₀ + ε` the
marked key's score is `q² - (k - q)² + δ`, and `(k - q)² = ((k - z₀) - ε)²`, so
the difference of two scores is the integer difference of squared distances,
the residual times the integer step between the keys, and the two offsets. -/
theorem upper_sub (ε δ δ' : ℝ) (z z' z₀ : ℤ) :
    dot ((z₀ : ℝ) + ε, 1) (markKey δ (z : ℝ)) - dot ((z₀ : ℝ) + ε, 1) (markKey δ' (z' : ℝ))
      = (((z' - z) * ((z - z₀) + (z' - z₀)) : ℤ) : ℝ) - 2 * ε * ((z' - z : ℤ) : ℝ)
        + (δ - δ') := by
  unfold dot markKey
  push_cast
  ring

/-- **The lower envelope's split.**  At `qy = -1` the score is
`(k + q)² - q² - δ`, convex in the key, so the same three terms appear with the
distances taken to `-z₀` and the offsets reversed. -/
theorem lower_sub (ε δ δ' : ℝ) (z z' z₀ : ℤ) :
    dot ((z₀ : ℝ) + ε, -1) (markKey δ (z : ℝ)) - dot ((z₀ : ℝ) + ε, -1) (markKey δ' (z' : ℝ))
      = (((z - z') * ((z + z₀) + (z' + z₀)) : ℤ) : ℝ) + 2 * ε * ((z - z' : ℤ) : ℝ)
        + (-δ + δ') := by
  unfold dot markKey
  push_cast
  ring

/-! ### A unit of integer separation survives the residual -/

/-- **The cancellation.**  The integer part of the difference is `J · s`, where
`J` is the step between the keys and `s` the sum of their signed distances; the
residual enters as `2εJ`, carrying the same `J`.  So the comparison is decided
by `s - 2ε` against the offsets, and one unit of `s` beats `2A` as soon as the
residual is inside `1/2 - A`.  This is what `4Kε < 1` misses: the factor `K`
divides out.

Source: `todo3.md` §4a; the argument is the integrality of `s`, so the bound is
independent of the magnitude of the keys. -/
theorem int_lt_of_drift {A ε δ δ' : ℝ} {J s : ℤ} (hε : |ε| < 1 / 2 - A) (hδ : |δ| ≤ A)
    (hδ' : |δ'| ≤ A) (h : 1 ≤ J * s) : 0 < (J : ℝ) * ((s : ℝ) - 2 * ε) + (δ - δ') := by
  have hA : 0 ≤ A := le_trans (abs_nonneg δ) hδ
  obtain ⟨hε₁, hε₂⟩ := abs_lt.mp hε
  obtain ⟨hδ₁, _⟩ := abs_le.mp hδ
  obtain ⟨_, hδ'₂⟩ := abs_le.mp hδ'
  have hJ : J ≠ 0 := by rintro rfl; simp at h
  rcases lt_or_gt_of_ne hJ with hneg | hpos
  · have hs : s < 0 := by by_contra hc; nlinarith [not_lt.mp hc]
    have h1 : (1 : ℝ) ≤ -(J : ℝ) := by exact_mod_cast (by omega : (1 : ℤ) ≤ -J)
    have h2 : (1 : ℝ) ≤ -(s : ℝ) := by exact_mod_cast (by omega : (1 : ℤ) ≤ -s)
    have := mul_le_mul_of_nonneg_right h1 (by linarith : (0 : ℝ) ≤ -((s : ℝ) - 2 * ε))
    nlinarith
  · have hs : 0 < s := by by_contra hc; nlinarith [not_lt.mp hc]
    have h1 : (1 : ℝ) ≤ (J : ℝ) := by exact_mod_cast hpos
    have h2 : (1 : ℝ) ≤ (s : ℝ) := by exact_mod_cast hs
    have := mul_le_mul_of_nonneg_right h1 (by linarith : (0 : ℝ) ≤ (s : ℝ) - 2 * ε)
    nlinarith

/-- The hypotheses of `int_lt_of_drift` at the shipped numbers: the offset
spread of `markKey`, the worst residual the released programs leave, and the
smallest separation two distinct keys can have. -/
example : |(1.907e-6 : ℝ)| < 1 / 2 - 0.4328 ∧ |(0 : ℝ)| ≤ 0.4328 ∧
    |(0.4 : ℝ)| ≤ 0.4328 ∧ (1 : ℤ) ≤ 1 * 1 := by
  norm_num

/-! ### The comparison, residual and all -/

/-- **The nearer key wins, residual notwithstanding.** -/
theorem upper_lt_of_sq_lt {A ε δ δ' : ℝ} {z z' z₀ : ℤ} (hε : |ε| < 1 / 2 - A) (hδ : |δ| ≤ A)
    (hδ' : |δ'| ≤ A) (h : (z - z₀) ^ 2 < (z' - z₀) ^ 2) :
    dot ((z₀ : ℝ) + ε, 1) (markKey δ' (z' : ℝ)) < dot ((z₀ : ℝ) + ε, 1) (markKey δ (z : ℝ)) := by
  have hI : 1 ≤ (z' - z) * ((z - z₀) + (z' - z₀)) := by nlinarith
  have hd := int_lt_of_drift hε hδ hδ' hI
  have e := upper_sub ε δ δ' z z' z₀
  push_cast at e hd
  linarith

/-- **And where the distances tie, the residual and the offsets decide.**  No
bound on `ε` is needed here: this is an identity, not an estimate. -/
theorem upper_lt_iff_of_sq_eq (ε δ δ' : ℝ) {z z' z₀ : ℤ} (h : (z - z₀) ^ 2 = (z' - z₀) ^ 2) :
    dot ((z₀ : ℝ) + ε, 1) (markKey δ (z : ℝ)) < dot ((z₀ : ℝ) + ε, 1) (markKey δ' (z' : ℝ)) ↔
      2 * ε * ((z - z' : ℤ) : ℝ) + δ < δ' := by
  have hz : ((z' - z) * ((z - z₀) + (z' - z₀)) : ℤ) = 0 := by nlinarith
  rw [← sub_lt_zero, upper_sub, hz]
  push_cast
  constructor <;> intro <;> linarith

/-- **The whole comparison at a near-integer query**, as an equivalence: the
squared distances order the keys, and the residual and the offsets are consulted
only where those tie.  `LiftCompare.upper_lt_iff` is this at `ε = 0`. -/
theorem upper_near_lt_iff {A ε δ δ' : ℝ} {z z' z₀ : ℤ} (hε : |ε| < 1 / 2 - A) (hδ : |δ| ≤ A)
    (hδ' : |δ'| ≤ A) :
    dot ((z₀ : ℝ) + ε, 1) (markKey δ (z : ℝ)) < dot ((z₀ : ℝ) + ε, 1) (markKey δ' (z' : ℝ)) ↔
      (z' - z₀) ^ 2 < (z - z₀) ^ 2 ∨
        ((z - z₀) ^ 2 = (z' - z₀) ^ 2 ∧ 2 * ε * ((z - z' : ℤ) : ℝ) + δ < δ') := by
  rcases lt_trichotomy ((z - z₀) ^ 2) ((z' - z₀) ^ 2) with hlt | heq | hgt
  · have hw := upper_lt_of_sq_lt hε hδ hδ' hlt
    constructor
    · intro hc; exact absurd hw (not_lt.mpr hc.le)
    · rintro (h | ⟨h, _⟩)
      · exact absurd h (not_lt.mpr hlt.le)
      · exact absurd h hlt.ne
  · rw [upper_lt_iff_of_sq_eq ε δ δ' heq]
    refine ⟨fun h => Or.inr ⟨heq, h⟩, ?_⟩
    rintro (h | ⟨-, h⟩)
    · exact absurd h (by rw [heq]; exact lt_irrefl _)
    · exact h
  · have hw := upper_lt_of_sq_lt hε hδ' hδ hgt
    exact ⟨fun _ => Or.inl hgt, fun _ => hw⟩

/-- **The lower envelope, the same way.**  The far key wins, the distances are
taken to `-z₀`, and where they tie the offsets enter reversed. -/
theorem lower_near_lt_iff {A ε δ δ' : ℝ} {z z' z₀ : ℤ} (hε : |ε| < 1 / 2 - A) (hδ : |δ| ≤ A)
    (hδ' : |δ'| ≤ A) :
    dot ((z₀ : ℝ) + ε, -1) (markKey δ (z : ℝ)) < dot ((z₀ : ℝ) + ε, -1) (markKey δ' (z' : ℝ)) ↔
      (z + z₀) ^ 2 < (z' + z₀) ^ 2 ∨
        ((z + z₀) ^ 2 = (z' + z₀) ^ 2 ∧ 2 * ε * ((z - z' : ℤ) : ℝ) + δ' < δ) := by
  have key : ∀ {w w' : ℤ} {d d' : ℝ}, |d| ≤ A → |d'| ≤ A → (w + z₀) ^ 2 < (w' + z₀) ^ 2 →
      dot ((z₀ : ℝ) + ε, -1) (markKey d (w : ℝ)) <
        dot ((z₀ : ℝ) + ε, -1) (markKey d' (w' : ℝ)) := by
    intro w w' d d' hd hd' hlt
    have hI : 1 ≤ (w' - w) * ((w + z₀) + (w' + z₀)) := by nlinarith
    have hdr := int_lt_of_drift (ε := -ε) (by rwa [abs_neg]) hd hd' hI
    have e := lower_sub ε d d' w w' z₀
    push_cast at e hdr
    linarith
  rcases lt_trichotomy ((z + z₀) ^ 2) ((z' + z₀) ^ 2) with hlt | heq | hgt
  · exact ⟨fun _ => Or.inl hlt, fun _ => key hδ hδ' hlt⟩
  · have hz : ((z - z') * ((z + z₀) + (z' + z₀)) : ℤ) = 0 := by nlinarith
    rw [← sub_lt_zero, lower_sub, hz]
    push_cast
    refine ⟨fun h => Or.inr ⟨heq, by linarith⟩, ?_⟩
    rintro (h | ⟨-, h⟩)
    · exact absurd h (by rw [heq]; exact lt_irrefl _)
    · linarith
  · have hw := key hδ' hδ hgt
    constructor
    · intro hc; exact absurd hw (not_lt.mpr hc.le)
    · rintro (h | ⟨h, _⟩)
      · exact absurd h (not_lt.mpr hgt.le)
      · exact absurd h.symm hgt.ne

/-- The hypotheses of `upper_lt_of_sq_lt`, `upper_near_lt_iff` and
`lower_near_lt_iff` together: the shipped spread, the measured residual, and a
query at the key `10`, with the key `12` two steps away. -/
example : |(1.907e-6 : ℝ)| < 1 / 2 - 0.4328 ∧ |(0.1 : ℝ)| ≤ 0.4328 ∧ |(0.4 : ℝ)| ≤ 0.4328 ∧
    ((10 : ℤ) - 10) ^ 2 < ((12 : ℤ) - 10) ^ 2 ∧ ((10 : ℤ) + 10) ^ 2 < ((12 : ℤ) + 10) ^ 2 := by
  norm_num

/-- And the hypothesis of `upper_lt_iff_of_sq_eq`: the symmetric pair, which is
the only place the residual and the offsets are consulted. -/
example : ((9 : ℤ) - 10) ^ 2 = ((11 : ℤ) - 10) ^ 2 := by norm_num

/-! ### The shipped numbers -/

/-- **The window the shipped offset spread leaves for the residual.**  The
compiler's recency term is under `LATEST_ALPHA / log 2 = 0.3 / log 2`, so
`1/2 - A` is above `0.067` — five orders of magnitude over the residual the
released programs leave (`1.9 · 10⁻⁶`, one ulp of a query at `2^33`).

Source: `todo3.md` §0; `alm-vm`'s `queries:` line on the released model. -/
theorem the_shipped_window : (0.067 : ℝ) < 1 / 2 - 0.3 / Real.log 2 := by
  have h : (0.6931471803 : ℝ) < Real.log 2 := Real.log_two_gt_d9
  have hpos : (0 : ℝ) < Real.log 2 := by linarith
  have hlt : (0.3 : ℝ) / Real.log 2 < 0.433 := by rw [div_lt_iff₀ hpos]; nlinarith
  linarith

/-- The measured residual is inside it, which is the hypothesis `hε` of every
theorem above at the shipped constants. -/
example : |(1.907e-6 : ℝ)| < 1 / 2 - 0.3 / Real.log 2 := by
  have := the_shipped_window
  rw [abs_of_pos (by norm_num)]
  linarith

/-- **The §4b pair, at a query the scale left off the grid.**  `336860161` is
half of the abscissa `0x20202020` that `todo3.md` §4b names, where the stored
ordinate has lost the unit that separates the key from its neighbour.  The
comparison rebuilt from the abscissa decides it, and goes on deciding it when
the query misses the integer by anything inside the window. -/
theorem the_section_4a_query {A ε δ δ' : ℝ} (hε : |ε| < 1 / 2 - A) (hδ : |δ| ≤ A)
    (hδ' : |δ'| ≤ A) :
    dot (((336860161 : ℤ) : ℝ) + ε, 1) (markKey δ' ((336860162 : ℤ) : ℝ)) <
      dot (((336860161 : ℤ) : ℝ) + ε, 1) (markKey δ ((336860161 : ℤ) : ℝ)) :=
  upper_lt_of_sq_lt (z₀ := 336860161) hε hδ hδ' (by norm_num)

/-- Its hypotheses, at the shipped spread and the measured residual. -/
example : |(1.907e-6 : ℝ)| < 1 / 2 - 0.4328 ∧ |(0 : ℝ)| ≤ 0.4328 ∧ |(0.4 : ℝ)| ≤ 0.4328 := by
  norm_num

end LiftResidual
end ALM
end Transformer

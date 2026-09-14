/-
# The gap the runtime reports, in the unit the guard compares against

`vm-rs/alm-hull/src/gap.rs` is the diagnostic half of the pair: `ScoreGaps`
divides the distance between the winner and the best strictly-lower score by
one *key step*, and reads the quotient on three bands — near `1` the two keys
are genuinely different, near `10^-5` the compiler's `LATEST_ALPHA` separation
is what is holding the order up, near `10^-15` the matvec's rounding is
deciding the answer.  `Transformer.ALM.LatestWindow` already says what those
last two bands mean.  What nothing said is why the division is there.

It is there because the quotient is the hypothesis.  The step `observe` divides
by is `|q[1]|`, which is the same number `grid.rs::off_the_grid` takes as its
`margin`, and `Transformer.ALM.GuardSep` needs exactly `margin ≤ |a - b|` to
discharge `cmp_of_sep`.  So a reported gap of at least one key step is not a
sign that the query is healthy — it *is* the separation hypothesis, written in
the only scale-free way available (`keyGap_scale_free`), and
`retrieval_of_keyGap` runs it into the answer.

Below the band the same arithmetic says the opposite: `rounding_decides_below`
turns a gap under a rounding budget into the two perturbations that invert the
order, which is what `NOISE = 1e-9` is counting.  And an exact tie is not a gap
at all (`keyGap_self`) — `observe` drops it, because it is `HullMeta::last_seq`
that answers there and not this.

Source: `todo3.md` §2a and §4; `vm-rs/alm-hull/src/gap.rs`, `ScoreGaps::observe`
and its three tests; `vm-rs/alm-hull/src/grid.rs`, `off_the_grid`.
-/

import Transformer.ALM.GuardSep

namespace Transformer
namespace ALM

/-! ### The quotient `observe` forms -/

/-- The distance from the winner to the runner-up measured in key steps: what
`ScoreGaps::observe` stores, with `step` the query's own second coordinate. -/
noncomputable def keyGap (step best second : ℝ) : ℝ := (best - second) / step

/-- **The scale divides out.**  A head queried at scale `σ` reports the same
gap as the same head at unit scale, which is why the numbers `gap.rs` prints
are comparable across runs and across the two reference runtimes at all. -/
@[simp] theorem keyGap_scale_free {σ a b : ℝ} (hσ : σ ≠ 0) :
    keyGap σ (σ * a) (σ * b) = a - b := by
  unfold keyGap
  rw [← mul_sub, mul_comm, mul_div_assoc, div_self hσ, mul_one]

/-- An exact tie is not a gap, so `observe` returns without recording one:
there is nothing here for the guard to be asked about, and `last_seq` answers
instead. -/
@[simp] theorem keyGap_self (σ b : ℝ) : keyGap σ b b = 0 := by
  unfold keyGap; rw [sub_self, zero_div]

/-! ### One key step is the separation hypothesis -/

/-- **A whole step, at either scale.**  Two distinct parabolic integer scores
are a unit apart before the scale (`one_le_sScore_sub`) and the scale divides
out of the quotient, so the reported gap is at least `1`.  This is `gap.rs`'s
own first test, which runs it at `1` and at the shipped `√2 · 10^10`. -/
theorem one_le_keyGap {q j k : ℤ} {σ : ℝ} (hσ : 0 < σ)
    (h : sScore q j < sScore q k) :
    1 ≤ keyGap σ (σ * sScore q k) (σ * sScore q j) := by
  rw [keyGap_scale_free (ne_of_gt hσ)]
  linarith [one_le_sScore_sub h]

/-- **And a whole step is exactly the margin the guard compares against.**  The
step `gap.rs` divides by and the margin `grid.rs` tests against are the same
number, so "the gap is at least one key step" and "the scores are at least a
margin apart" are one statement in two notations — the second being the
hypothesis `Transformer.ALM.GuardSep.cmp_of_guard` consumes. -/
theorem one_le_keyGap_iff {σ best second : ℝ} (hσ : 0 < σ) :
    1 ≤ keyGap σ best second ↔ σ ≤ best - second := by
  unfold keyGap
  rw [le_div_iff₀ hσ, one_mul]

/-! ### What a healthy gap buys -/

/-- **From the reported gap to the answer.**  A query whose diagnostic gap is a
full key step and whose guard passed with a bit to spare has its two computed
scores in the order the exact ones are: the runner-up stays the runner-up
after rounding.  This is the whole of what the two files in `vm-rs` are for,
joined — `gap.rs` supplies the separation, `grid.rs` supplies the spacing, and
`cmp_of_guard` turns the pair into the comparison. -/
theorem retrieval_of_keyGap {p : ℕ} {E : ℤ} {σ best second best' second' δ₁ δ₂ : ℝ}
    (hσ : 0 < σ) (hguard : ulpOf p E < σ) (hgap : 1 ≤ keyGap σ best second)
    (h₁ : |best' - best| ≤ δ₁) (h₂ : |second' - second| ≤ δ₂)
    (hδ₁ : δ₁ ≤ ulpOf p E / 2) (hδ₂ : δ₂ ≤ ulpOf p E / 2) :
    second' < best' := by
  have hd : σ ≤ best - second := (one_le_keyGap_iff hσ).mp hgap
  have hsep : σ ≤ |best - second| := le_trans hd (le_abs_self _)
  have hiff := cmp_of_guard (p := p) (E := E) hguard h₁ h₂ hδ₁ hδ₂ hsep
  exact lt_of_not_ge fun hge => absurd (hiff.mp hge) (by linarith)

/-! ### And what a gap under the noise floor means -/

/-- **Below the floor the rounding decides.**  A gap narrower than a rounding
budget of `η` key steps is inverted by roundings inside that budget: the loser
comes out strictly ahead and nothing announces it.  That is what `NOISE = 1e-9`
counts — §2a's reading that a gap of that size is "the key path's own rounding
and not a separation anyone intended" — and it is the same arithmetic as
`Transformer.ALM.LatestClose.past_the_window_rounding_decides`, here with the
gap given rather than derived from a position. -/
theorem rounding_decides_below {σ η best second : ℝ} (hσ : 0 < σ)
    (hlt : second < best) (hgap : keyGap σ best second < η) :
    ∃ ν₁ ν₂ : ℝ, |ν₁| ≤ η * σ / 2 ∧ |ν₂| ≤ η * σ / 2 ∧ best + ν₁ < second + ν₂ := by
  have hd : best - second < η * σ := by
    unfold keyGap at hgap
    exact (div_lt_iff₀ hσ).mp hgap
  have hη : 0 < η * σ := by linarith
  refine ⟨-(η * σ / 2), η * σ / 2, ?_, ?_, by linarith⟩
  · rw [abs_neg, abs_of_nonneg (by linarith)]
  · rw [abs_of_nonneg (by linarith)]

/-- **The middle band, named.**  Two writes to one logical key at unit query
scale differ by nothing but the compiler's perturbation, so the gap `gap.rs`
reports for them is `α · (inv_log_pos r - inv_log_pos p)` exactly — the `10^-5`
the released weights show, and not a property of the key at all. -/
theorem keyGap_writeScore (α q : ℝ) (k : ℤ) (p r : ℕ) :
    keyGap 1 (writeScore α q k r 0) (writeScore α q k p 0)
      = α * (invLogPos r - invLogPos p) := by
  unfold keyGap writeScore
  rw [div_one]
  ring

/-! ### The hypotheses are satisfiable -/

/-- The two smallest keys at the shipped scale: `q = 1` scores `0` at `k = 0`
and `1` at `k = 1`, so the reported gap is one key step however large `σ` is.
These are the hypotheses of `one_le_keyGap`, `one_le_keyGap_iff` and
`keyGap_scale_free`. -/
example : (0 : ℝ) < 14142135623.730951 ∧ sScore 1 0 < sScore 1 1 ∧
    keyGap 14142135623.730951 (14142135623.730951 * sScore 1 1)
      (14142135623.730951 * sScore 1 0) = 1 := by
  have h0 : sScore 1 0 = 0 := by simp [sScore]
  have h1 : sScore 1 1 = 1 := by norm_num [sScore]
  refine ⟨by norm_num, by rw [h0, h1]; norm_num, ?_⟩
  rw [keyGap_scale_free (by norm_num), h0, h1]
  norm_num

/-- A guard with room and a full key step of gap, at unit scale, with a
half-ulp of rounding on each score: the hypotheses of `retrieval_of_keyGap`. -/
example : (0 : ℝ) < 1 ∧ ulpOf 53 0 < 1 ∧ 1 ≤ keyGap 1 1 0 ∧
    |(1 : ℝ) - 1| ≤ ulpOf 53 0 / 2 ∧ |(0 : ℝ) - 0| ≤ ulpOf 53 0 / 2 := by
  refine ⟨by norm_num, ?_, ?_, ?_, ?_⟩ <;> simp only [ulpOf, keyGap] <;> norm_num

/-- And a gap the rounding owns: `10^-15` key steps against a budget of
`10^-9`, which is §2a's noise floor on `hello`.  These are the hypotheses of
`rounding_decides_below`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1e-15 ∧ keyGap 1 1e-15 0 < 1e-9 := by
  refine ⟨by norm_num, by norm_num, ?_⟩
  simp only [keyGap]
  norm_num

end ALM
end Transformer

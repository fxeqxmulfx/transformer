/-
# What actually orders two writes to one key, and how long it keeps doing it

`Transformer.ALM.HullResolve` and `Transformer.ALM.SoftmaxLatest` describe
`TieBreak::LATEST` as the machine documents it: two entries whose scores are
*equal* are separated by the sequence number `HullMeta::last_seq` carries.  On
the released weights that branch is never taken.  `todo3.md` §2a measures it —
zero ties on latest-flagged heads across `hello`, and the closest runner-up
`6.026e-6` away — because a key is a matvec through the residual stream and the
matvec rounds, so two writes to one logical key do not arrive at the head with
the same key and there is no exact tie left to break.

What orders them instead is a perturbation the compiler folds into the key:
`graph/core.py:316` adds `LATEST_ALPHA * inv_log_pos` to `ky`, and
`evaluator.py:230` computes `inv_log_pos` at position `p` as
`1/log 2 - 1/log (p + 2)`, which increases with `p`.  So the released machine
runs on the perturbation and `last_seq` is its unreached fallback, and nothing
in this development said what the perturbation has to satisfy.

This file says it, as a two-sided window on the scale `α` against the rounding
`η` the key path introduces:

* `latest_wins` — the lower side.  While one step of the perturbation is wider
  than the rounding, the later of two writes to one logical key scores strictly
  higher, with no tie and no sequence number consulted.
* `distinct_keys_keep_their_order` — the upper side.  While the whole
  perturbation stays under the unit gap between distinct integer keys, it
  reorders nothing it was not meant to.
* `released_alpha_below_half` — and the shipped `LATEST_ALPHA = 0.3` clears the
  upper side by a factor of two, which is where §2a's `1/2` comes from.

The lower side, unlike the upper one, cannot be met forever — the step shrinks
and the rounding does not — and `Transformer.ALM.LatestClose` says where it
stops being met, and that the answer inverts when it does.

Source: `todo3.md` §2 and §2a; `transformer_vm/graph/core.py:167-170, 316`;
`transformer_vm/evaluator.py:230`.  Scores are stated at unit query scale,
which decides nothing an argmax sees — `vm-rs/alm-model/src/cache.rs` divides a
query back to it and `vm-rs/alm-hull/src/gap.rs` reports every gap in it.
-/

import Transformer.ALM.Defs
import Mathlib.Analysis.Complex.ExponentialBounds

namespace Transformer
namespace ALM

/-! ### `inv_log_pos`, as the released compiler computes it -/

/-- The perturbation dimension the compiler feeds the head: at position `p`,
`1 / log 2 - 1 / log (p + 2)` (`transformer_vm/evaluator.py:230`). -/
noncomputable def invLogPos (p : ℕ) : ℝ := 1 / Real.log 2 - 1 / Real.log ((p : ℝ) + 2)

/-- The logarithm it divides by is positive at every position, the shift by
two being there to keep it so. -/
lemma log_shift_pos (p : ℕ) : 0 < Real.log ((p : ℝ) + 2) := by
  have hp : (0 : ℝ) ≤ (p : ℝ) := Nat.cast_nonneg p
  exact Real.log_pos (by linarith)

@[simp] lemma invLogPos_zero : invLogPos 0 = 0 := by norm_num [invLogPos]

/-- It increases with the position, which is the direction that makes the later
of two writes score higher. -/
lemma invLogPos_lt_invLogPos {p r : ℕ} (h : p < r) : invLogPos p < invLogPos r := by
  have hp := log_shift_pos p
  have hpr : (p : ℝ) < (r : ℝ) := by exact_mod_cast h
  have hlog : Real.log ((p : ℝ) + 2) < Real.log ((r : ℝ) + 2) :=
    Real.log_lt_log (by linarith) (by linarith)
  have hinv : 1 / Real.log ((r : ℝ) + 2) < 1 / Real.log ((p : ℝ) + 2) := by gcongr
  simp only [invLogPos]; linarith

lemma invLogPos_nonneg (p : ℕ) : 0 ≤ invLogPos p := by
  rcases Nat.eq_zero_or_pos p with h | h
  · simp [h]
  · exact le_of_lt (by simpa using invLogPos_lt_invLogPos (p := 0) h)

/-- And it never reaches `1 / log 2`, which is what bounds the whole
perturbation however long the trace runs. -/
lemma invLogPos_lt (p : ℕ) : invLogPos p < 1 / Real.log 2 := by
  have h := log_shift_pos p
  have hpos : 0 < 1 / Real.log ((p : ℝ) + 2) := by positivity
  simp only [invLogPos]; linarith

/-! ### The score a write carries -/

/-- The score the head computes for a write of logical key `k` made at position
`p`: the paraboloid score at unit query scale, the compiler's perturbation, and
`ν` — whatever the matvec's rounding did to the key, in the same units. -/
noncomputable def writeScore (α q : ℝ) (k : ℤ) (p : ℕ) (ν : ℝ) : ℝ :=
  2 * (k : ℝ) * q - (k : ℝ) ^ 2 + α * invLogPos p + ν

/-- The perturbation and the rounding are the only things between it and the
published embedding `Transformer.ALM.sScore`. -/
lemma writeScore_eq (α : ℝ) (q k : ℤ) (p : ℕ) (ν : ℝ) :
    writeScore α (q : ℝ) k p ν = sScore q k + α * invLogPos p + ν := by
  simp only [writeScore, sScore]

/-! ### The lower side: one step of the perturbation beats the rounding -/

/-- **The later write wins, and no tie is involved.**  Two writes to one logical
key `k`, at positions `p` and `r`, each landing within `η` of the score the
exact key path would give: if one step of the perturbation across that interval
is wider than the rounding can be on both sides together, the later write scores
strictly higher.  `HullMeta::last_seq` is never reached, which is what §2a
measures on the released weights — zero latest-ties, and the machine still
correct.  That `p` precedes `r` is not assumed; the hypothesis says so, since
`invLogPos` increases. -/
theorem latest_wins {α q η ν₁ ν₂ : ℝ} {k : ℤ} {p r : ℕ}
    (h₁ : |ν₁| ≤ η) (h₂ : |ν₂| ≤ η)
    (hstep : 2 * η < α * (invLogPos r - invLogPos p)) :
    writeScore α q k p ν₁ < writeScore α q k r ν₂ := by
  have hb₁ := abs_le.mp h₁
  have hb₂ := abs_le.mp h₂
  have hstep' : 2 * η < α * invLogPos r - α * invLogPos p := by
    rw [mul_sub] at hstep; exact hstep
  simp only [writeScore]; linarith [hb₁.2, hb₂.1]

/-- The hypotheses hold with the rounding at its worst inside a budget that is
itself positive: a quarter of the step the perturbation buys. -/
example : ∃ η ν₁ ν₂ : ℝ, 0 < η ∧ |ν₁| ≤ η ∧ |ν₂| ≤ η ∧
    2 * η < 1 * (invLogPos 1 - invLogPos 0) ∧
    writeScore 1 0 0 0 ν₁ < writeScore 1 0 0 1 ν₂ := by
  have hd : 0 < invLogPos 1 - invLogPos 0 := by
    have := invLogPos_lt_invLogPos (p := 0) (r := 1) (by norm_num); linarith
  refine ⟨(invLogPos 1 - invLogPos 0) / 4, (invLogPos 1 - invLogPos 0) / 4,
    -((invLogPos 1 - invLogPos 0) / 4), by linarith, ?_, ?_, by linarith, ?_⟩
  · rw [abs_of_nonneg (by linarith)]
  · rw [abs_neg, abs_of_nonneg (by linarith)]
  · exact latest_wins (by rw [abs_of_nonneg (by linarith)])
      (by rw [abs_neg, abs_of_nonneg (by linarith)]) (by linarith)

/-! ### The upper side: distinct keys are not reordered -/

/-- Between integer keys at an integer query the scores differ by an integer,
so two distinct ones are a whole unit apart.  That unit is the room the
perturbation has to fit inside. -/
lemma one_le_sScore_sub {q j k : ℤ} (h : sScore q j < sScore q k) :
    sScore q j + 1 ≤ sScore q k := by
  have hz : sScore q k - sScore q j = (((k - j) * (2 * q - k - j) : ℤ) : ℝ) := by
    simp only [sScore]; push_cast; ring
  have hpos : (0 : ℝ) < (((k - j) * (2 * q - k - j) : ℤ) : ℝ) := by rw [← hz]; linarith
  have hone : (1 : ℤ) ≤ (k - j) * (2 * q - k - j) := by exact_mod_cast hpos
  have : (1 : ℝ) ≤ (((k - j) * (2 * q - k - j) : ℤ) : ℝ) := by exact_mod_cast hone
  linarith [hz ▸ this]

/-- **And the perturbation reorders nothing it was not meant to.**  The whole of
it is under `α / log 2` at every position, by `invLogPos_lt`; while that and the
rounding on both sides together stay inside the unit gap, a key that scored
lower still scores lower, wherever and whenever either was written.  So recency
is bought without spending correctness — the upper end of §2a's window. -/
theorem distinct_keys_keep_their_order {α η ν₁ ν₂ : ℝ} {q j k : ℤ} {p r : ℕ}
    (hα : 0 ≤ α) (h₁ : |ν₁| ≤ η) (h₂ : |ν₂| ≤ η)
    (hwin : α / Real.log 2 + 2 * η < 1) (hjk : sScore q j < sScore q k) :
    writeScore α (q : ℝ) j p ν₁ < writeScore α (q : ℝ) k r ν₂ := by
  have hb₁ := abs_le.mp h₁
  have hb₂ := abs_le.mp h₂
  have hgap := one_le_sScore_sub hjk
  have hup : α * invLogPos p ≤ α / Real.log 2 := by
    have := mul_le_mul_of_nonneg_left (invLogPos_lt p).le hα
    rwa [mul_one_div] at this
  have hlo : 0 ≤ α * invLogPos r := mul_nonneg hα (invLogPos_nonneg r)
  rw [writeScore_eq, writeScore_eq]; linarith [hb₁.2, hb₂.1]

/-- Satisfiable, and away from the boundary: a quarter of the unit gap spent on
the perturbation and a quarter on the rounding. -/
example : ∃ α η ν₁ ν₂ : ℝ, 0 ≤ α ∧ |ν₁| ≤ η ∧ |ν₂| ≤ η ∧
    α / Real.log 2 + 2 * η < 1 ∧ sScore 0 1 < sScore 0 0 ∧
    writeScore α ((0 : ℤ) : ℝ) 1 7 ν₁ < writeScore α ((0 : ℤ) : ℝ) 0 0 ν₂ := by
  have hlog : (0 : ℝ) < Real.log 2 := by linarith [Real.log_two_gt_d9]
  have hs : sScore (0 : ℤ) (1 : ℤ) < sScore (0 : ℤ) (0 : ℤ) := by norm_num [sScore]
  have hdiv : Real.log 2 / 4 / Real.log 2 = 1 / 4 := by
    field_simp
  refine ⟨Real.log 2 / 4, 1 / 8, 1 / 8, -(1 / 8), by linarith, by rw [abs_of_nonneg (by norm_num : (0:ℝ) ≤ 1/8)],
    by rw [abs_neg, abs_of_nonneg (by norm_num : (0:ℝ) ≤ 1/8)], by rw [hdiv]; norm_num, hs, ?_⟩
  exact distinct_keys_keep_their_order (by linarith) (by rw [abs_of_nonneg (by norm_num : (0:ℝ) ≤ 1/8)])
    (by rw [abs_neg, abs_of_nonneg (by norm_num : (0:ℝ) ≤ 1/8)]) (by rw [hdiv]; norm_num) hs

/-- **And the shipped constant is inside that side.**  `LATEST_ALPHA = 0.3`
(`transformer_vm/graph/core.py:170`) holds the whole perturbation below `1/2`,
half the gap it is not allowed to cross.  This is the `1/2` §2a names as the
upper end of the admissible window, and the reason six orders of magnitude of
`α` all reproduce the traces. -/
lemma released_alpha_below_half (p : ℕ) : (0.3 : ℝ) * invLogPos p < 1 / 2 := by
  have hlog : (0.6 : ℝ) < Real.log 2 := by linarith [Real.log_two_gt_d9]
  have h : (0.3 : ℝ) * invLogPos p < 0.3 * (1 / Real.log 2) :=
    mul_lt_mul_of_pos_left (invLogPos_lt p) (by norm_num)
  have h2 : (0.3 : ℝ) * (1 / Real.log 2) < 1 / 2 := by
    rw [mul_one_div, div_lt_iff₀ (by linarith)]; linarith
  linarith

end ALM
end Transformer

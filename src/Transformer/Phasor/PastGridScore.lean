module

public import Mathlib.Algebra.Order.Round
public import Transformer.Phasor.TemporalRecurrenceScore

/-! DiffGPT at 5a37c515, current_grokking.adapters.temporal_records and
RecurrenceTask.accumulate. The prefix-only center decode and reciprocal
scale determine the future value of any proposed native token. -/
@[expose] public section

namespace Transformer.Phasor.PastGridScore

/-- The source decoder evaluates the raw future using a decoded center
token and the signed displacement of the proposed symbol from that center. -/
def value (base step : ℝ) (center token : ℤ) : ℝ :=
  base + ((token : ℝ) - (center : ℝ)) * step

/-- Two distinct integer tokens one unit apart have decoded values at least
one grid step apart. -/
theorem gap (b s : ℝ) (c i j : ℤ) (pos : 0 ≤ s) (order : i < j) :
    s ≤ value b s c j - value b s c i := by
  have h : i + 1 ≤ j := Int.add_one_le_iff.mpr order
  have real : (i : ℝ) + 1 ≤ (j : ℝ) := by exact_mod_cast h
  unfold value
  nlinarith [mul_nonneg (show 0 ≤ (j : ℝ) - i - 1 by linarith) pos]

/-- The possibly shifted grid gives at most one token that rounds to the
specified integer. At unit spacing the designated token must be strictly
inside the rounding interval. Larger spacing also permits a tie at an end. -/
theorem unique_token (b s : ℝ) (c trueToken attempt raw : ℤ)
    (wide : 1 ≤ s) (target : |value b s c trueToken - raw| ≤ 1 / 2)
    (strict : 1 < s ∨ |value b s c trueToken - raw| < 1 / 2)
    (decoded : |value b s c attempt - raw| ≤ 1 / 2) : attempt = trueToken := by
  have ha := abs_le.mp decoded
  have ht := abs_le.mp target
  by_contra distinct
  rcases lt_or_gt_of_ne distinct with h | h
  · have hs := gap b s c attempt trueToken (by linarith) h
    rcases strict with hs' | ht'
    · linarith
    · have inside := abs_lt.mp ht'
      linarith
  · have hs := gap b s c trueToken attempt (by linarith) h
    rcases strict with hs' | ht'
    · linarith
    · have inside := abs_lt.mp ht'
      linarith

/-- Agreement with unrestricted token argmax is exact whenever the prefix
decoder recovers the source target and the grid satisfies the separation
bound. Any nearest-integer tie rule with the stated radius is allowed. -/
theorem score_iff (round : ℝ → ℤ) (b s : ℝ) (c target raw attempt : ℤ)
    (nearest : ∀ x, |x - (round x : ℝ)| ≤ 1 / 2)
    (recovered : round (value b s c target) = raw) (wide : 1 ≤ s)
    (separated : 1 < s ∨ |value b s c target - raw| < 1 / 2) :
    round (value b s c attempt) = raw ↔ attempt = target := by
  have hit := nearest (value b s c target)
  rw [recovered] at hit
  constructor
  · intro yes
    have seen := nearest (value b s c attempt)
    rw [yes] at seen
    exact unique_token b s c target attempt raw wide hit separated seen
  · rintro rfl
    exact recovered

example : (3 : ℝ) ≤ value 4 3 10 12 - value 4 3 10 11 := gap _ _ _ _ _ (by norm_num) (by decide)
example (i : ℤ) (h : |value 3 2 10 i - 5| ≤ 1/2) : i=11 :=
  unique_token 3 2 10 11 i 5 (by norm_num) (by norm_num [value]) (Or.inl (by norm_num)) h


/-- On a finer grid different tokens can round to the same raw value,
independently of the rounding tie convention. For these two tokens no
midpoint tie occurs. -/
theorem fine_grid (round : ℝ → ℤ) (nearest : ∀ x, |x - (round x : ℝ)| ≤ 1 / 2) :
    round (value 0 (2/5) 0 0) = 0 ∧ round (value 0 (2/5) 0 1) = 0 := by
  norm_num [value]
  constructor
  · have a := abs_le.mp (nearest 0)
    have lo : (-1 : ℤ) < round 0 := by exact_mod_cast (show (-1 : ℝ) < round 0 by linarith)
    have hi : round 0 < (1 : ℤ) := by exact_mod_cast (show (round 0 : ℝ) < 1 by linarith)
    omega
  · have a := abs_le.mp (nearest (2/5))
    have lo : (-1 : ℤ) < round (2/5) := by
      exact_mod_cast (show (-1 : ℝ) < round (2/5) by linarith)
    have hi : round (2/5) < (1 : ℤ) := by
      exact_mod_cast (show (round (2/5) : ℝ) < 1 by linarith)
    omega

/-- For any monotone quantizer, the native tokens that recover a fixed raw
value form a convex subset of the ordered vocabulary. -/
theorem between (round : ℝ → ℤ) (mono : Monotone round) (b s : ℝ) (pos : 0 ≤ s)
    (c i j k raw : ℤ) (ij : i ≤ j) (jk : j ≤ k)
    (left : round (value b s c i) = raw) (right : round (value b s c k) = raw) :
    round (value b s c j) = raw := by
  have inc : Monotone (value b s c) := by
    intro x y h
    have hc : (x : ℝ) ≤ y := by exact_mod_cast h
    unfold value
    exact add_le_add_right (mul_le_mul_of_nonneg_right (sub_le_sub_right hc _) pos) _
  exact le_antisymm (right ▸ mono (inc jk)) (left ▸ mono (inc ij))


example (i : ℤ) : round (value 4 1 10 i) = 5 ↔ i = 11 := by
  apply score_iff round 4 1 10 11 5 i abs_sub_round
  · norm_num [value, round_eq]
  · norm_num
  · right; norm_num [value]

example : round (value 0 (2/5) 0 0) = 0 ∧ round (value 0 (2/5) 0 1) = 0 :=
  fine_grid round abs_sub_round

example : round (value 2 (1/5) 0 1) = 2 := by
  have monotone : Monotone (round : ℝ → ℤ) := by
    intro x y h
    rw [round_eq, round_eq]
    exact Int.floor_mono (by linarith)
  apply between round monotone 2 (1/5) (by norm_num) 0 0 1 2 2 (by decide) (by decide)
  · norm_num [value, round_eq]
  · norm_num [value, round_eq]

end Transformer.Phasor.PastGridScore

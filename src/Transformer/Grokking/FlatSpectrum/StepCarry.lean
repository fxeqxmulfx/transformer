module

public import Transformer.Grokking.FlatSpectrum.StepLevel

/-!
# The carry of a quantizer of steps: why one step has two tokens

Source: `review/flat_spectrum.md`, *The carry of the codec* (added), items CC1 to CC3. A codec
that writes the steps of a series on a grid of scale `s`, and carries its rounding error to the
next row, writes for the step `d` after steps that sum to `n` a difference of two levels,
`⌊s (n + d) + θ'⌋ − ⌊s n + θ⌋` (`carriedStep`); `θ` and `θ'` are the offsets of the grid at the
two rows.

* The token is `⌊s d + θ' − θ⌋` or the next integer, and which of the two is a bit of the steps
  before it (`carriedStep_eq`, `carriedStep_eq_or`): the fractional parts of `s n + θ` and of
  `s d + θ' − θ` reach `1` together or do not.
* On a grid that holds the values, `s n` and `s d` integers, there is one token, `s d`
  (`carriedStep_of_int`); an integer scale is such a grid for every integer series
  (`carriedStep_intCast`).
* With the scale `511 / 192` and the offsets `0` and `1 / 2` the step `1` is written `3` after
  `n = 0` and `4` after `n = 3` (`carriedStep_two`): a step with two tokens.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

/-- The floor of a sum is the sum of the floors, and one more exactly when the two fractional
parts reach `1` together. -/
theorem floor_add_eq_ite (x y : ℝ) :
    ⌊x + y⌋ = ⌊x⌋ + ⌊y⌋ + if 1 ≤ Int.fract x + Int.fract y then 1 else 0 := by
  have hx := Int.floor_add_fract x
  have hy := Int.floor_add_fract y
  have hx0 := Int.fract_nonneg x
  have hy0 := Int.fract_nonneg y
  have hx1 := Int.fract_lt_one x
  have hy1 := Int.fract_lt_one y
  split_ifs with h
  · rw [Int.floor_eq_iff]
    push_cast
    constructor <;> linarith
  · rw [Int.floor_eq_iff]
    push_cast
    constructor <;> linarith

/-- **CC1**, `review/flat_spectrum.md`: the token, less its zero, of the step `d` after steps
that sum to `n`, on a grid of scale `s` anchored before the steps, with the offset `θ` at the
row before the step and `θ'` at its row: the difference of the two levels. -/
noncomputable def carriedStep (s θ θ' : ℝ) (n d : ℤ) : ℤ := ⌊s * (n + d) + θ'⌋ - ⌊s * n + θ⌋

/-- **CC2**, `review/flat_spectrum.md`: the token of a step is `⌊s d + θ' − θ⌋` and one bit of
the steps before it. -/
theorem carriedStep_eq (s θ θ' : ℝ) (n d : ℤ) :
    carriedStep s θ θ' n d = ⌊s * d + θ' - θ⌋ +
      if 1 ≤ Int.fract (s * n + θ) + Int.fract (s * d + θ' - θ) then 1 else 0 := by
  unfold carriedStep
  rw [show s * ((n : ℝ) + d) + θ' = (s * n + θ) + (s * d + θ' - θ) by ring, floor_add_eq_ite]
  ring

/-- **CC2**, `review/flat_spectrum.md`: so a step has at most two tokens, whatever came before
it. -/
theorem carriedStep_eq_or (s θ θ' : ℝ) (n d : ℤ) :
    carriedStep s θ θ' n d = ⌊s * d + θ' - θ⌋ ∨
      carriedStep s θ θ' n d = ⌊s * d + θ' - θ⌋ + 1 := by
  rw [carriedStep_eq]
  split_ifs
  · exact Or.inr rfl
  · exact Or.inl (add_zero _)

/-- **CC3**, `review/flat_spectrum.md`: on a grid that holds the values, `s n` and `s d`
integers, the step has one token, `s d`, at any two offsets in `[0, 1)`. -/
theorem carriedStep_of_int {s θ θ' : ℝ} {n d j k : ℤ} (hn : s * n = j) (hd : s * d = k)
    (hθ : 0 ≤ θ) (hθ1 : θ < 1) (hθ' : 0 ≤ θ') (hθ'1 : θ' < 1) :
    carriedStep s θ θ' n d = k := by
  unfold carriedStep
  rw [show s * ((n : ℝ) + d) + θ' = θ' + ((j + k : ℤ) : ℝ) by push_cast; rw [← hn, ← hd]; ring,
    show s * (n : ℝ) + θ = θ + ((j : ℤ) : ℝ) by rw [← hn]; ring, Int.floor_add_intCast,
    Int.floor_add_intCast, Int.floor_eq_zero_iff.2 ⟨hθ', hθ'1⟩, Int.floor_eq_zero_iff.2 ⟨hθ, hθ1⟩]
  ring

/-- The scale `1 / 2` holds the even values: after `n = 2` the step `4` is the token `2`. -/
example : carriedStep (1 / 2) 0 (1 / 2) 2 4 = 2 :=
  carriedStep_of_int (j := 1) (by norm_num) (by norm_num) le_rfl one_pos (by norm_num)
    (by norm_num)

/-- **CC3**, `review/flat_spectrum.md`: an integer scale `m` writes the step `d` of an integer
series with the one token `m d`, at any two offsets in `[0, 1)`. -/
theorem carriedStep_intCast (m : ℤ) {θ θ' : ℝ} (n d : ℤ)
    (hθ : 0 ≤ θ) (hθ1 : θ < 1) (hθ' : 0 ≤ θ') (hθ'1 : θ' < 1) :
    carriedStep (m : ℝ) θ θ' n d = m * d :=
  carriedStep_of_int (j := m * n) (by push_cast; ring) (by push_cast; ring) hθ hθ1 hθ' hθ'1

/-- The scale `2` and the offsets `0` and `1 / 2`: the step `d` is the token `2 d`. -/
example (n d : ℤ) : carriedStep 2 0 (1 / 2) n d = 2 * d := by
  have h := carriedStep_intCast 2 (θ := 0) (θ' := 1 / 2) n d le_rfl one_pos (by norm_num)
    (by norm_num)
  simpa using h

/-- **CC3**, `review/flat_spectrum.md`: the scale `511 / 192` of the task and the offsets `0`
and `1 / 2` write the step `1` with `3` after `n = 0` and with `4` after `n = 3`. -/
theorem carriedStep_two :
    carriedStep (511 / 192) 0 (1 / 2) 0 1 = 3 ∧ carriedStep (511 / 192) 0 (1 / 2) 3 1 = 4 := by
  unfold carriedStep
  constructor
  · rw [sub_eq_iff_eq_add, Int.floor_eq_iff, show ⌊(511 / 192 : ℝ) * (0 : ℤ) + 0⌋ = 0 by
      rw [Int.floor_eq_iff]; norm_num]
    norm_num
  · rw [sub_eq_iff_eq_add, Int.floor_eq_iff, show ⌊(511 / 192 : ℝ) * (3 : ℤ) + 0⌋ = 7 by
      rw [Int.floor_eq_iff]; norm_num]
    norm_num

end Transformer.Grokking.FlatSpectrum

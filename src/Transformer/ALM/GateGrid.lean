/-
# The gate is where the stream's range is squared

Three parts of `Alm::forward_timed` carry a theorem and, until this file, two
did not: the gated feed-forward and the two residual additions.  Their fidelity
rested on `alm-vm/tests/reference.rs` reproducing the released traces token for
token, which is evidence that the port is faithful and no evidence at all about
what happens past the end of those traces.

What happens is a wall, and it is the same wall `Transformer.ALM.ScoreWall`
finds on the other path, reached by a different route.  The attention score
`2qk - k^2` leaves the grid when the key passes `2^26.5`; the gate
`max(ff, 0) * ff'` multiplies two coordinates of the residual stream together,
so it leaves the grid when the *stream* passes `2^26.5`, which is
`94 906 265` — `the_ffn_wall` is that this bound is sharp to the unit, as
`todo3.md` §0 found it to be for the score.

The rest is small and worth having written down anyway.  `relu_isBinary` is
that the half of the gate everyone worries about — the branch — is exactly the
half that costs nothing: `max a 0` is `a` or `0` and is representable whenever
`a` is, at any magnitude.  `gate_eq_ite` is that the two spellings of it agree:
`transformer.cpp:315` writes `(ff[i] > 0 ? ff[i] : 0.0)` and `model.rs` writes
`.max(0.0)`.  That is the agreement over the reals; the two cases where the
reals have nothing to say — a `NaN` operand and a negative zero — are IEEE-754
questions, and `model.rs` argues them where they belong, at the code.

And `isBinary_add` is the residual addition: exact while the sum is under the
wall, which is the only thing that was ever being claimed for it.

Source: `vm-rs/alm-model/src/model.rs`, `Alm::forward_timed`;
`transformer_vm/model/transformer.cpp` lines 310-320; `todo3.md` §0 and §4.
-/

import Transformer.ALM.ScoreWall

namespace Transformer
namespace ALM

/-! ### The gate, as the two implementations spell it -/

/-- **The gated feed-forward's elementwise operation.**  The first half of the
projection's output is rectified and multiplies the second. -/
noncomputable def gate (a b : ℝ) : ℝ := max a 0 * b

/-- **The two spellings agree.**  A ternary on `> 0` and a `max` against zero
are the same function of a real number; the port changed the spelling and not
the value. -/
theorem gate_eq_ite (a b : ℝ) : gate a b = (if 0 < a then a else 0) * b := by
  unfold gate
  rcases lt_or_ge 0 a with h | h
  · rw [ite_eq_left h, max_eq_left h.le]
  · rw [ite_eq_right (not_lt.mpr h), max_eq_right h]

/-! ### The rectifier is free and the product is not -/

/-- **The branch costs no precision at any magnitude.**  `max a 0` is `a` or
`0`, and both are representable as soon as `a` is — there is no wall on this
half of the gate, which is worth saying because it is the half that looks like
it might have one. -/
theorem relu_isBinary {p : ℕ} {a : ℝ} (ha : IsBinary p a) : IsBinary p (max a 0) := by
  rcases le_or_gt a 0 with h | h
  · rw [max_eq_right h]
    exact ⟨0, 0, by positivity, by simp⟩
  · rwa [max_eq_left h.le]

/-- **The product squares the range.**  Both operands are coordinates of the
same residual stream, so a stream bounded by `K` gates to `K^2` and no
further. -/
theorem abs_gate_le {a b K : ℝ} (ha : |a| ≤ K) (hb : |b| ≤ K) : |gate a b| ≤ K ^ 2 := by
  have hK : 0 ≤ K := le_trans (abs_nonneg a) ha
  have hrelu : |max a 0| ≤ K := by
    rcases le_or_gt a 0 with h | h
    · rw [max_eq_right h, abs_zero]; exact hK
    · rwa [max_eq_left h.le]
  calc |gate a b| = |max a 0| * |b| := by rw [gate, abs_mul]
    _ ≤ K * K := mul_le_mul hrelu hb (abs_nonneg b) hK
    _ = K ^ 2 := (sq K).symm

/-- **And below the wall the gate is exact.**  Two integer coordinates gate to
an integer, and an integer under `2 ^ p` is its own significand. -/
theorem isBinary_gate {p : ℕ} {za zb : ℤ} (h : |za * zb| < 2 ^ p) :
    IsBinary p (gate (za : ℝ) (zb : ℝ)) := by
  rcases le_or_gt za 0 with hz | hz
  · have hz' : (za : ℝ) ≤ 0 := by exact_mod_cast hz
    have hgate : gate (za : ℝ) (zb : ℝ) = 0 := by
      rw [gate, max_eq_right hz', zero_mul]
    exact ⟨0, 0, by simp, by simpa using hgate⟩
  · have hz' : (0 : ℝ) ≤ (za : ℝ) := by exact_mod_cast hz.le
    have hgate : gate (za : ℝ) (zb : ℝ) = ((za * zb : ℤ) : ℝ) := by
      rw [gate, max_eq_left hz']; push_cast; ring
    rw [hgate]
    exact isBinary_intCast h

/-- **The residual addition is exact under the same wall.**  `x[i] += …` twice
a layer, and there is nothing else to it: the sum of two integers is an
integer, and the wall is where integers stop being representable. -/
theorem isBinary_add {p : ℕ} {z w : ℤ} (h : |z + w| < 2 ^ p) :
    IsBinary p ((z : ℝ) + (w : ℝ)) := by
  have : ((z : ℝ) + (w : ℝ)) = ((z + w : ℤ) : ℝ) := by push_cast; ring
  rw [this]
  exact isBinary_intCast h

/-! ### Where the wall stands -/

/-- **The feed-forward's wall, sharp to the unit.**  A stream coordinate of
`94 906 265` gates to a product still inside `2 ^ 53`; one more and it is
outside.  That is the same number `todo3.md` §0 measures for the score, which
is not a coincidence: both are `⌈√(2^53)⌉`, and both mechanisms are a product
of two quantities the stream carries. -/
theorem the_ffn_wall : (94906265 : ℤ) ^ 2 < 2 ^ 53 ∧ (2 : ℤ) ^ 53 < 94906266 ^ 2 := by
  constructor <;> norm_num

/-- **So the wall is a bound on the stream, not on the gate.**  Whatever the
weights are, a residual stream whose coordinates are integers inside the wall
gates exactly. -/
theorem isBinary_gate_of_wall {za zb : ℤ} (ha : |za| ≤ 94906265) (hb : |zb| ≤ 94906265) :
    IsBinary 53 (gate (za : ℝ) (zb : ℝ)) := by
  refine isBinary_gate ?_
  rw [abs_mul]
  calc |za| * |zb| ≤ 94906265 * 94906265 := by
        exact mul_le_mul ha hb (abs_nonneg _) (by norm_num)
    _ < 2 ^ 53 := by norm_num

/-! ### The hypotheses are satisfiable -/

/-- The wall itself, gated against itself: the largest pair the theorem
accepts. -/
example : |(94906265 : ℤ)| ≤ 94906265 ∧ |(-94906265 : ℤ)| ≤ 94906265 := by decide

/-- A stream coordinate and its rectification at the shipped scale, where the
gate is an ordinary product: `3` and `-5`. -/
example : |(3 : ℤ) * (-5)| < 2 ^ 53 ∧ |(3 : ℤ) + (-5)| < 2 ^ 53 := by decide

/-- The two bounds `abs_gate_le` takes, at a scale where the gate is an
ordinary product of small reals. -/
example : |(3 : ℝ)| ≤ 5 ∧ |(-4 : ℝ)| ≤ 5 := by norm_num [abs_le]

/-- And a representable operand for `relu_isBinary`, past the wall, where the
rectifier still costs nothing. -/
example : IsBinary 53 ((2 : ℝ) ^ 100) := ⟨1, 100, by norm_num, by norm_num⟩

end ALM
end Transformer

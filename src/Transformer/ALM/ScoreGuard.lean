/-
# The query-time guard, and why the scale almost cancels out of it

`vm-rs/alm-hull/src/grid.rs` carries the only thing in either runtime that
reports whether a query was answered or guessed: `off_the_grid(score, margin)`,
which fires when `ulp(score) > |margin|`.  `Transformer.ALM.ScoreWall` proves
what it is guarding against — past `2^53` the unit that separates two integer
scores has nowhere to be stored — but says nothing about the test itself, and
the test is not the absolute one.  The shipped model multiplies its queries by
a hard-attention scale of `√2 · 10^10`, so an absolute "is the score past
`2^53`" test fires at position `799` of a `1034`-token `hello` run whose every
answer is right; `grid.rs` compares against `|qy|` instead, and the docstring's
claim is that the scale then cancels "very nearly".

Very nearly is the part worth proving.  `clean_at_every_scale` and
`dirty_at_every_scale`: outside a band of two binades the guard's verdict does
not depend on the scale at all — a query with a factor of two to spare is clean
at every scale, one that fails by a factor of two fails at every scale.  Inside
the band the scale decides, and that band is the whole of the movement
`grid.rs` measures, `94 906 266` down to `73 966 031`, twenty-two per cent.

`guard_iff_below_wall` is the other end: at unit scale the test is exactly
`|score| < 2^p`, which is §4's `94 906 266` and the hypothesis the rest of the
float development is stated with.

Source: `todo3.md` §4 and §4a; `vm-rs/alm-hull/src/grid.rs`.
-/

import Transformer.ALM.ScoreWall

namespace Transformer
namespace ALM

/-! ### The binade a number sits in -/

/-- `x` lies in the binade of exponent `e`: `2^e ≤ |x| < 2^(e+1)`.  Every
nonzero real lies in exactly one, and it is the exponent `ulp` reads off. -/
def IsExp (e : ℤ) (x : ℝ) : Prop := (2 : ℝ) ^ e ≤ |x| ∧ |x| < (2 : ℝ) ^ (e + 1)

/-- The spacing of the `p`-bit representables inside that binade: what
`grid.rs::ulp` returns for a number of this size. -/
noncomputable def ulpOf (p : ℕ) (e : ℤ) : ℝ := (2 : ℝ) ^ (e - (p : ℤ) + 1)

lemma ulpOf_pos (p : ℕ) (e : ℤ) : 0 < ulpOf p e := zpow_pos (by norm_num) _

/-- The binade is unique, so `ulp` is a function of the number. -/
lemma isExp_unique {e e' : ℤ} {x : ℝ} (h : IsExp e x) (h' : IsExp e' x) : e = e' := by
  have h₁ : (2 : ℝ) ^ e < (2 : ℝ) ^ (e' + 1) := lt_of_le_of_lt h.1 h'.2
  have h₂ : (2 : ℝ) ^ e' < (2 : ℝ) ^ (e + 1) := lt_of_le_of_lt h'.1 h.2
  rw [zpow_lt_zpow_iff_right₀ (by norm_num : (1 : ℝ) < 2)] at h₁ h₂
  omega

/-- **Scaling adds the exponents, to within one.**  `2^e ≤ |x| < 2^(e+1)` and
`2^f ≤ |σ| < 2^(f+1)` put `|σx|` in `[2^(e+f), 2^(e+f+2))`, which is two
binades: the scale shifts the exponent by `f` or by `f + 1`, and which of the
two is where all the scale dependence of the guard lives. -/
theorem isExp_mul {e f : ℤ} {σ x : ℝ} (hσ : IsExp f σ) (hx : IsExp e x) :
    IsExp (e + f) (σ * x) ∨ IsExp (e + f + 1) (σ * x) := by
  have hpos : ∀ g : ℤ, (0 : ℝ) < (2 : ℝ) ^ g := fun g => zpow_pos (by norm_num) g
  have habs : |σ * x| = |σ| * |x| := abs_mul σ x
  have hlo : (2 : ℝ) ^ (e + f) ≤ |σ * x| := by
    rw [habs, zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    calc (2 : ℝ) ^ e * (2 : ℝ) ^ f ≤ |x| * |σ| := by
          exact mul_le_mul hx.1 hσ.1 (hpos f).le (abs_nonneg x)
      _ = |σ| * |x| := mul_comm _ _
  have hhi : |σ * x| < (2 : ℝ) ^ (e + f + 2) := by
    rw [habs, show e + f + 2 = (e + 1) + (f + 1) by ring,
      zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    calc |σ| * |x| = |x| * |σ| := mul_comm _ _
      _ < (2 : ℝ) ^ (e + 1) * (2 : ℝ) ^ (f + 1) := by
          exact mul_lt_mul'' hx.2 hσ.2 (abs_nonneg x) (abs_nonneg σ)
  rcases lt_or_ge |σ * x| ((2 : ℝ) ^ (e + f + 1)) with hmid | hmid
  · exact Or.inl ⟨hlo, hmid⟩
  · exact Or.inr ⟨hmid, by rw [show e + f + 1 + 1 = e + f + 2 by ring]; exact hhi⟩

/-! ### The guard, and the scale it has to survive -/

/-- **A query with a factor of two to spare is clean at every scale.**  If the
unscaled spacing is at most half the unscaled margin of `1`, then whatever
binade the scale `σ` puts the score in, the spacing there is still at most `σ`
— the margin `grid.rs` compares against, `|qy|`.  So the guard does not fire on
a query that the unit-scale argument of §4 answers with room, however the
compiler scaled it. -/
theorem clean_at_every_scale {p : ℕ} {e f E : ℤ} {σ s : ℝ}
    (hσ : IsExp f σ) (hs : IsExp e s) (hσpos : 0 < σ)
    (hclean : 2 * ulpOf p e ≤ 1) (hE : IsExp E (σ * s)) : ulpOf p E ≤ σ := by
  have hfσ : (2 : ℝ) ^ f ≤ σ := by rw [← abs_of_pos hσpos]; exact hσ.1
  have hmul : ulpOf p (e + f) = (2 : ℝ) ^ f * ulpOf p e := by
    unfold ulpOf
    rw [← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    ring_nf
  have hstep : ulpOf p (e + f + 1) = 2 * ((2 : ℝ) ^ f * ulpOf p e) := by
    unfold ulpOf
    rw [← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0), show e + f + 1 - (p : ℤ) + 1
      = 1 + (f + (e - (p : ℤ) + 1)) by ring, zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0),
      zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    norm_num
  have hfpos : (0 : ℝ) < (2 : ℝ) ^ f := zpow_pos (by norm_num) f
  rcases isExp_mul hσ hs with h | h
  · rw [isExp_unique hE h, hmul]
    nlinarith [ulpOf_pos p e]
  · rw [isExp_unique hE h, hstep]
    nlinarith [ulpOf_pos p e]

/-- **And one that fails by a factor of two fails at every scale.**  The same
two binades read the other way: a spacing above twice the unit margin stays
above the scaled margin wherever the scale moves it.  Between the two
statements lies a band of two binades, and that band is the whole of the
movement `grid.rs` reports — `94 906 266` at unit scale against `73 966 031`
at the shipped one. -/
theorem dirty_at_every_scale {p : ℕ} {e f E : ℤ} {σ s : ℝ}
    (hσ : IsExp f σ) (hs : IsExp e s) (hσpos : 0 < σ)
    (hdirty : 2 < ulpOf p e) (hE : IsExp E (σ * s)) : σ < ulpOf p E := by
  have hfσ : σ < 2 * (2 : ℝ) ^ f := by
    have := hσ.2
    rw [abs_of_pos hσpos, show f + 1 = 1 + f by ring,
      zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)] at this
    norm_num at this
    linarith
  have hmul : ulpOf p (e + f) = (2 : ℝ) ^ f * ulpOf p e := by
    unfold ulpOf
    rw [← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    ring_nf
  have hfpos : (0 : ℝ) < (2 : ℝ) ^ f := zpow_pos (by norm_num) f
  have hmono : ulpOf p (e + f) ≤ ulpOf p (e + f + 1) := by
    unfold ulpOf
    rw [zpow_le_zpow_iff_right₀ (by norm_num : (1 : ℝ) < 2)]
    omega
  have hbase : σ < ulpOf p (e + f) := by rw [hmul]; nlinarith
  rcases isExp_mul hσ hs with h | h
  · rw [isExp_unique hE h]; exact hbase
  · rw [isExp_unique hE h]; linarith

/-! ### And at unit scale it is §4's wall -/

/-- **At `qy = 1` the guard is `|score| < 2^p`.**  The margin of a parabolic
integer key is the query's own second coordinate, so at unit scale
`ulp(score) ≤ 1` is exactly the condition `Transformer.ALM.ScoreWall` is stated
with, and `94 906 266` is where it stops holding. -/
theorem guard_iff_below_wall {p : ℕ} {e : ℤ} {s : ℝ} (hs : IsExp e s) :
    ulpOf p e ≤ 1 ↔ |s| < (2 : ℝ) ^ (p : ℤ) := by
  have hiff : ulpOf p e ≤ 1 ↔ e + 1 ≤ (p : ℤ) := by
    unfold ulpOf
    rw [show (1 : ℝ) = (2 : ℝ) ^ (0 : ℤ) by norm_num,
      zpow_le_zpow_iff_right₀ (by norm_num : (1 : ℝ) < 2)]
    omega
  rw [hiff]
  constructor
  · intro h
    exact lt_of_lt_of_le hs.2 (by
      rw [zpow_le_zpow_iff_right₀ (by norm_num : (1 : ℝ) < 2)]; omega)
  · intro h
    have := lt_of_le_of_lt hs.1 h
    rw [zpow_lt_zpow_iff_right₀ (by norm_num : (1 : ℝ) < 2)] at this
    omega

/-! ### The hypotheses are satisfiable -/

/-- A score with room to spare and a scale to move it: `s = 2^51` at `p = 53`
leaves a spacing of `2^-1`, and the scale `2^34` is a binade like any other. -/
example : IsExp 51 ((2 : ℝ) ^ (51 : ℤ)) ∧ IsExp 34 ((2 : ℝ) ^ (34 : ℤ)) ∧
    (0 : ℝ) < (2 : ℝ) ^ (34 : ℤ) ∧ 2 * ulpOf 53 51 ≤ 1 := by
  refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩, by positivity, ?_⟩
  · rw [abs_of_pos (by positivity)]
  · rw [abs_of_pos (by positivity), zpow_lt_zpow_iff_right₀ (by norm_num : (1 : ℝ) < 2)]
    omega
  · rw [abs_of_pos (by positivity)]
  · rw [abs_of_pos (by positivity), zpow_lt_zpow_iff_right₀ (by norm_num : (1 : ℝ) < 2)]
    omega
  · unfold ulpOf
    norm_num

/-- And one past the wall, which no scale rescues: `s = 2^55` at `p = 53`
leaves a spacing of `8`. -/
example : IsExp 55 ((2 : ℝ) ^ (55 : ℤ)) ∧ 2 < ulpOf 53 55 := by
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · rw [abs_of_pos (by positivity)]
  · rw [abs_of_pos (by positivity), zpow_lt_zpow_iff_right₀ (by norm_num : (1 : ℝ) < 2)]
    omega
  · unfold ulpOf
    norm_num

end ALM
end Transformer

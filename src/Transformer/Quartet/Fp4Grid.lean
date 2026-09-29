/-
# `RTN_FP4` at the top of the E2M1 grid

Support for arXiv:2601.22813v2, §3.1 and §3.3.  Rounding to nearest onto the E2M1 grid
`±{0, 0.5, 1, 1.5, 2, 3, 4, 6}` is a finite computation between two consecutive grid points,
and the two gaps at the top, `(4, 6)` and `(-6, -4)`, are the ones the `MS-EDEN` counterexample
of `Transformer.Quartet.Section3_EdenBias` lives in: an argument above `5` rounds to `±6`, one
between `4` and `5` to `±4`.  `floorOn_ceilOn_of_bracket` is the general statement: between two
consecutive grid points the floor is the lower one and the ceiling the upper.
-/

import Transformer.Quartet.Section3_Grids

namespace Transformer
namespace Quartet

/-- `RTN` fixes a grid point. -/
theorem rtn_self {G : Set ℝ} {x : ℝ} (hx : x ∈ G) : rtn G x = x := by
  rw [rtn, floorOn_eq hx le_rfl fun _ _ h => h, ceilOn_eq hx le_rfl fun _ _ h => h]
  simp

/-- **Between two consecutive grid points** the floor is the lower one and the ceiling the upper
one. -/
theorem floorOn_ceilOn_of_bracket {G : Set ℝ} {a b x : ℝ} (ha : a ∈ G) (hb : b ∈ G)
    (hgap : ∀ y ∈ G, y ≤ a ∨ b ≤ y) (hax : a < x) (hxb : x < b) :
    floorOn G x = a ∧ ceilOn G x = b := by
  refine ⟨floorOn_eq ha hax.le fun y hy hyx => ?_, ceilOn_eq hb hxb.le fun y hy hxy => ?_⟩
  · rcases hgap y hy with h | h
    · exact h
    · exact absurd (h.trans hyx) (not_le.mpr hxb)
  · rcases hgap y hy with h | h
    · exact absurd (hax.trans_le hxy) (not_lt.mpr h)
    · exact h

/-- No E2M1 point lies strictly between `4` and `6`. -/
theorem fp4_gap_four_six : ∀ y ∈ fp4, y ≤ 4 ∨ 6 ≤ y := by
  intro y hy
  simp only [fp4, Set.mem_insert_iff, Set.mem_singleton_iff] at hy
  rcases hy with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;>
    first | (left; norm_num; done) | (right; norm_num)

/-- No E2M1 point lies strictly between `-6` and `-4`. -/
theorem fp4_gap_neg_six_four : ∀ y ∈ fp4, y ≤ -6 ∨ -4 ≤ y := by
  intro y hy
  simp only [fp4, Set.mem_insert_iff, Set.mem_singleton_iff] at hy
  rcases hy with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;>
    first | (left; norm_num; done) | (right; norm_num)

/-- Above `5`, `RTN_FP4` returns the top of the grid, `6`: rounded to nearest in `(4, 6]`, and
saturated above `6`. -/
theorem rtn_fp4_six {x : ℝ} (hx : 5 < x) : rtn fp4 x = 6 := by
  have h4 : (4 : ℝ) ∈ fp4 := by norm_num [fp4]
  have h6 : (6 : ℝ) ∈ fp4 := by norm_num [fp4]
  rcases lt_trichotomy x 6 with h | h | h
  · obtain ⟨hf, hc⟩ := floorOn_ceilOn_of_bracket h4 h6 fp4_gap_four_six (by linarith) h
    rw [rtn, hf, hc, ite_eq_right (by linarith)]
  · subst h; exact rtn_self h6
  · exact rtn_of_forall_le h6 (fun _ hy => (mem_Icc_of_mem_fp4 hy).2) h

/-- Strictly between `4` and `5`, `RTN_FP4` returns `4`. -/
theorem rtn_fp4_four {x : ℝ} (h₁ : 4 < x) (h₂ : x < 5) : rtn fp4 x = 4 := by
  have h4 : (4 : ℝ) ∈ fp4 := by norm_num [fp4]
  have h6 : (6 : ℝ) ∈ fp4 := by norm_num [fp4]
  obtain ⟨hf, hc⟩ := floorOn_ceilOn_of_bracket h4 h6 fp4_gap_four_six h₁ (by linarith)
  rw [rtn, hf, hc, ite_eq_left (by linarith)]

/-- Below `-5`, `RTN_FP4` returns the bottom of the grid, `-6`. -/
theorem rtn_fp4_neg_six {x : ℝ} (hx : 5 < x) : rtn fp4 (-x) = -6 := by
  have h4 : (-4 : ℝ) ∈ fp4 := by norm_num [fp4]
  have h6 : (-6 : ℝ) ∈ fp4 := by norm_num [fp4]
  rcases lt_trichotomy x 6 with h | h | h
  · obtain ⟨hf, hc⟩ := floorOn_ceilOn_of_bracket (x := -x) h6 h4 fp4_gap_neg_six_four
      (by linarith) (by linarith)
    rw [rtn, hf, hc, ite_eq_left (by linarith)]
  · subst h; exact rtn_self h6
  · exact rtn_of_le_forall h6 (fun _ hy => (mem_Icc_of_mem_fp4 hy).1) (by linarith)

/-- Strictly between `-5` and `-4`, `RTN_FP4` returns `-4`. -/
theorem rtn_fp4_neg_four {x : ℝ} (h₁ : 4 < x) (h₂ : x < 5) : rtn fp4 (-x) = -4 := by
  have h4 : (-4 : ℝ) ∈ fp4 := by norm_num [fp4]
  have h6 : (-6 : ℝ) ∈ fp4 := by norm_num [fp4]
  obtain ⟨hf, hc⟩ := floorOn_ceilOn_of_bracket (x := -x) h6 h4 fp4_gap_neg_six_four
    (by linarith) (by linarith)
  rw [rtn, hf, hc, ite_eq_right (by linarith)]

/-- The hypotheses of the four roundings are satisfiable: `5.5 ↦ 6`, `4.5 ↦ 4` and their
negatives, and `256` is a fixed point of `RTN_FP8`. -/
example : rtn fp4 (11 / 2) = 6 ∧ rtn fp4 (9 / 2) = 4 ∧ rtn fp4 (-(11 / 2)) = -6 ∧
    rtn fp4 (-(9 / 2)) = -4 ∧ floorOn fp4 (9 / 2) = 4 ∧ ceilOn fp4 (9 / 2) = 6 ∧
    rtn fp8 256 = 256 :=
  ⟨rtn_fp4_six (by norm_num), rtn_fp4_four (by norm_num) (by norm_num),
    rtn_fp4_neg_six (by norm_num), rtn_fp4_neg_four (by norm_num) (by norm_num),
    (floorOn_ceilOn_of_bracket (by norm_num [fp4]) (by norm_num [fp4]) fp4_gap_four_six
      (by norm_num) (by norm_num)).1,
    (floorOn_ceilOn_of_bracket (by norm_num [fp4]) (by norm_num [fp4]) fp4_gap_four_six
      (by norm_num) (by norm_num)).2,
    rtn_self ⟨by norm_num, 8, 5, by norm_num⟩⟩

end Quartet
end Transformer

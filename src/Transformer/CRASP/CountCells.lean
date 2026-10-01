/-
# Exact quotient cells for positional attention

arXiv:2506.16055v3, Appendix B.2 and F, `thm:rtfr_to_TLClmod`.
The rounded division depends only on three integer sums, so its predicate
is independent of how a position encoding produces those sums.
-/

import Transformer.CRASP.AttentionFormulas

namespace Transformer.CRASP.CountCells

universe u
variable {σ : Type u} {p s j : ℕ}

/-- The finite cell test for one attention coordinate (Appendix B.2). -/
def formula (D N V : LinearCount σ) (y : Fx p s) : Form σ :=
  Form.or
    (.and D.isZero (V.roundedQuotient RTfr.prefixLengthCount y))
    (.and (D.scale (-1)).ltZero ((N.scale (2 ^ s)).roundedQuotient D y))

/-- A cell predicate costs one counting level beyond its input predicates (B.2/F). -/
theorem formula_mem {D N V : LinearCount σ} (hD : D.Good j)
    (hN : N.Good j) (hV : V.Good j) (y : Fx p s) :
    formula D N V y ∈ TLCl σ (j + 1) := by
  apply Form.or_mem
  · exact Form.and_mem (LinearCount.isZero_mem hD)
      (LinearCount.roundedQuotient_mem hV (RTfr.prefixLengthCount_good j) y)
  · exact Form.and_mem (LinearCount.ltZero_mem (LinearCount.good_scale hD (-1)))
      (LinearCount.roundedQuotient_mem (LinearCount.good_scale hN (2 ^ s)) hD y)

variable [DecidableEq σ]

/-- Exact semantics of the cell predicate, including a zero denominator (B.2/F). -/
theorem sat_formula (D N V : LinearCount σ) (y : Fx p s)
    (w : List σ) (i : ℕ) (hDnonneg : 0 ≤ D.val w i) :
    (formula D N V y).sat w i = true ↔
      (if D.val w i = 0 then
        Fx.round p s (((V.val w i : ℝ) / (i + 1 : ℕ)) / 2 ^ s)
       else Fx.round p s ((N.val w i : ℝ) / (D.val w i : ℝ))) = y := by
  have hL : 0 < (RTfr.prefixLengthCount : LinearCount σ).val w i := by simp
  have hfall := LinearCount.sat_roundedQuotient V RTfr.prefixLengthCount y w i hL
  have hs : (2 : ℝ) ^ s ≠ 0 := by positivity
  change (Form.or (.and D.isZero (V.roundedQuotient RTfr.prefixLengthCount y))
    (.and (D.scale (-1)).ltZero ((N.scale (2 ^ s)).roundedQuotient D y))).sat w i = true ↔ _
  rw [Form.sat_or]
  simp only [Form.sat, Bool.or_eq_true, Bool.and_eq_true, LinearCount.sat_isZero,
    LinearCount.sat_ltZero, LinearCount.val_scale, neg_one_mul, decide_eq_true_eq]
  by_cases hz : D.val w i = 0
  · simp only [hz, neg_zero, lt_self_iff_false, false_and, or_false, true_and]
    rw [hfall]
    simp only [ite_true]
    rw [RTfr.val_prefixLengthCount]
    push_cast
    rfl
  · have hpos : 0 < D.val w i := by omega
    have hnorm := LinearCount.sat_roundedQuotient (N.scale (2 ^ s)) D y w i hpos
    simp only [hz, false_and, false_or, neg_lt_zero, hpos, true_and]
    rw [hnorm]
    simp only [ite_false]
    have he : (((N.scale (2 ^ s)).val w i : ℝ) / (D.val w i : ℝ)) / 2 ^ s =
        (N.val w i : ℝ) / (D.val w i : ℝ) := by
      rw [LinearCount.val_scale]
      push_cast
      field_simp
    rw [he]

/-- Nonnegative sums and counted-formula bounds occur for constants (B.2/F). -/
example : (LinearCount.mk 1 [] : LinearCount Bool).Good 0 ∧
    0 ≤ (LinearCount.mk 1 [] : LinearCount Bool).val [] 0 := by
  simp [LinearCount.Good, LinearCount.val]

end Transformer.CRASP.CountCells

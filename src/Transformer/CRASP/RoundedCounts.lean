/-
# A rounded quotient as comparisons of signed counts

arXiv:2506.16055v3, Appendix B.2, the division step in
`thm:rtfr_to_TLCl`. Enumerating the finite rounded values replaces long
division by their exact endpoint-aware intervals. No extra counting is
needed after the weighted sums have been formed.
-/

import Transformer.CRASP.LinearCounts
import Transformer.CRASP.FixedFinite
import Transformer.CRASP.FormulaBounds

namespace Transformer.CRASP.LinearCount

universe u
variable {σ : Type u} {p s k : ℕ}

/-- `N / D`, rounded in mantissa units, tested against a fixed value (B.2). -/
def roundedQuotient (N D : LinearCount σ) (y : Fx p s) : Form σ :=
  .and
    (if -2 ^ (p - 1) < y.m then .neg ((N.sub (D.scale y.m)).ltZero) else Form.truth true)
    (if y.m < 2 ^ (p - 1) - 1 then (N.sub (D.scale (y.m + 1))).ltZero else Form.truth true)

/-- The quotient test has only the depth of its input counts (Appendix B.2). -/
theorem roundedQuotient_mem {N D : LinearCount σ} (hN : N.Good k) (hD : D.Good k)
    (y : Fx p s) : N.roundedQuotient D y ∈ TLCl σ (k + 1) := by
  apply Form.and_mem
  · split_ifs
    · exact Form.neg_mem (ltZero_mem (good_sub hN (good_scale hD y.m)))
    · exact Form.truth_mem _ _
  · split_ifs
    · exact ltZero_mem (good_sub hN (good_scale hD (y.m + 1)))
    · exact Form.truth_mem _ _

variable [DecidableEq σ]

/-- Equality to zero, using the two possible signs (Appendix B.2). -/
def isZero (A : LinearCount σ) : Form σ :=
  .and (.neg A.ltZero) (.neg (A.scale (-1)).ltZero)

omit [DecidableEq σ] in
/-- The zero test uses the same weighted counts (Appendix B.2). -/
theorem isZero_mem {A : LinearCount σ} (hA : A.Good k) :
    A.isZero ∈ TLCl σ (k + 1) :=
  Form.and_mem (Form.neg_mem (ltZero_mem hA))
    (Form.neg_mem (ltZero_mem (good_scale hA (-1))))

/-- The zero test has its integer meaning (Appendix B.2). -/
theorem sat_isZero (A : LinearCount σ) (w : List σ) (i : ℕ) :
    A.isZero.sat w i = decide (A.val w i = 0) := by
  rw [Bool.eq_iff_iff]
  simp [isZero, Form.sat, sat_ltZero]
  omega

/-- The test agrees with saturated rounding of a positive-denominator ratio.
Source: arXiv:2506.16055v3, Appendix B.2, division in `thm:rtfr_to_TLCl`. -/
theorem sat_roundedQuotient (N D : LinearCount σ) (y : Fx p s)
    (w : List σ) (i : ℕ) (hD : 0 < D.val w i) :
    (N.roundedQuotient D y).sat w i = true ↔
      Fx.round p s (((N.val w i : ℝ) / (D.val w i : ℝ)) / 2 ^ s) = y := by
  have hs : (2 : ℝ) ^ s ≠ 0 := by positivity
  have hd : (0 : ℝ) < D.val w i := by exact_mod_cast hD
  have hlow : ¬ N.val w i - y.m * D.val w i < 0 ↔
      (y.m : ℝ) ≤ (N.val w i : ℝ) / (D.val w i : ℝ) := by
    rw [le_div_iff₀ hd]
    constructor
    · intro h
      have hz : y.m * D.val w i ≤ N.val w i := by omega
      exact_mod_cast hz
    · intro h
      have hz : y.m * D.val w i ≤ N.val w i := by exact_mod_cast h
      omega
  have hupp : N.val w i - (y.m + 1) * D.val w i < 0 ↔
      (N.val w i : ℝ) / (D.val w i : ℝ) < (y.m + 1 : ℤ) := by
    rw [div_lt_iff₀ hd]
    constructor
    · intro h
      have hz : N.val w i < (y.m + 1) * D.val w i := by omega
      exact_mod_cast hz
    · intro h
      have hz : N.val w i < (y.m + 1) * D.val w i := by exact_mod_cast h
      omega
  rw [Fx.round_eq_iff, div_mul_cancel₀ _ hs]
  simp only [roundedQuotient, Form.sat, Bool.and_eq_true]
  split_ifs <;> simp only [Form.sat, Form.sat_truth, sat_ltZero, val_sub,
    val_scale, decide_eq_true_eq]
  all_goals simp_all

/-- Both bound and positive-denominator hypotheses are satisfiable (B.2). -/
example : (LinearCount.mk 1 [] : LinearCount Bool).Good 0 ∧
    0 < (LinearCount.mk 1 [] : LinearCount Bool).val [true] 1 := by
  simp [Good, val]

end Transformer.CRASP.LinearCount

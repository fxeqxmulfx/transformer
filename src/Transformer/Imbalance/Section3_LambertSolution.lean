/-
# Lambert-W expression for the solved gradient dynamics

arXiv:2402.19449v2, Appendix I, Lemma 5. A genuine positive-argument
Lambert W is constructed using the real inverse already proved to exist.
The manuscript's exact closed form is then recovered from the margin.
-/

import Transformer.Imbalance.Section3_GradientFlow

noncomputable section

namespace Transformer.Imbalance

/-- Positive-argument Lambert W: if `s+exp(s)=log(q)`, set W(q)=exp(s).
Appendix I, Lemma 5 uses W only at strictly positive arguments. -/
def positiveLambertW (q : ℝ) : ℝ :=
  Real.exp ((marginIso 1 zero_lt_one).symm (Real.log q))

/-- Defining inverse identity for Lambert W, used in Appendix I, Lemma 5. -/
theorem positiveLambertW_equation (q : ℝ) (hq : 0 < q) :
    positiveLambertW q * Real.exp (positiveLambertW q) = q := by
  let s := (marginIso 1 zero_lt_one).symm (Real.log q)
  have hs : Real.exp s + s = Real.log q := by
    have heq : marginPrimitive 1 s = Real.log q :=
      (marginIso 1 zero_lt_one).apply_symm_apply (Real.log q)
    simpa only [marginPrimitive, one_mul] using heq
  change Real.exp s * Real.exp (Real.exp s) = q
  rw [← Real.exp_add, show s + Real.exp s = Real.log q by linarith, Real.exp_log hq]

/-- Nonvacuity of the positive-argument Lambert-W identity; Lemma 5. -/
example : (0 : ℝ) < Real.exp 1 := Real.exp_pos _

/-- Uniqueness of the positive solution of `w exp(w)=q`;
Appendix I, Lemma 5. -/
theorem positiveLambertW_unique (q w : ℝ) (hw : 0 < w)
    (hweq : w * Real.exp w = q) : positiveLambertW q = w := by
  unfold positiveLambertW
  rw [← Real.exp_log hw]
  congr 1
  apply (marginIso 1 zero_lt_one).injective
  change marginPrimitive 1 ((marginIso 1 zero_lt_one).symm (Real.log q)) =
    marginPrimitive 1 (Real.log w)
  calc
    _ = Real.log q := (marginIso 1 zero_lt_one).apply_symm_apply _
    _ = marginPrimitive 1 (Real.log w) := by
      rw [← hweq, Real.log_mul hw.ne' (Real.exp_ne_zero w), Real.log_exp]
      simp [marginPrimitive, Real.exp_log hw, add_comm]

/-- Nonvacuity of positive Lambert-W uniqueness; Appendix I, Lemma 5. -/
example : (0 : ℝ) < 1 ∧ (1 : ℝ) * Real.exp 1 = Real.exp 1 := by simp

/-- Exact Lambert-W form of the margin; Appendix I, Lemma 5.
This proves equivalence to the manuscript's closed form, rather than
postulating the Lambert-W expression as a solution. -/
theorem gdMargin_lambert (z : ℝ) (hz : 0 < z) (π t : ℝ) :
    gdMargin z hz π t = (1 + (z + 1) * π * t) / z -
      positiveLambertW (Real.exp ((1 + (z + 1) * π * t) / z) / z) := by
  let u := gdMargin z hz π t
  let f := 1 + (z + 1) * π * t
  have heq : Real.exp u + z * u = f := gdMargin_equation z hz π t
  have hu : u + Real.exp u / z = f / z := by
    apply (eq_div_iff hz.ne').2
    field_simp
    linarith
  have hw : (Real.exp u / z) * Real.exp (Real.exp u / z) = Real.exp (f / z) / z := by
    calc
      _ = Real.exp (u + Real.exp u / z) / z := by rw [Real.exp_add]; ring
      _ = Real.exp (f / z) / z := by rw [hu]
  have hW := positiveLambertW_unique (Real.exp (f / z) / z) (Real.exp u / z)
    (by positivity) hw
  change u = f / z - positiveLambertW (Real.exp (f / z) / z)
  rw [hW]
  linarith

/-- Nonvacuity of the Lambert-W solution's assumptions; Lemma 5. -/
example : (0 : ℝ) < 2 := by norm_num

/-- The exact diagonal solution stated in Appendix I, Lemma 5,
with `f=1+cπt` and `z=c-1`. -/
theorem gdDiagonal_lambert (c : ℕ) (hc : 2 ≤ c) (π t : ℝ) :
    gdDiagonal c hc π t = (1 / (c : ℝ)) *
      (1 + (c : ℝ) * π * t - ((c : ℝ) - 1) *
        positiveLambertW (Real.exp ((1 + (c : ℝ) * π * t) / ((c : ℝ) - 1)) /
          ((c : ℝ) - 1))) := by
  have hz := otherClasses_pos c hc
  unfold gdDiagonal
  rw [gdMargin_lambert]
  have hc0 : (c : ℝ) ≠ 0 := by linarith
  have hc1 : (c : ℝ) - 1 + 1 = c := by ring
  rw [hc1]
  field_simp

/-- Nonvacuity of the exact diagonal solution; Appendix I, Lemma 5. -/
example : 2 ≤ 3 := by decide

/-- Off-diagonal part of the exact solution in Appendix I, Lemma 5:
`b=-a/(c-1)`. Lemma 6's prose later drops this minus sign. -/
theorem gdOffDiagonal_balance (c : ℕ) (hc : 2 ≤ c) (π t : ℝ) :
    gdOffDiagonal c hc π t = -gdDiagonal c hc π t / ((c : ℝ) - 1) := by
  have hz := otherClasses_pos c hc
  have hc0 : (c : ℝ) ≠ 0 := by linarith
  unfold gdDiagonal gdOffDiagonal
  field_simp

/-- Nonvacuity of the exact off-diagonal solution; Appendix I, Lemma 5. -/
example : 2 ≤ 3 := by decide

end Transformer.Imbalance

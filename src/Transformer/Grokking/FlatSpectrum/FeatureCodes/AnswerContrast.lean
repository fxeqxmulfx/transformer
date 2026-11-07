module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.AnswerMass

/-!
# Bound the combined answer contrast using its coefficient range

Source: `review/flat_spectrum.md`, *Combined answer contrasts*, AC3–AC4.
Unit-mass representations have zero-total-mass error. Therefore a common
constant can be subtracted from a contrast's coefficients before bounding it.
The midpoint of their range gives half the range times the pooled L1 error.
This bound permits arbitrary balanced latent errors and common score offsets.
It is sharp even for two answers. Fitting the representation and controlling
its held-out error remain separate obligations.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open Finset

variable {A V : Type*} [Fintype A]

/-- Correct the total signed mass along a fixed reference vector, AC3.
When the reference sums to one, this is an affine projection onto unit mass. -/
def unitMass (p r : A → ℝ) (a : A) : ℝ := p a + (1 - ∑ b, p b) * r a

/-- The correction supplies exact unit mass without positivity assumptions,
AC3. The reference can be the pushforward of a uniform latent distribution. -/
theorem sum_unitMass (p r : A → ℝ) (hr : ∑ a, r a = 1) :
    ∑ a, unitMass p r a = 1 := by
  simp only [unitMass, sum_add_distrib, ← mul_sum, hr]
  ring

/-- Mass correction commutes with pooling when its reference is pooled too,
AC3. Uniform correction on both spaces would generally break this identity. -/
theorem answerMass_unitMass {S : Type*} [Fintype S] [DecidableEq A]
    (f : S → A) (p r : S → ℝ) (a : A) :
    answerMass f (unitMass p r) a = unitMass (answerMass f p) (answerMass f r) a := by
  simp only [unitMass, answerMass, sum_add_distrib, ← mul_sum]
  rw [sum_fiberwise univ f p]

/-- **Zero total error removes a common coefficient**, AC3. -/
theorem centered_contrast_error_le (d e : A → ℝ) (he : ∑ a, e a = 0)
    {center radius : ℝ} (hd : ∀ a, |d a - center| ≤ radius) :
    |∑ a, d a * e a| ≤ radius * ∑ a, |e a| := by
  have hc : ∑ a, d a * e a = ∑ a, (d a - center) * e a := by
    simp only [sub_mul, sum_sub_distrib, ← mul_sum, he, mul_zero, sub_zero]
  rw [hc, mul_sum]
  refine (abs_sum_le_sum_abs _ _).trans (sum_le_sum fun a _ => ?_)
  rw [abs_mul]
  exact mul_le_mul_of_nonneg_right (hd a) (abs_nonneg _)

/-- The midpoint of an interval containing the contrast coefficients gives
half its width as the error multiplier, AC3. -/
theorem range_contrast_error_le (d e : A → ℝ) (he : ∑ a, e a = 0)
    {lo hi : ℝ} (hd : ∀ a, lo ≤ d a ∧ d a ≤ hi) :
    |∑ a, d a * e a| ≤ ((hi - lo) / 2) * ∑ a, |e a| := by
  apply centered_contrast_error_le d e he (center := (lo + hi) / 2)
  intro a
  obtain ⟨hl, hu⟩ := hd a
  apply abs_le.mpr
  constructor <;> linarith

/-- **The actual decoded contrast error is bounded after answer pooling**,
AC4. The output class type `V` can include labels absent from the answer set.
Different competitors retain their own coefficient range and ideal gap. -/
theorem answer_contrast_error_le [DecidableEq A] (C : V → A → ℝ) (p : A → ℝ)
    (a : A) (t v : V) (hp : ∑ b, p b = 1) {lo hi : ℝ}
    (hC : ∀ b, lo ≤ C t b - C v b ∧ C t b - C v b ≤ hi) :
    |(answerScore C p t - answerScore C p v) - (C t a - C v a)| ≤
      ((hi - lo) / 2) * ∑ b, |p b - if b = a then 1 else 0| := by
  let e : A → ℝ := fun b => p b - if b = a then 1 else 0
  have he : ∑ b, e b = 0 := by simp [e, sum_sub_distrib, hp]
  have h := range_contrast_error_le (fun b => C t b - C v b) e he hC
  have hsplit : (answerScore C p t - answerScore C p v) - (C t a - C v a) =
      ∑ b, (C t b - C v b) * e b := by
    simp [answerScore, e, mul_sub, sub_mul, sum_sub_distrib]
  rw [hsplit]
  exact h

example := sum_unitMass (A := Bool) (fun a => if a then 2 else -3)
  (fun _ => 1 / 2) (by norm_num)
example := answerMass_unitMass (S := Bool × Bool) Prod.fst
  (fun s => if s.2 then 2 else -3) (fun _ => 1 / 4) true
example := centered_contrast_error_le (A := Bool) (fun a => if a then 11 else 9)
  (fun a => if a then 1 / 4 else -1 / 4) (by norm_num)
  (center := 10) (radius := 1) (by intro a; cases a <;> norm_num)
example := range_contrast_error_le (A := Bool) (fun a => if a then 11 else 9)
  (fun a => if a then 1 / 4 else -1 / 4) (by norm_num)
  (lo := 9) (hi := 11) (by intro a; cases a <;> norm_num)

example := answer_contrast_error_le (A := Bool) (V := Bool)
  (fun v a => (if v = a then 1 else 0) + 10000)
  (fun a => if a then 3 / 4 else 1 / 4) true true false (by norm_num)
  (lo := -1) (hi := 1) (by intro a; cases a <;> norm_num)

/-- The half-range coefficient is attained by a two-answer error, AC3.
An arbitrarily large common coefficient `b` has no effect on the error. -/
theorem range_bound_attained (b M : ℝ) :
    |∑ a : Bool, ((if a then 1 else -1) + b) * (if a then M else -M)| =
      ∑ a : Bool, |if a then M else -M| := by
  simp only [Fintype.sum_bool, Bool.false_eq_true, ite_false, ite_true, abs_neg]
  ring_nf
  rw [abs_mul]
  norm_num

example := range_bound_attained 10000 (1 / 4)

end Transformer.Grokking.FlatSpectrum.FeatureCodes

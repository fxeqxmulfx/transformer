import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic

/-!
# Likelihood-based compression estimator

arXiv:2505.24832v3, Section 2.3. The estimator uses a maximum of target
and reference likelihoods. This yields a nonnegative improvement over
the reference. A pointwise maximum of two distributions is not generally
normalized, so it is not by itself a realizable arithmetic-coding model.
An equal mixture is normalized and loses at most one ideal bit relative
to the maximum. Integer codeword and universal-interpreter overheads are
not identified with ideal negative log likelihood.
-/

namespace Transformer.Memorization

/-- Section 2.3: ideal code-length improvement over a positive-likelihood
reference, in bits. Statements interpreting it assume `reference > 0`. -/
noncomputable def likelihoodUnintended (target reference : ℝ) : ℝ :=
  (Real.log (max target reference) - Real.log reference) / Real.log 2

/-- Section 2.3: using the better of the two likelihoods makes the
estimated unintended memorization nonnegative. -/
theorem likelihoodUnintended_nonneg (target reference : ℝ) (href : 0 < reference) :
    0 ≤ likelihoodUnintended target reference := by
  apply div_nonneg _ (Real.log_pos (by norm_num)).le
  exact sub_nonneg.mpr (Real.log_le_log href (le_max_right _ _))

/-- Section 2.3: an ordinary reference likelihood of 1/2 is positive. -/
example : (0 : ℝ) < 1 / 2 := by norm_num

/-- Section 2.3: no improvement is assigned to a target likelihood no
better than the reference, including a zero-likelihood target. -/
theorem likelihoodUnintended_eq_zero (target reference : ℝ) (h : target ≤ reference) :
    likelihoodUnintended target reference = 0 := by
  simp [likelihoodUnintended, max_eq_right h]

/-- Section 2.3: zero target likelihood and reference likelihood 1/2
satisfy this premise. -/
example : (0 : ℝ) ≤ 1 / 2 := by norm_num

/-- Section 2.3: for a better target, the estimate is the log likelihood
ratio. Both positivity hypotheses needed for the logarithm are explicit. -/
theorem likelihoodUnintended_eq_ratio (target reference : ℝ)
    (href : 0 < reference) (hbetter : reference < target) :
    likelihoodUnintended target reference = Real.log (target / reference) / Real.log 2 := by
  rw [likelihoodUnintended, max_eq_left hbetter.le,
    Real.log_div (lt_trans href hbetter).ne' href.ne']

/-- Section 2.3: target 3/4 and reference 1/2 witness strict improvement. -/
example : (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) < 3 / 4 := by norm_num

/-- Section 2.3, coding correction: the equal-mixture ideal NLL is at
most one bit larger than the best-model ideal NLL. A realizable integer
arithmetic code also requires its usual rounding/framing overhead. -/
theorem mixture_code_overhead (p q : ℝ) (hp : 0 < p) (hq : 0 < q) :
    -Real.log ((p + q) / 2) / Real.log 2 ≤
      -Real.log (max p q) / Real.log 2 + 1 := by
  have hmax : 0 < max p q := lt_of_lt_of_le hp (le_max_left _ _)
  have hle : max p q / 2 ≤ (p + q) / 2 := by
    rcases le_total p q with h | h
    · rw [max_eq_right h]
      linarith
    · rw [max_eq_left h]
      linarith
  have hl := Real.log_le_log (div_pos hmax (by norm_num : (0 : ℝ) < 2)) hle
  rw [Real.log_div hmax.ne' (by norm_num : (2 : ℝ) ≠ 0)] at hl
  have hlog : 0 < Real.log 2 := Real.log_pos (by norm_num)
  apply (div_le_iff₀ hlog).mpr
  field_simp
  linarith

/-- Section 2.3: two positive likelihoods satisfy the mixture theorem. -/
example : (0 : ℝ) < 3 / 4 ∧ (0 : ℝ) < 1 / 4 := by norm_num

/-- Section 2.3, counterexample to treating the pointwise maximum as a
normalized arithmetic-coding distribution: two valid binary distributions
have maxima summing to 1.5. -/
theorem maximum_likelihood_not_normalized :
    let p : Fin 2 → ℚ := ![3 / 4, 1 / 4]
    let q : Fin 2 → ℚ := ![1 / 4, 3 / 4]
    (∀ i, 0 < p i ∧ 0 < q i) ∧ (∑ i, p i) = 1 ∧ (∑ i, q i) = 1 ∧
      1 < ∑ i, max (p i) (q i) := by
  dsimp
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro i
    fin_cases i <;> norm_num
  · norm_num [Fin.sum_univ_two]
  · norm_num [Fin.sum_univ_two]
  · norm_num [Fin.sum_univ_two]

end Transformer.Memorization

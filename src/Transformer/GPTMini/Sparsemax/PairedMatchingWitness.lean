import Transformer.GPTMini.Sparsemax.PairedMatchingMixture
import Transformer.GPTMini.Sparsemax.ContentPermutation

/-!
# The repaired head learns the two swapped key/value assignments

New answer-only witness for the contextual atomic architecture following
arXiv:2211.11052v1, §3.1 and Appendix A.4. The old content-only model has
sharp error floor 1/2 on these same inputs. A cap-one paired head instead
reads the original value immediately after the queried key, using actual
causal sparsemax Eq. (1) of arXiv:1602.02068v2. Its two original value
coordinates remain independently selectable, with no attention inverse.

Every pair of cap-one scalar answers is physically attainable. Consequently
the two-observation linear price reaches the universal output-box bound.
This exact oracle is specific to these observations, not a global numerical
search method for Basis or arbitrary unseen language.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.GPTMini.Convex Transformer.ConvexRecall
open scoped BigOperators

/-- Select the predecessor role of key one while retaining free original values a and b.
Source: the new binding witness after §3.1; it uses no supplied route loss. -/
def pairedBindingHead (a b : ℝ) : MatchingHead 5 2 1 :=
  (((fun v d => if v = 1 ∧ d = 1 then 1 else 0),
    (fun v d => if v = 1 ∧ d = 1 then 1 else 0)),
    fun v _ => if v = 3 then a else if v = 4 then b else 0)

/-- The independent original value bounds make the complete witness head feasible.
Source: the numerical head box of the new Appendix A.4 binding model. -/
theorem pairedBindingHead_mem (a b : ℝ) (ha : -1 ≤ a ∧ a ≤ 1) (hb : -1 ≤ b ∧ b ≤ 1) :
    pairedBindingHead a b ∈ matchingHeadBox 5 2 1 1 := by
  refine ⟨?_, ?_, ?_⟩ <;> intro v d
  · fin_cases v <;> fin_cases d <;> norm_num [pairedBindingHead]
  · fin_cases v <;> fin_cases d <;> norm_num [pairedBindingHead]
  · fin_cases v <;> norm_num [pairedBindingHead, ha.1, ha.2, hb.1, hb.2]

/-- Both freely chosen scalar value coordinates satisfy the feasibility premises. -/
example : pairedBindingHead (3 / 4) (-1 / 4) ∈ matchingHeadBox 5 2 1 1 :=
  pairedBindingHead_mem _ _ (by norm_num) (by norm_num)

/-- The score-one destination is the original value occurrence following key one, in either table.
Source: actual learned dot products of the repaired §3.1 encoder. -/
theorem pairedBinding_scores (a b : ℝ) (r : Fin 2) (j : Fin 6) :
    pairedHeadScores (H := 1) (pairedBindingHead a b) (pairedBindingTokens r) 5 j =
      if j = 2 then 1 else 0 := by
  rw [pairedHeadScores_formula]
  rw [Fin.sum_univ_two]
  fin_cases r <;> fin_cases j <;>
    norm_num [pairedBindingHead, pairedBindingTokens, pairedPrevious]

/-- Original variational sparsemax routes exactly to that occurrence, rather than to the key token.
Source: the unit-gap support certificate for sparsemax Eq. (1) and §2.2. -/
theorem pairedBinding_weights (a b : ℝ) (r : Fin 2) :
    sparseWeights (pairedHeadScores (H := 1) (pairedBindingHead a b) (pairedBindingTokens r) 5) 5 =
      basis 2 := by
  apply sparseWeights_eq_basis_of_gap _ _ _ (by decide)
  intro j hj hn
  rw [pairedBinding_scores, pairedBinding_scores, ite_eq_right hn]
  norm_num

/-- Original values, not a reparameterized target response, determine the two different answers.
Source: the genuine sparsemax/value product after the repaired §3.1 encoder. -/
theorem pairedBinding_headOutput (a b : ℝ) (r : Fin 2) :
    pairedHeadOutput (H := 1) (pairedBindingHead a b) (pairedBindingTokens r) 5 0 =
      if r = 0 then a else b := by
  rw [pairedHeadOutput_formula, pairedBinding_weights]
  fin_cases r <;> norm_num [basis, pairedBindingHead, pairedBindingTokens, Fin.sum_univ_succ]

/-- A feasible one-head convex state fits the exchanged tables' distinct ordinary answer labels.
Source: the new genuine-head binding repair following Appendix A.4, with no route targets. -/
theorem pairedBinding_fit : pairedMixtureSample (H := 1) pairedBindingTokens
    (fun _ => 5) (fun _ => 0) (Finsupp.single (pairedBindingHead 0 1) 1) = pairedBindingTarget := by
  rw [pairedMixture_single_output]
  ext r
  exact pairedBinding_headOutput 0 1 r

/-- The literal answer-only squared criterion on contextual physical mixtures.
Source: ordinary regression after Appendix A.4, on the same inputs as the old encoder. -/
def pairedBindingError {H : ℕ} (μ : MatchingMixture 5 (2 * H) 1) : ℝ :=
  ∑ r, (pairedMixtureSample pairedBindingTokens (fun _ => 5) (fun _ => 0) μ r - pairedBindingTarget r) ^ 2

/-- The new physical forward eliminates the old model's sharp error floor one half.
Source: the exact answer-only binding witness following Appendix A.4. -/
theorem pairedBinding_error_zero : pairedBindingError (H := 1)
    (Finsupp.single (pairedBindingHead 0 1) 1) = 0 := by
  unfold pairedBindingError
  rw [pairedBinding_fit]
  simp

/-- The fitting state belongs to the same convex probability domain used for training.
Source: the original numerical head box and unit mass after Appendix A.4. -/
theorem pairedBinding_state_mem : Finsupp.single (pairedBindingHead 0 1) (1 : ℝ) ∈
    matchingMixtureDomain 5 2 1 1 :=
  matchingMixture_single_mem _ _ (pairedBindingHead_mem _ _ (by norm_num) (by norm_num))

/-- All cap-one heads have supporting price at least minus the two absolute gradient entries.
Source: the true original-value box, independent of attention supports, after sparsemax Eq. (1). -/
theorem pairedBindingPrice_lower (g : Fin 2 → ℝ) (h : MatchingHead 5 2 1)
    (hh : h ∈ matchingHeadBox 5 2 1 1) :
    -(∑ r, |g r|) ≤ pairedHeadPrice (H := 1) pairedBindingTokens (fun _ => 5) (fun _ => 0) g h := by
  have hl (r : Fin 2) : -|g r| ≤ g r * pairedHeadOutput (H := 1) h (pairedBindingTokens r) 5 0 := by
    have hb := pairedHeadOutput_bounds (H := 1) 1 h hh (pairedBindingTokens r) 5 0
    by_cases hg : 0 ≤ g r
    · rw [abs_of_nonneg hg]
      have hm := mul_le_mul_of_nonneg_left hb.1 hg
      nlinarith
    · have hn : g r ≤ 0 := by linarith
      rw [abs_of_nonpos hn]
      have hm := mul_le_mul_of_nonpos_left hb.2 hn
      nlinarith
  change -(∑ r, |g r|) ≤ ∑ r, g r * pairedHeadOutput (H := 1) h (pairedBindingTokens r) 5 0
  have hs := Finset.sum_le_sum (s := Finset.univ) (fun r hr => hl r)
  simpa only [Finset.sum_neg_distrib] using hs

/-- A fitting original head and nonzero supporting functional satisfy every box-price premise. -/
example : -5 ≤ pairedHeadPrice (H := 1) pairedBindingTokens (fun _ => 5) (fun _ => 0)
    ![(2 : ℝ), -3] (pairedBindingHead 0 1) := by
  have hb := pairedBindingPrice_lower ![(2 : ℝ), -3] (pairedBindingHead 0 1)
    (pairedBindingHead_mem _ _ (by norm_num) (by norm_num))
  norm_num [Fin.sum_univ_two] at hb ⊢
  exact hb

/-- An analytic price head; a zero gradient may choose either value-box endpoint.
Source: the exact two-observation pricing repair after Appendix A.4. Python chooses zero there instead. -/
def pairedBindingPriceHead (g : Fin 2 → ℝ) : MatchingHead 5 2 1 :=
  pairedBindingHead (if 0 ≤ g 0 then -1 else 1) (if 0 ≤ g 1 then -1 else 1)

/-- The exact analytic price head stays inside the original physical parameter box.
Source: the independent value coordinates of the repaired binding witness after §3.1. -/
theorem pairedBindingPriceHead_mem (g : Fin 2 → ℝ) :
    pairedBindingPriceHead g ∈ matchingHeadBox 5 2 1 1 := by
  apply pairedBindingHead_mem <;> split_ifs <;> norm_num

/-- The physical contextual price attains the universal bound for every supporting functional.
Source: exact two-observation original-value optimization following Appendix A.4. -/
theorem pairedBinding_price_exact (g : Fin 2 → ℝ) : pairedHeadPrice (H := 1) pairedBindingTokens
    (fun _ => 5) (fun _ => 0) g (pairedBindingPriceHead g) = -(∑ r, |g r|) := by
  have he (x : ℝ) : x * (if 0 ≤ x then -1 else 1) = -|x| := by
    by_cases hx : 0 ≤ x
    · simp only [hx, ite_true, abs_of_nonneg hx, mul_neg, mul_one]
    · have hn : x ≤ 0 := by linarith
      simp only [hx, ite_false, abs_of_nonpos hn, neg_neg, mul_one]
  simp only [pairedHeadPrice, matchingOutputPairing, pairedHeadSample, pairedBindingPriceHead,
    pairedBinding_headOutput, Fin.sum_univ_two, LinearMap.coe_mk, AddHom.coe_mk, ite_true,
    show (1 : Fin 2) ≠ 0 from by decide, ite_false, he]
  ring

end Transformer.GPTMini.Sparsemax

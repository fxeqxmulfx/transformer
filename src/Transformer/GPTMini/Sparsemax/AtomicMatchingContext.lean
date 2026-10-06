import Transformer.GPTMini.Sparsemax.AtomicMatchingSelection

/-!
# Ordinary answer data can require genuine query-key matching

New two-context witness for the genuine atomic architecture following
arXiv:2211.11052v1, Appendix A.4, with sparsemax Eq. (1) from
arXiv:1602.02068v2. The contexts contain the same two tokens in opposite
orders and request different answers at their final position. Values
are shared across contexts, including when they are freely relearned.

A uniform head cannot distinguish the two answers for any values or
keys. Its total ordinary squared error has the sharp lower bound 1/8.
A bounded genuinely learned Q/K/value head has zero error. Any fitting
feasible mixture must contain an actual head that distinguishes these
orders. Matching therefore reaches the output criterion without a
supervised route label or an attention-canceling value inverse.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.GPTMini.Convex
open scoped BigOperators

/-- Reversing the observed two-token order keeps the multiset of visible tokens identical.
Source: the new two-context matching witness after sparsemax Eq. (1). -/
def matchingReverseTokens (j : Fin 2) : Fin 2 := if j = 0 then 1 else 0

/-- Two data contexts share the learned physical Q/K/value tables.
Source: the genuine multi-context atomic model following Appendix A.4. -/
def matchingOrderTokens (r j : Fin 2) : Fin 2 := if r = 0 then j else matchingReverseTokens j

/-- Literal final-position predictions in both observed contexts.
Source: the genuine causal sparsemax forward inside the new Appendix A.4 mixture. -/
def matchingOrderSample {H : ℕ} : MatchingMixture 2 H 1 →ₗ[ℝ] (Fin 2 → ℝ) :=
  matchingMixtureSample matchingOrderTokens (fun _ => 1) (fun _ => 0)

/-- Answer labels, with no attention destination information.
Source: the new ordinary-answer witness for the atomic model following Appendix A.4. -/
def matchingOrderTarget (r : Fin 2) : ℝ := if r = 0 then 1 else 1 / 2

/-- Ordinary squared answer error; both original value tables and matching are selectable.
Source: the new two-context criterion after Appendix A.4; no route objective is added. -/
def matchingOrderError {H : ℕ} (μ : MatchingMixture 2 H 1) : ℝ :=
  ∑ r, (matchingOrderSample μ r - matchingOrderTarget r) ^ 2

/-- A uniform-query atom still leaves keys and original values freely learnable.
Source: the genuine sparsemax zero-score specialization of Eq. (1). -/
def matchingZeroQueryHead {H : ℕ} (keys : Matrix (Fin 2) (Fin H) ℝ)
    (values : Matrix (Fin 2) (Fin 1) ℝ) : MatchingHead 2 H 1 := ((0, keys), values)

/-- All learned keys and values leave the zero-query head unable to distinguish the two orders.
Source: sparsemax §2.2, uniform weights for equal scores in the shared-value physical forward. -/
theorem matchingZeroQueryHead_order {H : ℕ} (keys : Matrix (Fin 2) (Fin H) ℝ)
    (values : Matrix (Fin 2) (Fin 1) ℝ) :
    matchingHeadOutput (matchingZeroQueryHead keys values) id 1 0 =
      matchingHeadOutput (matchingZeroQueryHead keys values) matchingReverseTokens 1 0 := by
  have hs (tokens : Fin 2 → Fin 2) :
      matchingHeadScores (matchingZeroQueryHead keys values) tokens 1 = fun _ => (0 : ℝ) := by
    ext j
    simp [matchingHeadScores, matchingZeroQueryHead]
  unfold matchingHeadOutput
  rw [hs, hs, sparseWeights_two_equal]
  simp [Fin.sum_univ_two, matchingZeroQueryHead, matchingReverseTokens]
  ring

/-- The learned query distinguishes the reversed context through original sparsemax attention.
Source: sparsemax Eq. (1), uniform scores when the final token is token zero. -/
theorem matchingScalarHead_reverse_output :
    matchingHeadOutput (matchingScalarHead 1) matchingReverseTokens 1 0 = 1 / 2 := by
  have hs : matchingHeadScores (matchingScalarHead 1) matchingReverseTokens 1 = fun _ => (0 : ℝ) := by
    ext j
    simp [matchingHeadScores, matchingScalarHead, matchingReverseTokens]
  unfold matchingHeadOutput
  rw [hs, sparseWeights_two_equal]
  norm_num [Fin.sum_univ_two, matchingScalarHead, matchingReverseTokens]

/-- One genuine learned head fits both answer labels with common original values.
Source: the new multi-context witness after Appendix A.4, without any supplied route labels. -/
theorem matchingScalarHead_order_fit :
    matchingOrderSample (Finsupp.single (matchingScalarHead 1) 1) = matchingOrderTarget := by
  rw [matchingOrderSample, matchingMixture_single_output]
  ext r
  fin_cases r
  · exact matchingScalarHead_one_output
  · exact matchingScalarHead_reverse_output

/-- Every state's literal squared prediction error is nonnegative.
Source: ordinary regression criterion in §3.1, here on the genuine two-context matching model. -/
theorem matchingOrderError_nonneg {H : ℕ} (μ : MatchingMixture 2 H 1) : 0 ≤ matchingOrderError μ :=
  Finset.sum_nonneg (fun _ _ => sq_nonneg _)

/-- The bounded learned head attains the global minimum of the ordinary two-context criterion.
Source: the new exact genuine-head answer fit following Appendix A.4. -/
theorem matchingOrderError_fit : matchingOrderError (Finsupp.single (matchingScalarHead 1) 1) = 0 := by
  unfold matchingOrderError
  rw [matchingScalarHead_order_fit]
  simp

/-- Uniform routing has positive best error even when original values and keys are relearned.
Source: the new matching-necessity witness following Appendix A.4; original values are unrestricted. -/
theorem matchingOrderError_uniform_lower {H : ℕ} (keys : Matrix (Fin 2) (Fin H) ℝ)
    (values : Matrix (Fin 2) (Fin 1) ℝ) :
    (1 / 8 : ℝ) ≤ matchingOrderError (Finsupp.single (matchingZeroQueryHead keys values) 1) := by
  unfold matchingOrderError matchingOrderSample
  rw [matchingMixture_single_output]
  simp only [Fin.sum_univ_two, matchingHeadSample, matchingOrderTarget,
    ite_true, show (1 : Fin 2) ≠ 0 from by decide, ite_false]
  change 1 / 8 ≤ (matchingHeadOutput (matchingZeroQueryHead keys values) id 1 0 - 1) ^ 2 +
    (matchingHeadOutput (matchingZeroQueryHead keys values) matchingReverseTokens 1 0 - 1 / 2) ^ 2
  rw [← matchingZeroQueryHead_order]
  nlinarith [sq_nonneg (matchingHeadOutput (matchingZeroQueryHead keys values) id 1 0 - 3 / 4)]

/-- A freely relearned constant value table realizes the sharp uniform error bound. -/
example : matchingOrderError (Finsupp.single (matchingZeroQueryHead (H := 1) 0
    (fun _ _ => (3 / 4 : ℝ))) 1) = 1 / 8 := by
  unfold matchingOrderError matchingOrderSample
  rw [matchingMixture_single_output]
  have hs (tokens : Fin 2 → Fin 2) :
      matchingHeadScores (matchingZeroQueryHead (H := 1) 0 (fun _ _ => (3 / 4 : ℝ))) tokens 1 =
        fun _ => (0 : ℝ) := by ext j; simp [matchingHeadScores, matchingZeroQueryHead]
  simp only [Fin.sum_univ_two, matchingHeadSample, matchingHeadOutput, hs, sparseWeights_two_equal]
  norm_num [Fin.sum_univ_two, matchingZeroQueryHead, matchingOrderTarget]

/-- Any fitting feasible state selects an actual bounded head that distinguishes the input orders.
Source: the new answer-driven matching consequence following Appendix A.4. -/
theorem matchingOrder_fit_selects_head {H : ℕ} (cap : ℝ) (μ : MatchingMixture 2 H 1)
    (hμ : μ ∈ matchingMixtureDomain 2 H 1 cap) (hs : matchingOrderSample μ = matchingOrderTarget) :
    ∃ h ∈ μ.support, h ∈ matchingHeadBox 2 H 1 cap ∧
      matchingHeadOutput h id 1 0 ≠ matchingHeadOutput h matchingReverseTokens 1 0 := by
  classical
  by_contra hn
  have he (h : MatchingHead 2 H 1) (hh : h ∈ μ.support) :
      matchingHeadOutput h id 1 0 = matchingHeadOutput h matchingReverseTokens 1 0 := by
    by_contra hne
    exact hn ⟨h, hh, hμ.2.1 h (Finsupp.mem_support_iff.mp hh), hne⟩
  have hp : matchingOrderSample μ 0 = matchingOrderSample μ 1 := by
    unfold matchingOrderSample
    rw [matchingMixtureSample_apply, matchingMixtureSample_apply]
    apply Finset.sum_congr rfl
    intro h hh
    exact congrArg (fun x => μ h * x) (he h hh)
  rw [hs] at hp
  norm_num [matchingOrderTarget] at hp

/-- A genuine fitting state inhabits every answer-driven head-selection premise. -/
example : ∃ h ∈ (Finsupp.single (matchingScalarHead 1) (1 : ℝ)).support,
    h ∈ matchingHeadBox 2 1 1 1 ∧
      matchingHeadOutput h id 1 0 ≠ matchingHeadOutput h matchingReverseTokens 1 0 :=
  matchingOrder_fit_selects_head 1 _
    (matchingMixture_single_mem _ _ (matchingScalarHead_mem _ (by norm_num))) matchingScalarHead_order_fit

end Transformer.GPTMini.Sparsemax

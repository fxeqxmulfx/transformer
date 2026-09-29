/-
# No exact convex reformulation of all squared-loss sublevels

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §2–§3.1, equation
`eq:attention_only_obj`.  This strengthens the epigraph obstruction:
even equality of attainable training-objective sublevels for every
nonnegative squared-loss weight and every target is impossible.  It allows
an arbitrary nonlinear decoder and an arbitrary changed parameter penalty,
provided the candidate objectives are convex on one convex parameter set.
The candidate parameter space may be any real vector space, including an
infinite-dimensional one.
-/

import Transformer.Convexifying.Section3_SquaredLossGap

namespace Transformer.Convexifying

/-- An objective sublevel for genuine softmax attention with any finite
number of trainable-score heads and the paper's four-matrix weight decay.
The loss is a squared loss on the linear difference of the two output rows.
Source: arXiv:2211.11052v1, `eq:attention_only_obj`, specialized. -/
def OriginalProjectedSublevel (K target bound : ℝ) : Prop :=
  ∃ heads : List ScalarFourWeights,
    originalMixturePenalty heads +
      K * (originalMixtureDifference heads - target) ^ 2 ≤ bound

/-- The corresponding objective sublevel for a proposed parameter model
with decoder `decode` and possibly changed penalty `penalty`.
Source: `eq:attention_only_obj`, abstract candidate reformulation. -/
def DecodedProjectedSublevel {E : Type*} [AddCommGroup E] [Module ℝ E] (C : Set E)
    (decode penalty : E → ℝ)
    (K target bound : ℝ) : Prop :=
  ∃ z ∈ C, penalty z + K * (decode z - target) ^ 2 ≤ bound

/-- **No exact equality of all convex squared-loss objective sublevels.**
The candidate decoder and penalty can be arbitrary functions.  If all
nonnegative weighted squared-loss objectives are convex on the same
parameter set, their sublevel attainability cannot agree with the original
trainable-score model for every weight, target, and bound.  This conclusion
does not assume equality of prediction-budget epigraphs.
Source: arXiv:2211.11052v1, §2–§3.1, `eq:attention_only_obj`. -/
theorem no_universal_squared_sublevel_equivalence {E : Type*} [AddCommGroup E] [Module ℝ E]
    (C : Set E) (decode penalty : E → ℝ)
    (hall : ∀ K : ℝ, 0 ≤ K → ∀ target : ℝ,
      ConvexOn ℝ C
        (fun z => penalty z + K * (decode z - target) ^ 2)) :
    ¬ ∀ K : ℝ, 0 ≤ K → ∀ target bound : ℝ,
      OriginalProjectedSublevel K target bound ↔
        DecodedProjectedSublevel C decode penalty K target bound := by
  intro hequiv
  have hpen : ConvexOn ℝ C penalty := by
    simpa using hall 0 (by norm_num) 0
  have hsq : ∀ target : ℝ,
      ConvexOn ℝ C (fun z => penalty z + (decode z - target) ^ 2) := by
    intro target
    simpa using hall 1 (by norm_num) target
  have hpen_nonneg (z : E) (hz : z ∈ C) : 0 ≤ penalty z := by
    by_contra hneg
    have hneg' : penalty z < 0 := lt_of_not_ge hneg
    have hcand : DecodedProjectedSublevel C decode penalty 0 0 (penalty z) := by
      exact ⟨z, hz, by simp⟩
    obtain ⟨heads, hcost⟩ := (hequiv 0 (by norm_num) 0 (penalty z)).2 hcand
    have hred0 := reducedMixtureCost_nonneg
      (heads.map fun h => (h.q * h.k, h.v * h.o))
    have hge := originalMixturePenalty_ge_reduced heads
    nlinarith
  have hzeroOrig : OriginalProjectedSublevel 1 0 0 := by
    refine ⟨[], ?_⟩
    simp [originalMixturePenalty, originalMixtureDifference]
  obtain ⟨z0, hz0, hobj0⟩ := (hequiv 1 (by norm_num) 0 0).1 hzeroOrig
  have hpen0 := hpen_nonneg z0 hz0
  have hr0 : penalty z0 = 0 := by nlinarith [sq_nonneg (decode z0)]
  have hf0 : decode z0 = 0 := by nlinarith [sq_nonneg (decode z0)]
  obtain ⟨heads1, hD1, hR1⟩ := original_budget_witness
  have hendpoint : OriginalProjectedSublevel 1000000 (1 / 6) 2 := by
    refine ⟨heads1, ?_⟩
    rw [hD1]
    norm_num at hR1 ⊢
    exact hR1
  obtain ⟨z1, hz1, hobj1⟩ :=
    (hequiv 1000000 (by norm_num) (1 / 6) 2).1 hendpoint
  have hpen1 := hpen_nonneg z1 hz1
  have hr1 : penalty z1 ≤ 2 := by
    nlinarith [sq_nonneg (decode z1 - 1 / 6)]
  have herr1 : (decode z1 - 1 / 6) ^ 2 ≤ 2 / 1000000 := by
    nlinarith
  let z := (99 / 100 : ℝ) • z0 + (1 / 100 : ℝ) • z1
  have ha : 0 ≤ (99 / 100 : ℝ) := by norm_num
  have hb : 0 ≤ (1 / 100 : ℝ) := by norm_num
  have hab : (99 / 100 : ℝ) + 1 / 100 = 1 := by norm_num
  have hz : z ∈ C := hpen.1 hz0 hz1 ha hb hab
  have hf : decode z =
      (99 / 100 : ℝ) * decode z0 + (1 / 100 : ℝ) * decode z1 :=
    decoder_jensen_of_all_square_targets C decode penalty hsq
      hz0 hz1 ha hb hab
  have hr : penalty z ≤
      (99 / 100 : ℝ) * penalty z0 + (1 / 100 : ℝ) * penalty z1 := by
    simpa [z] using hpen.2 hz0 hz1 ha hb hab
  have hrz : penalty z ≤ 1 / 50 := by rw [hr0] at hr; nlinarith
  have herrEq : decode z - 1 / 600 = (decode z1 - 1 / 6) / 100 := by
    rw [hf, hf0]
    ring
  have herrz : 200000 * (decode z - 1 / 600) ^ 2 ≤ 1 / 25000 := by
    rw [herrEq]
    nlinarith [herr1]
  have hcandFinal : DecodedProjectedSublevel C decode penalty
      200000 (1 / 600) (1 / 40) := by
    exact ⟨z, hz, by nlinarith [hrz, herrz]⟩
  obtain ⟨headsF, hobjF⟩ :=
    (hequiv 200000 (by norm_num) (1 / 600) (1 / 40)).2 hcandFinal
  have hLower := originalProjectedObjective_lower headsF
  unfold originalProjectedObjective projectedSquareLoss at hLower
  nlinarith

end Transformer.Convexifying

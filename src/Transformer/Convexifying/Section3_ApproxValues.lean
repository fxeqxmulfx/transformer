/-
# Approximate objective-value equivalence also fails

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §2–§3.1, equation
`eq:attention_only_obj`.  The preceding no-go theorem used exact equality of
objective sublevels.  Here even arbitrarily accurate transfer of every
sublevel between the original and a single convex candidate is refuted.
This allows arbitrary real-vector-space parameters, a nonlinear decoder,
and a changed fixed penalty.
-/

import Transformer.Convexifying.Section3_UniversalValues

namespace Transformer.Convexifying

/-- Every objective sublevel on either side can be matched on the other
side with any positive tolerance in the objective bound.  This is the
value-level equivalence condition used below.
Source: arXiv:2211.11052v1, `eq:attention_only_obj`, abstract candidate. -/
def ApproxProjectedSublevelTransfer {E : Type*} [AddCommGroup E] [Module ℝ E]
    (C : Set E) (decode penalty : E → ℝ) : Prop :=
  ∀ K : ℝ, 0 ≤ K → ∀ target bound ε : ℝ, 0 < ε →
    (OriginalProjectedSublevel K target bound →
      DecodedProjectedSublevel C decode penalty K target (bound + ε)) ∧
    (DecodedProjectedSublevel C decode penalty K target bound →
      OriginalProjectedSublevel K target (bound + ε))

/-- **No arbitrarily accurate equivalence of training values.**  A fixed
candidate whose objectives are convex for all nonnegative weighted squared
losses cannot transfer every original objective sublevel to within every
positive tolerance and vice versa.  The candidate vector space may be
infinite-dimensional, and its decoder and fixed penalty may be arbitrary.
Source: arXiv:2211.11052v1, §2–§3.1, `eq:attention_only_obj`. -/
theorem no_universal_approx_sublevel_transfer {E : Type*}
    [AddCommGroup E] [Module ℝ E]
    (C : Set E) (decode penalty : E → ℝ)
    (hall : ∀ K : ℝ, 0 ≤ K → ∀ target : ℝ,
      ConvexOn ℝ C (fun z => penalty z + K * (decode z - target) ^ 2)) :
    ¬ ApproxProjectedSublevelTransfer C decode penalty := by
  intro htransfer
  have hpen : ConvexOn ℝ C penalty := by
    simpa using hall 0 (by norm_num) 0
  have hsq : ∀ target : ℝ,
      ConvexOn ℝ C (fun z => penalty z + (decode z - target) ^ 2) := by
    intro target
    simpa using hall 1 (by norm_num) target
  have hpen_nonneg (z : E) (hz : z ∈ C) : 0 ≤ penalty z := by
    by_contra hneg
    have hneg' : penalty z < 0 := lt_of_not_ge hneg
    have heps : 0 < -penalty z / 2 := by linarith
    have hcand : DecodedProjectedSublevel C decode penalty 0 0 (penalty z) :=
      ⟨z, hz, by simp⟩
    obtain ⟨heads, hcost⟩ :=
      ((htransfer 0 (by norm_num) 0 (penalty z)
        (-penalty z / 2) heps).2 hcand)
    have hred0 := reducedMixtureCost_nonneg
      (heads.map fun h => (h.q * h.k, h.v * h.o))
    have hge := originalMixturePenalty_ge_reduced heads
    nlinarith
  let ε : ℝ := 1 / 1000000000
  have hε : 0 < ε := by dsimp [ε]; norm_num
  have hzeroOrig : OriginalProjectedSublevel 1000000 0 0 := by
    refine ⟨[], ?_⟩
    simp [originalMixturePenalty, originalMixtureDifference]
  obtain ⟨z0, hz0, hobj0⟩ :=
    (htransfer 1000000 (by norm_num) 0 0 ε hε).1 hzeroOrig
  simp only [sub_zero, zero_add] at hobj0
  have hpen0 := hpen_nonneg z0 hz0
  have hr0 : penalty z0 ≤ ε := by
    have hs : 0 ≤ 1000000 * decode z0 ^ 2 := by positivity
    calc
      penalty z0 ≤ penalty z0 + 1000000 * decode z0 ^ 2 :=
        le_add_of_nonneg_right hs
      _ ≤ ε := hobj0
  have hf0sq : decode z0 ^ 2 ≤ ε / 1000000 := by
    apply (le_div_iff₀ (by norm_num : (0 : ℝ) < 1000000)).2
    nlinarith only [hobj0, hpen0]
  obtain ⟨heads1, hD1, hR1⟩ := original_budget_witness
  have hendpoint : OriginalProjectedSublevel 1000000 (1 / 6) 2 := by
    refine ⟨heads1, ?_⟩
    rw [hD1]
    norm_num at hR1 ⊢
    exact hR1
  obtain ⟨z1, hz1, hobj1⟩ :=
    (htransfer 1000000 (by norm_num) (1 / 6) 2 ε hε).1 hendpoint
  have hpen1 := hpen_nonneg z1 hz1
  have hr1 : penalty z1 ≤ 2 + ε := by
    have hs : 0 ≤ 1000000 * (decode z1 - 1 / 6) ^ 2 := by positivity
    calc
      penalty z1 ≤ penalty z1 + 1000000 * (decode z1 - 1 / 6) ^ 2 :=
        le_add_of_nonneg_right hs
      _ ≤ 2 + ε := hobj1
  have hf1sq : (decode z1 - 1 / 6) ^ 2 ≤ (2 + ε) / 1000000 := by
    apply (le_div_iff₀ (by norm_num : (0 : ℝ) < 1000000)).2
    nlinarith only [hobj1, hpen1]
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
  have hrz : penalty z ≤ 1 / 50 + ε := by
    calc
      penalty z ≤ (99 / 100 : ℝ) * penalty z0 +
          (1 / 100 : ℝ) * penalty z1 := hr
      _ ≤ (99 / 100 : ℝ) * ε + (1 / 100 : ℝ) * (2 + ε) :=
        add_le_add
          (mul_le_mul_of_nonneg_left hr0 (by norm_num))
          (mul_le_mul_of_nonneg_left hr1 (by norm_num))
      _ = 1 / 50 + ε := by ring
  have herrEq : decode z - 1 / 600 =
      (99 / 100 : ℝ) * decode z0 +
        (1 / 100 : ℝ) * (decode z1 - 1 / 6) := by
    rw [hf]
    ring
  have herrSq : (decode z - 1 / 600) ^ 2 ≤
      2 * (((99 / 100 : ℝ) * decode z0) ^ 2 +
        ((1 / 100 : ℝ) * (decode z1 - 1 / 6)) ^ 2) := by
    rw [herrEq]
    nlinarith [sq_nonneg
      ((99 / 100 : ℝ) * decode z0 - (1 / 100 : ℝ) * (decode z1 - 1 / 6))]
  have herrz : 200000 * (decode z - 1 / 600) ^ 2 ≤ 1 / 10000 := by
    dsimp [ε] at hf0sq hf1sq
    nlinarith [herrSq, hf0sq, hf1sq]
  have hcandFinal : DecodedProjectedSublevel C decode penalty
      200000 (1 / 600) (1 / 40) := by
    exact ⟨z, hz, by dsimp [ε] at hrz; nlinarith [hrz, herrz]⟩
  obtain ⟨headsF, hobjF⟩ :=
    (htransfer 200000 (by norm_num) (1 / 600) (1 / 40) ε hε).2 hcandFinal
  have hLower := originalProjectedObjective_lower headsF
  unfold originalProjectedObjective projectedSquareLoss at hLower
  dsimp [ε] at hobjF
  nlinarith

end Transformer.Convexifying

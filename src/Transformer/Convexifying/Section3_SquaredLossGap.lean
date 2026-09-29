/-
# A convex squared loss detects the atomic-penalty gap

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §2–§3.1, equation
`eq:attention_only_obj` allows squared loss.  A squared loss on the linear
difference between two output rows separates the ordinary four-matrix
weight-decay problem from the convex genuine-softmax atomic mixture, even
when the ordinary model has any finite number of trainable-score heads.
-/

import Transformer.Convexifying.Section3_UniversalSquared

namespace Transformer.Convexifying

/-- Squared loss on the difference between the two output rows, with target
`1/600` and positive weight `200000`.  Source: equation
`eq:attention_only_obj`, an allowed convex squared-loss specialization. -/
noncomputable def projectedSquareLoss (d : ℝ) : ℝ :=
  200000 * (d - 1 / 600) ^ 2

/-- The projected squared loss is convex in its scalar prediction.
Source: equation `eq:attention_only_obj`, squared-loss case. -/
theorem projectedSquareLoss_convex (x y t : ℝ)
    (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    projectedSquareLoss ((1 - t) * x + t * y) ≤
      (1 - t) * projectedSquareLoss x + t * projectedSquareLoss y := by
  have h := squareLoss_convex x y (1 / 600) t ht0 ht1
  have hscaled := mul_le_mul_of_nonneg_left h (by norm_num : (0 : ℝ) ≤ 200000)
  unfold projectedSquareLoss
  dsimp [squareLoss] at hscaled
  convert hscaled using 1
  ring

/-- The same loss on a two-row output is convex because row difference is
linear.  Source: equation `eq:attention_only_obj`, squared-loss case. -/
noncomputable def twoRowProjectedSquareLoss (u : Fin 2 → ℝ) : ℝ :=
  projectedSquareLoss (u 0 - u 1)

/-- Convexity of the chosen loss in the complete two-row prediction.
Source: equation `eq:attention_only_obj`, squared-loss case. -/
theorem twoRowProjectedSquareLoss_convex (u v : Fin 2 → ℝ) (t : ℝ)
    (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    twoRowProjectedSquareLoss (fun r => (1 - t) * u r + t * v r) ≤
      (1 - t) * twoRowProjectedSquareLoss u +
        t * twoRowProjectedSquareLoss v := by
  change projectedSquareLoss
      (((1 - t) * u 0 + t * v 0) - ((1 - t) * u 1 + t * v 1)) ≤
    (1 - t) * projectedSquareLoss (u 0 - u 1) +
      t * projectedSquareLoss (v 0 - v 1)
  have heq :
      ((1 - t) * u 0 + t * v 0) - ((1 - t) * u 1 + t * v 1) =
      (1 - t) * (u 0 - u 1) + t * (v 0 - v 1) := by ring
  rw [heq]
  exact projectedSquareLoss_convex _ _ _ ht0 ht1

/-- The atomic prediction-budget set and the chosen squared loss together
form a convex optimization problem in projected output and budget.
Source: §3.1, candidate convexification of `eq:attention_only_obj`. -/
theorem atomicProjectedProblem_convex :
    ConvexOn ℝ atomicMixtureEpigraph
      (fun db => projectedSquareLoss db.1 + db.2) := by
  constructor
  · exact atomicMixtureEpigraph_convex
  · intro x _ y _ a b ha hb hab
    have hb1 : b ≤ 1 := by linarith
    have haeq : a = 1 - b := by linarith
    rw [haeq]
    have hloss := projectedSquareLoss_convex x.1 y.1 b hb hb1
    change projectedSquareLoss ((1 - b) * x.1 + b * y.1) +
        ((1 - b) * x.2 + b * y.2) ≤
      (1 - b) * (projectedSquareLoss x.1 + x.2) +
        b * (projectedSquareLoss y.1 + y.2)
    linarith

/-- The interpolation interval in the convexity theorem is nonempty. -/
example : 0 ≤ (1 / 2 : ℝ) ∧ (1 / 2 : ℝ) ≤ 1 := by norm_num

/-- Original four-matrix objective for the chosen squared loss and `β = 1`.
Source: `eq:attention_only_obj`, specialized to two rows and finite width. -/
noncomputable def originalProjectedObjective
    (heads : List ScalarFourWeights) : ℝ :=
  projectedSquareLoss (originalMixtureDifference heads) +
    originalMixturePenalty heads

/-- Atomic-mixture objective for the same squared loss and weighted
coefficient penalty.  Source: §3.1, candidate convexification. -/
noncomputable def atomicProjectedObjective (heads : List (ℝ × ℝ)) : ℝ :=
  projectedSquareLoss (reducedMixtureDifference heads) +
    atomicMixtureCost heads

/-- The projected original objective is exactly the chosen convex loss of
the full two-row genuine-softmax output plus the paper's weight decay.
Source: `eq:attention_only_obj`, specialized. -/
theorem originalProjectedObjective_eq_rows (heads : List ScalarFourWeights) :
    originalProjectedObjective heads =
      twoRowProjectedSquareLoss (originalMixtureRows heads) +
        originalMixturePenalty heads := by
  rw [originalProjectedObjective, twoRowProjectedSquareLoss,
    originalMixtureDifference_eq_rows]

/-- The atomic objective uses the same full-output loss, with the changed
weighted coefficient penalty.  Source: §3.1, candidate convexification. -/
theorem atomicProjectedObjective_eq_rows (heads : List (ℝ × ℝ)) :
    atomicProjectedObjective heads =
      twoRowProjectedSquareLoss (atomicMixtureRows heads) +
        atomicMixtureCost heads := by
  rw [atomicProjectedObjective, twoRowProjectedSquareLoss,
    reducedMixtureDifference_eq_atomicRows]

/-- The convex atomic model has a feasible objective value at most `1/50`.
Source: equation `eq:attention_only_obj`, candidate atomic penalty. -/
theorem atomicProjectedObjective_witness :
    atomicProjectedObjective [(Real.log 2, 1 / 100)] ≤ 1 / 50 := by
  have hout : reducedMixtureDifference [(Real.log 2, 1 / 100)] = 1 / 600 := by
    simpa [reducedMixtureDifference] using smallAtomicDifference
  unfold atomicProjectedObjective
  rw [hout]
  simpa [projectedSquareLoss, atomicMixtureCost] using smallAtomicCost

/-- Every finite original network pays at least `1/30` under the same
convex squared loss.  Thus changing to the homogeneous atomic penalty
changes an optimal training value, not merely a representation of the
feasible set.  Source: arXiv:2211.11052v1, `eq:attention_only_obj`,
specialized. -/
theorem originalProjectedObjective_lower (heads : List ScalarFourWeights) :
    1 / 30 ≤ originalProjectedObjective heads := by
  have hbound := originalMixtureDifference_quadratic heads
  have hcost := originalMixturePenalty_ge_reduced heads
  have hred0 := reducedMixtureCost_nonneg
    (heads.map fun h => (h.q * h.k, h.v * h.o))
  have hR0 : 0 ≤ originalMixturePenalty heads := le_trans hred0 hcost
  by_cases hlarge : 1 / 30 ≤ originalMixturePenalty heads
  · unfold originalProjectedObjective projectedSquareLoss
    nlinarith [sq_nonneg (originalMixtureDifference heads - 1 / 600)]
  · have hsmall : originalMixturePenalty heads ≤ 1 / 30 := le_of_not_ge hlarge
    have hR2 : originalMixturePenalty heads ^ 2 ≤ 1 / 900 := by
      nlinarith [mul_nonneg hR0 (sub_nonneg.mpr hsmall)]
    have hD : originalMixtureDifference heads ≤ 1 / 900 := by
      linarith [le_abs_self (originalMixtureDifference heads)]
    have hgap : 1 / 1800 ≤
        1 / 600 - originalMixtureDifference heads := by linarith
    have hsum : 0 ≤
        (1 / 600 - originalMixtureDifference heads) + 1 / 1800 := by linarith
    have hsq : (1 / 1800 : ℝ) ^ 2 ≤
        (originalMixtureDifference heads - 1 / 600) ^ 2 := by
      nlinarith [mul_nonneg (sub_nonneg.mpr hgap) hsum]
    unfold originalProjectedObjective projectedSquareLoss
    nlinarith

/-- **Strict training-objective gap for an allowed convex loss.**  The
original model has value at least `1/30` for every finite number of heads;
the convex genuine-softmax atomic model has a point of value at most `1/50`.
Source: arXiv:2211.11052v1, `eq:attention_only_obj`, specialized. -/
theorem projected_squared_loss_gap :
    (∀ heads : List ScalarFourWeights,
      1 / 30 ≤ originalProjectedObjective heads) ∧
    (∃ heads : List (ℝ × ℝ), atomicProjectedObjective heads ≤ 1 / 50) := by
  constructor
  · exact originalProjectedObjective_lower
  · exact ⟨[(Real.log 2, 1 / 100)], atomicProjectedObjective_witness⟩

end Transformer.Convexifying

/-
# The exact prediction-budget epigraph is nonconvex

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §2–§3.1, equations
`eq:attention_only` and `eq:attention_only_obj`.  The original four-matrix
weight decay cannot be represented by a convex prediction-budget epigraph,
even when any finite number of trainable-score softmax heads is allowed.
-/

import Transformer.Convexifying.Section3_AtomicPenaltyWeights

namespace Transformer.Convexifying

/-- Projected prediction and budget pairs attainable by any finite number
of genuine trainable-score heads with the paper's four-matrix weight decay.
Source: `eq:attention_only`–`eq:attention_only_obj`, finite-width extension. -/
def originalMixtureEpigraph : Set (ℝ × ℝ) :=
  {dc | ∃ heads : List ScalarFourWeights,
    originalMixtureDifference heads = dc.1 ∧
    originalMixturePenalty heads ≤ dc.2}

/-- The cheap point admitted by the convex atomic mixture is outside the
original prediction-budget epigraph, even with arbitrarily many finite
trainable-score heads.  Source: arXiv:2211.11052v1,
`eq:attention_only_obj`, specialized. -/
theorem smallAtomicPoint_not_original :
    ((1 / 600 : ℝ), (1 / 50 : ℝ)) ∉ originalMixtureEpigraph := by
  rintro ⟨heads, houtput, hcost⟩
  exact (not_le_of_gt (originalMixturePenalty_gap heads houtput)) hcost

/-- The genuine-softmax atomic mixture is convex but does not have the
same prediction-budget set as the paper's four-matrix weight-decay model.
The score product is trainable in both sets.  Source: arXiv:2211.11052v1,
§2–§3.1, equations `eq:attention_only`–`eq:attention_only_obj`. -/
theorem atomicMixture_not_exact :
    atomicMixtureEpigraph ≠ originalMixtureEpigraph := by
  intro hEq
  have hpoint : ((1 / 600 : ℝ), (1 / 50 : ℝ)) ∈ originalMixtureEpigraph := by
    rw [← hEq]
    exact smallAtomicPoint
  exact smallAtomicPoint_not_original hpoint

/-- A one-head prediction of `1/6` in the nonuniform output coordinate is
available with four-matrix weight decay at most `2`.
Source: arXiv:2211.11052v1, equations `eq:attention_only`–
`eq:attention_only_obj`, specialized. -/
theorem original_budget_witness : (1 / 6, 2) ∈ originalMixtureEpigraph := by
  let h : ScalarFourWeights := ⟨Real.log 2, 1, 1, 1⟩
  have hlog0 : 0 ≤ Real.log 2 := Real.log_nonneg (by norm_num)
  have hlog1 : Real.log 2 ≤ 1 := by
    have h' := Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 2)
    norm_num at h'
    exact h'
  have hsq : (Real.log 2) ^ 2 ≤ 1 := by
    nlinarith [mul_nonneg hlog0 (sub_nonneg.mpr hlog1)]
  refine ⟨[h], ?_, ?_⟩
  · simp [originalMixtureDifference, toyOutput_difference,
      reducedHeadDifference, centeredSoftmax, h,
      Real.exp_log (by norm_num : (0 : ℝ) < 2)]
    norm_num
  · simp [originalMixturePenalty, scalarFourWeightCost, h]
    nlinarith

/-- **The exact projected prediction-budget epigraph is nonconvex even when the
number of original softmax heads is arbitrary and finite.**  Mixing the
zero network with a genuine head that has difference `1/6` at budget `2`
would put `(1/600, 1/50)` in any convex epigraph.  The quadratic bound
above proves that no original network can attain this point.  Thus an
affine-output convex reformulation cannot preserve both predictions and the
four-matrix weight decay of equation `eq:attention_only_obj`.
Source: arXiv:2211.11052v1, §2–§3.1. -/
theorem originalMixtureEpigraph_not_convex :
    ¬ Convex ℝ originalMixtureEpigraph := by
  intro hc
  have hzero : (0, 0) ∈ originalMixtureEpigraph := by
    exact ⟨[], by simp [originalMixtureDifference],
      by simp [originalMixturePenalty]⟩
  have hmid := (convex_iff_forall_pos.mp hc) hzero original_budget_witness
    (show 0 < (99 / 100 : ℝ) by norm_num)
    (show 0 < (1 / 100 : ℝ) by norm_num)
    (show (99 / 100 : ℝ) + 1 / 100 = 1 by norm_num)
  have hpoint : ((1 / 600 : ℝ), (1 / 50 : ℝ)) ∈ originalMixtureEpigraph := by
    convert hmid using 1
    ext <;> norm_num [Prod.smul_mk, Prod.mk_add_mk]
  obtain ⟨heads, hout, hcost⟩ := hpoint
  have hgap := originalMixturePenalty_gap heads hout
  linarith

/-- No finite-dimensional convex parameter set with an affine map to
projected prediction-budget pairs can exactly represent the original,
arbitrarily wide, trainable-score attention model with its four-matrix
weight decay.
This is the epigraph obstruction established here; nonlinear recovery or a
different penalty lies outside its scope.  Source: arXiv:2211.11052v1,
§2–§3.1, equations `eq:attention_only`–`eq:attention_only_obj`. -/
theorem no_finite_affine_budget_lift {m : ℕ}
    (C : Set (Fin m → ℝ)) (f : (Fin m → ℝ) →ᵃ[ℝ] (ℝ × ℝ))
    (hC : Convex ℝ C) : f '' C ≠ originalMixtureEpigraph := by
  intro hEq
  have hconv : Convex ℝ originalMixtureEpigraph := by
    rw [← hEq]
    exact Convex.affine_image f hC
  exact originalMixtureEpigraph_not_convex hconv

/-- The convex-domain hypothesis in the preceding theorem is satisfiable. -/
example : Convex ℝ (Set.univ : Set (Fin 1 → ℝ)) := convex_univ

end Transformer.Convexifying

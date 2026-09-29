/-
# The genuine-softmax atomic mixture is convex

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §2–§3.1.  This is a candidate
convex surrogate for the attention-only model in `eq:attention_only_obj`:
atoms retain trainable softmax score products, while coefficients carry a
weighted total-variation cost.  The surrogate's prediction-budget set is
convex.  The following module proves that it does not preserve the paper's
four-matrix weight decay.
-/

import Transformer.Convexifying.Section3_AtomicPenalty

namespace Transformer.Convexifying

/-- Weighted atomic cost for a finite mixture of genuine softmax heads.
Source: arXiv:2211.11052v1, §3.1, candidate convexification of equation
`eq:attention_only_obj`. -/
def atomicMixtureCost (heads : List (ℝ × ℝ)) : ℝ :=
  (heads.map fun h => atomicHeadCost h.1 h.2).sum

/-- The full two-row output of the atomic mixture.  Each atom is a genuine
softmax head with trainable score product `p`, realized by `q = p`, `k = 1`,
and output coefficient `t`, realized by `v = t`, `o = 1`.
Source: arXiv:2211.11052v1, equation `eq:attention_only`, specialized. -/
noncomputable def atomicMixtureRows (heads : List (ℝ × ℝ)) : Fin 2 → ℝ :=
  fun r => (heads.map fun h => toyOutput 0 r h.1 1 h.2 1).sum

/-- The projected atomic prediction is exactly the difference of its two
full output rows.  Source: equation `eq:attention_only`, atomic mixture. -/
theorem reducedMixtureDifference_eq_atomicRows (heads : List (ℝ × ℝ)) :
    reducedMixtureDifference heads =
      atomicMixtureRows heads 0 - atomicMixtureRows heads 1 := by
  have hhead (p t : ℝ) :
      reducedHeadDifference p t =
        toyOutput 0 0 p 1 t 1 - toyOutput 0 1 p 1 t 1 := by
    simp [toyOutput, toyScores, toyData, rowSoftmax,
      reducedHeadDifference, centeredSoftmax, Fin.sum_univ_two, Real.exp_zero]
    ring
  induction heads with
  | nil => simp [reducedMixtureDifference, atomicMixtureRows]
  | cons h hs ih =>
      simp only [reducedMixtureDifference, atomicMixtureRows,
        List.map_cons, List.sum_cons] at ih ⊢
      rw [hhead]
      linarith

/-- Scale only the output coefficients, leaving trainable score atoms
unchanged.  Source: equation `eq:attention_only`, atomic mixture. -/
def scaleAtomicHeads (a : ℝ) (heads : List (ℝ × ℝ)) : List (ℝ × ℝ) :=
  heads.map fun h => (h.1, a * h.2)

/-- The atomic prediction is linear in its signed coefficient.
Source: equation `eq:attention_only`, atomic mixture. -/
private theorem reducedHeadDifference_scale (p t a : ℝ) :
    reducedHeadDifference p (a * t) = a * reducedHeadDifference p t := by
  simp [reducedHeadDifference]
  ring

/-- The weighted atomic cost is positively homogeneous in its signed
coefficient.  Source: equation `eq:attention_only_obj`, candidate penalty. -/
private theorem atomicHeadCost_scale (p t a : ℝ) (ha : 0 ≤ a) :
    atomicHeadCost p (a * t) = a * atomicHeadCost p t := by
  simp [atomicHeadCost, abs_mul, abs_of_nonneg ha]
  ring

/-- Scaling every coefficient scales the total prediction.
Source: equation `eq:attention_only`, atomic mixture. -/
theorem reducedMixtureDifference_scale (a : ℝ) (heads : List (ℝ × ℝ)) :
    reducedMixtureDifference (scaleAtomicHeads a heads) =
      a * reducedMixtureDifference heads := by
  induction heads with
  | nil => simp [scaleAtomicHeads, reducedMixtureDifference]
  | cons h hs ih =>
      simp only [scaleAtomicHeads, reducedMixtureDifference,
        List.map_cons, List.sum_cons] at ih ⊢
      rw [reducedHeadDifference_scale, ih]
      ring

/-- Scaling every coefficient by a nonnegative number scales its total
weighted atomic cost.  Source: candidate convexification of §3.1. -/
theorem atomicMixtureCost_scale (a : ℝ) (ha : 0 ≤ a)
    (heads : List (ℝ × ℝ)) :
    atomicMixtureCost (scaleAtomicHeads a heads) =
      a * atomicMixtureCost heads := by
  induction heads with
  | nil => simp [scaleAtomicHeads, atomicMixtureCost]
  | cons h hs ih =>
      simp only [scaleAtomicHeads, atomicMixtureCost,
        List.map_cons, List.sum_cons] at ih ⊢
      rw [atomicHeadCost_scale _ _ _ ha, ih]
      ring

/-- Nonnegative scaling factors required by the two cost-scaling lemmas
exist. -/
example : 0 ≤ (1 : ℝ) := by norm_num

/-- Concatenating atom lists adds their predictions.
Source: equation `eq:attention_only`, atomic mixture. -/
theorem reducedMixtureDifference_append (u v : List (ℝ × ℝ)) :
    reducedMixtureDifference (u ++ v) =
      reducedMixtureDifference u + reducedMixtureDifference v := by
  simp [reducedMixtureDifference, List.map_append, List.sum_append]

/-- Concatenating atom lists adds their costs.
Source: candidate convexification of equation `eq:attention_only_obj`. -/
theorem atomicMixtureCost_append (u v : List (ℝ × ℝ)) :
    atomicMixtureCost (u ++ v) = atomicMixtureCost u + atomicMixtureCost v := by
  simp [atomicMixtureCost, List.map_append, List.sum_append]

/-- Achievable prediction-budget pairs under the positively homogeneous
atomic cost.  Score parameters range over all real numbers, so query and
key scores are not fixed.  Source: §3.1, candidate convexification. -/
def atomicMixtureEpigraph : Set (ℝ × ℝ) :=
  {dc | ∃ heads : List (ℝ × ℝ),
    reducedMixtureDifference heads = dc.1 ∧ atomicMixtureCost heads ≤ dc.2}

/-- The finite atomic prediction-budget epigraph is convex.  Both score
products remain free; convex combinations scale coefficients and concatenate
the two finite atom lists.  Source: §3.1, candidate convexification. -/
theorem atomicMixtureEpigraph_convex : Convex ℝ atomicMixtureEpigraph := by
  apply convex_iff_forall_pos.mpr
  intro x hx y hy a b ha hb hab
  change (∃ heads : List (ℝ × ℝ),
    reducedMixtureDifference heads = x.1 ∧ atomicMixtureCost heads ≤ x.2) at hx
  change (∃ heads : List (ℝ × ℝ),
    reducedMixtureDifference heads = y.1 ∧ atomicMixtureCost heads ≤ y.2) at hy
  obtain ⟨u, huout, hucost⟩ := hx
  obtain ⟨v, hvout, hvcost⟩ := hy
  refine ⟨scaleAtomicHeads a u ++ scaleAtomicHeads b v, ?_, ?_⟩
  · rw [reducedMixtureDifference_append,
      reducedMixtureDifference_scale, reducedMixtureDifference_scale,
      huout, hvout]
    simp
  · rw [atomicMixtureCost_append, atomicMixtureCost_scale _ (le_of_lt ha),
      atomicMixtureCost_scale _ (le_of_lt hb)]
    have h₁ : 0 ≤ a * (x.2 - atomicMixtureCost u) :=
      mul_nonneg (le_of_lt ha) (sub_nonneg.mpr hucost)
    have h₂ : 0 ≤ b * (y.2 - atomicMixtureCost v) :=
      mul_nonneg (le_of_lt hb) (sub_nonneg.mpr hvcost)
    change a * atomicMixtureCost u + b * atomicMixtureCost v ≤
      a * x.2 + b * y.2
    linarith

/-- The convex atomic model contains the genuine `log 2` softmax atom at
prediction difference `1/600` and budget `1/50`.
Source: equation `eq:attention_only`, candidate convexification. -/
theorem smallAtomicPoint :
    ((1 / 600 : ℝ), (1 / 50 : ℝ)) ∈ atomicMixtureEpigraph := by
  refine ⟨[(Real.log 2, 1 / 100)], ?_, ?_⟩
  · simpa [reducedMixtureDifference] using smallAtomicDifference
  · simpa [atomicMixtureCost] using smallAtomicCost

end Transformer.Convexifying

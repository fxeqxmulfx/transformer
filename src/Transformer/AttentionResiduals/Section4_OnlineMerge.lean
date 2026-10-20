module

public import Transformer.AttentionResiduals.Section3_Blocks

/-!
# Exact two-phase attention merging

arXiv:2603.15031v1, §4.2, Algorithm 1: cached inter-block attention is
merged with a sequential partial-block source using online softmax.
The stored numerator is unnormalized; the mass is the sum of shifted
exponentials. The proof uses exact reals, with no floating-point claim.
Every intermediate reference shift may be arbitrary; the merge chooses
their maximum. The embedding guarantees a positive normalization mass
even at the first block and when the second source group is empty.
-/

@[expose] public section

open scoped BigOperators
open Transformer.Residual

namespace Transformer.AttentionResiduals

variable {ι κ : Type*} [Fintype ι] [Fintype κ]
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Shifted softmax mass, `ell` in arXiv:2603.15031v1, Algorithm 1. -/
noncomputable def shiftedMass (scores : ι → ℝ) (m : ℝ) : ℝ :=
  ∑ i, Real.exp (scores i - m)

/-- Shifted, unnormalized value numerator, `o` in Algorithm 1.
Its shift can be a maximum, or any real number in exact arithmetic. -/
noncomputable def shiftedNumerator (scores : ι → ℝ) (values : ι → E) (m : ℝ) : E :=
  weightedSum (fun i => Real.exp (scores i - m)) values

/-- A nonempty source group always has positive shifted mass. Source:
arXiv:2603.15031v1, §4.2, the softmax statistics in Algorithm 1. -/
theorem shiftedMass_pos [Nonempty ι] (scores : ι → ℝ) (m : ℝ) :
    0 < shiftedMass scores m := by
  exact Finset.sum_pos (fun i _ => Real.exp_pos (scores i - m)) Finset.univ_nonempty

/-- Changing the reference maximum rescales a group's mass.
Source: arXiv:2603.15031v1, §4.2, online softmax merge. -/
theorem rescale_mass (scores : ι → ℝ) (m M : ℝ) :
    Real.exp (m - M) * shiftedMass scores m = shiftedMass scores M := by
  unfold shiftedMass
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [← Real.exp_add]
  congr 1
  ring

/-- Changing the reference maximum rescales the unnormalized numerator.
Source: arXiv:2603.15031v1, §4.2, online softmax merge. -/
theorem rescale_numerator (scores : ι → ℝ) (values : ι → E) (m M : ℝ) :
    Real.exp (m - M) • shiftedNumerator scores values m =
      shiftedNumerator scores values M := by
  unfold shiftedNumerator weightedSum
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [← mul_smul, ← Real.exp_add]
  congr 2
  ring

/-- The attention output equals the unnormalized numerator divided by its
shifted mass. Source: arXiv:2603.15031v1, §4.2, Algorithm 1.
This also fixes the meaning of the algorithm's `o` and `ell` statistics. -/
theorem attend_from_statistics [Nonempty ι] (scores : ι → ℝ)
    (values : ι → E) (m : ℝ) :
    attend scores values = (shiftedMass scores m)⁻¹ • shiftedNumerator scores values m := by
  have hm : shiftedMass scores m = depthPartition scores / Real.exp m := by
    simp only [shiftedMass, depthPartition, Real.exp_sub, Finset.sum_div]
  have hw : ∀ i, depthWeight scores i = Real.exp (scores i - m) / shiftedMass scores m := by
    intro i
    rw [depthWeight, Real.exp_sub, hm]
    field_simp
  unfold attend shiftedNumerator weightedSum
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [hw, div_eq_mul_inv, mul_smul]
  exact smul_comm _ _ _

/-- Mass for the union of two disjoint source groups is additive.
Source: arXiv:2603.15031v1, §4.2, Phase 1 plus Phase 2 sources. -/
theorem shiftedMass_union (s : ι → ℝ) (t : κ → ℝ) (m : ℝ) :
    shiftedMass (Sum.elim s t) m = shiftedMass s m + shiftedMass t m := by
  unfold shiftedMass
  rw [Fintype.sum_sum_type]
  rfl

/-- Unnormalized numerators for disjoint source groups are additive.
Source: arXiv:2603.15031v1, §4.2, Phase 1 plus Phase 2 values. -/
theorem shiftedNumerator_union (s : ι → ℝ) (t : κ → ℝ)
    (v : ι → E) (w : κ → E) (m : ℝ) :
    shiftedNumerator (Sum.elim s t) (Sum.elim v w) m =
      shiftedNumerator s v m + shiftedNumerator t w m := by
  unfold shiftedNumerator weightedSum
  rw [Fintype.sum_sum_type]
  rfl

/-- The online merge uses a common maximum and rescales both groups'
unnormalized statistics. Source: arXiv:2603.15031v1, Algorithm 1.
For Block AttnRes the second group consists of one partial-block source. -/
noncomputable def onlineMerge (s : ι → ℝ) (t : κ → ℝ)
    (v : ι → E) (w : κ → E) (m₁ m₂ : ℝ) : E :=
  let m := max m₁ m₂
  let a := Real.exp (m₁ - m)
  let b := Real.exp (m₂ - m)
  (a * shiftedMass s m₁ + b * shiftedMass t m₂)⁻¹ •
    (a • shiftedNumerator s v m₁ + b • shiftedNumerator t w m₂)

/-- The online merge denominator is positive even if its second source
group is empty. Block AttnRes always has a nonempty first group because
the token embedding is retained. Source: arXiv:2603.15031v1, §4.2,
Algorithm 1. Division is therefore well-defined in exact real arithmetic. -/
theorem online_denominator_pos [Nonempty ι] (s : ι → ℝ) (t : κ → ℝ)
    (m₁ m₂ : ℝ) :
    0 < Real.exp (m₁ - max m₁ m₂) * shiftedMass s m₁ +
      Real.exp (m₂ - max m₁ m₂) * shiftedMass t m₂ := by
  have hp := shiftedMass_pos s m₁
  have hn : 0 ≤ shiftedMass t m₂ := by
    exact Finset.sum_nonneg fun i _ => (Real.exp_pos (t i - m₂)).le
  positivity

/-- Two-phase evaluation is exactly full attention over the same sources.
The first phase's scores may be precomputed because its query and stored
block keys are independent of the sequential current-block computation.
Source: arXiv:2603.15031v1, §4.2, Algorithm 1. -/
theorem onlineMerge_eq_attend [Nonempty ι] (s : ι → ℝ) (t : κ → ℝ)
    (v : ι → E) (w : κ → E) (m₁ m₂ : ℝ) :
    onlineMerge s t v w m₁ m₂ = attend (Sum.elim s t) (Sum.elim v w) := by
  unfold onlineMerge
  dsimp only
  rw [rescale_mass, rescale_mass, rescale_numerator, rescale_numerator]
  rw [← shiftedMass_union, ← shiftedNumerator_union]
  exact (attend_from_statistics _ _ _).symm

/-- A singleton second group gives exactly the partial-block merge used by
Block AttnRes. Source: arXiv:2603.15031v1, Algorithm 1, Phase 2. -/
theorem merge_partial_block [Nonempty ι] (scores : ι → ℝ) (values : ι → E)
    (score : ℝ) (p : E) (m : ℝ) :
    onlineMerge scores (fun _ : Unit => score) values (fun _ => p) m score =
      attend (Sum.elim scores (fun _ : Unit => score))
        (Sum.elim values (fun _ => p)) := by
  apply onlineMerge_eq_attend

end Transformer.AttentionResiduals

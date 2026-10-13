import Transformer.Grokking.Geometry.Decisions

/-!
# A balanced incorrect classifier with perfect structural energy

Source: Nanda et al., arXiv:2301.05217v1, section 5.1, keeps correctness
checks through restricted loss. Counterclaim investigated: perfect
invariant energy in the fixed-orbit observer at 4436290, even after
removing row shifts and global class bias, certifies a correct rule.

Two equally sized cells have different target labels. A binary readout
swaps both labels and is constant within each cell. Each row's class mean
and each class's global input mean are zero, so both centering operations
leave it unchanged. Every cell has two points, total energy is positive,
and the structural fraction is one, but every cell's answer is wrong.

The readout is not input independent. This is a finite-logit counterexample
to a metric interpretation, not a claim that GPTMini weights or native
AdamW generate this particular classifier. It explains the need for a
correct margin or restricted loss alongside structural measurements.
-/

namespace Transformer.Grokking.Geometry

open scoped BigOperators
open Transformer.Grokking.NaiveLoss

/-- Balanced binary scores for the opposite of the cell's target label.
Source: the counterexample to symmetry-only success in the fixed-orbit
adaptation of arXiv:2301.05217v1, section 5.1. -/
def swappedScores (q k : Bool) : ℝ := if k = q then -1 else 1

/-- Both removed biases are already zero. Source: the centering policy
at 4436290, adapting restricted logits in arXiv:2301.05217v1, section 5.1.
Thus the counterexample is not eliminated by discarding constant logits. -/
theorem swapped_scores_bias_free :
    (∀ q : Bool, meanOver Finset.univ (swappedScores q) = 0) ∧
    (∀ k : Bool, meanOver Finset.univ (fun q => meanOver Finset.univ
      (fun _ : Bool => swappedScores q k)) = 0) := by
  constructor
  · intro q
    cases q <;> norm_num [meanOver, swappedScores, Fintype.sum_bool]
  · intro k
    cases k <;> norm_num [meanOver, swappedScores, Fintype.sum_bool]

/-- Averaging the actual cell points preserves these incorrect scores.
Source: fixed-cell averaging at 4436290, adapting arXiv:2301.05217v1,
section 5.1; no orthogonality premise is supplied. -/
theorem swapped_scores_cell_mean (q k : Bool) :
    cellMean (fun _ : Bool => (Finset.univ : Finset Bool))
      (fun q _ k => swappedScores q k) q k = swappedScores q k := by
  unfold cellMean
  apply meanOver_const
  exact ⟨false, Finset.mem_univ false⟩

/-- The selected logits have eight units of energy and zero residual.
Source: the energy policy at 4436290, adapting arXiv:2301.05217v1,
section 5.1; the fraction is therefore defined at positive energy. -/
theorem swapped_scores_energy :
    cellTotalEnergy Finset.univ Finset.univ
      (fun _ : Bool => (Finset.univ : Finset Bool)) (fun q _ k => swappedScores q k) = 8 ∧
    cellResidualEnergy Finset.univ Finset.univ
      (fun _ : Bool => (Finset.univ : Finset Bool)) (fun q _ k => swappedScores q k) = 0 := by
  constructor
  · norm_num [cellTotalEnergy, energyOver, swappedScores, Fintype.sum_bool]
  · norm_num [cellResidualEnergy, residualOver, energyOver, meanOver,
      swappedScores, Fintype.sum_bool]

/-- Each cell strictly predicts the opposite label and fails its own
target. Source: the correctness check required in arXiv:2301.05217v1,
section 5.1; this failure is not an argmax tie. -/
theorem swapped_scores_wrong (q : Bool) :
    ¬StrictCorrect (swappedScores q) q ∧ StrictCorrect (swappedScores q) (!q) := by
  constructor
  · intro h
    have he := h (!q) (Bool.not_ne_self q)
    cases q <;> norm_num [swappedScores] at he
  · intro k hk
    cases q <;> cases k
    · norm_num [swappedScores]
    · exact False.elim (hk rfl)
    · exact False.elim (hk rfl)
    · norm_num [swappedScores]

/-- Each correct target loses to its opposite by two logit units.
Source: the missing correctness condition in the symmetry-only inference
for the adaptation of arXiv:2301.05217v1, section 5.1, at 4436290. -/
theorem swapped_scores_target_gap (q : Bool) :
    swappedScores q q - swappedScores q (!q) = -2 := by
  cases q <;> norm_num [swappedScores]

/-- A perfect structural fraction cannot supply the positive-margin
premise needed by the cleanup certificate. Source: the counterexample
for the adaptation of arXiv:2301.05217v1, section 5.1, at 4436290. -/
theorem swapped_scores_no_positive_margin (q : Bool) (margin : ℝ) (hm : 0 < margin) :
    ¬MarginAtLeast (swappedScores q) q margin := by
  intro h
  have hg := h (!q) (Bool.not_ne_self q)
  rw [swapped_scores_target_gap] at hg
  linarith

example : 0 < (1 : ℝ) := by norm_num

/-- Perfect invariant energy can coexist with every projected answer
being wrong. Source: counterexample to symmetry-only success for the
orbit observer at 4436290, adapting arXiv:2301.05217v1, section 5.1. -/
theorem perfect_fraction_with_every_answer_wrong :
    cellEnergyFraction Finset.univ Finset.univ
      (fun _ : Bool => (Finset.univ : Finset Bool)) (fun q _ k => swappedScores q k) = 1 ∧
    ∀ q : Bool, ¬StrictCorrect
      (cellMean (fun _ : Bool => (Finset.univ : Finset Bool)) (fun q _ k => swappedScores q k) q) q := by
  obtain ⟨ht, hr⟩ := swapped_scores_energy
  have hp : 0 < cellTotalEnergy Finset.univ Finset.univ
      (fun _ : Bool => (Finset.univ : Finset Bool)) (fun q _ k => swappedScores q k) := by
    rw [ht]
    norm_num
  constructor
  · exact (cell_fraction_one_iff _ _ _ _ hp).mpr hr
  · intro q
    have he : cellMean (fun _ : Bool => (Finset.univ : Finset Bool))
        (fun q _ k => swappedScores q k) q = swappedScores q := by
      funext k
      exact swapped_scores_cell_mean q k
    rw [he]
    exact (swapped_scores_wrong q).1

/-- Positive energy, complete two-point coverage and both removed biases
being zero still do not certify correctness from a fraction of one.
Source: the numerical guards at 4436290, adapting the structural measures
of arXiv:2301.05217v1, section 5.1; no dynamics restriction is imposed. -/
theorem balanced_wrong_geometry_witness :
    (∀ q : Bool, meanOver Finset.univ (swappedScores q) = 0) ∧
    (∀ k : Bool, meanOver Finset.univ (fun q => meanOver Finset.univ
      (fun _ : Bool => swappedScores q k)) = 0) ∧
    2 ≤ (Finset.univ : Finset Bool).card ∧
    0 < cellTotalEnergy Finset.univ Finset.univ
      (fun _ : Bool => (Finset.univ : Finset Bool)) (fun q _ k => swappedScores q k) ∧
    cellEnergyFraction Finset.univ Finset.univ
      (fun _ : Bool => (Finset.univ : Finset Bool)) (fun q _ k => swappedScores q k) = 1 ∧
    ∀ q : Bool, ¬StrictCorrect
      (cellMean (fun _ : Bool => (Finset.univ : Finset Bool)) (fun q _ k => swappedScores q k) q) q := by
  obtain ⟨hrow, hclass⟩ := swapped_scores_bias_free
  have ht := swapped_scores_energy.1
  obtain ⟨hf, hw⟩ := perfect_fraction_with_every_answer_wrong
  refine ⟨hrow, hclass, ?_, ?_, hf, hw⟩
  · norm_num
  · rw [ht]
    norm_num

end Transformer.Grokking.Geometry

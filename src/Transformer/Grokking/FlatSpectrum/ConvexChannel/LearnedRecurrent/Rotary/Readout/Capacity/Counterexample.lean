module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Witness

/-!
# Zero fixed-weight gap can coexist with strict joint CE improvement

Source: `review/flat_spectrum.md`, CM104. The full actual CE decreases on two
opposite ordered prefixes when capacity moves between retained components.
This counterexample concerns conditional training optimization. It supplies
neither unseen answers nor a claim about benchmark accuracy.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity

open Pair Optimality

/-- Moving capacity raises the actual probability of each original target,
CM104; both original classes remain in the softmax denominator. -/
theorem moved_probability (b : Bool) :
    Geometry.probability (jointScore allocationModel parked [b, !b]) b <
      Geometry.probability (jointScore allocationModel moved [b, !b]) b := by
  have he : 1 < Real.exp 16 := Real.one_lt_exp_iff.mpr (by norm_num)
  have hp : (1 / 2 : ℝ) < Real.exp 16 / (Real.exp 16 + 1) := by
    apply (div_lt_div_iff₀ (by norm_num) (by positivity)).mpr
    linarith
  simp only [Geometry.probability, parked_scores, moved_scores, Fintype.sum_bool]
  cases b <;> simpa [add_comm, one_add_one_eq_two] using hp

/-- The same-budget capacity reallocation strictly lowers actual ordinary CE
on each original ordered record, CM104. -/
theorem moved_ce (b : Bool) :
    Landscape.Neural.crossEntropy (jointScore allocationModel moved [b, !b]) b <
      Landscape.Neural.crossEntropy (jointScore allocationModel parked [b, !b]) b := by
  rw [Geometry.crossEntropy_eq_neg_log_probability,
    Geometry.crossEntropy_eq_neg_log_probability]
  have hp : 0 < Geometry.probability (jointScore allocationModel parked [b, !b]) b := by
    norm_num [Geometry.probability, parked_scores]
  have h := Real.log_lt_log hp (moved_probability b)
  linarith

/-- Strict improvement holds for the entire indexed mean, CM104. Neither
the classes nor the records nor the retained state functions change. -/
theorem moved_mean : jointLoss 1 allocationModel allocationPrefix allocationTarget moved <
    jointLoss 1 allocationModel allocationPrefix allocationTarget parked := by
  have h₁ := moved_ce true
  have h₂ := moved_ce false
  unfold jointLoss meanLoss
  rw [Fin.sum_univ_two, Fin.sum_univ_two]
  norm_num only [Nat.cast_one]
  change (1 / 2 : ℝ) *
      (Landscape.Neural.crossEntropy (jointScore allocationModel moved [true, false]) true +
        Landscape.Neural.crossEntropy (jointScore allocationModel moved [false, true]) false) <
    (1 / 2 : ℝ) *
      (Landscape.Neural.crossEntropy (jointScore allocationModel parked [true, false]) true +
        Landscape.Neural.crossEntropy (jointScore allocationModel parked [false, true]) false)
  norm_num only [Bool.not_true, Bool.not_false] at h₁ h₂
  linarith

/-- The larger gap detects improvement that the old fixed-weight gap misses,
CM104, using strict actual CE improvement rather than a postulated coefficient. -/
theorem parked_joint_gap_pos :
    0 < jointGap 1 16 allocationModel allocationPrefix allocationTarget parked := by
  have h := joint_gap_lower 1 16 allocationModel allocationPrefix allocationTarget
    parked moved moved_mem
  have h' := moved_mean
  linarith

/-- Concrete CM104 counterexample to treating a zero fixed-weight gap as an
obstruction to reallocation: feasible nonzero complex rows, zero old gap,
positive joint gap and strictly lower actual training CE. -/
theorem fixed_gap_misses_reallocation :
    parked ∈ jointDomain 16 ∧ moved ∈ jointDomain 16 ∧
    gap 1 16 (fun j : Bool => if j then 0 else 1) allocationModel
      allocationPrefix allocationTarget allocationRows = 0 ∧
    0 < jointGap 1 16 allocationModel allocationPrefix allocationTarget parked ∧
    jointLoss 1 allocationModel allocationPrefix allocationTarget moved <
      jointLoss 1 allocationModel allocationPrefix allocationTarget parked :=
  ⟨parked_mem, moved_mem, fixed_gap_zero, parked_joint_gap_pos, moved_mean⟩

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity

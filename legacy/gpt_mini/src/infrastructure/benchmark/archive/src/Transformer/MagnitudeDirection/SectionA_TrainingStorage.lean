/-
# Softplus storage for the training correction

User-requested convergence extension of arXiv:2606.25971v2, §3.1 and
Appendix A, Algorithm 2. A rejected fused step is replaced in weight
space. Rescaling all effective row gains by one positive factor restores
the fixed direction norm, preserves their ratios and keeps column gains.
The model's weight is unchanged by this storage repair.
-/

import Transformer.MagnitudeDirection.SectionA_FullAlgorithm
import Transformer.MagnitudeDirection.Section3_Representation

noncomputable section

namespace Transformer.MagnitudeDirection

variable {m n : ℕ}

/-- Unit effective row/column gains for fused initialization,
arXiv:2606.25971v2, §4.1.3 and Appendix A, Algorithm 2. -/
def unitGainState (W : Matrix (Fin m) (Fin n) ℝ) : FullState m n :=
  ⟨W, fun _ => Real.log (Real.exp 1 - 1), fun _ => Real.log (Real.exp 1 - 1)⟩

/-- Unit initialization recovers every matrix entry exactly,
arXiv:2606.25971v2, §4.1.3 and Appendix A, Algorithm 2, line 2. -/
theorem unitGainState_direction (W : Matrix (Fin m) (Fin n) ℝ) :
    fullDirection softplus (unitGainState W) = W := by
  simp only [fullDirection, unitGainState, softplus_unit_initialization, unfuse_one]

/-- The raw softplus inverse also recovers the original raw parameter.
Source: arXiv:2606.25971v2, §4.1.3, storage-repair extension. -/
theorem softplus_inverse_left (x : ℝ) :
    Real.log (Real.exp (softplus x) - 1) = x := by
  rw [softplus, Real.exp_log (by positivity : 0 < 1 + Real.exp x)]
  simp

/-- A nonzero fused matrix has a nonzero recovered direction with positive
softplus gains. Source: arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem fullDirection_ne_zero (s : FullState m n) (hW : s.weight ≠ 0) :
    fullDirection softplus s ≠ 0 := by
  intro hD
  have hf := fuse_unfuse (fun i => softplus (s.rawRow i))
    (fun j => softplus (s.rawCol j)) s.weight
    (fun i => (softplus_pos _).ne') (fun j => (softplus_pos _).ne')
  change fuse _ _ (fullDirection softplus s) = s.weight at hf
  rw [hD] at hf
  apply hW
  exact hf.symm.trans (by ext i j; simp [fuse])

/-- Nonzero fused storage exists, arXiv:2606.25971v2, Appendix A. -/
example : (unitGainState (1 : Matrix (Fin 1) (Fin 1) ℝ)).weight ≠ 0 := by norm_num [unitGainState]

/-- Restore the direction sphere after a replacement fused step. Every row
gain is multiplied by `norm(old direction)/c`; all column gains and fused
entries are preserved. This is an added algorithm correction.
Source: extension of arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
def rebalanceStorage (c : ℝ) (s : FullState m n) : FullState m n :=
  let a := frobeniusNorm (fullDirection softplus s) / c
  ⟨s.weight, fun i => Real.log (Real.exp (a * softplus (s.rawRow i)) - 1), s.rawCol⟩

/-- Storage repair preserves the exact model weight and column parameters,
arXiv:2606.25971v2, §3.1, fused-weight training extension. -/
theorem rebalanceStorage_preserves (c : ℝ) (s : FullState m n) :
    (rebalanceStorage c s).weight = s.weight ∧
      (rebalanceStorage c s).rawCol = s.rawCol := ⟨rfl, rfl⟩

/-- Effective row gains are changed by one positive common factor;
fine-grained row ratios are preserved by the repair. Source: extension
of arXiv:2606.25971v2, Appendix A, Algorithm 2 and §4.1.3. -/
theorem rebalanceStorage_row (c : ℝ) (s : FullState m n)
    (hc : 0 < c) (hW : s.weight ≠ 0) (i : Fin m) :
    softplus ((rebalanceStorage c s).rawRow i) =
      frobeniusNorm (fullDirection softplus s) / c * softplus (s.rawRow i) := by
  have ha := div_pos ((frobeniusNorm_pos_iff _).mpr (fullDirection_ne_zero s hW)) hc
  exact softplus_inverse _ (mul_pos ha (softplus_pos _))

/-- Positive radius and nonzero storage satisfy the repair premises,
arXiv:2606.25971v2, Appendix A, training extension. -/
example : (0 : ℝ) < 1 ∧ (unitGainState (1 : Matrix (Fin 1) (Fin 1) ℝ)).weight ≠ 0 := by
  norm_num [unitGainState]

/-- The recovered repaired direction is exactly the radial projection.
This bridges fused-space convergence to the actual MD representation.
Source: extension of arXiv:2606.25971v2, Appendix A, Algorithm 2, lines 2, 8. -/
theorem rebalanceStorage_direction (c : ℝ) (s : FullState m n)
    (hc : 0 < c) (hW : s.weight ≠ 0) :
    fullDirection softplus (rebalanceStorage c s) =
      matrixProject c (fullDirection softplus s) := by
  have hn := (frobeniusNorm_pos_iff _).mpr (fullDirection_ne_zero s hW)
  ext i j
  change s.weight i j /
    (softplus ((rebalanceStorage c s).rawRow i) * softplus (s.rawCol j)) = _
  rw [rebalanceStorage_row c s hc hW]
  dsimp [matrixProject, fullDirection, unfuse]
  field_simp [(softplus_pos (s.rawRow i)).ne', (softplus_pos (s.rawCol j)).ne',
    hn.ne', hc.ne']

/-- Repair inputs are genuinely nonzero, arXiv:2606.25971v2, Appendix A. -/
example : (0 : ℝ) < 2 ∧ (unitGainState (1 : Matrix (Fin 1) (Fin 1) ℝ)).weight ≠ 0 := by
  norm_num [unitGainState]

/-- The repaired direction has the prescribed positive sphere radius,
without any assumption about the optimizer's gain outputs.
Source: extension of arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem rebalanceStorage_norm (c : ℝ) (s : FullState m n)
    (hc : 0 < c) (hW : s.weight ≠ 0) :
    frobeniusNorm (fullDirection softplus (rebalanceStorage c s)) = c := by
  rw [rebalanceStorage_direction c s hc hW]
  exact matrixProject_norm c _ hc.le (fullDirection_ne_zero s hW)

/-- The norm-restoration assumptions are satisfiable,
arXiv:2606.25971v2, Appendix A, training extension. -/
example : (0 : ℝ) < 1 ∧ (unitGainState (1 : Matrix (Fin 1) (Fin 1) ℝ)).weight ≠ 0 := by
  norm_num [unitGainState]

/-- An already valid MD state is preserved, including both raw gain vectors.
Thus an accepted paper step is kept exactly by the correction.
Source: extension of arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem rebalanceStorage_fixed (c : ℝ) (s : FullState m n)
    (hc : 0 < c) (hD : frobeniusNorm (fullDirection softplus s) = c) :
    rebalanceStorage c s = s := by
  cases s with
  | mk W row col =>
    simp only [rebalanceStorage, hD, div_self hc.ne', one_mul, softplus_inverse_left]

/-- Valid positive-sphere storage exists, arXiv:2606.25971v2, Appendix A. -/
example : (0 : ℝ) < 1 ∧
    frobeniusNorm (fullDirection softplus (unitGainState (1 : Matrix (Fin 1) (Fin 1) ℝ))) = 1 := by
  rw [unitGainState_direction, frobeniusNorm_eq_sqrt]
  norm_num

end Transformer.MagnitudeDirection

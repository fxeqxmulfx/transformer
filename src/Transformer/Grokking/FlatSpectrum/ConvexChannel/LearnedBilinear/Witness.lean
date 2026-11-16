module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Geometry

/-!
# Both learned encoders change their observable object geometry

Source: `review/flat_spectrum.md`, CM86. Two positive 2-by-2 matrices give
different covariances in both actual recovered encoders, with readout 5 and 10.
A separate exact comparison shows that the imaginary learned weights affect
the real output. These are feasible parameter comparisons, not training runs.
The raw three-object expressivity witness is in `Parity.lean`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Witness

open scoped BigOperators Matrix ComplexOrder MatrixOrder Matrix.Norms.Elementwise

/-- The complex row used in the inhabited two-object geometry witness, CM86. -/
def row : Matrix Unit Bool ℂ := fun _ o => if o then Complex.I else 1

/-- A nonnegative diagonal change in the joint state, CM86. -/
def diagonalChange (b o : Bool) : ℂ := if o = b then (if b then 8 else 3) else 0

/-- The two complete complex matrices [[4,i],[-i,1]] and [[1,i],[-i,9]],
CM86. They are joint parameters, not supplied data or labels. -/
def joint (b : Bool) : Matrix Bool Bool ℂ :=
  row.conjTranspose * row + Matrix.diagonal (diagonalChange b)

/-- Both joint points are positive, inhabiting the covariance hypotheses,
CM86. The model itself permits all rectangular matrices. -/
theorem joint_psd (b : Bool) : (joint b).PosSemidef := by
  apply (Matrix.posSemidef_conjTranspose_mul_self row).add
  apply Matrix.PosSemidef.diagonal
  intro o
  cases o <;> cases b <;> norm_num [diagonalChange]

/-- The actual learned readout magnitude changes from 5 to 10, CM86. -/
theorem readout_joint (b : Bool) : readout (joint b) = if b then 10 else 5 := by
  have h := nuclearNorm_eq_trace_of_mul_self (X := joint b) (joint_psd b)
    (by rw [(joint_psd b).isHermitian.eq])
  have hr := congrArg Complex.re h
  cases b <;> norm_num [readout, joint, row, diagonalChange, Matrix.trace,
    Matrix.diag, Matrix.mul_apply, Matrix.conjTranspose_apply, Matrix.diagonal] at hr ⊢ <;>
    exact hr

/-- Both actual normalized encoder covariances equal the changed joint
geometry divided by its learned magnitude, CM86. -/
theorem representation_covariances (b : Bool) :
    leftRepresentation (joint b) * (leftRepresentation (joint b)).conjTranspose =
        (if b then (10 : ℝ)⁻¹ else 5⁻¹) • joint b ∧
    rightRepresentation (joint b) * (rightRepresentation (joint b)).conjTranspose =
        (if b then (10 : ℝ)⁻¹ else 5⁻¹) • joint b := by
  have h := positive_representation_covariance (joint_psd b)
  cases b <;> simpa only [readout_joint, Bool.false_eq_true, ↓reduceIte] using h

/-- The relative object energies change in both learned representations,
CM86. This observable cannot be changed by a hidden unitary rotation alone. -/
theorem covariance_entries :
    (leftRepresentation (joint false) * (leftRepresentation (joint false)).conjTranspose)
        false false = 4 / 5 ∧
    (leftRepresentation (joint true) * (leftRepresentation (joint true)).conjTranspose)
        false false = 1 / 10 ∧
    (rightRepresentation (joint false) * (rightRepresentation (joint false)).conjTranspose)
        false false = 4 / 5 ∧
    (rightRepresentation (joint true) * (rightRepresentation (joint true)).conjTranspose)
        false false = 1 / 10 := by
  rw [(representation_covariances false).1, (representation_covariances true).1,
    (representation_covariances false).2, (representation_covariances true).2]
  norm_num [joint, row, diagonalChange, Matrix.mul_apply, Matrix.conjTranspose_apply,
    Matrix.diagonal, Complex.real_smul]

/-- The actual learned query geometry changes, CM86. -/
theorem left_geometry_changes :
    leftRepresentation (joint false) * (leftRepresentation (joint false)).conjTranspose ≠
      leftRepresentation (joint true) * (leftRepresentation (joint true)).conjTranspose := by
  intro h
  have he := congrArg (fun M : Matrix Bool Bool ℂ => (M false false).re) h
  rw [covariance_entries.1, covariance_entries.2.1] at he
  norm_num at he

/-- The actual learned context geometry changes as well, CM86. -/
theorem right_geometry_changes :
    rightRepresentation (joint false) * (rightRepresentation (joint false)).conjTranspose ≠
      rightRepresentation (joint true) * (rightRepresentation (joint true)).conjTranspose := by
  intro h
  have he := congrArg (fun M : Matrix Bool Bool ℂ => (M false false).re) h
  rw [covariance_entries.2.2.1, covariance_entries.2.2.2] at he
  norm_num at he

/-- Removing the imaginary learned entries changes the actual real output
from -1 to 0 on the same complex encoder inputs, CM86. -/
theorem imaginary_entries_contribute :
    (forward (joint false) (Pi.single false 1) (Pi.single true 1)).re = -1 ∧
      (forward (Matrix.diagonal (fun b : Bool => if b then (1 : ℂ) else 4))
        (Pi.single false 1) (Pi.single true 1)).re = 0 := by
  simp [forward_eq, joint, row, diagonalChange, Matrix.mulVec, dotProduct,
    Matrix.mul_apply, Matrix.conjTranspose_apply, Matrix.diagonal]

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Witness

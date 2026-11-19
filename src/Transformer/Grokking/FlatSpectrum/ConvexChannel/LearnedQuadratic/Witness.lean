module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedQuadratic.Sequence

/-!
# A raw-sequence witness changes representations, readouts and predictions

Source: `review/convex/moments.md`, CM85. Two feasible PSD points use nonzero
imaginary learned weights. Their balanced representations and readouts change,
their original-object predictions reverse, and both desired answer CEs fall.
These are exact feasible comparisons, not an optimizer trajectory or a scored
stress-test run. The original inputs are the one-object Boolean sequences.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedQuadratic.Witness

open scoped BigOperators Matrix ComplexOrder MatrixOrder Matrix.Norms.Elementwise
open LearnedHinge (Coordinate Prefix)

abbrev D := Coordinate Bool 1

/-- A physical complex row used to construct the two inhabited points, CM85. -/
def row : Matrix Unit D ℂ := fun _ d =>
  match d with | none => 0 | some (_, o) => if o then Complex.I else 1

/-- The changing nonnegative diagonal, CM85. It is not input or target data. -/
def diagonalChange (b : Bool) (d : D) : ℂ :=
  match d with | none => 0 | some (_, o) => if o = b then (if b then 8 else 3) else 0

/-- A positive physical factor with a complex off-diagonal entry, CM85. -/
def physical (b : Bool) : Matrix D D ℂ :=
  row.conjTranspose * row + Matrix.diagonal (diagonalChange b)

/-- Both full physical factors are positive semidefinite, CM85. -/
theorem physical_psd (b : Bool) : (physical b).PosSemidef := by
  apply (Matrix.posSemidef_conjTranspose_mul_self row).add
  apply Matrix.PosSemidef.diagonal
  intro d
  fin_cases d <;> cases b <;> norm_num [diagonalChange]

/-- Exact ambient norms determine distinct learned readout amplitudes, CM85.
The ambient matrix norm is the full entrywise supremum norm. -/
theorem physical_norm (b : Bool) : ‖physical b‖ = if b then 9 else 4 := by
  apply le_antisymm
  · apply (Matrix.norm_le_iff (by cases b <;> norm_num)).mpr
    intro d e
    fin_cases d <;> fin_cases e <;> cases b <;>
      norm_num [physical, row, diagonalChange, Matrix.mul_apply,
        Matrix.conjTranspose_apply, Matrix.diagonal]
  · have h := Matrix.norm_entry_le_entrywise_sup_norm (physical b)
      (i := some (0, b)) (j := some (0, b))
    cases b <;> norm_num [physical, row, diagonalChange, Matrix.mul_apply,
      Matrix.conjTranspose_apply, Matrix.diagonal] at h ⊢ <;> exact h

/-- The whole joint matrix for the positive branch, CM85. -/
def joint (b : Bool) : Matrix D D ℂ := (physical b).conjTranspose * physical b

/-- The canonical recovery returns these actual complex physical factors,
CM85, making both internal stages inspectable. -/
theorem factor_joint (b : Bool) : factor (joint b) = physical b :=
  factor_of_gram (physical_psd b)

/-- Learned readout and two distinct representation coordinates change,
CM85. The imaginary entry is retained in both representations. -/
theorem stages_change :
    LearnedHinge.amplitude (factor (joint false)) = 2 ∧
    LearnedHinge.amplitude (factor (joint true)) = 3 ∧
    LearnedHinge.representation (factor (joint false)) (some (0, false)) (some (0, false)) = 2 ∧
    LearnedHinge.representation (factor (joint true)) (some (0, false)) (some (0, false)) = 1 / 3 ∧
    LearnedHinge.representation (factor (joint false)) (some (0, false)) (some (0, true)) =
      Complex.I / 2 ∧
    LearnedHinge.representation (factor (joint true)) (some (0, false)) (some (0, true)) =
      Complex.I / 3 := by
  have ha : LearnedHinge.amplitude (factor (joint false)) = 2 := by
    norm_num [LearnedHinge.amplitude, factor_joint, physical_norm]
  have hb : LearnedHinge.amplitude (factor (joint true)) = 3 := by
    norm_num [LearnedHinge.amplitude, factor_joint, physical_norm]
  have hra : LearnedHinge.representation (factor (joint false)) = (1 / 2 : ℝ) • physical false := by
    rw [LearnedHinge.representation, ha, factor_joint]
    norm_num
  have hrb : LearnedHinge.representation (factor (joint true)) = (1 / 3 : ℝ) • physical true := by
    rw [LearnedHinge.representation, hb, factor_joint]
    norm_num
  refine ⟨ha, hb, ?_⟩
  rw [hra, hrb]
  norm_num [physical, row, diagonalChange, Matrix.mul_apply, Matrix.conjTranspose_apply,
    Matrix.diagonal, div_eq_mul_inv, mul_comm]

/-- The internal representations are not related by any common complex
scale, CM85. The learned direction changes as well as the readout magnitude. -/
theorem representations_not_proportional :
    ¬ ∃ z : ℂ, LearnedHinge.representation (factor (joint true)) =
      z • LearnedHinge.representation (factor (joint false)) := by
  rintro ⟨z, hz⟩
  have hd := congrArg (fun W : Matrix D D ℂ => W (some (0, false)) (some (0, false))) hz
  have hi := congrArg (fun W : Matrix D D ℂ => W (some (0, false)) (some (0, true))) hz
  simp only [Matrix.smul_apply, smul_eq_mul, stages_change.2.2.1,
    stages_change.2.2.2.1] at hd
  simp only [Matrix.smul_apply, smul_eq_mul, stages_change.2.2.2.2.1,
    stages_change.2.2.2.2.2] at hi
  have hd' := congrArg Complex.re hd
  have hi' := congrArg Complex.im hi
  norm_num at hd' hi'
  linarith

/-- Both original output objects and readout signs remain present, CM85. -/
noncomputable def parameters (b : Bool) : Parameters D Bool :=
  fun c s => if c then (if s then joint b else (4 : ℝ) • 1) else 0

/-- The changed points lie in the same label-independent full domain, CM85. -/
theorem parameters_mem (b : Bool) : parameters b ∈ domain := by
  intro c s
  cases c <;> cases s
  · exact Matrix.PosSemidef.zero
  · exact Matrix.PosSemidef.zero
  · exact Matrix.PosSemidef.one.smul (by norm_num : (0 : ℝ) ≤ 4)
  · exact Matrix.posSemidef_conjTranspose_mul_self (physical b)

/-- Actual immutable one-object input, CM85. -/
def tokens (x : Bool) : Prefix Bool 1 := ⟨[x], by simp⟩

/-- The complete recovered raw forward has these literal original-object
logits, CM85. Imaginary source coordinates contribute to the squared energies. -/
theorem raw_logits (b x c : Bool) : sequenceLogits (parameters b) (tokens x) c =
    if c then (if b then (if x then 74 else -6) else (if x then -6 else 9)) else 0 := by
  rw [sequenceLogits_eq, logits_eq (parameters_mem b)]
  cases b <;> cases x <;> cases c <;>
    norm_num [parameters, quadratic, inputRow, LearnedHinge.Projection.rawRow,
      tokens, joint, physical, row, diagonalChange, Matrix.mul_apply,
      Matrix.conjTranspose_apply, Matrix.diagonal, Matrix.mulVec, dotProduct,
      Fintype.sum_option, Fintype.sum_prod_type, Fin.sum_univ_succ]

/-- Removing the imaginary physical weights changes the actual recovered
network's energy, CM85. They are not unused annotations of a real forward. -/
theorem imaginary_contributes :
    network (factor (joint false)) (inputRow (tokens false)) = 17 ∧
    network ((physical false).map (fun z => (z.re : ℂ))) (inputRow (tokens false)) = 16 := by
  rw [factor_joint]
  simp only [network_eq]
  norm_num [energy, physical, row, diagonalChange, inputRow, tokens,
    LearnedHinge.Projection.rawRow, Matrix.map_apply, Matrix.mul_apply,
    Matrix.conjTranspose_apply, Matrix.diagonal, Matrix.mulVec, dotProduct,
    Fintype.sum_option, Fintype.sum_prod_type, Fin.sum_univ_succ, Complex.normSq]

/-- Relative preferences reverse on the two raw inputs, CM85; the change
cannot be achieved by a common positive confidence rescaling. -/
theorem predictions_reverse :
    sequenceLogits (parameters false) (tokens false) false <
      sequenceLogits (parameters false) (tokens false) true ∧
    sequenceLogits (parameters true) (tokens false) true <
      sequenceLogits (parameters true) (tokens false) false ∧
    sequenceLogits (parameters false) (tokens true) true <
      sequenceLogits (parameters false) (tokens true) false ∧
    sequenceLogits (parameters true) (tokens true) false <
      sequenceLogits (parameters true) (tokens true) true := by
  norm_num [raw_logits]

/-- Both original-object CEs strictly improve at the second feasible point,
CM85. No internal representation is a supervised target. -/
theorem losses_improve (x : Bool) :
    sequenceLoss (tokens x) x (parameters true) < sequenceLoss (tokens x) x (parameters false) := by
  unfold sequenceLoss
  rw [ConvexRecall.crossEntropy_relative, ConvexRecall.crossEntropy_relative]
  cases x <;> simp only [Fintype.sum_bool, raw_logits] <;> norm_num only [Bool.false_eq_true,
    Bool.true_eq_false, ↓reduceIte, sub_zero, sub_self, zero_sub, neg_neg, Real.exp_zero] <;>
    apply Real.log_lt_log (by positivity)
  · exact add_lt_add_left (Real.exp_lt_exp.mpr (by norm_num : (-6 : ℝ) < 9)) 1
  · exact add_lt_add_right (Real.exp_lt_exp.mpr (by norm_num : (-74 : ℝ) < 6)) 1

example := network_factor (Matrix.posSemidef_conjTranspose_mul_self (physical false))
  (inputRow (tokens true))
example := logits_combination (parameters_mem false) (parameters_mem true)
  (inputRow (tokens true)) (1 / 2) (1 / 2) (by norm_num) (by norm_num) (by norm_num)
example (x : Bool) : nextObject (parameters true) (tokens x) false = x := by
  apply nextObject_of_unique
  intro c hc
  cases x <;> cases c <;> norm_num [raw_logits] at *

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedQuadratic.Witness

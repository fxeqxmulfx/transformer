module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Basic
public import Mathlib.Analysis.Matrix.Order

/-!
# Learned squared complex responses and their exact joint Gram coordinates

Source: `review/convex/moments.md`, CM85. A complete positive semidefinite
matrix recovers a square-root factor, learned representations, and a learned
readout. Inference applies the representations before the squared-modulus
activation and the readout. The Gram identity is proved, not substituted as
the definition of that computation. There is no fixed rank constraint or
activation-mask enumeration. The recovered width is the full input dimension.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedQuadratic

open scoped BigOperators Matrix ComplexOrder MatrixOrder Matrix.Norms.Elementwise

variable {D H : Type*} [Fintype D] [Fintype H]

/-- Squared magnitudes of all actual complex hidden responses, CM85. -/
noncomputable def energy (W : Matrix H D ℂ) (x : D → ℂ) : ℝ :=
  ∑ h, Complex.normSq ((W *ᵥ x) h)

/-- The real quadratic form used to certify the learned computation, CM85. -/
noncomputable def quadratic (M : Matrix D D ℂ) (x : D → ℂ) : ℝ :=
  (star x ⬝ᵥ (M *ᵥ x)).re

/-- All hidden responses contribute to their exact Gram form, CM85.
This holds for arbitrary complex weights and arbitrary complex inputs. -/
theorem energy_gram (W : Matrix H D ℂ) (x : D → ℂ) :
    energy W x = quadratic (W.conjTranspose * W) x := by
  unfold quadratic
  rw [← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec,
    Matrix.vecMul_conjTranspose, star_star]
  simp only [dotProduct, Complex.re_sum, Pi.star_apply]
  apply Finset.sum_congr rfl
  intro h hh
  exact congrArg Complex.re (Complex.normSq_eq_conj_mul_self (z := (W *ᵥ x) h))

/-- Real rescaling of learned representations has degree two, CM85. -/
theorem energy_smul (a : ℝ) (W : Matrix H D ℂ) (x : D → ℂ) :
    energy (a • W) x = a ^ 2 * energy W x := by
  simp only [energy, Matrix.smul_mulVec, Pi.smul_apply, Complex.real_smul,
    Complex.normSq_mul, Complex.normSq_ofReal, ← Finset.mul_sum, pow_two]

/-- The complete learned two-stage energy, CM85. The readout is the square
of the balanced amplitude; the representation is recovered from the same W. -/
noncomputable def network (W : Matrix H D ℂ) (x : D → ℂ) : ℝ :=
  LearnedHinge.amplitude W ^ 2 * energy (LearnedHinge.representation W) x

/-- The actual balanced representation/activation/readout equals the
original factor energy at every input, including W = 0, CM85. -/
theorem network_eq (W : Matrix H D ℂ) (x : D → ℂ) :
    network W x = energy W x := by
  unfold network
  rw [← energy_smul, LearnedHinge.recover]

/-- Nonlinear factor recovery from the whole joint matrix, CM85. -/
noncomputable def factor [DecidableEq D] (M : Matrix D D ℂ) : Matrix D D ℂ :=
  CFC.sqrt M

/-- The recovered full-width factor has exactly the original PSD Gram
matrix, CM85. No low-rank approximation or unimplemented pricing oracle occurs. -/
theorem factor_gram [DecidableEq D] {M : Matrix D D ℂ} (hM : M.PosSemidef) :
    (factor M).conjTranspose * factor M = M := by
  have hs : (factor M).PosSemidef := Matrix.nonneg_iff_posSemidef.mp (CFC.sqrt_nonneg M)
  rw [hs.isHermitian.eq]
  exact CFC.sqrt_mul_sqrt_self M hM.nonneg

/-- The actual recovered learned path has the certified joint quadratic
output at every input, CM85. This is the convex-coordinate identity. -/
theorem network_factor [DecidableEq D] {M : Matrix D D ℂ} (hM : M.PosSemidef)
    (x : D → ℂ) : network (factor M) x = quadratic M x := by
  rw [network_eq, energy_gram, factor_gram hM]

/-- Every finite physical squared-response network maps to a feasible
joint point, CM85. The recovered width can differ from its physical width. -/
theorem physical_coverage [DecidableEq D] (W : Matrix H D ℂ) (x : D → ℂ) :
    network (factor (W.conjTranspose * W)) x = energy W x := by
  rw [network_factor (Matrix.posSemidef_conjTranspose_mul_self W), energy_gram]

/-- A positive Hermitian physical factor is its Gram matrix's recovered
factor, CM85. General physical factors need not be returned in the same gauge. -/
theorem factor_of_gram [DecidableEq D] {W : Matrix D D ℂ} (hW : W.PosSemidef) :
    factor (W.conjTranspose * W) = W := by
  unfold factor
  apply CFC.sqrt_unique
  · rw [hW.isHermitian.eq]
  · exact hW.nonneg

/-- Absorb each nonnegative learned physical readout into its row, CM85. -/
noncomputable def weightedFactors (a : H → ℝ) (W : Matrix H D ℂ) : Matrix H D ℂ :=
  fun h d => (Real.sqrt (a h) : ℂ) * W h d

/-- All independent positive physical readouts are retained exactly, CM85.
This does not assume equal amplitudes across the physical hidden rows. -/
theorem weighted_energy (a : H → ℝ) (ha : ∀ h, 0 ≤ a h) (W : Matrix H D ℂ) (x : D → ℂ) :
    energy (weightedFactors a W) x = ∑ h, a h * Complex.normSq ((W *ᵥ x) h) := by
  unfold energy
  apply Finset.sum_congr rfl
  intro h hh
  have hr : (weightedFactors a W *ᵥ x) h = (Real.sqrt (a h) : ℂ) * (W *ᵥ x) h := by
    simp only [Matrix.mulVec, dotProduct, weightedFactors, mul_assoc, Finset.mul_sum]
  rw [hr, Complex.normSq_mul, Complex.normSq_ofReal, Real.mul_self_sqrt (ha h)]

/-- Exact function coverage for arbitrary finite physical width and
nonnegative independent readouts, CM85. Signed outputs use the two branches. -/
theorem weighted_coverage [DecidableEq D] (a : H → ℝ) (ha : ∀ h, 0 ≤ a h)
    (W : Matrix H D ℂ) (x : D → ℂ) :
    network (factor ((weightedFactors a W).conjTranspose * weightedFactors a W)) x =
      ∑ h, a h * Complex.normSq ((W *ᵥ x) h) := by
  rw [physical_coverage, weighted_energy a ha]

/-- Quadratic readings are linear in every joint matrix coordinate, CM85.
The input is fixed data; the matrix is fully trained. -/
theorem quadratic_combination (M N : Matrix D D ℂ) (x : D → ℂ) (a b : ℝ) :
    quadratic (a • M + b • N) x = a * quadratic M x + b * quadratic N x := by
  simp only [quadratic, Matrix.add_mulVec, Matrix.smul_mulVec,
    dotProduct_add, dotProduct_smul]
  simp

example := factor_gram (D := Bool) (M := 0) Matrix.PosSemidef.zero
example := network_factor (D := Bool) (M := 0) Matrix.PosSemidef.zero
  (fun b => if b then Complex.I else 1)
example := factor_of_gram (D := Bool) (W := 0) Matrix.PosSemidef.zero
example := weighted_energy (H := Bool) (D := Bool) (fun h => if h then 2 else 3)
  (by intro h; cases h <;> norm_num) (fun h d => if h = d then 1 else Complex.I)
  (fun d => if d then 1 else Complex.I)
example := weighted_coverage (H := Bool) (D := Bool) (fun h => if h then 2 else 3)
  (by intro h; cases h <;> norm_num) (fun h d => if h = d then 1 else Complex.I)
  (fun d => if d then 1 else Complex.I)

end Transformer.Grokking.Convex.ConvexChannel.LearnedQuadratic

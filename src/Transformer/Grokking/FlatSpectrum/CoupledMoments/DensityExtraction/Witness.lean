module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction.Centering
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.FourierExtraction.Witness

/-!
# Mixed densities, a genuine basis rotation, and a collision limitation

Source: `review/flat_spectrum.md`, CM21, examples. The modulo-13 feasible set
contains mixed matrices. The modulo-2 collision permits an off-diagonal
density, refuting the diagonal conclusion without separated differences.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction

open Finset Matrix FourierExtraction
open scoped ComplexOrder

/-- A genuine two-mode probability mixture in the three-mode example, CM21. -/
noncomputable def mixedWeights : Fin 3 → ℝ := ![1 / 2, 1 / 2, 0]

/-- The example has nonnegative weights of total mass one, CM21. -/
theorem mixed_weights : Weights mixedWeights := by
  constructor
  · intro i
    fin_cases i <;> norm_num [mixedWeights]
  · norm_num [mixedWeights, Fin.sum_univ_three]

/-- Mixed densities remain feasible: purity was not hidden in the domain,
CM21. The corresponding vector mixture would fail the CM20 amplitude test. -/
theorem mixed_feasible :
    Feasible (basis id tripleFrequency) (diagonal fun i => (mixedWeights i : ℂ)) := by
  exact diagonal_feasible id tripleFrequency mixed_weights

/-- This feasible density is none of the pure coefficient projectors, CM21. -/
theorem mixed_not_pure :
    ∀ i, (diagonal fun j => (mixedWeights j : ℂ)) ≠ projector i := by
  intro i he
  have h := congrArg (fun M : Matrix (Fin 3) (Fin 3) ℂ => M 0 0) he
  fin_cases i <;> norm_num [mixedWeights, projector, Pi.single_apply] at h

/-- An exact rational rotation mixes the first two hidden modes, CM21. -/
noncomputable def rotation : Matrix (Fin 3) (Fin 3) ℂ :=
  !![3 / 5, -(4 / 5), 0; 4 / 5, 3 / 5, 0; 0, 0, 1]

/-- The rotation preserves column inner products, CM21. -/
theorem rotation_left : rotation.conjTranspose * rotation = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [rotation, Matrix.mul_apply, Fin.sum_univ_three, Matrix.one_apply,
      Complex.conj_ofNat]

/-- The rotation has the inverse required by the coordinate equivalence, CM21. -/
theorem rotation_right : rotation * rotation.conjTranspose = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [rotation, Matrix.mul_apply, Fin.sum_univ_three, Matrix.one_apply,
      Complex.conj_ofNat]

/-- All supplied columns are centered despite mixing the latent directions,
CM21. Thus the nonconstant extraction hypotheses are inhabited. -/
theorem rotated_centered : ∀ i, ∑ x, (basis id tripleFrequency * rotation) x i = 0 := by
  apply centered_mul
  intro j
  simp only [basis, id_eq, sum_stdAddChar_mul]
  fin_cases j <;> norm_num [tripleFrequency] <;> decide

/-- A rotated exact three-dimensional input really admits an optimizer whose
relative-phase column distinguishes all thirteen rows, CM21. -/
theorem witness_extraction :
    ∃ M, LexMaximum (basis id tripleFrequency * rotation) (0 : ZMod 13) 1 M ∧
      ∀ a, Function.Injective (fun z => reading (basis id tripleFrequency * rotation) M z a) := by
  have : Fact (Nat.Prime 13) := ⟨by decide⟩
  exact supplied_extraction _ (Equiv.refl _) triple_separated rotation rotation_left
    rotation_right rfl rotated_centered (by decide)

/-- Rank-one vector lifting agrees with the actual amplitude, CM21. This
identity is used only to exhibit the collision counterexample, not as a
restriction on the convex matrix variable. -/
theorem vector_reading {q : ℕ} [NeZero q] {ι : Type*} [Fintype ι]
    (k : ι → ZMod q) (a : ι → ℂ) (x : ZMod q) :
    reading (basis id k) (vecMulVec a (star a)) x x = (Complex.normSq (wave k a x) : ℂ) := by
  rw [fourier_diagonal, wave_power, sum_comm]
  apply sum_congr rfl
  intro i hi
  apply sum_congr rfl
  intro j hj
  simp only [Matrix.vecMulVec_apply, Pi.star_apply]
  ring

/-- The CM20 collision vector defines a PSD density with nonzero cross terms. -/
noncomputable def collisionDensity : Matrix (Fin 2) (Fin 2) ℂ :=
  vecMulVec collisionCoefficient (star collisionCoefficient)

/-- The colliding example satisfies every lifted constraint, CM21. -/
theorem collision_feasible : Feasible (basis id collisionFrequency) collisionDensity := by
  refine ⟨⟨Matrix.posSemidef_vecMulVec_self_star _, ?_⟩, fun x => ?_⟩
  · norm_num [collisionDensity, Matrix.trace, Matrix.diag, Matrix.vecMulVec_apply,
      collisionCoefficient, Fin.sum_univ_two, Complex.ext_iff, Complex.conj_ofNat]
  · rw [collisionDensity, vector_reading, collision_flat]
    norm_num

/-- Separation cannot be dropped from the diagonal conclusion: this feasible
matrix has a nonzero off-diagonal entry, CM21. -/
theorem collision_cross_term : collisionDensity 0 1 ≠ 0 := by
  norm_num [collisionDensity, Matrix.vecMulVec_apply, collisionCoefficient, Complex.ext_iff]

/-- The colliding feasible density is not any diagonal probability mixture,
CM21. Thus the main characterization would be false without separation. -/
theorem collision_not_diagonal :
    ¬ ∃ w : Fin 2 → ℝ, collisionDensity = diagonal (fun i => (w i : ℂ)) := by
  rintro ⟨w, hw⟩
  have he := congrArg (fun M : Matrix (Fin 2) (Fin 2) ℂ => M 0 1) hw
  exact collision_cross_term (by simpa using he)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction

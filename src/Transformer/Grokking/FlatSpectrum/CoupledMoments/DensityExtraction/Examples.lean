module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction.Witness

/-!
# Inhabited instances of every conditional extraction theorem

Source: `review/flat_spectrum.md`, CM21. The examples use the separated
three-mode subspace, a mixed density, and a rotation that mixes two modes.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction

open Finset Matrix FourierExtraction

example := reading_mix (basis id tripleFrequency) (projector 0) (projector 1) (1 / 2) (1 / 2)
  (0 : ZMod 13) 1
example := density_convex (ι := Fin 3)
example := feasible_convex (basis id tripleFrequency)
example := reading_eq (basis id tripleFrequency) (projector 0) (0 : ZMod 13) 1
example := reading_diagonal (basis id tripleFrequency) (fun i => (mixedWeights i : ℂ))
  (0 : ZMod 13) 1
example := diagonal_density mixedWeights
example := diagonal_decomposition mixedWeights
example := fourier_diagonal tripleFrequency (projector 0) (0 : ZMod 13)
example := diagonal_coefficient tripleFrequency (projector 0) (1 : ZMod 13)
example := flat_off_diagonal triple_separated _ mixed_feasible.2
  (by decide : (0 : Fin 3) ≠ 1)
example := diagonal_feasible id tripleFrequency mixed_weights
example := feasible_iff_diagonal (Equiv.refl (ZMod 13)) triple_separated
  (diagonal fun i => (mixedWeights i : ℂ))
example := projector_feasible id tripleFrequency 0
example := feasible_hull (Equiv.refl (ZMod 13)) triple_separated
example := projector_reading id tripleFrequency 0 (0 : ZMod 13) 1
example := diagonal_reading id tripleFrequency mixedWeights (0 : ZMod 13) 1
example := density_conjugate rotation _ rotation_left mixed_feasible.1
example := reading_change (basis id tripleFrequency) rotation (projector 0) (0 : ZMod 13) 1
example := undo_conjugate rotation (projector 0) rotation_left
example := feasible_change (basis id tripleFrequency) rotation (projector 0)
  rotation_left rotation_right
example := rotated_projector_feasible id tripleFrequency rotation rotation_left rotation_right 0
example := rotated_hull (Equiv.refl (ZMod 13)) triple_separated rotation rotation_left rotation_right
example := exists_lex_vertex (fun i : Fin 3 => ZMod.stdAddChar (tripleFrequency i))
example := weighted_bound mixed_weights (fun _ => (1 : ℝ)) 1 (by intros; rfl)
example := weighted_support mixed_weights (fun _ => (1 : ℝ)) 1 (by intros; rfl)
  (by simpa using mixed_weights.2)

example : (![0, 1, 0] : Fin 3 → ℝ) = Pi.single 1 1 := by
  have hw : Weights (![0, 1, 0] : Fin 3 → ℝ) := by
    constructor
    · intro j
      fin_cases j <;> norm_num
    · norm_num [Fin.sum_univ_three]
  refine lex_weights_pure hw ![0, 1, Complex.I] ?_ 1 ?_ ?_ ?_ ?_
  · intro u v he
    fin_cases u <;> fin_cases v <;> simp_all [Complex.ext_iff]
  · intro j
    fin_cases j <;> norm_num
  · intro j hj
    fin_cases j <;> simp_all
  · norm_num [Fin.sum_univ_three]
  · norm_num [Fin.sum_univ_three]

example := exists_fourier_max (Equiv.refl (ZMod 13)) triple_separated 0 1

example : Function.Injective (fun i : Fin 3 =>
    ZMod.stdAddChar (((0 : ZMod 13) - 1) * tripleFrequency i)) := by
  have : Fact (Nat.Prime 13) := ⟨by decide⟩
  exact score_injective (Equiv.refl _) triple_separated
    (zero_ne_one : (0 : ZMod 13) ≠ 1)

example : ∃ M : Matrix (Fin 3) (Fin 3) ℂ, ∃ i, M = projector i := by
  have : Fact (Nat.Prime 13) := ⟨by decide⟩
  obtain ⟨i, hi⟩ := exists_fourier_max (Equiv.refl (ZMod 13)) triple_separated 0 1
  exact ⟨projector i, fourier_max_pure (Equiv.refl _) triple_separated
    (zero_ne_one : (0 : ZMod 13) ≠ 1) _ hi⟩

example := lex_change (basis id tripleFrequency) rotation (projector 0)
  rotation_left rotation_right (0 : ZMod 13) 1
example := exists_rotated_max (Equiv.refl (ZMod 13)) triple_separated rotation
  rotation_left rotation_right 0 1

example : ∃ M : Matrix (Fin 3) (Fin 3) ℂ,
    (∃ i, M = rotation.conjTranspose * projector i * rotation) ∧
      Function.Injective (fun z => reading (basis id tripleFrequency * rotation) M z (0 : ZMod 13)) := by
  have : Fact (Nat.Prime 13) := ⟨by decide⟩
  obtain ⟨i, hi⟩ := exists_rotated_max (Equiv.refl (ZMod 13)) triple_separated rotation
    rotation_left rotation_right 0 1
  refine ⟨_, rotated_max_pure (Equiv.refl _) triple_separated rotation rotation_left
    rotation_right (zero_ne_one : (0 : ZMod 13) ≠ 1) _ hi, ?_⟩
  exact extracted_column_injective (Equiv.refl _) triple_separated
    (centered_nonzero (Equiv.refl _) tripleFrequency rotation rotation_right rotated_centered)
    rotation rotation_left rotation_right (zero_ne_one : (0 : ZMod 13) ≠ 1) _ hi 0

example := centered_mul (basis id tripleFrequency * rotation) rotation rotated_centered 0
example := centered_nonzero (Equiv.refl (ZMod 13)) tripleFrequency rotation rotation_right
  rotated_centered
example := vector_reading collisionFrequency collisionCoefficient (0 : ZMod 2)

-- `witness_extraction` instantiates `supplied_extraction` with centered rotated columns.

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction

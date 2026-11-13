module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityBounds.Witness

/-!
# Closed instances of every conditional error-bound theorem

Source: `review/flat_spectrum.md`, CM22. Positive tolerances, actual feature
perturbations, and a mixed approximate optimizer with a positive gap appear
separately, so exact geometry is not silently equated with estimated geometry.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityBounds

open Finset Matrix DensityExtraction FourierExtraction

example := relaxed_convex (basis id tripleFrequency) (1 / 100)
example := exact_relaxed mixed_feasible (by norm_num : (0 : ℝ) ≤ 1 / 100)
example := relaxed_zero (basis id tripleFrequency) (diagonal fun i => (mixedWeights i : ℂ))
example := density_weights mixed_feasible.1
example := density_diagonal_le_one mixed_feasible.1 (0 : Fin 3)
example := density_entry_le_one collision_feasible.1 (0 : Fin 2) 1
example := reading_diagonal_real (basis id collisionFrequency) collision_feasible.1 (0 : ZMod 2)
example := relaxed_interval (basis id tripleFrequency) (1 / 100)
  (diagonal fun i => (mixedWeights i : ℂ))
example := error_coefficient triple_separated (diagonal fun i => (mixedWeights i : ℂ))
  (by decide : (0 : Fin 3) ≠ 1)
example := off_diagonal_le (Equiv.refl (ZMod 13)) triple_separated _ 0 mixed_relaxed.2
  (by decide : (0 : Fin 3) ≠ 1)
example := diagonal_reading_error id tripleFrequency (diagonal fun i => (mixedWeights i : ℂ))
  (by norm_num : (0 : ℝ) ≤ 1 / 100)
  (by intro i j hij; simp [diagonal_apply_ne _ hij]) (0 : ZMod 13) 1
example := product_error (11 / 10) (11 / 10) 1 1
  (by norm_num : (0 : ℝ) ≤ 1 / 10) (by norm_num : (0 : ℝ) ≤ 1)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)

example := reading_perturbation perturbedBasis (basis id twoFrequency) two_feasible.1
  (by norm_num : (0 : ℝ) ≤ 1 / 10) (by norm_num : (0 : ℝ) ≤ 1)
  (fun x i => (perturbed_error x i).le)
  (by intro x i; simp [basis, ZMod.stdAddChar_apply, Circle.norm_coe]) (1 : ZMod 4) 0
example := relaxed_transfer perturbedBasis (basis id twoFrequency) (perturbed_admitted 0)
  (by norm_num : (0 : ℝ) ≤ 1 / 10) (by norm_num : (0 : ℝ) ≤ 1)
  (fun x i => (perturbed_error x i).le)
  (by intro x i; simp [basis, ZMod.stdAddChar_apply, Circle.norm_coe])
-- `perturbed_admitted` supplies the nonzero-error instance of `reference_admitted`.
example := perturbed_error 1 0
example := perturbed_admitted 0
example := two_gap 1 (by decide)

example := weighted_gap two_weights (![1, 0] : Fin 2 → ℝ) 0 1
  (by intro j hj; fin_cases j <;> simp_all)
example := relaxed_score_error (Equiv.refl (ZMod 4)) two_separated
  (exact_relaxed two_feasible le_rfl) 1 0
example := score_concentration (Equiv.refl (ZMod 4)) two_separated
  (η := 1 / 4) (exact_relaxed two_feasible le_rfl) 1 0 0 two_gap
  (by norm_num [diagonal_reading, twoFrequency, twoWeights, Fin.sum_univ_two, character_four_one])
example := two_near _ (exact_relaxed two_feasible le_rfl)

example := objective_transfer (basis id twoFrequency) (basis id twoFrequency) two_feasible.1
  (ε := 0) (δ := 0) (R := 1) (η := 1 / 4) (by norm_num) (by norm_num)
  (by intros; simp) (by intro x i; simp [basis, ZMod.stdAddChar_apply, Circle.norm_coe])
  (by norm_num) (1 : ZMod 4) 0 two_near two_feasible

-- A mixed optimizer with mass `1/4` outside the best mode meets the combined
-- bound with equality at zero feature/diagonal error and objective gap `1/4`.
example : (1 : ℝ) * (1 - twoWeights 0) ≤ 1 / 4 := by
  have h := rotated_concentration (Equiv.refl (ZMod 4)) two_separated
    (basis id twoFrequency) (1 : Matrix (Fin 2) (Fin 2) ℂ) (by simp) (by simp)
    (δ := 0) (R := 1) (η := 1 / 4)
    (exact_relaxed two_feasible le_rfl) (by norm_num) (by norm_num)
    (by intros; simp) (by intro x i; simp [basis, ZMod.stdAddChar_apply, Circle.norm_coe])
    (by norm_num) 1 0 0 two_gap two_near
  simpa using h

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityBounds

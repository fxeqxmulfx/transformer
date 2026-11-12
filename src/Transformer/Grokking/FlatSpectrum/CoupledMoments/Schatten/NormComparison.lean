module

public import Transformer.Grokking.FlatSpectrum.SpectralBall.Basic
public import Transformer.Grokking.FlatSpectrum.FlatCapacity

/-!
# Comparing the two spectral budgets

Source: `review/flat_spectrum.md`, CM5. An operator bound R implies Frobenius energy at
most `card(n) R^2`, by applying the vector-energy bound to each coordinate vector. The
rank/Frobenius nuclear inequality then gives nuclear norm at most `card(n) R`. Thus the
large sum budget used by the full-rank control does not shrink the old operator-ball domain.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

open Matrix Finset
open scoped Matrix.Norms.L2Operator

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]

/-- Summing the operator energy bound over coordinate vectors bounds total energy (CM5). -/
theorem frobenius_le_card_sq {X : Matrix m n ℂ} {R : ℝ} (h : ‖X‖ ≤ R) :
    frobeniusSq X ≤ Fintype.card n * R ^ 2 := by
  have hcol (j : n) : ∑ i, ‖X i j‖ ^ 2 ≤ R ^ 2 := by
    simpa [Pi.single_apply, Matrix.col, apply_ite] using SpectralBall.energy_le h (Pi.single j 1)
  calc
    frobeniusSq X = ∑ j, ∑ i, ‖X i j‖ ^ 2 := Finset.sum_comm
    _ ≤ ∑ _j : n, R ^ 2 := Finset.sum_le_sum fun j _ => hcol j
    _ = Fintype.card n * R ^ 2 := by simp

/-- The nuclear budget `card(n) R` is redundant under operator norm at most R (CM5). -/
theorem nuclear_le_card_operator {X : Matrix m n ℂ} {R : ℝ} (h : ‖X‖ ≤ R) :
    nuclearNorm X ≤ Fintype.card n * R := by
  have hR : 0 ≤ R := (norm_nonneg X).trans h
  have hc : (0 : ℝ) ≤ Fintype.card n := Nat.cast_nonneg _
  have hr : (X.rank : ℝ) ≤ Fintype.card n := by exact_mod_cast rank_le_card_width X
  apply (sq_le_sq₀ (nuclearNorm_nonneg X) (mul_nonneg hc hR)).mp
  calc
    nuclearNorm X ^ 2 ≤ X.rank * frobeniusSq X := nuclearNorm_sq_le_rank_mul_frobeniusSq X
    _ ≤ (Fintype.card n : ℝ) * frobeniusSq X :=
      mul_le_mul_of_nonneg_right hr (frobeniusSq_nonneg X)
    _ ≤ (Fintype.card n : ℝ) * (Fintype.card n * R ^ 2) :=
      mul_le_mul_of_nonneg_left (frobenius_le_card_sq h) hc
    _ = ((Fintype.card n : ℝ) * R) ^ 2 := by ring

example := frobenius_le_card_sq (X := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (R := 2)
  (by norm_num)

example := nuclear_le_card_operator (X := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (R := 2)
  (by norm_num)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

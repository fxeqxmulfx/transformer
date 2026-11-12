module

public import Mathlib.Analysis.CStarAlgebra.Matrix
public import Transformer.Grokking.FlatSpectrum.ConvexRecall.Loss
public import Transformer.Grokking.FlatSpectrum.Training.CrossEntropy
public import Transformer.Grokking.FlatSpectrum.ContractionDual

/-!
# Convex complex matrices with an upper spectral bound

Source: `review/flat_spectrum.md`, SB1. The parameters are the complex entries themselves,
with unit entry disks and a bound on the Euclidean operator norm. This is an upper bound,
not an equality of singular values or a product of trained factors. Ordinary cross-entropy
of fixed real-linear probes is jointly convex in all parameters.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.SpectralBall

open Matrix Finset Training
open scoped Matrix.Norms.L2Operator ComplexOrder

variable {m n S C : Type*} [Fintype m] [Fintype n] [DecidableEq n]

/-- Unit entry disks intersected with the L2 operator-norm ball (SB1). -/
def Feasible (R : ℝ) (X : Matrix m n ℂ) : Prop :=
  (∀ i j, ‖X i j‖ ≤ 1) ∧ ‖X‖ ≤ R

/-- Both constraints are convex in the complete matrix (SB1). -/
theorem feasible_convex (R : ℝ) : Convex ℝ {X : Matrix m n ℂ | Feasible R X} := by
  intro X hX Y hY a b ha hb hab
  constructor
  · intro i j
    calc
      ‖a • X i j + b • Y i j‖ ≤ a * ‖X i j‖ + b * ‖Y i j‖ := by
        simpa [norm_smul, Real.norm_of_nonneg ha, Real.norm_of_nonneg hb]
          using norm_add_le (a • X i j) (b • Y i j)
      _ ≤ a * 1 + b * 1 := add_le_add (mul_le_mul_of_nonneg_left (hX.1 i j) ha)
        (mul_le_mul_of_nonneg_left (hY.1 i j) hb)
      _ = 1 := by simpa using hab
  · calc
      ‖a • X + b • Y‖ ≤ a * ‖X‖ + b * ‖Y‖ := by
        simpa [norm_smul, Real.norm_of_nonneg ha, Real.norm_of_nonneg hb]
          using norm_add_le (a • X) (b • Y)
      _ ≤ a * R + b * R := add_le_add (mul_le_mul_of_nonneg_left hX.2 ha)
        (mul_le_mul_of_nonneg_left hY.2 hb)
      _ = R := by rw [← add_mul, hab, one_mul]

/-- Zero is feasible for every nonnegative radius (SB1). -/
theorem feasible_zero {R : ℝ} (hR : 0 ≤ R) : Feasible R (0 : Matrix m n ℂ) := by
  constructor
  · intro i j
    simp
  · simpa using hR

example := feasible_zero (m := Fin 2) (n := Fin 2) (R := 2) zero_le_two

variable [Fintype S] [Fintype C]

omit [DecidableEq n] in
/-- Fixed probes make ordinary empirical CE jointly convex (SB1). -/
theorem ceLoss_convex (A : S → C → Matrix m n ℂ) (y : S → C) :
    ConvexOn ℝ Set.univ (ceLoss A y) := by
  refine ⟨convex_univ, ?_⟩
  intro X _ Y _ a b ha hb hab
  simp only [ceLoss, smul_eq_mul, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro s _
  have h := (ConvexRecall.convex_crossEntropy (y s)).2 (Set.mem_univ _)
    (Set.mem_univ _) ha hb hab (x := fun c => pairing (A s c) X)
      (y := fun c => pairing (A s c) Y)
  simpa only [pairing_add_right, pairing_smul_right, Pi.add_def, Pi.smul_def,
    smul_eq_mul] using h

omit [Fintype S] [Fintype C] in
/-- The actual L2 bound controls every vector's squared energy (SB1). -/
theorem energy_le {R : ℝ} {X : Matrix m n ℂ} (h : ‖X‖ ≤ R) (v : n → ℂ) :
    ∑ i, ‖(X *ᵥ v) i‖ ^ 2 ≤ R ^ 2 * ∑ j, ‖v j‖ ^ 2 := by
  have hR : 0 ≤ R := (norm_nonneg X).trans h
  have hv := (l2_opNorm_mulVec X (WithLp.toLp 2 v)).trans
    (mul_le_mul_of_nonneg_right h (norm_nonneg _))
  have hs := sq_le_sq₀ (norm_nonneg _) (mul_nonneg hR (norm_nonneg _)) |>.2 hv
  simpa [mul_pow, EuclideanSpace.norm_sq_eq, EuclideanSpace.equiv] using hs

example := energy_le (X := (0 : Matrix (Fin 2) (Fin 2) ℂ)) (R := 2)
  (by norm_num) (fun _ => 1)

end Transformer.Grokking.FlatSpectrum.SpectralBall

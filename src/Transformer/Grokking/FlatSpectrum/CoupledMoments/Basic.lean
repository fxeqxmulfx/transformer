module

public import Transformer.Grokking.FlatSpectrum.SpectralBall.Basic

/-!
# Complex class moments of one shared label distribution

Source: `review/flat_spectrum.md`, CM1. A probability vector is learned at each input cell.
Every complex code matrix is a fixed linear image of these same probabilities, with its own
operator-norm upper bound. There are no independent, potentially inconsistent moment entries
and no products of learned factors. The spectral constraints and class probabilities are
jointly convex. For unit codes the old entry disks follow from the simplex constraint.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments

open Matrix Finset
open scoped Matrix.Norms.L2Operator

/-- Shared class coordinates, equivalently complex Fourier coordinates when the code is
invertible. The real probability parametrization avoids redundant imaginary coordinates. -/
abbrev Table (m n C : Type*) := m → n → C → ℝ

variable {m n C F : Type*} [Fintype m] [Fintype n] [Fintype C] [DecidableEq n]

/-- Each input cell carries one class distribution (CM1). -/
def Rows (P : Table m n C) : Prop :=
  (∀ i j c, 0 ≤ P i j c) ∧ ∀ i j, ∑ c, P i j c = 1

/-- A complex reading of the shared class distribution (CM1). -/
noncomputable def moment (φ : C → ℂ) (P : Table m n C) : Matrix m n ℂ :=
  fun i j => ∑ c, (P i j c : ℂ) * φ c

/-- Simplexes intersected with spectral upper bounds on fixed complex moments (CM1). -/
def Feasible (φ : F → C → ℂ) (R : F → ℝ) (P : Table m n C) : Prop :=
  Rows P ∧ ∀ f, ‖moment (φ f) P‖ ≤ R f

omit [Fintype m] [Fintype n] [DecidableEq n] in
/-- Every complex moment is real-linear in all learned coordinates (CM1). -/
theorem moment_linear (φ : C → ℂ) (P Q : Table m n C) (a b : ℝ) :
    moment φ (a • P + b • Q) = a • moment φ P + b • moment φ Q := by
  ext i j
  simp [moment, Complex.real_smul, Finset.sum_add_distrib, Finset.mul_sum, add_mul, mul_assoc]

omit [Fintype m] [Fintype n] [DecidableEq n] in
/-- The probability constraints form a convex set (CM1). -/
theorem rows_convex : Convex ℝ {P : Table m n C | Rows P} := by
  intro P hP Q hQ a b ha hb hab
  constructor
  · intro i j c
    exact add_nonneg (mul_nonneg ha (hP.1 i j c)) (mul_nonneg hb (hQ.1 i j c))
  · intro i j
    change (∑ c, (a * P i j c + b * Q i j c)) = 1
    simp only [Finset.sum_add_distrib, ← Finset.mul_sum, hP.2 i j, hQ.2 i j, mul_one]
    exact hab

/-- Coupling all codes through the same distribution preserves convexity (CM1). -/
theorem feasible_convex (φ : F → C → ℂ) (R : F → ℝ) :
    Convex ℝ {P : Table m n C | Feasible φ R P} := by
  intro P hP Q hQ a b ha hb hab
  refine ⟨rows_convex hP.1 hQ.1 ha hb hab, ?_⟩
  intro f
  rw [moment_linear]
  calc
    ‖a • moment (φ f) P + b • moment (φ f) Q‖ ≤
        a * ‖moment (φ f) P‖ + b * ‖moment (φ f) Q‖ := by
      simpa [norm_smul, Real.norm_of_nonneg ha, Real.norm_of_nonneg hb]
        using norm_add_le (a • moment (φ f) P) (b • moment (φ f) Q)
    _ ≤ a * R f + b * R f := add_le_add (mul_le_mul_of_nonneg_left (hP.2 f) ha)
      (mul_le_mul_of_nonneg_left (hQ.2 f) hb)
    _ = R f := by rw [← add_mul, hab, one_mul]

omit [Fintype m] [Fintype n] [DecidableEq n] in
/-- Unit-modulus codes automatically obey the entry-disk bounds (CM1). -/
theorem moment_norm_le_one {φ : C → ℂ} {P : Table m n C}
    (hφ : ∀ c, ‖φ c‖ ≤ 1) (hP : Rows P) (i : m) (j : n) : ‖moment φ P i j‖ ≤ 1 := by
  calc
    _ ≤ ∑ c, ‖(P i j c : ℂ) * φ c‖ := norm_sum_le _ _
    _ ≤ ∑ c, P i j c := Finset.sum_le_sum fun c _ => by
      rw [norm_mul, Complex.norm_real, Real.norm_of_nonneg (hP.1 i j c)]
      exact mul_le_of_le_one_right (hP.1 i j c) (hφ c)
    _ = 1 := hP.2 i j

omit [Fintype m] [Fintype n] [DecidableEq n] in
/-- Every class coordinate is at most one (CM1). -/
theorem coordinate_le_one {P : Table m n C} (hP : Rows P) (i : m) (j : n) (c : C) :
    P i j c ≤ 1 := by
  have h := Finset.single_le_sum (fun d (_ : d ∈ Finset.univ) => hP.1 i j d)
    (Finset.mem_univ c)
  simpa [hP.2 i j] using h

variable [DecidableEq C]

/-- A table of one-hot class distributions, used as a realizability witness (CM1). -/
def pointMass (labels : m → n → C) : Table m n C :=
  fun i j c => if c = labels i j then 1 else 0

omit [Fintype m] [Fintype n] [DecidableEq n] in
/-- One-hot tables satisfy the probability constraints (CM1). -/
theorem rows_pointMass (labels : m → n → C) : Rows (pointMass labels) := by
  constructor
  · intro i j c
    simp only [pointMass]
    split_ifs <;> norm_num
  · intro i j
    simp [pointMass]

omit [Fintype m] [Fintype n] [DecidableEq n] in
/-- One-hot distributions give the actual complex label code (CM1). -/
theorem moment_pointMass (φ : C → ℂ) (labels : m → n → C) :
    moment φ (pointMass labels) = fun i j => φ (labels i j) := by
  ext i j
  simp [moment, pointMass, apply_ite]

example := moment_norm_le_one (φ := fun b : Bool => if b then (1 : ℂ) else -1)
  (by intro b; cases b <;> norm_num) (rows_pointMass (fun (_ : Fin 1) (_ : Fin 1) => true)) 0 0

example := coordinate_le_one (rows_pointMass (fun (_ : Fin 1) (_ : Fin 1) => true)) 0 0 true

omit [DecidableEq C] in
/-- Uniform class mass supplies a feasible initialization for centered codes (CM1). -/
noncomputable def uniform : Table m n C := fun _ _ _ => (Fintype.card C : ℝ)⁻¹

omit [Fintype m] [Fintype n] [DecidableEq n] [DecidableEq C] in
/-- Uniform class mass is a probability vector for every nonempty vocabulary (CM1). -/
theorem rows_uniform [Nonempty C] : Rows (uniform : Table m n C) := by
  constructor
  · intro i j c
    exact inv_nonneg.mpr (Nat.cast_nonneg _)
  · intro i j
    have hc : (Fintype.card C : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
    simp [uniform, hc]

omit [Fintype m] [Fintype n] [DecidableEq n] [DecidableEq C] in
/-- Every centered code has zero reading under uniform probabilities (CM1). -/
theorem moment_uniform_zero {φ : C → ℂ} (hφ : ∑ c, φ c = 0) :
    moment φ (uniform : Table m n C) = 0 := by
  ext i j
  simp only [moment, uniform, ← Finset.mul_sum, hφ, mul_zero, Matrix.zero_apply]

omit [DecidableEq C] in
/-- Centered codes and nonnegative radii give a nonempty feasible domain (CM1). -/
theorem feasible_uniform [Nonempty C] {φ : F → C → ℂ} {R : F → ℝ}
    (hφ : ∀ f, ∑ c, φ f c = 0) (hR : ∀ f, 0 ≤ R f) :
    Feasible φ R (uniform : Table m n C) := by
  refine ⟨rows_uniform, ?_⟩
  intro f
  rw [moment_uniform_zero (hφ f), norm_zero]
  exact hR f

example := moment_uniform_zero (m := Fin 1) (n := Fin 1)
  (φ := fun c : Bool => if c then (1 : ℂ) else -1) (by simp)

example := feasible_uniform (m := Fin 1) (n := Fin 1)
  (φ := fun (_ : Fin 1) (c : Bool) => if c then Complex.I else -Complex.I)
  (R := fun _ => 1) (by intro f; simp) (by intro f; norm_num)

end Transformer.Grokking.FlatSpectrum.CoupledMoments

module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Updates.Forward

/-!
# Exact zero-support storage for complete learned parameters

Source: `review/convex/moments.md`, CM88. A missing storage entry denotes an
exact zero in the unchanged complete parameter domain. Zero context columns
contribute nothing to the query Gram matrix and have zero recovered context
rows. No small nonzero singular value or learned coefficient is discarded.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Updates

open scoped BigOperators Matrix
open LearnedHinge (Prefix)
open Identification

variable {O : Type*} [Fintype O] [DecidableEq O] {T N : ℕ}

/-- The complete record-weighted coordinate pullback, CM88. Its scalar
inputs may be analytic CE residuals or measured finite loss differences. -/
noncomputable def pullback (tokens : Fin N → Prefix O (T + 1))
    (scalar : Fin N → O → ℝ) (z : RealCoordinate O T) : ℝ :=
  (N : ℝ)⁻¹ * ∑ i, response (tokens i) z * scalar i z.2.1

omit [Fintype O] in
/-- Coordinates with no response on any batch record have exactly zero
update, CM88. All class residuals and all repeated records are still defined. -/
theorem pullback_zero (tokens : Fin N → Prefix O (T + 1)) (scalar : Fin N → O → ℝ)
    (z : RealCoordinate O T) (hz : ∀ i, response (tokens i) z = 0) :
    pullback tokens scalar z = 0 := by
  unfold pullback
  simp only [hz, zero_mul, Finset.sum_const_zero, mul_zero]

omit [Fintype O] in
/-- Exact zero coordinates remain zero after a complete batch update when
none of its records uses them, CM88. This justifies implicit zero storage,
not removal of any possible coordinate from the model. -/
theorem zero_update (r : RealCoordinate O T → ℝ) (tokens : Fin N → Prefix O (T + 1))
    (scalar : Fin N → O → ℝ) (rate : ℝ) (z : RealCoordinate O T)
    (hr : r z = 0) (hz : ∀ i, response (tokens i) z = 0) :
    r z - rate * pullback tokens scalar z = 0 := by
  rw [hr, pullback_zero tokens scalar z hz, mul_zero, sub_zero]

variable {Q D : Type*} [Fintype Q] [Fintype D] [DecidableEq Q] [DecidableEq D]

omit [DecidableEq D] in
/-- A zero joint context column gives a zero complete recovered context
row, CM88. This uses the actual balanced spectral reconstruction. -/
theorem rightFactor_zero_column (M : Matrix Q D ℂ) (d : D) (hz : ∀ q, M q d = 0) :
    ∀ q, rightFactor M d q = 0 := by
  intro q
  simp [rightFactor, Matrix.mul_apply, Matrix.conjTranspose_apply, hz]

/-- Normalizing the learned context encoder preserves its exact zero rows,
CM88. Its full domain and all singular directions remain unchanged. -/
theorem rightRepresentation_zero_column (M : Matrix Q D ℂ) (d : D)
    (hz : ∀ q, M q d = 0) : ∀ q, rightRepresentation M d q = 0 := by
  intro q
  simp [rightRepresentation, rightFactor_zero_column M d hz]

omit [Fintype Q] [DecidableEq Q] [DecidableEq D] in
/-- Summing only exactly nonzero stored columns gives the entire query
Gram matrix, CM88. The omitted summands are proved zero, not approximated. -/
theorem gram_supported_columns (M : Matrix Q D ℂ) (stored : Finset D)
    (hz : ∀ d, d ∉ stored → ∀ q, M q d = 0) (q k : Q) :
    (M * M.conjTranspose) q k = ∑ d ∈ stored, M q d * star (M k d) := by
  rw [Matrix.mul_apply]
  simp only [Matrix.conjTranspose_apply]
  symm
  apply Finset.sum_subset (Finset.subset_univ stored)
  intro d hd hn
  rw [hz d hn q, zero_mul]

example : pullback (fun _ : Fin 2 => repeated) (fun _ _ => 1)
    (false, true, some true, none) = 0 := by
  apply pullback_zero
  intro i
  simp [response, queryRow, repeated]

example : (0 : RealCoordinate Bool 4 → ℝ) (false, true, some true, none) -
    1 * pullback (fun _ : Fin 2 => repeated) (fun _ _ => 1)
      (false, true, some true, none) = 0 := by
  apply zero_update
  · rfl
  · intro i
    simp [response, queryRow, repeated]

example : ∀ q : Bool, rightFactor (fun (_ : Bool) (_ : Bool) => (0 : ℂ)) false q = 0 :=
  rightFactor_zero_column _ false (by simp)

example : ∀ q : Bool, rightRepresentation (fun (_ : Bool) (_ : Bool) => (0 : ℂ)) false q = 0 :=
  rightRepresentation_zero_column _ false (by simp)

example : ((0 : Matrix Bool Bool ℂ) * (0 : Matrix Bool Bool ℂ).conjTranspose) false true =
    ∑ d ∈ (∅ : Finset Bool), (0 : Matrix Bool Bool ℂ) false d * star ((0 : Matrix Bool Bool ℂ) true d) :=
  gram_supported_columns 0 ∅ (by simp) false true

end Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Updates

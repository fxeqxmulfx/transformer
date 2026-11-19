module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Interpolation.Basic
public import Mathlib.LinearAlgebra.Dimension.Constructions
public import Mathlib.LinearAlgebra.Dimension.StrongRankCondition

/-!
# A dimension lower bound for exact affine compression of the whole family

Source: `review/convex/moments.md`, CM84. An affine margin readout that covers
every feasible model must cover every real full-word table. Its linear part
is therefore surjective and needs at least as many real coordinates as full
words. The encoder into those coordinates may be arbitrary and nonlinear.
This is not a lower bound for one structured task or every nonlinear convex
model; affine logits are not a necessary condition for convex cross-entropy.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Interpolation

open PatternReduction

variable {O V : Type*} [Fintype O] [DecidableEq O] {T : ℕ}
  [AddCommGroup V] [Module ℝ V]

/-- Exact affine coverage of all feasible margins forces surjectivity onto
all full-word tables, CM84, even with an arbitrary encoding of model points. -/
theorem affine_readout_surjective (base selected : O) (hne : base ≠ selected)
    (readout : V →ₗ[ℝ] ((Fin T → O) → ℝ)) (offset : (Fin T → O) → ℝ)
    (hcover : ∀ p : ReducedSequenceParameters O T,
      p ∈ complexDomain rawInput Set.univ diagonal →
      ∃ v : V, ∀ word : Fin T → O,
        readout v word + offset word = sequenceLogits p (fullPrefix word) selected -
          sequenceLogits p (fullPrefix word) base) :
    Function.Surjective readout := by
  intro values
  obtain ⟨p, hp, hmargin⟩ := margin_interpolation base selected hne (values + offset)
  obtain ⟨v, hv⟩ := hcover p hp
  refine ⟨v, ?_⟩
  funext word
  have h := hv word
  rw [hmargin] at h
  change readout v word + offset word = values word + offset word at h
  linarith

/-- An exact finite-dimensional affine compression of the whole current
family needs at least |O|^T real dimensions, CM84. The conclusion concerns
the function class, not naive activation-pattern enumeration alone. -/
theorem affine_dimension_bound [Module.Finite ℝ V]
    (base selected : O) (hne : base ≠ selected)
    (readout : V →ₗ[ℝ] ((Fin T → O) → ℝ)) (offset : (Fin T → O) → ℝ)
    (hcover : ∀ p : ReducedSequenceParameters O T,
      p ∈ complexDomain rawInput Set.univ diagonal →
      ∃ v : V, ∀ word : Fin T → O,
        readout v word + offset word = sequenceLogits p (fullPrefix word) selected -
          sequenceLogits p (fullPrefix word) base) :
    Fintype.card O ^ T ≤ Module.finrank ℝ V := by
  have h := LinearMap.finrank_le_finrank_of_surjective
    (affine_readout_surjective base selected hne readout offset hcover)
  simpa only [Module.finrank_pi, Fintype.card_fun, Fintype.card_fin] using h

/-- For an actual array of real coordinates, the same exact-coverage lower
bound applies to its retained coordinate count, CM84. -/
theorem affine_coordinate_bound {D : Type*} [Fintype D]
    (base selected : O) (hne : base ≠ selected)
    (readout : (D → ℝ) →ₗ[ℝ] ((Fin T → O) → ℝ)) (offset : (Fin T → O) → ℝ)
    (hcover : ∀ p : ReducedSequenceParameters O T,
      p ∈ complexDomain rawInput Set.univ diagonal →
      ∃ v : D → ℝ, ∀ word : Fin T → O,
        readout v word + offset word = sequenceLogits p (fullPrefix word) selected -
          sequenceLogits p (fullPrefix word) base) :
    Fintype.card O ^ T ≤ Fintype.card D := by
  simpa only [Module.finrank_pi] using
    affine_dimension_bound base selected hne readout offset hcover

example := affine_readout_surjective (T := 2) false true (by decide)
  LinearMap.id (fun word => if word 0 then 1 else 0) (by
    intro p hp
    refine ⟨fun word => sequenceLogits p (fullPrefix word) true -
      sequenceLogits p (fullPrefix word) false - (if word 0 then 1 else 0), ?_⟩
    intro word
    dsimp
    ring)
example := affine_dimension_bound (T := 2) false true (by decide)
  LinearMap.id (fun word => if word 0 then 1 else 0) (by
    intro p hp
    refine ⟨fun word => sequenceLogits p (fullPrefix word) true -
      sequenceLogits p (fullPrefix word) false - (if word 0 then 1 else 0), ?_⟩
    intro word
    dsimp
    ring)
example := affine_coordinate_bound (D := Fin 2 → Bool) false true (by decide)
  LinearMap.id (fun word => if word 0 then 1 else 0) (by
    intro p hp
    refine ⟨fun word => sequenceLogits p (fullPrefix word) true -
      sequenceLogits p (fullPrefix word) false - (if word 0 then 1 else 0), ?_⟩
    intro word
    dsimp
    ring)

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Interpolation

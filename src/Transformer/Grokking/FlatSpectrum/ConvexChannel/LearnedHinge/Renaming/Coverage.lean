module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Renaming.Convexity

/-!
# Exact physical coverage and retained cost of the paired architecture

Source: `review/convex/moments.md`, CM78. The new function class is the paired
average of the original physical phased family. CM76's exact lift still
covers that class. This is an explicit restriction to renaming-equivariant
predictions when the renaming is involutive, not preservation of every old
prediction. Two forward calls share one parameter point; its exponential
complete-domain dimension is unchanged.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Renaming

open PatternReduction

variable {O J : Type*} [Fintype O] [DecidableEq O] [Fintype J] {T : ℕ}

/-- The fully shared physical paired network keeps both independent
representation and signed readout factors in each original pass, CM78. -/
noncomputable def factoredPairedForward (σ : Equiv.Perm O)
    (w : J → ObjectWeights O T) (a : J → O → ℝ) (tokens : Prefix O T) (c : O) : ℂ :=
  (1 / 2 : ℂ) * (factoredForward w a (rawInput tokens) c +
    factoredForward w a (rawInput (renamed σ tokens)) (σ c))

/-- Every declared physical paired function has exactly the same whole
complex predictions after the complete joint lift, CM78. The earlier
class/sign widening remains; arbitrary free complex transformers are excluded. -/
theorem paired_covers (σ : Equiv.Perm O) (w : J → ObjectWeights O T) (a : J → O → ℝ) :
    ∃ p : ReducedSequenceParameters O T,
      p ∈ complexDomain rawInput Set.univ diagonal ∧
      ∀ tokens : Prefix O T, pairedForward σ p tokens = factoredPairedForward σ w a tokens := by
  obtain ⟨p, hp, hpred⟩ := reducedSequence_covers w a
  refine ⟨p, hp, ?_⟩
  intro tokens
  funext c
  simp only [pairedForward, factoredPairedForward, hpred]

/-- The paired model retains all original real trainable coordinates,
CM78. Only the number of forward calls changes, not its parameter index.
This formula omits optimizer state and is not a native memory admission. -/
theorem paired_real_coordinate_count (O : Type*) [Fintype O] [DecidableEq O] (T : ℕ) :
    Fintype.card (Bool × ScalarPattern (Prefix O T) × O × Bool × Coordinate O T) =
      4 * Fintype.card O * (1 + T * Fintype.card O) * 2 ^ Fintype.card (Prefix O T) :=
  reducedSequence_real_coordinate_count O T

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Renaming

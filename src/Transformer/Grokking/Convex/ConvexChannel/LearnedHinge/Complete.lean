module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Sequence
public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Patterns

/-!
# Complete raw-prefix coverage and convex next-object CE

Source: `review/convex/moments.md`, CM75. Use the entire finite bounded-prefix
universe as the activation region. One full-pattern domain then has convex
ordinary next-object CE at every admitted prefix and contains the exact
predictions of every original finite-width phased complex ReLU network.
The global statement pays the full pattern and class-copy storage cost.
This is a mathematical construction, not an admitted native stress run or
an efficient representation and learning algorithm.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge

variable {O J : Type*} [Fintype O] [DecidableEq O] [Fintype J] {T : ℕ}

/-- Every joint coordinate in the complete raw-prefix pattern expansion,
CM75. No class, sign, or activation pattern is omitted. -/
abbrev CompleteParameters (J O : Type*) (T : ℕ) :=
  Parameters (J × Pattern (Prefix O T)) O (ObjectWeights O T)

/-- Ordinary next-object CE is convex at every admitted raw prefix on this
one complete domain, CM75. There is no training-only membership premise;
the price is the complete all-prefix activation constraint system. -/
theorem completeSequenceLoss_convex (tokens : Prefix O T) (target : O) :
    ConvexOn ℝ (complexDomain rawInput Set.univ (fun jm x => jm.2 x) :
      Set (CompleteParameters J O T)) (sequenceLoss tokens target) :=
  sequenceLoss_convex (O := O) (J := J × Pattern (Prefix O T)) (T := T)
    Set.univ (fun (jm : J × Pattern (Prefix O T)) x => jm.2 x) (Set.mem_univ tokens) target

/-- Every independently specified representation and signed class readout
of the declared physical family is represented on every bounded raw prefix,
CM75. The recovered network has the explicitly wider class-specific width. -/
theorem completeSequence_covers (w : J → ObjectWeights O T) (a : J → O → ℝ) :
    ∃ p : CompleteParameters J O T,
      p ∈ complexDomain rawInput Set.univ (fun jm x => jm.2 x) ∧
      ∀ tokens : Prefix O T, complexForward p (rawInput tokens) = factoredForward w a (rawInput tokens) := by
  let input : Prefix O T → ObjectWeights O T →ₗ[ℝ] ℂ := rawInput
  refine ⟨expanded input w a, ?_, ?_⟩
  · exact expanded_mem input w a
  · intro tokens
    exact expanded_forward input w a (input tokens)

/-- The global raw-prefix statement retains the full exponential storage
factor, CM75. It does not turn a finite existence proof into a compact model. -/
theorem completeSequence_real_coordinate_count (O J : Type*)
    [Fintype O] [DecidableEq O] [Fintype J] (T : ℕ) :
    Fintype.card (Bool × (J × Pattern (Prefix O T)) × O × Bool × Coordinate O T) =
      4 * Fintype.card J * Fintype.card O * (1 + T * Fintype.card O) *
        2 ^ (2 * Fintype.card (Prefix O T)) := by
  rw [sequence_real_coordinate_count, Fintype.card_prod, pattern_count]
  ring

example := completeSequenceLoss_convex (J := Unit)
  (⟨[true, false], by decide⟩ : Prefix Bool 2) true
example := completeSequence_covers (J := Unit) (O := Bool) (T := 2)
  (fun _ _ => 1 + Complex.I) (fun _ c => if c then 2 else -3)

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge

module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.ComplexLoss
public import Mathlib.Data.Fintype.Option
public import Mathlib.Data.Set.Finite.List

/-!
# Immutable object sequences and a learned complex next-object model

Source: `review/convex/moments.md`, CM75. Input objects have no supplied
arithmetic features, answer programs, or target-dependent encoding. A
position/object index addresses a learned complex embedding coordinate;
one extra coordinate supplies a learned bias. Actual inference sums those
learned values, applies the complex activation, and multiplies the learned
readouts. The target appears only in the usual next-object CE.

The context cap is part of the architecture. Inputs exceeding it are not
silently truncated: they cannot inhabit `Prefix`. Position-specific learned
coordinates retain order. The output vocabulary is the same object type.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge

open scoped BigOperators

/-- A raw prefix admitted by the declared architecture, CM75. -/
abbrev Prefix (O : Type*) (T : ℕ) := {tokens : List O // tokens.length ≤ T}

/-- The entire bounded raw-prefix universe is finite for finite objects,
CM75. This does not enumerate it during training or assert small storage. -/
noncomputable instance prefixFintype (O : Type*) [Finite O] (T : ℕ) : Fintype (Prefix O T) :=
  (List.finite_length_le O T).fintype

/-- Raw object/position addresses and the bias address, CM75. -/
abbrev Coordinate (O : Type*) (T : ℕ) := Option (Fin T × O)

/-- All object representations remain learned complex values, CM75. -/
abbrev ObjectWeights (O : Type*) (T : ℕ) := Coordinate O T → ℂ

/-- Read learned embeddings from the actual immutable prefix, CM75.
The input object identity only selects a coordinate; its value is learned. -/
noncomputable def rawInput {O : Type*} {T : ℕ} (tokens : Prefix O T) :
    ObjectWeights O T →ₗ[ℝ] ℂ :=
  LinearMap.proj none + ∑ i : Fin T,
    match tokens.val[i.val]? with
    | none => 0
    | some o => LinearMap.proj (some (i, o))

/-- Actual prefix processing reads every visible learned position/object
embedding and the learned bias, CM75. No feature vector is supplied by a task. -/
theorem rawInput_apply {O : Type*} {T : ℕ} (tokens : Prefix O T) (w : ObjectWeights O T) :
    rawInput tokens w = w none + ∑ i : Fin T,
      match tokens.val[i.val]? with
      | none => 0
      | some o => w (some (i, o)) := by
  simp only [rawInput, LinearMap.add_apply, LinearMap.proj_apply, LinearMap.sum_apply]
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  cases tokens.val[i.val]? <;> rfl

/-- The complete interface outputs logits over the original object type,
CM75. Prefix data are constant and every model weight is in `p`. -/
noncomputable def sequenceLogits {O J : Type*} [Fintype O] [Fintype J] {T : ℕ}
    (p : Parameters J O (ObjectWeights O T)) (tokens : Prefix O T) : O → ℝ :=
  complexLogits p (rawInput tokens)

/-- The ordinary next-object training loss for the actual raw-sequence
complex network, CM75. No internal states are supervised. -/
noncomputable def sequenceLoss {O J : Type*} [Fintype O] [Fintype J] {T : ℕ}
    (tokens : Prefix O T) (target : O) (p : Parameters J O (ObjectWeights O T)) : ℝ :=
  Landscape.Neural.crossEntropy (sequenceLogits p tokens) target

/-- Every admitted prefix has normalized full-vocabulary probabilities,
CM75. A witness object only establishes that the vocabulary is nonempty. -/
theorem sequence_probability_sum {O J : Type*} [Fintype O] [Fintype J] {T : ℕ}
    (p : Parameters J O (ObjectWeights O T)) (tokens : Prefix O T) (witness : O) :
    ∑ c, Geometry.probability (sequenceLogits p tokens) c = 1 :=
  complexProbability_sum p (rawInput tokens) witness

/-- Actual next-object CE is convex on the complete raw-input cone, CM75.
For a guarantee at every admitted prefix, `region` must be `Set.univ`;
restricting it to observed training prefixes does not prove more. -/
theorem sequenceLoss_convex {O J : Type*} [Fintype O] [Fintype J] {T : ℕ}
    (region : Set (Prefix O T)) (mask : J → Prefix O T × Bool → Bool)
    {tokens : Prefix O T} (ht : tokens ∈ region) (target : O) :
    ConvexOn ℝ (complexDomain rawInput region mask : Set (Parameters J O (ObjectWeights O T)))
      (sequenceLoss tokens target) := by
  unfold sequenceLoss sequenceLogits
  have h := complexLoss_convex (E := ObjectWeights O T) (X := Prefix O T) (J := J) (C := O)
    (rawInput (O := O) (T := T)) region mask ht target
  unfold complexLoss at h
  exact h

/-- Complete real-coordinate storage before optimizer buffers and recovered
factor caches, CM75. Both components, all classes, and both signs are charged. -/
theorem sequence_real_coordinate_count (O J : Type*) [Fintype O] [Fintype J] (T : ℕ) :
    Fintype.card (Bool × J × O × Bool × Coordinate O T) =
      4 * Fintype.card J * Fintype.card O * (1 + T * Fintype.card O) := by
  simp only [Fintype.card_prod, Fintype.card_bool, Fintype.card_option, Fintype.card_fin]
  ring

example := sequenceLoss_convex (O := Bool) (J := Unit) (T := 3)
  Set.univ (fun _ _ => true) (tokens := ⟨[true, false], by decide⟩) (Set.mem_univ _) true

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge

module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Reader
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Renaming.Prediction

/-!
# A learned complex sequence predicts the next original object

Source: `review/convex/moments.md`, CM86. Both numerical encoders and the
readout are recovered from the jointly trained rectangular matrices. Inference
performs raw learned lookups, a complex interaction and the learned readout;
only then are real logits and ordinary original-object probabilities formed.
The proof identities do not replace the definition of this physical path.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear

open scoped BigOperators Matrix
open LearnedHinge (Prefix)

variable {O : Type*} [Fintype O] [DecidableEq O] {T : ℕ}

/-- All classes, query objects, positions and ordered pairs are learned,
CM86. The same parameters are shared across all admitted input prefixes. -/
abbrev SequenceParameters (O : Type*) (T : ℕ) :=
  Parameters (QueryCoordinate O) (ContextCoordinate O T) O

/-- Actual raw learned complex computation for one original output object,
CM86. Both encoders are applied before their complex interaction and readout. -/
noncomputable def sequenceBranch
    (M : Matrix (QueryCoordinate O) (ContextCoordinate O T) ℂ)
    (tokens : Prefix O (T + 1)) : ℂ :=
  (readout M : ℂ) * (1 + Complex.I) *
    (star (fun h => readQuery tokens ((leftRepresentation M).conjTranspose h)) ⬝ᵥ
      (fun h => readContext (past tokens) ((rightRepresentation M).conjTranspose h)))

/-- The actual learned raw path has the exact two-encoder interpretation,
CM86. Repeated adjacent pairs have their full multiplicity. -/
theorem sequenceBranch_eq
    (M : Matrix (QueryCoordinate O) (ContextCoordinate O T) ℂ)
    (tokens : Prefix O (T + 1)) :
    sequenceBranch M tokens = forward M (queryRow tokens) (contextRow (past tokens)) := by
  have hq : (fun h => readQuery tokens ((leftRepresentation M).conjTranspose h)) =
      (leftRepresentation M).conjTranspose *ᵥ queryRow tokens := by
    funext h
    exact (queryRow_read tokens _).symm
  have hc : (fun h => readContext (past tokens) ((rightRepresentation M).conjTranspose h)) =
      (rightRepresentation M).conjTranspose *ᵥ contextRow (past tokens) := by
    funext h
    exact (contextRow_read (past tokens) _).symm
  rw [sequenceBranch, hq, hc]
  rfl

/-- The output classes are precisely the same constant objects as the input,
CM86. Real parts are taken after the complete learned complex computation. -/
noncomputable def sequenceLogits (p : SequenceParameters O T)
    (tokens : Prefix O (T + 1)) : O → ℝ := fun c => (sequenceBranch (p c) tokens).re

/-- Raw original-object logits retain the joint linear interpretation, CM86. -/
theorem sequenceLogits_eq (p : SequenceParameters O T) (tokens : Prefix O (T + 1)) :
    sequenceLogits p tokens = logits p (queryRow tokens) (contextRow (past tokens)) := by
  funext c
  simp only [sequenceLogits, logits, sequenceBranch_eq]

/-- Only the ordinary next-original-object CE receives the target, CM86. -/
noncomputable def sequenceLoss (tokens : Prefix O (T + 1)) (target : O)
    (p : SequenceParameters O T) : ℝ :=
  Landscape.Neural.crossEntropy (sequenceLogits p tokens) target

/-- Joint convexity of ordinary answer CE through both raw learned encoders,
CM86. This does not assert convexity in independently varied physical factors. -/
theorem sequenceLoss_convex (tokens : Prefix O (T + 1)) (target : O) :
    ConvexOn ℝ Set.univ (sequenceLoss tokens target : SequenceParameters O T → ℝ) := by
  have h : sequenceLoss tokens target = loss (queryRow tokens) (contextRow (past tokens)) target := by
    funext p
    simp only [sequenceLoss, loss, sequenceLogits_eq]
  rw [h]
  exact loss_convex _ _ _

/-- Every indexed raw minibatch keeps repeated records and short tails,
CM86. This is a loss, not an assertion that an empty batch makes an update. -/
noncomputable def sequenceBatchLoss {N : ℕ} (tokens : Fin N → Prefix O (T + 1))
    (target : Fin N → O) (p : SequenceParameters O T) : ℝ :=
  (N : ℝ)⁻¹ * ∑ i, sequenceLoss (tokens i) (target i) p

/-- The exact mean raw minibatch loss remains jointly convex, CM86. -/
theorem sequenceBatchLoss_convex {N : ℕ} (tokens : Fin N → Prefix O (T + 1))
    (target : Fin N → O) :
    ConvexOn ℝ Set.univ (sequenceBatchLoss tokens target : SequenceParameters O T → ℝ) := by
  have h : sequenceBatchLoss tokens target =
      batchLoss (fun i => queryRow (tokens i)) (fun i => contextRow (past (tokens i))) target := by
    funext p
    simp only [sequenceBatchLoss, batchLoss, sequenceLoss, loss, sequenceLogits_eq]
  rw [h]
  exact batchLoss_convex _ _ _

/-- Every original output object has positive probability, CM86. -/
theorem sequence_probability_pos (p : SequenceParameters O T)
    (tokens : Prefix O (T + 1)) (c : O) :
    0 < Geometry.probability (sequenceLogits p tokens) c := Geometry.probability_pos _ _

/-- The probabilities normalize over the complete original vocabulary,
CM86. The witness certifies that at least one object exists. -/
theorem sequence_probability_sum (p : SequenceParameters O T)
    (tokens : Prefix O (T + 1)) (witness : O) :
    ∑ c, Geometry.probability (sequenceLogits p tokens) c = 1 := Geometry.sum_probability _ witness

/-- Return one original constant object, CM86. Native inference must scan
all objects and specify a deterministic tie order; this mathematical selection
allows any maximizer. The witness is not a supervised answer. -/
noncomputable def nextObject (p : SequenceParameters O T)
    (tokens : Prefix O (T + 1)) (witness : O) : O :=
  Classical.choose (LearnedHinge.Renaming.winner_exists (sequenceLogits p tokens) witness)

/-- The returned original object maximizes the actual raw learned logits, CM86. -/
theorem nextObject_max (p : SequenceParameters O T)
    (tokens : Prefix O (T + 1)) (witness c : O) :
    sequenceLogits p tokens c ≤ sequenceLogits p tokens (nextObject p tokens witness) :=
  Classical.choose_spec (LearnedHinge.Renaming.winner_exists (sequenceLogits p tokens) witness) c

/-- A strict winner is the returned original object, CM86. No latent label
or decoder equivalence class is used. -/
theorem nextObject_of_unique (p : SequenceParameters O T)
    (tokens : Prefix O (T + 1)) (witness target : O)
    (h : ∀ c, c ≠ target → sequenceLogits p tokens c < sequenceLogits p tokens target) :
    nextObject p tokens witness = target := by
  by_contra hne
  exact (not_lt_of_ge (nextObject_max p tokens witness target)) (h _ hne)

omit [DecidableEq O] in
/-- Complete dense real parameter count, before factors, optimizer or batch
buffers, CM86. Both complex components and all original classes remain. -/
theorem sequence_real_coordinate_count :
    Fintype.card (Bool × O × QueryCoordinate O × ContextCoordinate O T) =
      2 * Fintype.card O * (1 + Fintype.card O) *
        (1 + T * Fintype.card O + (if 2 ≤ T then 1 else 0) * Fintype.card O ^ 2) := by
  rw [Fintype.card_prod, Fintype.card_bool, Fintype.card_prod, Fintype.card_prod,
    query_coordinate_count, context_coordinate_count]
  ring

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear

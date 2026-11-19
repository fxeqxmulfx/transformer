module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedQuadratic.Model
public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Projection.Raw
public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Renaming.Prediction

/-!
# Immutable sequences read learned complex quadratic representations

Source: `review/convex/moments.md`, CM85. Objects only address learned bias
and position/object weights. Inference sums those learned values in each
hidden row, squares both complex components, and multiplies a learned
complex readout. The original object is the only supervised target.
The row representation below certifies that computation; it is not a
task-supplied arithmetic feature map or a full-prefix parameter table.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedQuadratic

open scoped BigOperators Matrix ComplexOrder MatrixOrder Matrix.Norms.Elementwise
open LearnedHinge (Prefix Coordinate ObjectWeights rawInput)

variable {O : Type*} [Fintype O] [DecidableEq O] {T : ℕ}

/-- The exact address row of the immutable sequence, used in proofs, CM85. -/
def inputRow (tokens : Prefix O T) : Coordinate O T → ℂ :=
  fun d => (LearnedHinge.Projection.rawRow tokens d : ℝ)

/-- The address row reads precisely the learned raw embeddings, CM85.
Every visible position and the learned bias remain in the actual computation. -/
theorem inputRow_read (tokens : Prefix O T) (w : ObjectWeights O T) :
    w ⬝ᵥ inputRow tokens = rawInput tokens w := by
  rw [LearnedHinge.rawInput_apply]
  unfold dotProduct
  simp only [Fintype.sum_option, inputRow, LearnedHinge.Projection.rawRow,
    Complex.ofReal_one, mul_one, Fintype.sum_prod_type]
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  have choose (v : Option O) :
      (∑ o, w (some (i, o)) * ((if v = some o then 1 else 0 : ℝ) : ℂ)) =
        match v with | none => 0 | some o => w (some (i, o)) := by
    cases v with
    | none => simp
    | some o =>
      simp only [Option.some.injEq]
      calc
        _ = ∑ k, if o = k then w (some (i, k)) else 0 := by
          apply Finset.sum_congr rfl
          intro k hk
          by_cases h : o = k <;> simp [h]
        _ = _ := Fintype.sum_ite_eq o _
  exact choose _

/-- Actual raw learned complex inference in one signed output branch, CM85. -/
noncomputable def sequenceBranch (M : Matrix (Coordinate O T) (Coordinate O T) ℂ)
    (tokens : Prefix O T) : ℂ :=
  (LearnedHinge.amplitude (factor M) ^ 2 : ℝ) * (1 + Complex.I) *
    ((∑ h, Complex.normSq
      (rawInput tokens (LearnedHinge.representation (factor M) h))) : ℝ)

/-- The raw learned computation agrees with the general quadratic network,
CM85. This includes the factor and readout, not only the final logit. -/
theorem sequenceBranch_eq (M : Matrix (Coordinate O T) (Coordinate O T) ℂ)
    (tokens : Prefix O T) : sequenceBranch M tokens = branch M (inputRow tokens) := by
  unfold sequenceBranch branch energy
  congr 2
  apply Finset.sum_congr rfl
  intro h hh
  congr 1
  exact (inputRow_read tokens (LearnedHinge.representation (factor M) h)).symm

/-- Logits are indexed by the same immutable object set as the input, CM85. -/
noncomputable def sequenceLogits (p : Parameters (Coordinate O T) O)
    (tokens : Prefix O T) : O → ℝ :=
  fun c => (sequenceBranch (p c true) tokens - sequenceBranch (p c false) tokens).re

/-- The entire original-object forward has the proved joint interpretation,
CM85. No class, hidden coordinate, or complex component is removed. -/
theorem sequenceLogits_eq (p : Parameters (Coordinate O T) O) (tokens : Prefix O T) :
    sequenceLogits p tokens = logits p (inputRow tokens) := by
  funext c
  simp only [sequenceLogits, logits, sequenceBranch_eq]

/-- Usual next-original-object CE through the raw learned network, CM85. -/
noncomputable def sequenceLoss (tokens : Prefix O T) (target : O)
    (p : Parameters (Coordinate O T) O) : ℝ :=
  Landscape.Neural.crossEntropy (sequenceLogits p tokens) target

/-- Ordinary raw-sequence CE is jointly convex at every admitted prefix,
CM85. The same domain works for all prefixes, without enumeration. -/
theorem sequenceLoss_convex (tokens : Prefix O T) (target : O) :
    ConvexOn ℝ (domain : Set (Parameters (Coordinate O T) O)) (sequenceLoss tokens target) := by
  have hfun : sequenceLoss tokens target = loss (inputRow tokens) target := by
    funext p
    simp only [sequenceLoss, loss, sequenceLogits_eq]
  rw [hfun]
  exact loss_convex (inputRow tokens) target

/-- The original-object prediction probabilities retain the whole vocabulary,
CM85. A witness object establishes that the vocabulary is nonempty. -/
theorem sequence_probability_sum (p : Parameters (Coordinate O T) O)
    (tokens : Prefix O T) (witness : O) :
    ∑ c, Geometry.probability (sequenceLogits p tokens) c = 1 := by
  rw [sequenceLogits_eq]
  exact probability_sum p (inputRow tokens) witness

/-- Return an original immutable object, CM85. The fixed witness establishes
vocabulary nonemptiness; it is not the supervised target. A native implementation
must scan the full vocabulary and declare a deterministic tie order. -/
noncomputable def nextObject (p : Parameters (Coordinate O T) O)
    (tokens : Prefix O T) (witness : O) : O :=
  Classical.choose (LearnedHinge.Renaming.winner_exists (sequenceLogits p tokens) witness)

/-- The returned object maximizes the actual learned logits, CM85. -/
theorem nextObject_max (p : Parameters (Coordinate O T) O)
    (tokens : Prefix O T) (witness c : O) :
    sequenceLogits p tokens c ≤ sequenceLogits p tokens (nextObject p tokens witness) :=
  Classical.choose_spec (LearnedHinge.Renaming.winner_exists (sequenceLogits p tokens) witness) c

/-- A strict original-object winner is returned under every tie selection,
CM85. The target enters only this correctness statement and the ordinary loss. -/
theorem nextObject_of_unique (p : Parameters (Coordinate O T) O)
    (tokens : Prefix O T) (witness target : O)
    (h : ∀ c, c ≠ target → sequenceLogits p tokens c < sequenceLogits p tokens target) :
    nextObject p tokens witness = target := by
  by_contra hne
  exact (not_lt_of_ge (nextObject_max p tokens witness target)) (h _ hne)

/-- Full dense real storage count, before factor and optimizer buffers,
CM85. Hermitian redundancy is retained; this is not a low-rank parameter count. -/
theorem sequence_real_coordinate_count (O : Type*) [Fintype O] (T : ℕ) :
    Fintype.card (Bool × O × Bool × Coordinate O T × Coordinate O T) =
      4 * Fintype.card O * (1 + T * Fintype.card O) ^ 2 := by
  simp only [Fintype.card_prod, Fintype.card_bool, Fintype.card_option, Fintype.card_fin]
  ring

end Transformer.Grokking.Convex.ConvexChannel.LearnedQuadratic

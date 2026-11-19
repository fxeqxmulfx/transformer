module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Unseen

/-!
# Two shared learned passes under an involutive object renaming

Source: `review/convex/moments.md`, CM78. Both passes use the same complete
learned representation and readout. The architecture averages whole complex
outputs after aligning output objects. This imposes an explicit equivariance
restriction; it does not preserve arbitrary unsymmetrized model predictions.
Objects and targets remain immutable, and targets are used only by ordinary CE.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Renaming

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {O J : Type*} {T : ℕ}

/-- Apply a fixed object renaming to every original sequence position, CM78.
The length and all positions are retained. This supplies no feature values. -/
def renamed (σ : Equiv.Perm O) (tokens : Prefix O T) : Prefix O T :=
  ⟨tokens.val.map σ, by simpa using tokens.property⟩

/-- Renaming twice restores the entire immutable sequence, CM78. -/
theorem renamed_involutive (σ : Equiv.Perm O) (hσ : Function.Involutive σ) :
    Function.Involutive (renamed (T := T) σ) := by
  intro tokens
  apply Subtype.ext
  change (tokens.val.map σ).map σ = tokens.val
  simp only [List.map_map]
  have he : (σ ∘ σ) = id := funext hσ
  rw [he, List.map_id]

/-- The identity renaming retains the complete sequence, CM78. -/
theorem renamed_identity (tokens : Prefix O T) : renamed (Equiv.refl O) tokens = tokens := by
  apply Subtype.ext
  exact List.map_id _

variable [Fintype O] [Fintype J]

/-- Two actual complex forwards share every learned coordinate, CM78.
The fixed average is architecture wiring, not an independent learned head. -/
noncomputable def pairedForward (σ : Equiv.Perm O)
    (p : Parameters J O (ObjectWeights O T)) (tokens : Prefix O T) (c : O) : ℂ :=
  (1 / 2 : ℂ) * (complexForward p (rawInput tokens) c +
    complexForward p (rawInput (renamed σ tokens)) (σ c))

/-- Original next-object logits retain the complete complex computation, CM78. -/
noncomputable def pairedLogits (σ : Equiv.Perm O)
    (p : Parameters J O (ObjectWeights O T)) (tokens : Prefix O T) (c : O) : ℝ :=
  (pairedForward σ p tokens c).re

/-- Ordinary full-vocabulary next-object CE; no internal target is supplied, CM78. -/
noncomputable def pairedLoss (σ : Equiv.Perm O) (tokens : Prefix O T) (target : O)
    (p : Parameters J O (ObjectWeights O T)) : ℝ := crossEntropy (pairedLogits σ p tokens) target

/-- The actual complex paired network is equivariant for every learned point,
CM78. This does not need an observed-data or activation-cone assumption. -/
theorem pairedForward_equivariant (σ : Equiv.Perm O) (hσ : Function.Involutive σ)
    (p : Parameters J O (ObjectWeights O T)) (tokens : Prefix O T) (c : O) :
    pairedForward σ p (renamed σ tokens) (σ c) = pairedForward σ p tokens c := by
  unfold pairedForward
  rw [renamed_involutive σ hσ, hσ]
  rw [add_comm]

/-- Taking real logits preserves the proved full complex equivariance, CM78. -/
theorem pairedLogits_equivariant (σ : Equiv.Perm O) (hσ : Function.Involutive σ)
    (p : Parameters J O (ObjectWeights O T)) (tokens : Prefix O T) (c : O) :
    pairedLogits σ p (renamed σ tokens) (σ c) = pairedLogits σ p tokens c := by
  exact congrArg Complex.re (pairedForward_equivariant σ hσ p tokens c)

/-- A base prediction already respecting the chosen pair is unchanged,
CM78. Arbitrary base functions are not claimed to satisfy this premise. -/
theorem pairedForward_preserves (σ : Equiv.Perm O)
    (p : Parameters J O (ObjectWeights O T)) (tokens : Prefix O T) (c : O)
    (h : complexForward p (rawInput (renamed σ tokens)) (σ c) =
      complexForward p (rawInput tokens) c) :
    pairedForward σ p tokens c = complexForward p (rawInput tokens) c := by
  rw [pairedForward, h]
  ring

/-- Real logits are the aligned average of two actual learned passes, CM78.
No imaginary feature or class is removed from either original pass. -/
theorem pairedLogits_average (σ : Equiv.Perm O)
    (p : Parameters J O (ObjectWeights O T)) (tokens : Prefix O T) (c : O) :
    pairedLogits σ p tokens c =
      (sequenceLogits p tokens c + sequenceLogits p (renamed σ tokens) (σ c)) / 2 := by
  simp [pairedLogits, pairedForward, Complex.mul_re, sequenceLogits, complexLogits]
  ring

/-- Every original output object keeps a strictly positive probability, CM78. -/
theorem pairedProbability_pos (σ : Equiv.Perm O)
    (p : Parameters J O (ObjectWeights O T)) (tokens : Prefix O T) (c : O) :
    0 < Geometry.probability (pairedLogits σ p tokens) c := Geometry.probability_pos _ _

/-- The same full-vocabulary softmax normalizes, CM78. The witness only
establishes that the original object set is nonempty. -/
theorem pairedProbability_sum (σ : Equiv.Perm O)
    (p : Parameters J O (ObjectWeights O T)) (tokens : Prefix O T) (witness : O) :
    ∑ c, Geometry.probability (pairedLogits σ p tokens) c = 1 := Geometry.sum_probability _ witness

example := renamed_involutive (T := 2) (Equiv.refl Bool) (fun _ => rfl)
example := pairedForward_equivariant (J := Unit) (T := 2) (Equiv.refl Bool)
  (fun _ => rfl) 0 ⟨[false, true], by decide⟩ true
example := pairedLogits_equivariant (J := Unit) (T := 2) (Equiv.refl Bool)
  (fun _ => rfl) 0 ⟨[false, true], by decide⟩ true
example := pairedForward_preserves (J := Unit) (T := 2) (Equiv.refl Bool) 0
  ⟨[false, true], by decide⟩ true (by rw [renamed_identity]; rfl)

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Renaming

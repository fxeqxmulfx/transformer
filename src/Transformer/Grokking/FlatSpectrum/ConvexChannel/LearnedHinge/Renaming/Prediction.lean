module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Renaming.Identification
public import Mathlib.Data.Finset.Max

/-!
# Returning an original immutable object from the learned sequence model

Source: `review/convex/moments.md`, CM78. Full-vocabulary logits are internal.
The mathematical next-object prediction is a maximizing original object.
Any tie selection obeys the same low-CE and transfer guarantees. Classical
choice here states existence; a native implementation must specify its tie
order and scan all output objects. No target enters the predictor.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Renaming

variable {O J : Type*} [Fintype O] [Fintype J] {T : ℕ}

/-- A maximum exists in the original nonempty finite object set, CM78. -/
theorem winner_exists (logits : O → ℝ) (witness : O) :
    ∃ c, ∀ d, logits d ≤ logits c := by
  obtain ⟨c, hc, hmax⟩ := Finset.exists_max_image Finset.univ logits ⟨witness, Finset.mem_univ _⟩
  exact ⟨c, fun d => hmax d (Finset.mem_univ _)⟩

/-- Return one original object. The fixed witness only establishes nonemptiness;
it is not an observed or missing target and is not read by the network, CM78. -/
noncomputable def predict (σ : Equiv.Perm O) (p : Parameters J O (ObjectWeights O T))
    (tokens : Prefix O T) (witness : O) : O :=
  Classical.choose (winner_exists (pairedLogits σ p tokens) witness)

/-- The returned original object maximizes every retained logit, CM78. -/
theorem predict_max (σ : Equiv.Perm O) (p : Parameters J O (ObjectWeights O T))
    (tokens : Prefix O T) (witness : O) (c : O) :
    pairedLogits σ p tokens c ≤ pairedLogits σ p tokens (predict σ p tokens witness) :=
  Classical.choose_spec (winner_exists (pairedLogits σ p tokens) witness) c

/-- A strict winner is the returned immutable object for every tie rule,
CM78. The target is used only by this correctness theorem. -/
theorem predict_of_unique (σ : Equiv.Perm O) (p : Parameters J O (ObjectWeights O T))
    (tokens : Prefix O T) (witness target : O)
    (h : ∀ c, c ≠ target → pairedLogits σ p tokens c < pairedLogits σ p tokens target) :
    predict σ p tokens witness = target := by
  by_contra hne
  exact (not_lt_of_ge (predict_max σ p tokens witness target)) (h _ hne)

/-- A low ordinary CE fit returns the actual target object, CM78. -/
theorem predict_of_loss (σ : Equiv.Perm O) (p : Parameters J O (ObjectWeights O T))
    (tokens : Prefix O T) (witness target : O) (hfit : pairedLoss σ tokens target p < Real.log 2) :
    predict σ p tokens witness = target :=
  predict_of_unique σ p tokens witness target (low_loss_unique σ p tokens target hfit)

/-- The next-object output transfers to the renamed prefix without that
prefix's target entering the model, CM78. Actual task correctness also needs
the answer symmetry stated in `task_transfer`. -/
theorem predict_renamed (σ : Equiv.Perm O) (hσ : Function.Involutive σ)
    (p : Parameters J O (ObjectWeights O T)) (tokens : Prefix O T) (witness target : O)
    (hfit : pairedLoss σ tokens target p < Real.log 2) :
    predict σ p (renamed σ tokens) witness = σ target :=
  predict_of_unique σ p (renamed σ tokens) witness (σ target)
    (renamed_unique σ hσ p tokens target hfit)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Renaming

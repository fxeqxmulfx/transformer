module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Fallback.Batch

/-!
# Preserve the prior head except on newly covered inputs

Source: `review/flat_spectrum.md`, CM39. Both support predicates are fixed
input functions. The prior block remains active on its own support and where
the supplemental decoder also refuses. Only newly covered inputs use the
supplemental block. These are exact logit identities, without target premises.
The CM38 feature bounds, charged budgets and affine convex domain apply to
these actual features, with the supplemental block frozen.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.CoverageExtension

open Fallback

variable {K L X C S : Type*} [Fintype K] [Fintype L]

/-- Keep the prior unless it refuses and the supplement accepts, CM39.
Neither support predicate depends on the trained coefficients or targets. -/
def keepPrior (priorKnown supplementKnown : X → Bool) (x : X) : Bool :=
  priorKnown x || !supplementKnown x

/-- Prior support and joint refusal both preserve the actual prior
logits, CM39. Agreement with the supplemental head is not required. -/
theorem prior_logits (priorKnown supplementKnown : X → Bool)
    (prior : K → X → ℝ) (supplement : L → X → ℝ)
    (W : K → C → ℝ) (V : L → C → ℝ) {x : X}
    (hx : priorKnown x = true ∨ supplementKnown x = false) :
    headLogits (features (keepPrior priorKnown supplementKnown) prior supplement)
      (coefficients W V) x = headLogits prior W x := by
  rw [logits]
  rcases hx with hp | hs
  · simp [keepPrior, hp]
  · simp [keepPrior, hs]

/-- Only a prior refusal and a supplemental acceptance activate the
supplemental logits, CM39. Acceptance is not a correctness certificate. -/
theorem newly_covered_logits (priorKnown supplementKnown : X → Bool)
    (prior : K → X → ℝ) (supplement : L → X → ℝ)
    (W : K → C → ℝ) (V : L → C → ℝ) {x : X}
    (hp : priorKnown x = false) (hs : supplementKnown x = true) :
    headLogits (features (keepPrior priorKnown supplementKnown) prior supplement)
      (coefficients W V) x = headLogits supplement V x := by
  rw [logits]
  simp [keepPrior, hp, hs]

/-- Any actual logit change is confined to newly covered inputs, CM39.
The converse need not hold: the two heads may already agree there. -/
theorem changed_only_newly (priorKnown supplementKnown : X → Bool)
    (prior : K → X → ℝ) (supplement : L → X → ℝ)
    (W : K → C → ℝ) (V : L → C → ℝ) {x : X}
    (hchange : headLogits (features (keepPrior priorKnown supplementKnown) prior supplement)
      (coefficients W V) x ≠ headLogits prior W x) :
    priorKnown x = false ∧ supplementKnown x = true := by
  cases hp : priorKnown x with
  | true => exact False.elim (hchange (prior_logits _ _ _ _ _ _ (Or.inl hp)))
  | false =>
    cases hs : supplementKnown x with
    | false => exact False.elim (hchange (prior_logits _ _ _ _ _ _ (Or.inr hs)))
    | true => exact ⟨rfl, rfl⟩

variable [Fintype C]

/-- Every prior-covered original record retains its actual logits, so
the normalized raw-label CE is the prior's entire objective function, CM39.
Record IDs preserve repeated inputs and the original batch denominator. -/
theorem batch_loss_eq (priorKnown supplementKnown : X → Bool)
    (prior : K → X → ℝ) (supplement : L → X → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S)
    (covered : ∀ s ∈ batch, priorKnown (inputs s) = true)
    (W : K → C → ℝ) (V : L → C → ℝ) :
    batchLoss (features (keepPrior priorKnown supplementKnown) prior supplement)
      inputs labels batch (coefficients W V) = batchLoss prior inputs labels batch W := by
  apply Fallback.batch_loss_eq
  intro s hs
  simp [keepPrior, covered s hs]

/-- Exact function equality transfers the Fréchet derivative in every
prior coefficient, CM39. The supplement is frozen; the two support predicates
are fixed. At nondifferentiability points the library convention is retained. -/
theorem batch_fderiv_eq (priorKnown supplementKnown : X → Bool)
    (prior : K → X → ℝ) (supplement : L → X → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S)
    (covered : ∀ s ∈ batch, priorKnown (inputs s) = true)
    (V : L → C → ℝ) (W : K → C → ℝ) :
    fderiv ℝ (fun H => batchLoss
      (features (keepPrior priorKnown supplementKnown) prior supplement)
      inputs labels batch (coefficients H V)) W =
      fderiv ℝ (batchLoss prior inputs labels batch) W := by
  apply Fallback.batch_fderiv_eq
  intro s hs
  simp [keepPrior, covered s hs]

end Transformer.Grokking.FlatSpectrum.ConvexDepth.CoverageExtension

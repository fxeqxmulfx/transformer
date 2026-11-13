module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel.Coverage

/-!
# Ordinary training CE cannot distinguish the refined unseen defaults

Source: `review/flat_spectrum.md`, CM36. Derive equality of the actual observed
CE trajectories from CM33's normalized gradient row formula. The two models
may have different latent types and Fourier blocks. Unitarity transfers the
identity to actual complex logits; no equality of unseen targets is assumed.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel

open KernelCodes ConvexDepth Matrix
open Landscape.Neural (crossEntropy)

variable {O Z W A : Type*} [DecidableEq O] [Fintype A] [DecidableEq A]

/-- Equal initial observed rows remain equal through the same actual CE
schedule even with different default spaces, CM36. The batch denominator,
raw label, rate and clipping are all inherited from the executed step. -/
theorem observed_trace_seen_eq (bound : ℝ) (defaults : Z → A → ℝ) (other : W → A → ℝ)
    (labels : O → A) (batches : ℕ → Finset O) (rates : ℕ → ℝ)
    (initial : Matrix (O ⊕ Z) A ℝ) (alternate : Matrix (O ⊕ W) A ℝ)
    (hi : ∀ o, initial (.inl o) = alternate (.inl o)) (n : ℕ) :
    ∀ o, observedCETrace bound defaults labels batches rates initial n (.inl o) =
      observedCETrace bound other labels batches rates alternate n (.inl o) := by
  induction n with
  | zero => exact hi
  | succ n ih =>
    intro o
    rw [observedCETrace, observedCETrace, observedCEStep_row, observedCEStep_row, ih o]

variable {X : Type*} [DecidableEq X]
variable [Fintype Z] [DecidableEq Z] [Fintype W] [DecidableEq W]

/-- Different latent maps and representatives produce identical observed
complex logits under the same table-CE schedule and raw-margin initialization,
CM36. Fourier block sizes are allowed to differ. -/
theorem refined_seen_logits_eq (observed : Finset X) (latent : X → Z) (otherLatent : X → W)
    (labels : ↥observed → A) (representative : Z → A) (otherRepresentative : W → A)
    (batches : ℕ → Finset ↥observed) (rates : ℕ → ℝ) (n : ℕ)
    {x : X} (hx : x ∈ observed) :
    headLogits (complexFeatures (fiberFourier (refinedBlock observed latent)))
      (complexCoefficients (refinedCEHead observed latent labels representative batches rates n))
      (refinedState observed latent x) =
    headLogits (complexFeatures (fiberFourier (refinedBlock observed otherLatent)))
      (complexCoefficients (refinedCEHead observed otherLatent labels otherRepresentative batches rates n))
      (refinedState observed otherLatent x) := by
  simp only [refinedCEHead, tableHead_logits _ (refinedBlock_unitary observed latent),
    tableHead_logits _ (refinedBlock_unitary observed otherLatent), refinedState, dite_eq_left hx]
  exact observed_trace_seen_eq 32 (latentDefaultMargins representative)
    (latentDefaultMargins otherRepresentative) labels batches rates
    (refinedRawMargins observed labels representative)
    (refinedRawMargins observed labels otherRepresentative) (fun _ => rfl) n ⟨x, hx⟩

variable {R I J C D K L : Type*} [CommRing R]
variable [Fintype I] [Fintype J] [DecidableEq I] [DecidableEq J]
variable {M : Submodule R (Coordinates I J C → R)}
variable {N : Submodule R (Coordinates I J D → R)}

/-- The original and grouped complete-code heads are instances of the
observed-logit identity, CM36. No relationship between the two codes or their
unseen predictions is smuggled into the hypotheses. -/
theorem vectorCE_seen_logits_eq (raw : Module.Basis K R M) (grouped : Module.Basis L R N)
    (observed : Finset (I × J)) (labels : ↥observed → A)
    (representative : CodeStates raw → A) (otherRepresentative : CodeStates grouped → A)
    (batches : ℕ → Finset ↥observed) (rates : ℕ → ℝ) (n : ℕ)
    {x : I × J} (hx : x ∈ observed) :
    vectorCELogits raw observed labels representative batches rates n x =
      vectorCELogits grouped observed labels otherRepresentative batches rates n x := by
  unfold vectorCELogits
  exact refined_seen_logits_eq observed _ _ labels representative otherRepresentative
    batches rates n hx

/-- Empirical ordinary raw-vocabulary CE is exactly tied for these actual
complex trajectories, CM36. Normalizing by the common observation count does
not select one unseen completion over the other. -/
theorem vectorCE_training_loss_eq (raw : Module.Basis K R M) (grouped : Module.Basis L R N)
    (observed : Finset (I × J)) (labels : ↥observed → A)
    (representative : CodeStates raw → A) (otherRepresentative : CodeStates grouped → A)
    (batches : ℕ → Finset ↥observed) (rates : ℕ → ℝ) (n : ℕ) :
    (∑ x : ↥observed, crossEntropy
      (vectorCELogits raw observed labels representative batches rates n x.val) (labels x)) /
      observed.card =
    (∑ x : ↥observed, crossEntropy
      (vectorCELogits grouped observed labels otherRepresentative batches rates n x.val) (labels x)) /
      observed.card := by
  congr 1
  apply Finset.sum_congr rfl
  intro x _
  rw [vectorCE_seen_logits_eq raw grouped observed labels representative otherRepresentative
    batches rates n x.property]

example := observed_trace_seen_eq 32 (fun z c : Bool => if c = z then 24 else 0)
  (fun (_ : Unit) (c : Bool) => if c then 24 else 0) id (fun _ => Finset.univ) (fun _ => 1)
  (fun (s : Bool ⊕ Bool) c => if c = s.elim id id then 24 else 0)
  (fun (s : Bool ⊕ Unit) c => if c = s.elim id (fun _ => true) then 24 else 0)
  (fun _ => rfl) 3

example := refined_seen_logits_eq ({false} : Finset Bool) id (fun _ => ())
  (fun x => !x.val) id (fun _ => true) (fun _ => Finset.univ) (fun _ => 1) 3
  (x := false) (by simp)

example := vectorCE_seen_logits_eq cornerRawBasis cornerGroupedBasis cornerObserved cornerHeadLabels
  (fun _ => (1, false)) (fun _ => (0, false)) cornerBatches (fun _ => 1) 4
  (x := (true, false)) (by decide)

example := vectorCE_training_loss_eq cornerRawBasis cornerGroupedBasis cornerObserved cornerHeadLabels
  (fun _ => (1, false)) (fun _ => (0, false)) (fun _ => Finset.univ) (fun _ => 1) 0

end Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel

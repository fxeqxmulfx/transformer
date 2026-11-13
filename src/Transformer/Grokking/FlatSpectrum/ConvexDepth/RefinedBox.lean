module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.BoxUpdate
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.FiberFourier
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.QuotientReadout

/-!
# Feasible block updates for the refined quotient model

Source: `review/flat_spectrum.md`, CM32. The actual model uses only the
latent-fiber Fourier blocks, projects proposed observed logits to[-32,32],
and fixes default rows at margin24. Every proposal produces a bounded complex
head with the same unseen logits. CE is convex in feasible table coordinates;
the clipping operation is an optimization projection, not a learned activation.
Correct unseen answers still require the explicit quotient-identification premise.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open Matrix
open Landscape.Neural (crossEntropy)

variable {X Z C D : Type*} [DecidableEq X] [Fintype Z] [DecidableEq Z] [DecidableEq C]

/-- Fixed margin24 representatives of the learned latent codes, CM32.
The native constructor must learn or declare these before CE updates. -/
def latentDefaultMargins (representative : Z → C) : Z → C → ℝ :=
  fun z c => if c = representative z then 24 else 0

/-- Actual bounded complex coefficients after a projected table proposal.
Input identities and the latent map determine the Fourier blocks, CM32. -/
noncomputable def refinedBoxHead (observed : Finset X) (latent : X → Z)
    (representative : Z → C) (proposal : Matrix (↥observed ⊕ Z) C ℝ) :
    Matrix (↥observed ⊕ Z) C ℂ :=
  tableHead (fiberFourier (refinedBlock observed latent))
    (boundedTableProposal 32 (latentDefaultMargins representative) proposal)

/-- Every proposal fits the actual complex budget, with all observations
and defaults charged. No hypothesis asserts update feasibility, CM32. -/
theorem refinedBox_budget (observed : Finset X) (latent : X → Z)
    (representative : Z → C) (proposal : Matrix (↥observed ⊕ Z) C ℝ) :
    complexCoefficients (refinedBoxHead observed latent representative proposal) ∈
      headBudget (64 * ((observed.card : ℝ) + (Fintype.card Z : ℝ))) := by
  have h := boundedTableProposal_budget _ (refinedBlock_unitary observed latent)
    (latentDefaultMargins representative)
    (by intro z c; by_cases hc : c = representative z <;> norm_num [latentDefaultMargins, hc])
    proposal
  simpa only [refinedBoxHead, Fintype.card_coe] using h

/-- Every projected proposal keeps its exact declared default at an
unobserved input, in the actual real/imaginary forward pass, CM32. -/
theorem refinedBox_unseen_logits (observed : Finset X) (latent : X → Z)
    (representative : Z → C) (proposal : Matrix (↥observed ⊕ Z) C ℝ)
    {x : X} (hx : x ∉ observed) :
    headLogits (complexFeatures (fiberFourier (refinedBlock observed latent)))
      (complexCoefficients (refinedBoxHead observed latent representative proposal))
      (refinedState observed latent x) = latentDefaultMargins representative (latent x) := by
  rw [refinedBoxHead, tableHead_logits _ (refinedBlock_unitary observed latent)]
  simp only [refinedState, dite_eq_right hx]
  rfl

/-- Default preservation entails preservation of a compatible pooled
readout. Unequal group sizes are covered by the vocabulary bound, CM32. -/
theorem refinedBox_unseen_pooled [Fintype C] [DecidableEq D]
    (observed : Finset X) (latent : X → Z) (representative : Z → C)
    (quotient : C → D) (code : Z → D)
    (hr : ∀ z, quotient (representative z) = code z) (hcard : Fintype.card C ≤ 1024)
    (proposal : Matrix (↥observed ⊕ Z) C ℝ) {x : X} (hx : x ∉ observed)
    {d : D} (hd : d ≠ code (latent x)) :
    quotientScore quotient
      (headLogits (complexFeatures (fiberFourier (refinedBlock observed latent)))
        (complexCoefficients (refinedBoxHead observed latent representative proposal))
        (refinedState observed latent x)) d <
    quotientScore quotient
      (headLogits (complexFeatures (fiberFourier (refinedBlock observed latent)))
        (complexCoefficients (refinedBoxHead observed latent representative proposal))
        (refinedState observed latent x)) (code (latent x)) := by
  rw [refinedBox_unseen_logits _ _ _ _ hx]
  rw [← hr (latent x)] at hd ⊢
  exact quotientScore_margin_strict _ _ _ (native_margin_exceeds_vocabulary hcard) hd

/-- When the latent code separately identifies the target quotient, every
projected update retains the correct unseen decoded answer, CM32. The target
is used only in this theorem's identification premise, never in the model. -/
theorem refinedBox_unseen_decoded [Fintype C] [DecidableEq D]
    (observed : Finset X) (latent : X → Z) (representative : Z → C)
    (quotient : C → D) (code : Z → D)
    (hr : ∀ z, quotient (representative z) = code z) (hcard : Fintype.card C ≤ 1024)
    (proposal : Matrix (↥observed ⊕ Z) C ℝ) (target : X → C) {x : X} (hx : x ∉ observed)
    (hi : code (latent x) = quotient (target x)) {d : D} (hd : d ≠ quotient (target x)) :
    quotientScore quotient
      (headLogits (complexFeatures (fiberFourier (refinedBlock observed latent)))
        (complexCoefficients (refinedBoxHead observed latent representative proposal))
        (refinedState observed latent x)) d <
    quotientScore quotient
      (headLogits (complexFeatures (fiberFourier (refinedBlock observed latent)))
        (complexCoefficients (refinedBoxHead observed latent representative proposal))
        (refinedState observed latent x)) (quotient (target x)) := by
  rw [← hi] at hd ⊢
  exact refinedBox_unseen_pooled _ _ _ _ _ hr hcard proposal hx hd

/-- The recorded raw assignment defines an initializer table. Observed
labels enter the head, never the fixed feature extractor, CM32. -/
def refinedRawMargins (observed : Finset X) (labels : ↥observed → C)
    (representative : Z → C) : Matrix (↥observed ⊕ Z) C ℝ :=
  fun s c => if c = refinedLabels observed labels representative s then 24 else 0

/-- Clipping preserves the initial recorded raw margins at every observed
input. A latent collision no longer forces shared logits, CM32. -/
theorem refinedBox_initial_seen (observed : Finset X) (latent : X → Z)
    (labels : ↥observed → C) (representative : Z → C) {x : X} (hx : x ∈ observed) :
    headLogits (complexFeatures (fiberFourier (refinedBlock observed latent)))
      (complexCoefficients (refinedBoxHead observed latent representative
        (refinedRawMargins observed labels representative))) (refinedState observed latent x) =
      fun c => if c = labels ⟨x, hx⟩ then 24 else 0 := by
  rw [refinedBoxHead, tableHead_logits _ (refinedBlock_unitary observed latent)]
  funext c
  by_cases hc : c = labels ⟨x, hx⟩ <;>
    norm_num [refinedState, hx, boundedTableProposal, boxClip, refinedRawMargins, refinedLabels, hc]

/-- The bounded block initializer escapes the colliding-code log2 CE
floor using original raw labels and ordinary full-vocabulary CE, CM32. -/
theorem refinedBox_initial_ce [Fintype C] (observed : Finset X) (latent : X → Z)
    (labels : ↥observed → C) (representative : Z → C) (hcard : Fintype.card C ≤ 1024)
    {x : X} (hx : x ∈ observed) :
    crossEntropy
      (headLogits (complexFeatures (fiberFourier (refinedBlock observed latent)))
        (complexCoefficients (refinedBoxHead observed latent representative
          (refinedRawMargins observed labels representative))) (refinedState observed latent x))
      (labels ⟨x, hx⟩) < Real.log 2 := by
  rw [refinedBox_initial_seen _ _ _ _ hx]
  exact native_margin_ce_lt_log_two _ hcard

/-- The trained objective is convex on the feasible table domain. The
code is fixed and both complex coefficient coordinates are linear in T.
This does not claim convexity as a function of an unclipped proposal, CM32. -/
theorem refinedBox_loss_convex [Fintype C] {S : Type*} [Fintype S]
    (observed : Finset X) (latent : X → Z) (representative : Z → C)
    (inputs : S → X) (labels : S → C) :
    ConvexOn ℝ (defaultTableDomain 32 (latentDefaultMargins representative))
      (fun T => headLoss (complexFeatures (fiberFourier (refinedBlock observed latent)))
        (fun s => refinedState observed latent (inputs s)) labels
        (complexCoefficients (tableHead (fiberFourier (refinedBlock observed latent)) T))) := by
  refine ⟨defaultTableDomain_convex _ _, ?_⟩
  intro T hT V hV a b ha hb hab
  exact (tableHead_loss_convex _ (refinedBlock_unitary observed latent) _ labels 32).2
    hT.1 hV.1 ha hb hab

example := refinedBox_budget ({false} : Finset Bool) id (id : Bool → Bool) (fun _ _ => 1000)

example := refinedBox_unseen_logits ({false} : Finset Bool) id (id : Bool → Bool)
  (fun _ _ => 1000) (x := true) (by simp)

example := refinedBox_unseen_pooled ({false} : Finset Bool) id id (id : Bool → Bool) id
  (fun _ => rfl) (by decide) (fun _ _ => 1000) (x := true) (by simp) (d := false) (by decide)

example := refinedBox_unseen_decoded ({false} : Finset Bool) id id (id : Bool → Bool) id
  (fun _ => rfl) (by decide) (fun _ _ => 1000) id
  (x := true) (by simp) rfl (d := false) (by decide)

example := refinedBox_initial_seen ({false} : Finset Bool) id
  (fun x => !x.val) (id : Bool → Bool) (x := false) (by simp)

example := refinedBox_initial_ce ({false} : Finset Bool) id
  (fun x => !x.val) (id : Bool → Bool) (by decide) (x := false) (by simp)

example := refinedBox_loss_convex ({false} : Finset Bool) id (id : Bool → Bool)
  (fun s : Bool => s) (fun s => !s)

end Transformer.Grokking.FlatSpectrum.ConvexDepth

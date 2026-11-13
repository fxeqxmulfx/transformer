module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedCE
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.RefinedBox

/-!
# Observed-label preservation in the actual refined complex model

Source: `review/flat_spectrum.md`, CM33. Start from the recorded margin24
table and execute normalized full/minibatch ordinary-CE steps. Every finite
iterate fits the64N complex budget, preserves every observed raw winner and
retains the declared unseen logits. Unknown target identification and finite
precision remain separate; the theorem does not infer them from training CE.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open Matrix
open Landscape.Neural (crossEntropy)

variable {X Z C : Type*} [DecidableEq C]

/-- The native fixed defaults satisfy the32-box at every latent code,
CM33. This is independent of the unknown target on an unobserved input. -/
theorem latentDefaultMargins_bound (representative : Z → C) :
    ∀ z c, |latentDefaultMargins representative z c| ≤ 32 := by
  intro z c
  by_cases hc : c = representative z <;> norm_num [latentDefaultMargins, hc]

/-- The actual raw-margin initializer satisfies both the box and the
fixed-default equalities, CM33. Observed labels can vary between inputs. -/
theorem refinedRawMargins_feasible (observed : Finset X) (labels : ↥observed → C)
    (representative : Z → C) : refinedRawMargins observed labels representative ∈
      defaultTableDomain 32 (latentDefaultMargins representative) := by
  constructor
  · intro s c
    by_cases hc : c = refinedLabels observed labels representative s <;>
      norm_num [refinedRawMargins, hc]
  · intro z
    rfl

variable [DecidableEq X] [Fintype Z] [DecidableEq Z] [Fintype C]

/-- Actual complex coefficients after a finite schedule of derived,
normalized CE updates in bounded table coordinates, CM33. -/
noncomputable def refinedCEHead (observed : Finset X) (latent : X → Z)
    (labels : ↥observed → C) (representative : Z → C)
    (batches : ℕ → Finset ↥observed) (rates : ℕ → ℝ) (n : ℕ) :
    Matrix (↥observed ⊕ Z) C ℂ :=
  tableHead (fiberFourier (refinedBlock observed latent))
    (observedCETrace 32 (latentDefaultMargins representative) labels batches rates
      (refinedRawMargins observed labels representative) n)

/-- Every finite iterate satisfies the same actual complex-coordinate
budget. Observed inputs and every latent default remain charged, CM33. -/
theorem refinedCE_budget (observed : Finset X) (latent : X → Z)
    (labels : ↥observed → C) (representative : Z → C)
    (batches : ℕ → Finset ↥observed) (rates : ℕ → ℝ) (n : ℕ) :
    complexCoefficients (refinedCEHead observed latent labels representative batches rates n) ∈
      headBudget (64 * ((observed.card : ℝ) + (Fintype.card Z : ℝ))) := by
  have hf := observedCETrace_feasible (by norm_num : (0 : ℝ) ≤ 32)
    (latentDefaultMargins representative) (latentDefaultMargins_bound representative)
    labels batches rates _ (refinedRawMargins_feasible observed labels representative) n
  have h := tableHead_native_budget _ (refinedBlock_unitary observed latent) _ hf.1
  simpa only [refinedCEHead, Fintype.card_sum, Fintype.card_coe, Nat.cast_add] using h

/-- The actual complex logits keep every observed raw target ahead of
each rival by at least24 under any nonnegative-rate minibatch schedule, CM33. -/
theorem refinedCE_seen_margin (observed : Finset X) (latent : X → Z)
    (labels : ↥observed → C) (representative : Z → C)
    (batches : ℕ → Finset ↥observed) (rates : ℕ → ℝ) (hr : ∀ n, 0 ≤ rates n)
    (n : ℕ) {x : X} (hx : x ∈ observed) {c : C} (hc : c ≠ labels ⟨x, hx⟩) :
    24 ≤ headLogits (complexFeatures (fiberFourier (refinedBlock observed latent)))
      (complexCoefficients (refinedCEHead observed latent labels representative batches rates n))
      (refinedState observed latent x) (labels ⟨x, hx⟩) -
    headLogits (complexFeatures (fiberFourier (refinedBlock observed latent)))
      (complexCoefficients (refinedCEHead observed latent labels representative batches rates n))
      (refinedState observed latent x) c := by
  have h := observedCETrace_margin (by norm_num : (0 : ℝ) ≤ 32)
    (latentDefaultMargins representative) (latentDefaultMargins_bound representative)
    labels batches rates hr _ (refinedRawMargins_feasible observed labels representative)
    n ⟨x, hx⟩ c
  simpa [refinedCEHead, tableHead_logits _ (refinedBlock_unitary observed latent),
    refinedState, hx, refinedRawMargins, refinedLabels, hc] using h

/-- Every observed raw prediction stays correct through actual complex
forward multiplication, CM33. This is not a statement about unobserved labels. -/
theorem refinedCE_seen_winner (observed : Finset X) (latent : X → Z)
    (labels : ↥observed → C) (representative : Z → C)
    (batches : ℕ → Finset ↥observed) (rates : ℕ → ℝ) (hr : ∀ n, 0 ≤ rates n)
    (n : ℕ) {x : X} (hx : x ∈ observed) {c : C} (hc : c ≠ labels ⟨x, hx⟩) :
    headLogits (complexFeatures (fiberFourier (refinedBlock observed latent)))
      (complexCoefficients (refinedCEHead observed latent labels representative batches rates n))
      (refinedState observed latent x) c <
    headLogits (complexFeatures (fiberFourier (refinedBlock observed latent)))
      (complexCoefficients (refinedCEHead observed latent labels representative batches rates n))
      (refinedState observed latent x) (labels ⟨x, hx⟩) := by
  have h := refinedCE_seen_margin observed latent labels representative batches rates hr n hx hc
  linarith

/-- Every observed actual CE remains below the colliding-code log2 floor,
CM33. Its premise bounds the original vocabulary, not a selected class subset. -/
theorem refinedCE_seen_ce (observed : Finset X) (latent : X → Z)
    (labels : ↥observed → C) (representative : Z → C) (hcard : Fintype.card C ≤ 1024)
    (batches : ℕ → Finset ↥observed) (rates : ℕ → ℝ) (hr : ∀ n, 0 ≤ rates n)
    (n : ℕ) {x : X} (hx : x ∈ observed) :
    crossEntropy
      (headLogits (complexFeatures (fiberFourier (refinedBlock observed latent)))
        (complexCoefficients (refinedCEHead observed latent labels representative batches rates n))
        (refinedState observed latent x)) (labels ⟨x, hx⟩) < Real.log 2 := by
  have h := observedCETrace_loss_le (by norm_num : (0 : ℝ) ≤ 32)
    (latentDefaultMargins representative) (latentDefaultMargins_bound representative)
    labels batches rates hr _ (refinedRawMargins_feasible observed labels representative)
    n ⟨x, hx⟩
  have hsmall := native_margin_ce_lt_log_two (labels ⟨x, hx⟩) hcard
  have hfit := h.trans_lt hsmall
  simpa [refinedCEHead, tableHead_logits _ (refinedBlock_unitary observed latent),
    refinedState, hx, refinedRawMargins, refinedLabels] using hfit

/-- All unobserved inputs keep their declared default logits after every
finite update. Agreement with the unseen target still needs identification. -/
theorem refinedCE_unseen_logits (observed : Finset X) (latent : X → Z)
    (labels : ↥observed → C) (representative : Z → C)
    (batches : ℕ → Finset ↥observed) (rates : ℕ → ℝ) (n : ℕ) {x : X} (hx : x ∉ observed) :
    headLogits (complexFeatures (fiberFourier (refinedBlock observed latent)))
      (complexCoefficients (refinedCEHead observed latent labels representative batches rates n))
      (refinedState observed latent x) = latentDefaultMargins representative (latent x) := by
  rw [refinedCEHead, tableHead_logits _ (refinedBlock_unitary observed latent)]
  simp only [refinedState, dite_eq_right hx]
  exact observedCETrace_default (by norm_num : (0 : ℝ) ≤ 32)
    (latentDefaultMargins representative) (latentDefaultMargins_bound representative)
    labels batches rates _ (refinedRawMargins_feasible observed labels representative) n (latent x)

example := latentDefaultMargins_bound (id : Bool → Bool)

example := refinedRawMargins_feasible ({false} : Finset Bool) (fun x => !x.val) (id : Bool → Bool)

example := refinedCE_budget ({false} : Finset Bool) id (fun x => !x.val) (id : Bool → Bool)
  (fun _ => Finset.univ) (fun _ => 1) 7

example := refinedCE_seen_margin ({false} : Finset Bool) id (fun x => !x.val) (id : Bool → Bool)
  (fun _ => Finset.univ) (fun _ => 1) (by intro n; norm_num) 7
  (x := false) (by simp) (c := false) (by decide)

example := refinedCE_seen_winner ({false} : Finset Bool) id (fun x => !x.val) (id : Bool → Bool)
  (fun _ => Finset.univ) (fun _ => 1) (by intro n; norm_num) 7
  (x := false) (by simp) (c := false) (by decide)

example := refinedCE_seen_ce ({false} : Finset Bool) id (fun x => !x.val) (id : Bool → Bool)
  (by decide) (fun _ => Finset.univ) (fun _ => 1) (by intro n; norm_num) 7
  (x := false) (by simp)

example := refinedCE_unseen_logits ({false} : Finset Bool) id
  (fun x => !x.val) (id : Bool → Bool) (fun _ => ∅) (fun _ => 1) 0
  (x := true) (by simp)

end Transformer.Grokking.FlatSpectrum.ConvexDepth

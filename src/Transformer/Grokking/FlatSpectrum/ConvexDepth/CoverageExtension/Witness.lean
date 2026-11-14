module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.CoverageExtension.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ComplexHeadBudget

/-!
# A flat complex extension changes only one of three unseen inputs

Source: `review/flat_spectrum.md`, CM39. Two observed inputs determine the
support tests of two distinct state maps. The supplemental head disagrees
with the prior on all three unseen inputs, but the extension changes only
the newly covered one. Both stored complex blocks use the flat Fourier code.
This is a model-class witness, not a benchmark or an inferred target function.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.CoverageExtension

open Fallback

/-- The original inputs used to bind both support tests, CM39. -/
def observed : Finset (Fin 5) := {0, 1}

/-- Input2 aliases observed input0; inputs3 and4 have unseen states, CM39. -/
def priorState (x : Fin 5) : Fin 4 :=
  if x = 0 ∨ x = 2 then 0 else if x = 1 then 1 else if x = 3 then 2 else 3

/-- Inputs2 and3 alias observed input1; input4 has an unseen state, CM39. -/
def supplementState (x : Fin 5) : Fin 4 := if x = 0 then 0 else if x = 4 then 2 else 1

/-- Actual observed-state membership for the prior, CM39. -/
def priorKnown (x : Fin 5) : Bool := decide (priorState x ∈ observed.image priorState)

/-- Actual observed-state membership for the supplement, CM39. -/
def supplementKnown (x : Fin 5) : Bool :=
  decide (supplementState x ∈ observed.image supplementState)

/-- Prior margin labels, fixed independently of the support gate, CM39. -/
def priorLabels (s : Fin 4) : Bool := decide (s = 1)

/-- Supplemental margin labels agreeing on the two observations, CM39. -/
def supplementLabels (s : Fin 4) : Bool := decide (s ≠ 0)

/-- Actual realified prior complex coefficients, CM39. -/
noncomputable def priorHead : Fin 4 × Bool → Bool → ℝ :=
  complexCoefficients (complexInitializer fourier4 priorLabels 24)

/-- Actual realified supplemental complex coefficients, CM39. -/
noncomputable def supplementHead : Fin 4 × Bool → Bool → ℝ :=
  complexCoefficients (complexInitializer fourier4 supplementLabels 24)

/-- Prior input features from the flat Fourier code, CM39. -/
noncomputable def priorFeatures (k : Fin 4 × Bool) (x : Fin 5) : ℝ :=
  complexFeatures fourier4 k (priorState x)

/-- Supplemental input features from the flat Fourier code, CM39. -/
noncomputable def supplementFeatures (k : Fin 4 × Bool) (x : Fin 5) : ℝ :=
  complexFeatures fourier4 k (supplementState x)

/-- The exact combined complex forward pass, CM39. -/
noncomputable def extensionLogits (x : Fin 5) : Bool → ℝ :=
  headLogits (features (keepPrior priorKnown supplementKnown) priorFeatures supplementFeatures)
    (coefficients priorHead supplementHead) x

/-- The prior's actual Fourier product yields its declared margins, CM39. -/
theorem prior_margin (x : Fin 5) : headLogits priorFeatures priorHead x =
    fun c => if c = priorLabels (priorState x) then 24 else 0 := by
  exact complexInitializer_logits fourier4 fourier4_unitary priorLabels 24 (priorState x)

/-- The supplement's actual Fourier product yields its declared margins, CM39. -/
theorem supplement_margin (x : Fin 5) : headLogits supplementFeatures supplementHead x =
    fun c => if c = supplementLabels (supplementState x) then 24 else 0 := by
  exact complexInitializer_logits fourier4 fourier4_unitary supplementLabels 24 (supplementState x)

/-- Both complex blocks are charged and the supplemental block is
frozen in the actual affine convex domain, CM39. -/
theorem witness_budget : coefficients priorHead supplementHead ∈
    parentDomain supplementHead 512 := by
  have hp := complexInitializer_native_budget fourier4 fourier4_unitary priorLabels
  have hs := complexInitializer_native_budget fourier4 fourier4_unitary supplementLabels
  have h := coefficients_mem hp hs
  norm_num at h
  exact h

/-- The supplement disagrees on every unseen input, yet the extension
preserves the prior on old support and on joint refusal, CM39. -/
theorem witness_preserved :
    extensionLogits 2 = headLogits priorFeatures priorHead 2 ∧
    extensionLogits 4 = headLogits priorFeatures priorHead 4 ∧
    (∀ x ∈ ({2, 3, 4} : Finset (Fin 5)),
      headLogits priorFeatures priorHead x true < headLogits supplementFeatures supplementHead x true) := by
  refine ⟨prior_logits _ _ _ _ _ _ (Or.inl (by decide)),
    prior_logits _ _ _ _ _ _ (Or.inr (by decide)), ?_⟩
  intro x hx
  simp only [Finset.mem_insert, Finset.mem_singleton] at hx
  rcases hx with rfl | rfl | rfl <;>
    rw [prior_margin, supplement_margin] <;>
    norm_num [priorLabels, supplementLabels, priorState, supplementState]

/-- The newly covered input strictly changes its predicted class, CM39.
Its correct unknown target is deliberately not asserted. -/
theorem witness_new : extensionLogits 3 false < extensionLogits 3 true ∧
    headLogits priorFeatures priorHead 3 true < headLogits priorFeatures priorHead 3 false := by
  have h : extensionLogits 3 = headLogits supplementFeatures supplementHead 3 :=
    newly_covered_logits _ _ _ _ _ _ (by decide) (by decide)
  rw [h, supplement_margin, prior_margin]
  norm_num [priorLabels, supplementLabels, priorState, supplementState]

example := prior_logits priorKnown supplementKnown priorFeatures supplementFeatures
  priorHead supplementHead (x := 2) (Or.inl (by decide))

example := prior_logits priorKnown supplementKnown priorFeatures supplementFeatures
  priorHead supplementHead (x := 4) (Or.inr (by decide))

example := newly_covered_logits priorKnown supplementKnown priorFeatures supplementFeatures
  priorHead supplementHead (x := 3) (by decide) (by decide)

example : priorKnown 3 = false ∧ supplementKnown 3 = true := by
  apply changed_only_newly priorKnown supplementKnown priorFeatures supplementFeatures
    priorHead supplementHead
  intro h
  have ht := congrFun h true
  change extensionLogits 3 true = _ at ht
  have hd := witness_preserved.2.2 (3 : Fin 5) (by decide)
  have hs : extensionLogits 3 = headLogits supplementFeatures supplementHead 3 :=
    newly_covered_logits _ _ _ _ _ _ (by decide) (by decide)
  rw [← hs] at hd
  linarith

-- Three original record IDs retain the repeated input0 and the shorter tail.
private def input (s : Fin 3) : Fin 5 := if s = 2 then 1 else 0
private def label (s : Fin 3) : Bool := decide (s = 2)

private theorem covered (s : Fin 3) : priorKnown (input s) = true := by
  fin_cases s <;> decide

example := batch_loss_eq priorKnown supplementKnown priorFeatures supplementFeatures
  input label (Finset.univ : Finset (Fin 3)) (fun s _ => covered s) priorHead supplementHead

example := batch_loss_eq priorKnown supplementKnown priorFeatures supplementFeatures
  input label ({0, 1} : Finset (Fin 3)) (fun s _ => covered s) priorHead supplementHead

example := batch_loss_eq priorKnown supplementKnown priorFeatures supplementFeatures
  input label ({2} : Finset (Fin 3)) (fun s _ => covered s) priorHead supplementHead

example := batch_fderiv_eq priorKnown supplementKnown priorFeatures supplementFeatures
  input label ({2} : Finset (Fin 3)) (fun s _ => covered s) supplementHead priorHead

example := batchLoss_convex (keepPrior priorKnown supplementKnown)
  priorFeatures supplementFeatures input label ({2} : Finset (Fin 3)) supplementHead 512

example := complex_features_bound (keepPrior priorKnown supplementKnown) fourier4 fourier4
  fourier4_unitary fourier4_unitary priorState supplementState (3 : Fin 5)

example := complex_logits_bound (keepPrior priorKnown supplementKnown) fourier4 fourier4
  fourier4_unitary fourier4_unitary priorState supplementState witness_budget.1 (3 : Fin 5) true

end Transformer.Grokking.FlatSpectrum.ConvexDepth.CoverageExtension

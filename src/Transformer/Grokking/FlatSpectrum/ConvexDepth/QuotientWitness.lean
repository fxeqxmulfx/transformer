module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.QuotientDomain

/-!
# Four observed raw classes, a repaired collision and ambiguous unseen aliases

Source: `review/flat_spectrum.md`, CM31. Inputs are three Booleans. The first
selects observed versus held-out inputs; the second is the latent code and
the third an alias. All four raw classes occur among four observations.
Two targets agree there and have the same quotient everywhere, but disagree
on held-out aliases. The refined complex head fits the observations and
both targets' quotient, yet is raw-wrong for one target at an unseen input.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open Landscape.Neural (crossEntropy)

/-- Four observed categorical inputs, independent of their labels, CM31. -/
def quotientAliasObserved : Finset (Bool × Bool × Bool) :=
  Finset.univ.filter (fun x => x.1 = false)

/-- A fixed latent map that discards the raw alias bit, CM31. -/
def quotientAliasLatent (x : Bool × Bool × Bool) : Bool := x.2.1

/-- The original observed raw labels, all four of which occur, CM31. -/
def quotientAliasLabels (x : ↥quotientAliasObserved) : Bool × Bool := x.val.2

/-- A declared representative for each latent fiber, CM31. -/
def quotientAliasRepresentative (z : Bool) : Bool × Bool := (z, false)

/-- Two target tables with the same observations and quotient. Their
unseen alias bits differ; neither table is a feature-extractor input. -/
def quotientAliasTarget (flip : Bool) (x : Bool × Bool × Bool) : Bool × Bool :=
  (x.2.1, if x.1 = true then Bool.xor x.2.2 flip else x.2.2)

/-- Every raw class is observed, so the ambiguity is not caused by an
absent output class, CM31. -/
theorem quotientAlias_all_classes (c : Bool × Bool) :
    ∃ x : ↥quotientAliasObserved, quotientAliasLabels x = c := by
  refine ⟨⟨(false, c), ?_⟩, rfl⟩
  simp [quotientAliasObserved]

/-- The two nonconstant targets agree on all four observations, share a
quotient everywhere and disagree on a genuinely unobserved input, CM31. -/
theorem quotientAlias_ambiguity :
    quotientAliasObserved.card = 4 ∧
    (∀ x ∈ quotientAliasObserved, quotientAliasTarget false x = quotientAliasTarget true x) ∧
    (∀ x, (quotientAliasTarget false x).1 = (quotientAliasTarget true x).1) ∧
    (true, true, true) ∉ quotientAliasObserved ∧
    quotientAliasTarget false (true, true, true) ≠ quotientAliasTarget true (true, true, true) := by
  decide

/-- The unrefined latent head cannot fit two different observed aliases
with mean CE below log2, irrespective of its trainable coefficients, CM31. -/
theorem quotientAlias_collision (head : Bool → Bool × Bool → ℝ) :
    Real.log 2 ≤
      (crossEntropy (head false) (false, false) +
        crossEntropy (head false) (false, true)) / 2 := by
  exact latent_collision_mean_ce quotientAliasLatent head (quotientAliasTarget false)
    (x := (false, false, false)) (y := (false, false, true)) rfl (by decide)

/-- Two observed aliases sharing their latent code become genuinely
different feature states. Their labels are not used by the extractor. -/
theorem quotientAlias_separated :
    quotientAliasLatent (false, false, false) = quotientAliasLatent (false, false, true) ∧
    refinedState quotientAliasObserved quotientAliasLatent (false, false, false) ≠
      refinedState quotientAliasObserved quotientAliasLatent (false, false, true) := by
  refine ⟨rfl, ?_⟩
  exact refinedState_separates _ _ (by decide) (by decide) (by decide)

/-- The refinement fits the same raw observations for both targets,
including the two colliding aliases from the preceding bound, CM31. -/
theorem quotientAlias_seen (flip : Bool) (x : ↥quotientAliasObserved) :
    refinedPrediction quotientAliasObserved quotientAliasLatent quotientAliasLabels
      quotientAliasRepresentative x = quotientAliasTarget flip x := by
  rw [refinedPrediction_seen _ _ _ _ x.property]
  have hx : x.val.1 = false := (Finset.mem_filter.mp x.property).2
  simp [quotientAliasLabels, quotientAliasTarget, hx]

/-- Each actual complex-head training loss is below the unrefined pair's
log2 floor, with all four raw classes in the CE denominator, CM31. -/
theorem quotientAlias_seen_ce (x : ↥quotientAliasObserved) :
    crossEntropy
      (refinedComplexLogits quotientAliasObserved quotientAliasLatent quotientAliasLabels
        quotientAliasRepresentative 24 x) (quotientAliasLabels x) < Real.log 2 := by
  exact refinedComplex_ce_small_seen _ _ _ _ (by decide) x.property

/-- The saved head's pooled scores give the correct quotient for either
target on all eight inputs. The raw alias remains unidentifiable, CM31. -/
theorem quotientAlias_pooled (flip : Bool) (x : Bool × Bool × Bool) {d : Bool}
    (hd : d ≠ (quotientAliasTarget flip x).1) :
    quotientScore Prod.fst
      (refinedComplexLogits quotientAliasObserved quotientAliasLatent quotientAliasLabels
        quotientAliasRepresentative 24 x) d <
    quotientScore Prod.fst
      (refinedComplexLogits quotientAliasObserved quotientAliasLatent quotientAliasLabels
        quotientAliasRepresentative 24 x) (quotientAliasTarget flip x).1 := by
  exact refinedComplex_pooled_decoded quotientAliasObserved quotientAliasLatent
    quotientAliasLabels quotientAliasRepresentative Prod.fst id
    (fun _ => rfl) (fun _ => rfl) (by decide) (quotientAliasTarget flip) x rfl hd

/-- A strict wrong raw prediction survives at a held-out input despite
complete observed-class coverage and correct pooled decoding, CM31. -/
theorem quotientAlias_raw_wrong :
    refinedComplexLogits quotientAliasObserved quotientAliasLatent quotientAliasLabels
      quotientAliasRepresentative 24 (true, true, true)
      (quotientAliasTarget false (true, true, true)) <
    refinedComplexLogits quotientAliasObserved quotientAliasLatent quotientAliasLabels
      quotientAliasRepresentative 24 (true, true, true) (true, false) := by
  have h := refinedComplex_strict quotientAliasObserved quotientAliasLatent quotientAliasLabels
    quotientAliasRepresentative (a := 24) (by norm_num) (true, true, true)
    (c := quotientAliasTarget false (true, true, true)) (by decide)
  exact h

/-- Four observed-input states and two latent defaults give six actual
Fourier states, rather than an uncounted lookup table, CM31. -/
theorem quotientAlias_states : Fintype.card (↥quotientAliasObserved ⊕ Bool) = 6 := by
  rw [refinedState_card]
  decide

/-- The actual six-state code has unit singular values, CM31. -/
theorem quotientAlias_singularValues (i : ↥quotientAliasObserved ⊕ Bool) :
    singularValues (categoricalFourierOn (↥quotientAliasObserved ⊕ Bool)) i = 1 :=
  categoricalFourierOn_singularValues _ i

/-- Margin24 uses at most384 of the real/imaginary L1 budget per class in
the concrete four-class witness, CM31. -/
theorem quotientAlias_budget :
    complexCoefficients
      (refinedComplexHead quotientAliasObserved quotientAliasLabels
        quotientAliasRepresentative 24) ∈
        headBudget 384 := by
  have h := refinedComplex_budget quotientAliasObserved quotientAliasLabels
    quotientAliasRepresentative
  have hc : quotientAliasObserved.card = 4 := by decide
  norm_num [hc] at h
  exact h

example := quotientAlias_collision (fun z c => if c.1 = z then 1 else -1)

example := quotientAlias_seen false ⟨(false, false, true), by decide⟩

example := quotientAlias_seen_ce ⟨(false, false, true), by decide⟩

example := quotientAlias_pooled false (true, true, true) (d := false) (by decide)

example := quotientAlias_pooled true (true, true, true) (d := false) (by decide)

end Transformer.Grokking.FlatSpectrum.ConvexDepth

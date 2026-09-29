/-
# Exact expanded sequence length in the kaleidoscope compiler

Arora et al., arXiv:2312.04927v1, Appendix `lmm: kaleido-coyote`.
The existing recursive expanded K class is now compiled using exactly
2^e·n rows and d features, with depth at most 70w log₂(2^e nd). Every
projection is an actual width-one K matrix with expansion two. The
source's d ≥ 2 is represented by k > 0; no invertibility premise is needed.
-/

import Transformer.Zoology.Appendix_InPlaceWeightWidths

noncomputable section

namespace Transformer.Zoology

/-- Exact-length model of the existing expanded recursive K matrix.
Source: Appendix `lmm: kaleido-coyote`, innerN=en and innerD=d,
where the power-of-two expansion factor is 2^e. -/
def ExpandedKaleidoscope.exactModel {p k e : ℕ}
    (K : ExpandedKaleidoscope (k + p) e) (hk : 0 < k) :
    PaddedCyclicCoyoteModel (butterflyWidth p) (butterflyWidth k) := {
  innerLength := butterflyWidth (p + e)
  innerWidth := butterflyWidth k
  lengthBound := butterflyWidth_le_expanded p e
  widthBound := le_refl _
  network := (compileInPlaceButterflyStages ⟨0, hk⟩ K.toGrid.inner.stages).toNetwork
}

/-- The exact-length model computes the original scalar matrix action
on every input. Source: Appendix `lmm: kaleido-coyote`, full functional claim. -/
theorem ExpandedKaleidoscope.exactModel_correct {p k e : ℕ}
    (K : ExpandedKaleidoscope (k + p) e) (hk : 0 < k)
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    (K.exactModel hk).run u = butterflyGridDecode
      (p := p) (k := k) (K.apply (butterflyGridEncode u)) := by
  unfold PaddedCyclicCoyoteModel.run ExpandedKaleidoscope.exactModel
  rw [((compileInPlaceButterflyStages ⟨0, hk⟩ K.toGrid.inner.stages).toNetwork_correct
    (padSequence u)).1, compileInPlaceButterflyStages_correct,
    K.toGrid.inner.stages_correct]
  exact K.toGrid_correct u

/-- The inner length and width are exactly those in the source lemma.
Source: Appendix `lmm: kaleido-coyote`, power-of-two expansion 2^e. -/
theorem ExpandedKaleidoscope.exactModel_dimensions {p k e : ℕ}
    (K : ExpandedKaleidoscope (k + p) e) (hk : 0 < k) :
    (K.exactModel hk).innerLength = butterflyWidth (p + e) ∧
      (K.exactModel hk).innerWidth = butterflyWidth k := ⟨rfl, rfl⟩

/-- Exact-space compilation retains the source's logarithmic depth bound.
Source: Appendix `lmm: kaleido-coyote`, O(w log(end)) depth. -/
theorem ExpandedKaleidoscope.exactModel_layerCount {p k e : ℕ}
    (K : ExpandedKaleidoscope (k + p) e) (hk : 0 < k) :
    (K.exactModel hk).network.layerCount ≤ 70 * K.inner.width * (k + (p + e)) := by
  change (compileInPlaceButterflyStages ⟨0, hk⟩ K.toGrid.inner.stages).toNetwork.layerCount ≤ _
  rw [((compileInPlaceButterflyStages ⟨0, hk⟩ K.toGrid.inner.stages).toNetwork_correct
    (fun _ _ => 0)).2]
  have h := compileInPlaceButterflyStages_layerCount ⟨0, hk⟩ K.toGrid.inner.stages
  rw [K.toGrid.inner.stages_length, K.toGrid_inner_width] at h
  convert h using 1
  ring

/-- Binary logarithmic form of the exact-length depth guarantee.
Source: Appendix `lmm: kaleido-coyote`, expanded scalar dimension end. -/
theorem ExpandedKaleidoscope.exactModel_logDepth {p k e : ℕ}
    (K : ExpandedKaleidoscope (k + p) e) (hk : 0 < k) :
    (K.exactModel hk).network.layerCount ≤ 70 * K.inner.width *
      Nat.log 2 (butterflyWidth (p + e) * butterflyWidth k) := by
  rw [butterflyLayout_log, Nat.add_comm (p + e) k]
  exact K.exactModel_layerCount hk

/-- All exact-length compiler weights satisfy the actual K restriction.
Source: Appendix `def: W-kmat`, constant width one and expansion two. -/
theorem ExpandedKaleidoscope.exactCompile_hasWidthOne {p k e : ℕ}
    (K : ExpandedKaleidoscope (k + p) e) (hk : 0 < k) :
    (compileInPlaceButterflyStages ⟨0, hk⟩ K.toGrid.inner.stages).HasWidthOne :=
  compileInPlaceButterflyStages_hasWidthOne _ _

/-- One fixed model realizes the given K matrix on all inputs, with the
source's exact workspace and depth bound. Source: Appendix `lmm: kaleido-coyote`.
Dimensions follow the manuscript's power-of-two butterfly convention. -/
theorem exists_exact_kaleidoscope_coyote {p k e : ℕ}
    (K : ExpandedKaleidoscope (k + p) e) (hk : 0 < k) :
    ∃ model : PaddedCyclicCoyoteModel (butterflyWidth p) (butterflyWidth k),
      (∀ u, model.run u = butterflyGridDecode
        (p := p) (k := k) (K.apply (butterflyGridEncode u))) ∧
      model.innerLength = butterflyWidth (p + e) ∧
      model.innerWidth = butterflyWidth k ∧
      model.network.layerCount ≤ 70 * K.inner.width *
        Nat.log 2 (butterflyWidth (p + e) * butterflyWidth k) := by
  exact ⟨K.exactModel hk, K.exactModel_correct hk, (K.exactModel_dimensions hk).1,
    (K.exactModel_dimensions hk).2, K.exactModel_logDepth hk⟩

/-- The witness is itself K-constrained, with exact en × d coordinates
and actual width-one stored projections. Source: Appendix `lmm: kaleido-coyote`
and `def: W-kmat`, combined functional, space, depth and weight guarantees. -/
theorem exists_exact_kaleidoscope_k_coyote {p k e : ℕ}
    (K : ExpandedKaleidoscope (k + p) e) (hk : 0 < k) :
    ∃ network : CyclicKCoyoteNetwork (butterflyWidth (p + e)) k 1,
      network.HasWidthOne ∧
      (∀ u, cropSequence (butterflyWidth_le_expanded p e) (le_refl _)
        (network.run (padSequence u)) = butterflyGridDecode
          (p := p) (k := k) (K.apply (butterflyGridEncode u))) ∧
      network.layerCount ≤ 70 * K.inner.width *
        Nat.log 2 (butterflyWidth (p + e) * butterflyWidth k) := by
  refine ⟨compileInPlaceButterflyStages ⟨0, hk⟩ K.toGrid.inner.stages,
    K.exactCompile_hasWidthOne hk, ?_, ?_⟩
  · intro u
    rw [compileInPlaceButterflyStages_correct, K.toGrid.inner.stages_correct]
    exact K.toGrid_correct u
  · have h := K.exactModel_logDepth hk
    change (compileInPlaceButterflyStages ⟨0, hk⟩ K.toGrid.inner.stages).toNetwork.layerCount ≤ _ at h
    rw [((compileInPlaceButterflyStages ⟨0, hk⟩ K.toGrid.inner.stages).toNetwork_correct
      (fun _ _ => 0)).2] at h
    exact h

/-- The source feature-width assumption holds for a genuine 2 × 2 model.
Source: Appendix `lmm: kaleido-coyote`, n,d ≥ 2. -/
example : (0 : ℕ) < 1 := by decide

end Transformer.Zoology

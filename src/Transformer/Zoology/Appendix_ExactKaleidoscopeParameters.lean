/-
# Scalar storage of the exact-length K compiler

Arora et al., arXiv:2312.04927v1, Appendix `prop: single-baseconv`,
`def: W-kmat`, and `lmm: kaleido-coyote`. The count inspects the stored
K factors, filters, and biases of every layer. Exact sequence length does
not require larger or unrepresented dense projection weights.
-/

import Transformer.Zoology.Appendix_ExactKaleidoscopeCoyote

noncomputable section

namespace Transformer.Zoology

/-- Exact stored parameter count for any actual width-one K layer stack.
Source: Appendix `prop: single-baseconv`, including the K representation. -/
theorem CyclicKCoyoteNetwork.parameterCount_of_widthOne {n k : ℕ}
    (N : CyclicKCoyoteNetwork n k 1) (h : N.HasWidthOne) :
    N.parameterCount = N.layerCount *
      (3 * n * butterflyWidth k + 4 * (k + 1) * butterflyWidth (k + 1)) := by
  have hm : N.layers.map KCoyoteParameters.parameterCount =
      N.layers.map (fun _ =>
        3 * n * butterflyWidth k + 4 * (k + 1) * butterflyWidth (k + 1)) := by
    apply List.map_congr_left
    intro p hp
    rw [kCoyote_parameterCount_eq, h p hp]
    simp
  unfold CyclicKCoyoteNetwork.parameterCount
  rw [hm]
  simp [CyclicKCoyoteNetwork.layerCount]

/-- Exact scalar storage of the actual existing expanded K matrix's compiler.
Source: Appendix `prop: single-baseconv` and `lmm: kaleido-coyote`. -/
theorem ExpandedKaleidoscope.exactCompile_parameterCount {p k e : ℕ}
    (K : ExpandedKaleidoscope (k + p) e) (hk : 0 < k) :
    (compileInPlaceButterflyStages ⟨0, hk⟩ K.toGrid.inner.stages).parameterCount =
      (compileInPlaceButterflyStages ⟨0, hk⟩ K.toGrid.inner.stages).layerCount *
        (3 * butterflyWidth (p + e) * butterflyWidth k +
          4 * (k + 1) * butterflyWidth (k + 1)) := by
  exact CyclicKCoyoteNetwork.parameterCount_of_widthOne _ (K.exactCompile_hasWidthOne hk)

/-- Near-linear scalar storage per layer and logarithmic number of layers.
Source: Appendix `prop: single-baseconv` and `lmm: kaleido-coyote`; this
counts stored parameters and does not assert an executable runtime bound. -/
theorem ExpandedKaleidoscope.exactCompile_parameterBound {p k e : ℕ}
    (K : ExpandedKaleidoscope (k + p) e) (hk : 0 < k) :
    (compileInPlaceButterflyStages ⟨0, hk⟩ K.toGrid.inner.stages).parameterCount ≤
      (70 * K.inner.width * (k + (p + e))) *
        (3 * butterflyWidth (p + e) * butterflyWidth k +
          4 * (k + 1) * butterflyWidth (k + 1)) := by
  rw [K.exactCompile_parameterCount hk]
  have h := K.exactModel_layerCount hk
  change (compileInPlaceButterflyStages ⟨0, hk⟩ K.toGrid.inner.stages).toNetwork.layerCount ≤ _ at h
  rw [((compileInPlaceButterflyStages ⟨0, hk⟩ K.toGrid.inner.stages).toNetwork_correct
    (fun _ _ => 0)).2] at h
  exact Nat.mul_le_mul_right _ h

/-- The storage premise holds for an actual one-layer 2 × 2 model.
Source: Appendix `lmm:primitives`, identity projection. -/
example : (featurePairProjectionNetwork (n := 2) (0 : Fin 1)
    {a := 1, b := 0, c := 0, d := 1}).HasWidthOne :=
  featurePairProjectionNetwork_hasWidthOne _ _

/-- The source feature condition in the expanded compiler is nonempty.
Source: Appendix `lmm: kaleido-coyote`, d ≥ 2. -/
example : (0 : ℕ) < 1 := by decide

end Transformer.Zoology

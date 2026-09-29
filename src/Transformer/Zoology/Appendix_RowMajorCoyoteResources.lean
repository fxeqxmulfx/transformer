/-
# Explicit logarithmic depth, workspace, and K-weight bounds

Arora et al., arXiv:2312.04927v1, Appendix `prop: butterfly-hyena`,
`lmm: kaleido-coyote`, and `prop: single-baseconv`. These bounds refer to
the verified primary-form matrix compiler, not the dense feature-memory
circuit compiler. The latter still needs a width-sensitive factorization.
-/

import Transformer.Zoology.Appendix_ExpandedRowMajorCoyote

namespace Transformer.Zoology

/-- The number of binary digits of the full row-major dimension.
Source: Appendix `def: butterfly`, log₂(nd) block sizes. -/
theorem butterflyLayout_log (p k : ℕ) :
    Nat.log 2 (butterflyWidth p * butterflyWidth k) = p + k := by
  rw [butterflyWidth_eq_pow, butterflyWidth_eq_pow, ← pow_add]
  exact Nat.log_pow (by decide) _

/-- Exact logarithmic depth for the expanded primary-form hierarchy.
Source: Appendix `lmm: kaleido-coyote`, O(w log(end)) depth. -/
theorem ExpandedRowMajorKaleidoscope.model_logDepth {p k e : ℕ}
    (K : ExpandedRowMajorKaleidoscope p k e) :
    K.model.network.layerCount = 6 * K.inner.width *
      Nat.log 2 (butterflyWidth (p + e) * butterflyWidth k) := by
  rw [K.model_layerCount, butterflyLayout_log]
  ring

/-- External n × d layout, ninefold expanded sequence workspace, and
unchanged features. Source: Appendix `lmm: kaleido-coyote`, larger
workspace: this construction proves 9en. The exact en bound for d ≥ 2
is proved in Appendix_ExactKaleidoscopeCoyote by an in-place compiler. -/
theorem ExpandedRowMajorKaleidoscope.model_dimensions {p k e : ℕ}
    (K : ExpandedRowMajorKaleidoscope p k e) :
    K.model.innerLength = 9 * butterflyWidth (p + e) ∧
      K.model.innerWidth = butterflyWidth k := ⟨rfl, rfl⟩

/-- All layer weights satisfy the K restriction with hierarchy width one
and expansion two, rather than relying on arbitrary dense projections.
Source: Appendix `def: W-kmat`, permitted polylogarithmic weight bounds. -/
theorem ExpandedRowMajorKaleidoscope.compile_weightWidths {p k e : ℕ}
    (K : ExpandedRowMajorKaleidoscope p k e) :
    ((compileBinaryButterflyStages K.inner.stages).layers.map fun layer =>
      layer.weight.inner.width) =
        List.replicate (6 * K.inner.width * (k + (p + e))) 1 := by
  rw [compileBinaryButterflyStages_weightWidths, K.inner.stages_length]
  congr 1
  ring

/-- A concrete row swap is realized on the original two-token, two-feature
shape. Source: Appendix `prop: butterfly-hyena`, nondegenerate row factor. -/
example (u : RealSequence (butterflyWidth 1) (butterflyWidth 1)) :
    (binaryButterflyProgramModel
      [({axis := .inl 0, main := fun _ _ => 0, off := fun _ _ => 1} :
        BinaryButterflyStage 1 1)]).run u =
      fun i q => u (if i = 0 then 1 else 0) q := by
  rw [binaryButterflyProgramModel_correct]
  funext i q
  fin_cases i <;>
    (simp [applyBinaryButterflyStages, BinaryButterflyStage.apply, butterflyToggleIndex]; rfl)

/-- A concrete feature swap uses the same feature width two, with no
dense unrestricted weight. Source: Appendix `prop: butterfly-hyena`,
nondegenerate feature factor. -/
example (u : RealSequence (butterflyWidth 1) (butterflyWidth 1)) :
    (binaryButterflyProgramModel
      [({axis := .inr 0, main := fun _ _ => 0, off := fun _ _ => 1} :
        BinaryButterflyStage 1 1)]).run u =
      fun i q => u i (if q = 0 then 1 else 0) := by
  rw [binaryButterflyProgramModel_correct]
  funext i q
  fin_cases q <;>
    (simp [applyBinaryButterflyStages, BinaryButterflyStage.apply, butterflyToggleIndex]; rfl)

end Transformer.Zoology

/-
# K-constrained resource bounds for shifted diagonal sums

Arora et al., arXiv:2312.04927v1, Appendix `prop: butterfly-hyena`,
`def: W-kmat`, and `prop: single-baseconv`. The three-layer construction
has unchanged feature width, workspace length `9m`, and actual stored
K-weight/filter/bias parameter counts. No dense weight bound is assumed.
-/

import Transformer.Zoology.Appendix_ShiftSumSimulation
import Transformer.Zoology.Appendix_CyclicKNetwork

namespace Transformer.Zoology

/-- Store all three zero-weight layers as genuine K-constrained layers.
Source: Appendix `def: W-kmat` and `prop: butterfly-hyena`. -/
def shiftSumKNetwork {m k : ℕ} [NeZero m]
    (coeff : Fin 3 → RealSequence m (butterflyWidth k))
    (shift : Fin 3 → Fin m) : CyclicKCoyoteNetwork (9 * m) k 1 :=
  { layers := (shiftSumNetwork coeff shift).layers.map
      zeroWeightCyclicKParameters }

/-- Decoding the three K weights returns exactly the ordinary workspace
construction. Source: Appendix `def: W-kmat`, zero-weight representation. -/
theorem shiftSumKNetwork_toNetwork {m k : ℕ} [NeZero m]
    (coeff : Fin 3 → RealSequence m (butterflyWidth k))
    (shift : Fin 3 → Fin m) :
    (shiftSumKNetwork coeff shift).toNetwork = shiftSumNetwork coeff shift := by
  change ({layers :=
    [(zeroWeightCyclicKParameters shiftSumCopyParameters).toParameters,
      (zeroWeightCyclicKParameters
        (shiftSumBranchParameters coeff shift)).toParameters,
      (zeroWeightCyclicKParameters shiftSumCollectParameters).toParameters]} :
      CyclicCoyoteNetwork (9 * m) (butterflyWidth k)) = _
  rw [zeroWeightCyclicKParameters_correct shiftSumCopyParameters rfl,
    zeroWeightCyclicKParameters_correct (shiftSumBranchParameters coeff shift) rfl,
    zeroWeightCyclicKParameters_correct shiftSumCollectParameters rfl]
  rfl

/-- The K-constrained stack computes the exact shifted-diagonal sum and
clears all workspace rows. Source: Appendix `prop: butterfly-hyena`,
functional construction with `def: W-kmat` respected. -/
theorem shiftSumKNetwork_correct {m k : ℕ} [NeZero m]
    (coeff : Fin 3 → RealSequence m (butterflyWidth k))
    (shift : Fin 3 → Fin m) (u : RealSequence m (butterflyWidth k)) :
    (shiftSumKNetwork coeff shift).run (shiftSumPad u) =
      shiftSumPad (shiftedDiagonalSum coeff shift u) := by
  rw [← ((shiftSumKNetwork coeff shift).toNetwork_correct
    (shiftSumPad u)).1, shiftSumKNetwork_toNetwork]
  exact shiftSumNetwork_correct coeff shift u

/-- The K-constrained construction retains the exact three-layer depth.
Source: Appendix `prop: butterfly-hyena`, constant depth. -/
theorem shiftSumKNetwork_layerCount {m k : ℕ} [NeZero m]
    (coeff : Fin 3 → RealSequence m (butterflyWidth k))
    (shift : Fin 3 → Fin m) :
    (shiftSumKNetwork coeff shift).layerCount = 3 := rfl

/-- Exact stored scalar parameter count for the three-layer construction.
It is linear in workspace volume with an additional butterfly-size term.
Source: Appendix `prop: single-baseconv` and `def: W-kmat`. -/
theorem shiftSumKNetwork_parameterCount {m k : ℕ} [NeZero m]
    (coeff : Fin 3 → RealSequence m (butterflyWidth k))
    (shift : Fin 3 → Fin m) :
    (shiftSumKNetwork coeff shift).parameterCount =
      3 * (27 * m * butterflyWidth k +
        4 * (k + 1) * butterflyWidth (k + 1)) := by
  simp only [CyclicKCoyoteNetwork.parameterCount, shiftSumKNetwork,
    shiftSumNetwork, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    zeroWeightCyclicKParameters_parameterCount]
  ring

/-- External-shape model for the shifted-diagonal sum: sequence workspace
is expanded by nine, and the feature dimension is unchanged.
Source: Appendix `prop: butterfly-hyena`, resource statement. -/
def shiftSumKModel {m k : ℕ} [NeZero m]
    (coeff : Fin 3 → RealSequence m (butterflyWidth k))
    (shift : Fin 3 → Fin m) : PaddedCyclicCoyoteModel m (butterflyWidth k) := {
  innerLength := 9 * m
  innerWidth := butterflyWidth k
  lengthBound := by omega
  widthBound := le_refl _
  network := (shiftSumKNetwork coeff shift).toNetwork
}

/-- The padded K-constrained model computes the prescribed shifted sum
on its original `m × 2^k` external layout. Source: Appendix
`prop: butterfly-hyena`, exact functional and dimension components. -/
theorem shiftSumKModel_correct {m k : ℕ} [NeZero m]
    (coeff : Fin 3 → RealSequence m (butterflyWidth k))
    (shift : Fin 3 → Fin m) (u : RealSequence m (butterflyWidth k)) :
    (shiftSumKModel coeff shift).run u = shiftedDiagonalSum coeff shift u := by
  have hpad : (padSequence u : RealSequence (9 * m) (butterflyWidth k)) =
      shiftSumPad u := by
    funext i q
    simp [padSequence, shiftSumPad]
  unfold PaddedCyclicCoyoteModel.run shiftSumKModel
  rw [hpad, shiftSumKNetwork_toNetwork, shiftSumNetwork_correct]
  funext i q
  simp [cropSequence, shiftSumPad]

end Transformer.Zoology

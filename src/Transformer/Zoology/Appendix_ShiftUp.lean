/-
# Upward shift using a cyclic impulse and output mask

Arora et al., arXiv:2312.04927v1, Appendix Proposition `prop: prim-shift`.
The printed three-layer proof incorrectly says that cyclic convolution by
the impulse at zero reverses the input. The permitted cyclic-convolution
variant can perform the required up-shift directly: use an impulse at the
negative offset and mask the wrapped output rows. This proves the corrected
one-layer construction below.
-/

import Transformer.Zoology.Appendix_Shift
import Transformer.Zoology.Appendix_Network

open scoped BigOperators

namespace Transformer.Zoology

/-- A cyclic impulse shifts by its index.
Source: Appendix Proposition `prop: prim-shift`, cyclic-convolution primitive. -/
theorem cyclicConvolution_impulse {n d : ℕ} (u : RealSequence n d)
    (s : Fin n) :
    cyclicConvolution u (impulse s) = fun i q => u (i - s) q := by
  classical
  funext i q
  unfold cyclicConvolution
  rw [Finset.sum_eq_single s]
  · simp [impulse]
  · intro k _ hks
    simp [impulse, hks]
  · intro hs
    exact (hs (Finset.mem_univ s)).elim

/-- Shift values toward smaller indices by `s` positions, filling the last
`s` rows with zeros.  Source: Appendix Proposition `prop: prim-shift`. -/
def shiftUp {n d : ℕ} (u : RealSequence n d)
    (s : Fin n) : RealSequence n d :=
  fun i q => if i.val + s.val < n then u (i + s) q else 0

/-- Cyclic-convolution parameters for a true up-shift: an impulse at `-s`
performs the wraparound, and a fixed gate removes the wrapped rows.
Source: Appendix Proposition `prop: prim-shift`, corrected construction. -/
def shiftUpCyclicParameters {n d : ℕ} (s : Fin n) :
    CoyoteParameters n d := {
  weight := fun _ _ => 0
  filter := impulse (-s)
  bias₁ := fun i _ => if i.val + s.val < n then 1 else 0
  bias₂ := fun _ _ => 0
}

private theorem fin_neg_neg {n : ℕ} (s : Fin n) : -(-s) = s := by
  have hn : NeZero n := ⟨Nat.ne_of_gt s.pos⟩
  ext
  simp

/-- One cyclic-convolution Coyote layer computes the up-shift.
Source: Appendix Proposition `prop: prim-shift`, corrected construction. -/
theorem coyote_cyclic_realizes_shiftUp {n d : ℕ}
    (u : RealSequence n d) (s : Fin n) :
    coyoteLayerCyclic (shiftUpCyclicParameters s) u = shiftUp u s := by
  funext i q
  simp only [coyoteLayerCyclic, shiftUpCyclicParameters]
  rw [show cyclicConvolution u (impulse (-s)) i q = u (i - (-s)) q from
    congrFun (congrFun (cyclicConvolution_impulse u (-s)) i) q]
  have hshift : i - (-s) = i + s := by
    rw [Fin.sub_eq_add_neg, fin_neg_neg]
  by_cases h : i.val + s.val < n
  · simp [linearProjection, shiftUp, h, hshift]
  · simp [linearProjection, shiftUp, h]

/-- Three permitted Coyote layers compute the up-shift: one cyclic layer
between two identity layers. This matches the proposition's layer count
while correcting its erroneous identity-filter reversal calculation.
Source: Appendix Proposition `prop: prim-shift`, up-shift half. -/
theorem coyote_three_layers_shiftUp {n d : ℕ} (u : RealSequence n d)
    (s : Fin n) :
    coyoteLayer (linearCoyoteParameters (identityWeight d))
      (coyoteLayerCyclic (shiftUpCyclicParameters s)
        (coyoteLayer (linearCoyoteParameters (identityWeight d)) u)) =
      shiftUp u s := by
  rw [coyote_identity, coyote_cyclic_realizes_shiftUp, coyote_identity]

end Transformer.Zoology

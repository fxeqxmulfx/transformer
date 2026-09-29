/-
# Two-layer block addition primitive

Arora et al., arXiv:2312.04927v1, Appendix Proposition `prop: prim-add`
and Appendix `sec: primitives`. The first layer uses the sum of impulses at
zero and at the block length, then keeps only the second block. The second
layer inserts ones in the first block, as the primitive's stated output does.
-/

import Transformer.Zoology.Appendix_Shift
import Transformer.Zoology.Appendix_Network

open scoped BigOperators

namespace Transformer.Zoology

/-- The convolution filter whose two taps are at zero and `s`.
Source: Appendix Proposition `prop: prim-add`, first filter `1+X^n`. -/
def addBlockFilter {N d : ℕ} [NeZero N] (s : Fin N) : RealSequence N d :=
  fun i q => impulse 0 i q + impulse s i q

/-- Convolution with `1+X^s` adds the input to its down-shift.
Source: Appendix Proposition `prop: prim-add`, polynomial calculation. -/
theorem causalConvolution_addBlockFilter {N d : ℕ} [NeZero N]
    (u : RealSequence N d) (s : Fin N) :
    causalConvolution u (addBlockFilter s) =
      fun i q => u i q + shiftDown u s i q := by
  funext i q
  calc
    causalConvolution u (addBlockFilter s) i q =
        (∑ k : Fin N, if k ≤ i then impulse 0 k q * u (i - k) q else 0) +
        (∑ k : Fin N, if k ≤ i then impulse s k q * u (i - k) q else 0) := by
      unfold causalConvolution addBlockFilter
      rw [← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro k _
      by_cases hk : k ≤ i <;> simp [hk, add_mul]
    _ = u i q + shiftDown u s i q := by
      change causalConvolution u (impulse 0) i q +
        causalConvolution u (impulse s) i q = _
      rw [causalConvolution_impulse, causalConvolution_impulse, shiftDown_zero]

/-- First-layer parameters: preserve only the second block of the sum.
Source: Appendix Proposition `prop: prim-add`, first layer. -/
def addBlockFirstParameters {N d : ℕ} [NeZero N]
    (s : Fin N) : CoyoteParameters N d := {
  weight := fun _ _ => 0
  filter := addBlockFilter s
  bias₁ := fun i _ => if s.val ≤ i.val ∧ i.val < 2 * s.val then 1 else 0
  bias₂ := fun _ _ => 0
}

/-- Second-layer parameters: insert ones in the first block and copy the
sum in the second block. Source: Appendix Proposition `prop: prim-add`. -/
def addBlockSecondParameters {N d : ℕ} [NeZero N]
    (s : Fin N) : CoyoteParameters N d := {
  weight := fun _ _ => 0
  filter := impulse 0
  bias₁ := fun _ _ => 1
  bias₂ := fun i _ => if i.val < s.val then 1 else 0
}

/-- The output specified for the `add_n` primitive. The first block is one,
the second block contains its old contents plus the first input block, and
all later positions are zero. Source: Appendix `sec: primitives`, `Add`. -/
def addBlockOutput {N d : ℕ} (u : RealSequence N d) (s : Fin N) :
    RealSequence N d :=
  fun i q =>
    if i.val < s.val then 1 else
    if i.val < 2 * s.val then u i q + shiftDown u s i q else 0

/-- Two Coyote layers realize the paper's running-sum block addition for
every representable block offset. Source: Appendix Proposition
`prop: prim-add`, corrected explicit model. -/
theorem coyote_two_layers_addBlock {N d : ℕ} [NeZero N]
    (u : RealSequence N d) (s : Fin N) :
    coyoteLayer (addBlockSecondParameters s)
      (coyoteLayer (addBlockFirstParameters s) u) =
        addBlockOutput u s := by
  funext i q
  have hfirst : coyoteLayer (addBlockFirstParameters s) u i q =
      (if s.val ≤ i.val ∧ i.val < 2 * s.val then 1 else 0) *
        (u i q + shiftDown u s i q) := by
    simp [coyoteLayer, addBlockFirstParameters, linearProjection,
      causalConvolution_addBlockFilter]
  rw [show coyoteLayer (addBlockSecondParameters s)
      (coyoteLayer (addBlockFirstParameters s) u) i q =
      coyoteLayer (addBlockFirstParameters s) u i q +
        (if i.val < s.val then 1 else 0) by
      simp [coyoteLayer, addBlockSecondParameters, linearProjection,
        causalConvolution_impulse, shiftDown_zero]]
  rw [hfirst]
  by_cases hlow : i.val < s.val
  · have hnot : ¬(s.val ≤ i.val ∧ i.val < 2 * s.val) := by omega
    simp only [addBlockOutput, hlow, hnot, ite_true, ite_false, zero_mul,
      zero_add]
  · by_cases hhigh : i.val < 2 * s.val
    · have hmid : s.val ≤ i.val ∧ i.val < 2 * s.val := by omega
      simp only [addBlockOutput, hlow, hhigh, hmid, true_and,
        ite_true, ite_false, one_mul, add_zero]
    · simp only [addBlockOutput, hlow, hhigh, and_false, ite_false,
        zero_mul, add_zero]

/-- A positive block length `n` satisfying `2n ≤ N` supplies a valid shift
index and hence the claimed two-layer construction. Source: Appendix
Proposition `prop: prim-add`, resource and input-size hypotheses. -/
theorem coyote_two_layers_addBlock_of_length {N d : ℕ} [NeZero N]
    (u : RealSequence N d) (n : ℕ) (hn : 0 < n) (hfit : 2 * n ≤ N) :
    ∃ s : Fin N, s.val = n ∧
      coyoteLayer (addBlockSecondParameters s)
        (coyoteLayer (addBlockFirstParameters s) u) =
          addBlockOutput u s := by
  have hlt : n < N := by omega
  refine ⟨⟨n, hlt⟩, rfl, ?_⟩
  exact coyote_two_layers_addBlock u _

/-- The block-size hypotheses in the running-sum theorem are satisfiable. -/
example : ∃ s : Fin 2, 0 < s.val ∧ 2 * s.val ≤ 2 := by
  refine ⟨1, ?_⟩
  norm_num

end Transformer.Zoology

/-
# The zero projection is a permitted K-matrix

Arora et al., arXiv:2312.04927v1, Appendix `def: W-kmat` and
`lmm:primitives`. Several Coyote constructions use a zero linear weight.
We exhibit an expanded kaleidoscope representation of that weight, so the
constructions need not appeal to an unrestricted dense matrix.
-/

import Transformer.Zoology.Appendix_KCoyote

namespace Transformer.Zoology

/-- A zero butterfly factor, used at the outermost level to annihilate
both half-vectors. Source: Appendix `def: butterfly`. -/
def zeroButterflyFactor (m : ℕ) : ButterflyFactor m := {
  upperLeft := fun _ => 0
  upperRight := fun _ => 0
  lowerLeft := fun _ => 0
  lowerRight := fun _ => 0
}

/-- A concrete butterfly tree of each power-of-two width, used to fill
the irrelevant lower levels under a zero outer factor. -/
def auxiliaryButterflyTree : (k : ℕ) → ButterflyTree k
  | 0 => .leaf
  | k + 1 => .node (zeroButterflyFactor (butterflyWidth k))
      (auxiliaryButterflyTree k) (auxiliaryButterflyTree k)

/-- A zero butterfly matrix of width `2^(k+1)`.
Source: Appendix `def: butterfly`, recursive display. -/
def zeroButterflyTree (k : ℕ) : ButterflyTree (k + 1) :=
  .node (zeroButterflyFactor (butterflyWidth k))
    (auxiliaryButterflyTree k) (auxiliaryButterflyTree k)

/-- Its action is exactly zero on every input vector.
Source: Appendix `def: butterfly`, outer zero factor. -/
theorem zeroButterflyTree_apply (k : ℕ)
    (u : Fin (butterflyWidth (k + 1)) → ℝ) :
    (zeroButterflyTree k).apply u = 0 := by
  funext i
  simp [zeroButterflyTree, ButterflyTree.apply,
    zeroButterflyFactor, ButterflyFactor.apply]

/-- A `BB*` factor with a zero left butterfly is zero, independently of
the right butterfly. Source: Appendix `def: kaleidoscope`. -/
def zeroBBStar (k : ℕ) : BBStar (k + 1) := {
  left := zeroButterflyTree k
  right := auxiliaryButterflyTree (k + 1)
}

/-- The zero factor annihilates every vector.
Source: Appendix `def: kaleidoscope`, class `BB*`. -/
theorem zeroBBStar_apply (k : ℕ)
    (u : Fin (butterflyWidth (k + 1)) → ℝ) :
    (zeroBBStar k).apply u = 0 := by
  exact zeroButterflyTree_apply k _

/-- One expanded `BB*` factor representing the zero operator at an
external feature width `2^k`. Source: Appendix `def: kaleidoscope`,
expanded class `(BB*)^w_e`. -/
def zeroExpandedKaleidoscope (k : ℕ) : ExpandedKaleidoscope k 1 := {
  inner := {factors := [zeroBBStar k]}
}

/-- The expanded zero K-matrix annihilates every input vector.
Source: Appendix `def: W-kmat`, zero-weight case. -/
theorem zeroExpandedKaleidoscope_apply (k : ℕ)
    (u : Fin (butterflyWidth k) → ℝ) :
    (zeroExpandedKaleidoscope k).apply u = 0 := by
  change cropButterfly ((zeroBBStar k).apply (padButterfly u)) = 0
  rw [zeroBBStar_apply]
  rfl

/-- The corresponding Coyote linear weight has every entry zero.
Source: Appendix `lmm:primitives`, zero-weight constructions under the
paper's K-matrix restriction. -/
theorem zero_kaleidoscopeWeight (k : ℕ)
    (i j : Fin (butterflyWidth k)) :
    kaleidoscopeWeight (zeroExpandedKaleidoscope k) i j = 0 := by
  simp [kaleidoscopeWeight, ExpandedKaleidoscope.matrix,
    zeroExpandedKaleidoscope_apply]

end Transformer.Zoology

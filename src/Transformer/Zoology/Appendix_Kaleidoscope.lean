/-
# The kaleidoscope matrix hierarchy

Arora et al., arXiv:2312.04927v1, Appendix `def: kaleidoscope`.
This implementation uses power-of-two matrix sizes, as required by the
paper's butterfly factors.  A `BB*` factor is a butterfly matrix multiplied
by the transpose of another, and a width-`w` hierarchy element is a
product of `w` such factors.  Expansion takes the upper-left corner of a
larger power-of-two matrix.
-/

import Transformer.Zoology.Appendix_ButterflyTree

open scoped BigOperators

namespace Transformer.Zoology

/-- The ordinary matrix of a recursively represented butterfly operator.
Source: Appendix `def: butterfly`. -/
def ButterflyTree.matrix {k : ℕ} (tree : ButterflyTree k)
    (i j : Fin (butterflyWidth k)) : ℝ :=
  tree.apply (fun t => if t = j then 1 else 0) i

/-- One factor of the paper's `BB*` class.  Real `*` is transpose.
Source: Appendix `def: kaleidoscope`. -/
structure BBStar (k : ℕ) where
  left : ButterflyTree k
  right : ButterflyTree k

/-- Apply `B₁ B₂ᵀ` to a vector.
Source: Appendix `def: kaleidoscope`. -/
def BBStar.apply {k : ℕ} (factor : BBStar k)
    (u : Fin (butterflyWidth k) → ℝ) :
    Fin (butterflyWidth k) → ℝ :=
  factor.left.apply fun j =>
    ∑ i, factor.right.matrix i j * u i

/-- Two butterfly trees, each with its own factor parameters.
Source: Appendix `def: kaleidoscope`. -/
def BBStar.parameterCount {k : ℕ} (factor : BBStar k) : ℕ :=
  factor.left.parameterCount + factor.right.parameterCount

/-- Each `BB*` factor has `4k·2^k` scalar coefficients in its two trees.
Source: Appendix `def: butterfly` and `def: kaleidoscope`. -/
theorem BBStar.parameterCount_eq {k : ℕ} (factor : BBStar k) :
    factor.parameterCount = 4 * k * butterflyWidth k := by
  unfold BBStar.parameterCount
  rw [factor.left.parameterCount_eq, factor.right.parameterCount_eq]
  ring

/-- A product of `BB*` factors of common matrix size.
Source: Appendix `def: kaleidoscope`, class `(BB*)^w`. -/
structure Kaleidoscope (k : ℕ) where
  factors : List (BBStar k)

/-- Product width is the number of `BB*` factors.
Source: Appendix `def: kaleidoscope`. -/
def Kaleidoscope.width {k : ℕ} (K : Kaleidoscope k) : ℕ :=
  K.factors.length

/-- Apply the factors in their listed order.
Source: Appendix `def: kaleidoscope`. -/
def Kaleidoscope.apply {k : ℕ} (K : Kaleidoscope k)
    (u : Fin (butterflyWidth k) → ℝ) :
    Fin (butterflyWidth k) → ℝ :=
  K.factors.foldl (fun state factor => factor.apply state) u

/-- Total real coefficients in all stored butterfly factors.
Source: Appendix `def: kaleidoscope`. -/
def Kaleidoscope.parameterCount {k : ℕ} (K : Kaleidoscope k) : ℕ :=
  (K.factors.map BBStar.parameterCount).sum

/-- The unexpanded width-`w` class stores `4wk·2^k` factor coefficients.
Source: Appendix `def: kaleidoscope`. -/
theorem Kaleidoscope.parameterCount_eq {k : ℕ} (K : Kaleidoscope k) :
    K.parameterCount = K.width * (4 * k * butterflyWidth k) := by
  rcases K with ⟨fs⟩
  change (fs.map BBStar.parameterCount).sum =
    fs.length * (4 * k * butterflyWidth k)
  induction fs with
  | nil => simp
  | cons f fs ih =>
      simp only [List.map_cons, List.sum_cons, List.length_cons]
      rw [f.parameterCount_eq]
      rw [ih]
      ring

/-- A power-of-two expansion of the dimension is at least as large as the
original dimension.  Source: Appendix `def: kaleidoscope`, `e > 0`. -/
theorem butterflyWidth_le_expanded (k e : ℕ) :
    butterflyWidth k ≤ butterflyWidth (k + e) := by
  rw [butterflyWidth_eq_pow, butterflyWidth_eq_pow]
  exact Nat.pow_le_pow_of_le (by norm_num : 1 < 2) (Nat.le_add_right k e)

/-- Pad an input vector by zeros before applying the expanded hierarchy.
Source: Appendix `def: kaleidoscope`, matrix `Sᵀ`. -/
def padButterfly {k e : ℕ} (u : Fin (butterflyWidth k) → ℝ) :
    Fin (butterflyWidth (k + e)) → ℝ :=
  fun i => if h : i.val < butterflyWidth k then u ⟨i.val, h⟩ else 0

/-- Extract the upper-left output coordinates after an expanded hierarchy.
Source: Appendix `def: kaleidoscope`, matrix `S`. -/
def cropButterfly {k e : ℕ}
    (u : Fin (butterflyWidth (k + e)) → ℝ) :
    Fin (butterflyWidth k) → ℝ :=
  fun i => u ⟨i.val, lt_of_lt_of_le i.isLt (butterflyWidth_le_expanded k e)⟩

/-- The paper's expansion class `(BB*)^w_e`, for power-of-two expansion.
Source: Appendix `def: kaleidoscope`. -/
structure ExpandedKaleidoscope (k e : ℕ) where
  inner : Kaleidoscope (k + e)

/-- Apply the upper-left corner of the expanded matrix.
Source: Appendix `def: kaleidoscope`, `M = S E Sᵀ`. -/
def ExpandedKaleidoscope.apply {k e : ℕ} (K : ExpandedKaleidoscope k e)
    (u : Fin (butterflyWidth k) → ℝ) :
    Fin (butterflyWidth k) → ℝ :=
  cropButterfly (K.inner.apply (padButterfly u))

/-- Zero padding followed by cropping is exactly the identity map.
Source: Appendix `def: kaleidoscope`, matrices `S` and `Sᵀ`. -/
theorem crop_pad_butterfly {k e : ℕ} (u : Fin (butterflyWidth k) → ℝ) :
    cropButterfly (padButterfly (e := e) u) = u := by
  funext i
  simp [cropButterfly, padButterfly, i.isLt]

end Transformer.Zoology

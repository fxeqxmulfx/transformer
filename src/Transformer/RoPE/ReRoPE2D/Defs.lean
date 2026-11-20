module

public import Transformer.RoPE.Basic
public import Transformer.RoPE.RoFormer.Section3_Rotary

/-!
# 2D ReRoPE: axial RoPE of vision encoders

2D ReRoPE is in no paper. ReRoPE (Su, kexue.fm/archives/9708, formalized in
`Transformer.RoPE.ReRoPE`) clamps RoPE's relative position at a window, for causal language
models. `Transformer.RoPE.ReRoPE2D` carries it to the axial 2D RoPE of vision encoders, whose
patches have a row and a column. This file defines that RoPE.

* `axialRotary θ ax x`: axial 2D RoPE at the position `x = (row, column)`, block diagonal as
  RoFormer's `rotary` (arXiv:2104.09864v5, Eq. (15)). The plane `l` follows the axis `ax l` and
  turns by `x (ax l) θ_l`; `axialRotary_const`: at the position `(m, m)` it is `rotary θ m`;
* `axialRotary_dotProduct`: `(R_x q)ᵀ(R_y k) = qᵀ R_{y−x} k`, as Eq. (16) in one dimension;
* `dotProduct_axialRotary_eq_sum`: the score is a row score plus a column score. The planes of
  the axis `a` (`axisPart`) see only the coordinate `x a`, through the one-dimensional
  `rotary θ (x a)`;
* `kimiAxis`, `kimiFreq`: Kimi-K3's vision encoder, MoonViT (`Rope2DPosEmbRepeated`,
  `papers/Kimi-K3/modeling_kimi_k3.py`, lines 341–434, and `apply_rope`, lines 172–193).
  `apply_rope` multiplies the complex coordinate `l`, the real pair `(2l, 2l + 1)` of the head
  and RoFormer's plane `l`, by `cis` of its angle. The complex coordinate `2i` turns with the
  column `x_pos = flat_pos % max_width`, the coordinate `2i + 1` with the row, both at
  `base^{−4i/d}`. The docstring of `_precompute_freqs_cis` assigns `2i` to the height; its code
  assigns the column;
* `glmAxis`, `glmFreq`: GLM-5.3-Flash's vision encoder (`Glm5NextVisionRotaryEmbedding`,
  `papers/GLM-5.3-Flash/modeling_glm5_next.py`, lines 1737–1795, and
  `apply_rotary_pos_emb_vision`, lines 1585–1596). `rotate_half` pairs the coordinates `l` and
  `l + d/2`; the first `d/4` planes turn with the first position coordinate (`h`, as the source's
  comment says), the others with the second (`w`), at `base^{−4(l mod d/4)/d}`. The position ids
  come from `get_vision_position_ids`, which is not in `papers/`.

With `d = 2p` coordinates both encoders turn every plane ("no partial rotation involved"), and
the two axes share one set of frequencies. A coordinate order other than RoFormer's is a
permutation of both `q` and `k`, which keeps every dot product.
-/

@[expose] public section

open scoped BigOperators
open Matrix Transformer.RoPE.RoFormer

namespace Transformer.RoPE.ReRoPE2D

variable {p : ℕ}

/-- Axial 2D RoPE at the position `x = (row, column)`: block diagonal, the plane `l` turned by
`x (ax l) θ_l`. -/
noncomputable def axialRotary (θ : Fin p → ℝ) (ax : Fin p → Fin 2) (x : Fin 2 → ℝ) :
    Matrix (Fin 2 × Fin p) (Fin 2 × Fin p) ℝ :=
  blockDiagonal fun l => rotation (x (ax l) * θ l)

theorem axialRotary_add (θ : Fin p → ℝ) (ax : Fin p → Fin 2) (x y : Fin 2 → ℝ) :
    axialRotary θ ax (x + y) = axialRotary θ ax x * axialRotary θ ax y := by
  simp only [axialRotary, ← blockDiagonal_mul, Pi.add_apply, add_mul, rotation_add]

theorem axialRotary_transpose (θ : Fin p → ℝ) (ax : Fin p → Fin 2) (x : Fin 2 → ℝ) :
    (axialRotary θ ax x)ᵀ = axialRotary θ ax (-x) := by
  simp only [axialRotary, blockDiagonal_transpose, rotation_transpose, Pi.neg_apply, neg_mul]

theorem axialRotary_zero (θ : Fin p → ℝ) (ax : Fin p → Fin 2) : axialRotary θ ax 0 = 1 := by
  simp only [axialRotary, Pi.zero_apply, zero_mul, rotation_zero]
  exact blockDiagonal_one

/-- At the position `(m, m)` axial RoPE is RoFormer's one-dimensional `rotary θ m`. -/
theorem axialRotary_const (θ : Fin p → ℝ) (ax : Fin p → Fin 2) (m : ℝ) :
    axialRotary θ ax (fun _ => m) = rotary θ m := by
  simp only [axialRotary, rotary]

/-- `(R_x q)ᵀ(R_y k) = qᵀ R_{y−x} k`: the score depends on the positions through `y − x`. -/
theorem axialRotary_dotProduct (θ : Fin p → ℝ) (ax : Fin p → Fin 2) (x y : Fin 2 → ℝ)
    (q k : Fin 2 × Fin p → ℝ) :
    (axialRotary θ ax x *ᵥ q) ⬝ᵥ (axialRotary θ ax y *ᵥ k) =
      q ⬝ᵥ (axialRotary θ ax (y - x) *ᵥ k) := by
  rw [mulVec_dotProduct, mulVec_mulVec, axialRotary_transpose, ← axialRotary_add, neg_add_eq_sub]

/-- Each coordinate of the plane `l` turns with the coordinate `x (ax l)` of the position. -/
theorem axialRotary_mulVec_apply (θ : Fin p → ℝ) (ax : Fin p → Fin 2) (x : Fin 2 → ℝ)
    (k : Fin 2 × Fin p → ℝ) (r : Fin 2) (l : Fin p) :
    (axialRotary θ ax x *ᵥ k) (r, l) = (rotary θ (x (ax l)) *ᵥ k) (r, l) := by
  simp [axialRotary, rotary, mulVec, dotProduct, Fintype.sum_prod_type, blockDiagonal_apply,
    ite_mul]

/-- The coordinates of `v` on the planes of the axis `a`, and `0` on the others. -/
def axisPart (ax : Fin p → Fin 2) (a : Fin 2) (v : Fin 2 × Fin p → ℝ) : Fin 2 × Fin p → ℝ :=
  fun rl => if ax rl.2 = a then v rl else 0

/-- The parts of the two axes add up to the vector. -/
theorem sum_axisPart (ax : Fin p → Fin 2) (v : Fin 2 × Fin p → ℝ) : ∑ a, axisPart ax a v = v := by
  ext rl
  simp [axisPart, Finset.sum_apply]

/-- The planes of the axis `a` see only the coordinate `x a` of the position. -/
theorem axisPart_dotProduct (θ : Fin p → ℝ) (ax : Fin p → Fin 2) (a : Fin 2) (x : Fin 2 → ℝ)
    (q k : Fin 2 × Fin p → ℝ) :
    axisPart ax a q ⬝ᵥ (axialRotary θ ax x *ᵥ k) = axisPart ax a q ⬝ᵥ (rotary θ (x a) *ᵥ k) := by
  simp only [dotProduct]
  refine Finset.sum_congr rfl fun rl _ => ?_
  obtain ⟨r, l⟩ := rl
  simp only [axisPart]
  split_ifs with h
  · rw [axialRotary_mulVec_apply, h]
  · rw [zero_mul, zero_mul]

/-- **The axial score is a row score plus a column score**: on the planes of each axis it is the
one-dimensional RoPE score (RoFormer's `rotary`) at that axis's coordinate. -/
theorem dotProduct_axialRotary_eq_sum (θ : Fin p → ℝ) (ax : Fin p → Fin 2) (x : Fin 2 → ℝ)
    (q k : Fin 2 × Fin p → ℝ) :
    q ⬝ᵥ (axialRotary θ ax x *ᵥ k) = ∑ a, axisPart ax a q ⬝ᵥ (rotary θ (x a) *ᵥ k) := by
  conv_lhs => rw [← sum_axisPart ax q]
  rw [sum_dotProduct]
  exact Finset.sum_congr rfl fun a _ => axisPart_dotProduct θ ax a x q k

/-- Kimi-K3's MoonViT: the complex coordinate `2i` turns with the column (axis `1`), `2i + 1`
with the row (axis `0`). -/
def kimiAxis (l : Fin p) : Fin 2 :=
  if l.val % 2 = 0 then 1 else 0

/-- Kimi-K3's MoonViT: the complex coordinate `l` turns at `base^{−4i/d}`, `i = ⌊l/2⌋` and
`d = 2p`, that is `base^{−2⌊l/2⌋/p}`: each axis gets the grid `frequency base p` of a rotary
dimension `d/2`. -/
noncomputable def kimiFreq (base : ℝ) (l : Fin p) : ℝ :=
  frequency base p (l.val / 2)

/-- GLM-5.3-Flash: the first `d/4 = p/2` planes turn with the first position coordinate (axis
`0`), the others with the second (axis `1`). -/
def glmAxis (l : Fin p) : Fin 2 :=
  if l.val < p / 2 then 0 else 1

/-- GLM-5.3-Flash: the plane `l` turns at `base^{−4(l mod d/4)/d}`, with `d = 2p` that is
`base^{−2(l mod p/2)/p}`: each axis gets the grid `frequency base p` of a rotary dimension
`d/2`. -/
noncomputable def glmFreq (base : ℝ) (l : Fin p) : ℝ :=
  frequency base p (l.val % (p / 2))

/-- In MoonViT the planes `0` and `1` turn at the frequency `1`, with the column and the row. -/
example : kimiAxis (0 : Fin 32) = 1 ∧ kimiAxis (1 : Fin 32) = 0 ∧
    kimiFreq 10000 (0 : Fin 32) = 1 ∧ kimiFreq 10000 (1 : Fin 32) = 1 := by
  simp [kimiAxis, kimiFreq, frequency]

/-- In GLM-5.3-Flash the planes `0` and `p/2` turn at the frequency `1`, with the two axes. -/
example : glmAxis (0 : Fin 32) = 0 ∧ glmAxis (16 : Fin 32) = 1 ∧
    glmFreq 10000 (0 : Fin 32) = 1 ∧ glmFreq 10000 (16 : Fin 32) = 1 := by
  simp [glmAxis, glmFreq, frequency]

end Transformer.RoPE.ReRoPE2D

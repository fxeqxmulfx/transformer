module

public import Transformer.RoPE.ReRoPE.Section2_Fusion
public import Transformer.RoPE.RoFormer.Section3_Properties

/-!
# ReRoPE, 计算 (Computation): two attention matrices

Su, *Transformer升级之路：12、无限外推的ReRoPE？*, kexue.fm/archives/9708, section 计算
(Computation). The rotation `R^t` is RoFormer's `rotary θ t` (arXiv:2104.09864v5, Eq. (15)), and
the first attention matrix
`a⁽¹⁾_{i,j} = (R^i q_i)ᵀ(R^j k_j) = q_iᵀ R^{j−i} k_j` is `RoFormer.rotary_dotProduct`.

* `leaky_score`: Leaky ReRoPE's second matrix, positions spaced `1/k`,
  `a⁽²⁾_{i,j} = (R^{(i−w)/k+w} q_i)ᵀ(R^{j/k} k_j) = q_iᵀ R^{(j−i+w)/k−w} k_j`;
* `rectified_score`: ReRoPE's second matrix `a⁽²⁾_{i,j} = (R^w q_i)ᵀ k_j` is `q_iᵀ R^{−w} k_j`.
  The source prints `q_iᵀ R^w k_j`, and `exists_rectified_score_ne` shows that the two differ;
  the decoding form `R^{max(j−i,−w)}` below has the right sign;
* `merge_leaky`, `merge_rectified`: `a_{i,j} = a⁽¹⁾_{i,j}` for `i − j < w` and `a⁽²⁾_{i,j}`
  otherwise is the RoPE score at the relative position of the matrix (leaky-rerope), (rerope);
* `merge_rectified_eq_max`, `merge_leaky_eq_max`: token-by-token decoding needs one matrix,
  `q_iᵀ(R^{max(j−i,−w)} k_j)` and `q_iᵀ(R^{max(j−i,(j−i+w)/k−w)} k_j)`; the second needs the
  `k ≥ 1` of an interpolation, which the source leaves implicit. For `k < 1` the maximum picks
  the other branch.

Why one attention matrix does not suffice outside decoding is `Section3_SinglePass`.
-/

@[expose] public section

open Matrix
open scoped Real

namespace Transformer.RoPE.ReRoPE

open RoFormer

variable {p : ℕ}

/-- Computation: Leaky ReRoPE's second attention matrix,
`a⁽²⁾_{i,j} = (R^{(i−w)/k+w} q_i)ᵀ(R^{j/k} k_j) = q_iᵀ R^{(j−i+w)/k−w} k_j`. -/
theorem leaky_score (θ : Fin p → ℝ) (q key : Fin 2 × Fin p → ℝ) (w k i j : ℝ) :
    (rotary θ ((i - w) / k + w) *ᵥ q) ⬝ᵥ (rotary θ (j / k) *ᵥ key) =
      q ⬝ᵥ (rotary θ ((j - i + w) / k - w) *ᵥ key) := by
  rw [rotary_dotProduct]
  congr 3
  ring

/-- Computation: ReRoPE's second attention matrix
`a⁽²⁾_{i,j} = (R^w q_i)ᵀ k_j = q_iᵀ R^{−w} k_j`. The source prints `q_iᵀ R^w k_j`; the rotation
moves to the key transposed, `(R^w)ᵀ = R^{−w}`. -/
theorem rectified_score (θ : Fin p → ℝ) (q key : Fin 2 × Fin p → ℝ) (w : ℝ) :
    (rotary θ w *ᵥ q) ⬝ᵥ key = q ⬝ᵥ (rotary θ (-w) *ᵥ key) := by
  rw [mulVec_dotProduct, rotary_transpose]

/-- Computation: the printed `(R^w q_i)ᵀ k_j = q_iᵀ R^w k_j` is false. With one pair,
`θ = π/2`, `w = 1`, `q = e_1` and `k = e_2`, the left side is `1` and the right side `−1`. -/
theorem exists_rectified_score_ne :
    ∃ (θ : Fin 1 → ℝ) (q key : Fin 2 × Fin 1 → ℝ) (w : ℝ),
      (rotary θ w *ᵥ q) ⬝ᵥ key ≠ q ⬝ᵥ (rotary θ w *ᵥ key) := by
  refine ⟨fun _ => π / 2, Pi.single (0, 0) 1, Pi.single (1, 0) 1, 1, ?_⟩
  simp [rotary, blockDiagonal_apply, rotation]
  norm_num

/-- Computation: the merged matrix, `a⁽¹⁾` for `i − j < w` and Leaky ReRoPE's `a⁽²⁾` otherwise, is
the RoPE score at Leaky ReRoPE's relative position, `q_iᵀ R^{−leakyPosition w k (i−j)} k_j`. -/
theorem merge_leaky (θ : Fin p → ℝ) (q key : Fin 2 × Fin p → ℝ) (w k i j : ℝ) :
    (if i - j < w then (rotary θ i *ᵥ q) ⬝ᵥ (rotary θ j *ᵥ key)
      else (rotary θ ((i - w) / k + w) *ᵥ q) ⬝ᵥ (rotary θ (j / k) *ᵥ key)) =
      q ⬝ᵥ (rotary θ (-leakyPosition w k (i - j)) *ᵥ key) := by
  unfold leakyPosition
  split_ifs
  · rw [rotary_dotProduct, neg_sub]
  · rw [leaky_score]
    congr 3
    ring

/-- Computation: the merged matrix, `a⁽¹⁾` for `i − j < w` and ReRoPE's `a⁽²⁾` otherwise, is the
RoPE score at ReRoPE's relative position, `q_iᵀ R^{−min(i−j, w)} k_j`. -/
theorem merge_rectified (θ : Fin p → ℝ) (q key : Fin 2 × Fin p → ℝ) (w i j : ℝ) :
    (if i - j < w then (rotary θ i *ᵥ q) ⬝ᵥ (rotary θ j *ᵥ key) else (rotary θ w *ᵥ q) ⬝ᵥ key) =
      q ⬝ᵥ (rotary θ (-rectifiedPosition w (i - j)) *ᵥ key) := by
  unfold rectifiedPosition
  split_ifs with h
  · rw [rotary_dotProduct, min_eq_left h.le, neg_sub]
  · rw [rectified_score, min_eq_right (not_lt.mp h)]

/-- Computation, decoding: the merged ReRoPE matrix is the single matrix
`q_iᵀ(R^{max(j−i,−w)} k_j)`. -/
theorem merge_rectified_eq_max (θ : Fin p → ℝ) (q key : Fin 2 × Fin p → ℝ) (w i j : ℝ) :
    (if i - j < w then (rotary θ i *ᵥ q) ⬝ᵥ (rotary θ j *ᵥ key) else (rotary θ w *ᵥ q) ⬝ᵥ key) =
      q ⬝ᵥ (rotary θ (max (j - i) (-w)) *ᵥ key) := by
  rw [merge_rectified, rectifiedPosition, ← max_neg_neg, neg_sub]

/-- Computation, decoding: for `k ≥ 1` the merged Leaky ReRoPE matrix is the single matrix
`q_iᵀ(R^{max(j−i,(j−i+w)/k−w)} k_j)`. -/
theorem merge_leaky_eq_max (θ : Fin p → ℝ) (q key : Fin 2 × Fin p → ℝ) (w i j : ℝ) {k : ℝ}
    (hk : 1 ≤ k) :
    (if i - j < w then (rotary θ i *ᵥ q) ⬝ᵥ (rotary θ j *ᵥ key)
      else (rotary θ ((i - w) / k + w) *ᵥ q) ⬝ᵥ (rotary θ (j / k) *ᵥ key)) =
      q ⬝ᵥ (rotary θ (max (j - i) ((j - i + w) / k - w)) *ᵥ key) := by
  have hk0 : 0 < k := lt_of_lt_of_le one_pos hk
  have hpos : -leakyPosition w k (i - j) = max (j - i) ((j - i + w) / k - w) := by
    rw [leakyPosition]
    split_ifs with h
    · have h1 : (j - i + w) / k ≤ j - i + w := div_le_self (by linarith) hk
      rw [max_eq_left (by linarith)]
      ring
    · have h1 : j - i + w ≤ (j - i + w) / k := by
        rw [le_div_iff₀ hk0]
        nlinarith
      rw [max_eq_right (by linarith)]
      ring
  rw [merge_leaky, hpos]

example (θ : Fin 64 → ℝ) (q key : Fin 2 × Fin 64 → ℝ) (i j : ℝ) :
    (if i - j < 128 then (rotary θ i *ᵥ q) ⬝ᵥ (rotary θ j *ᵥ key)
      else (rotary θ ((i - 128) / 16 + 128) *ᵥ q) ⬝ᵥ (rotary θ (j / 16) *ᵥ key)) =
      q ⬝ᵥ (rotary θ (max (j - i) ((j - i + 128) / 16 - 128)) *ᵥ key) :=
  merge_leaky_eq_max θ q key 128 i j (by norm_num)

end Transformer.RoPE.ReRoPE

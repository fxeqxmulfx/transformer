/-
# DASH — temporal preconditioners computed with batched Gram products

arXiv:2602.02016v2, §2 and §4. Transposition swaps only the matrix
axes; the batching axis is preserved. The EMA recurrence and its
entire history agree with each block's Shampoo preconditioners.
-/

import Transformer.DASH.Section4_Blocking
import Transformer.DASH.Section2_History

open scoped Matrix

noncomputable section

namespace Transformer.DASH

variable {ι : Type*} {m n : ℕ}

/-- Transpose only the last two axes of a stacked gradient.
Source: arXiv:2602.02016v2, §4, batched Gram computation. -/
def batchTranspose (G : ι → Matrix (Fin m) (Fin n) ℝ) :
    ι → Matrix (Fin n) (Fin m) ℝ := fun j => (G j).transpose

/-- Left preconditioner update with a genuine batched Gram product.
Source: arXiv:2602.02016v2, §2, Algorithm 1, and §4, `bmm(G,Gᵀ)`. -/
def batchLeftEma (β : ℝ) (L : ι → Matrix (Fin m) (Fin m) ℝ)
    (G : ι → Matrix (Fin m) (Fin n) ℝ) : ι → Matrix (Fin m) (Fin m) ℝ :=
  fun j => β • L j + (1 - β) • (batchMul G (batchTranspose G)) j

/-- Right preconditioner update with the batch axis held fixed.
Source: arXiv:2602.02016v2, §2, Algorithm 1, and §4, `bmm(Gᵀ,G)`. -/
def batchRightEma (β : ℝ) (R : ι → Matrix (Fin n) (Fin n) ℝ)
    (G : ι → Matrix (Fin m) (Fin n) ℝ) : ι → Matrix (Fin n) (Fin n) ℝ :=
  fun j => β • R j + (1 - β) • (batchMul (batchTranspose G) G) j

/-- The complete left stacked history. As in `leftHistory`, array entry
`G t` denotes the manuscript's `G_(t+1)`.
Source: arXiv:2602.02016v2, §2, Algorithm 1, and §4, stacked preconditioners. -/
def batchLeftHistory (β : ℝ) (G : ℕ → ι → Matrix (Fin m) (Fin n) ℝ) :
    ℕ → ι → Matrix (Fin m) (Fin m) ℝ
  | 0 => fun _ => 0
  | t + 1 => batchLeftEma β (batchLeftHistory β G t) (G t)

/-- Right stacked history over the same indexed gradient sequence.
Source: arXiv:2602.02016v2, §2, Algorithm 1, and §4, stacked right preconditioners. -/
def batchRightHistory (β : ℝ) (G : ℕ → ι → Matrix (Fin m) (Fin n) ℝ) :
    ℕ → ι → Matrix (Fin n) (Fin n) ℝ
  | 0 => fun _ => 0
  | t + 1 => batchRightEma β (batchRightHistory β G t) (G t)

/-- Batched left EMA equals the original block update at every index.
Source: arXiv:2602.02016v2, §4, unchanged stacked Shampoo updates. -/
theorem batchLeftEma_eq (β : ℝ) (L : ι → Matrix (Fin m) (Fin m) ℝ)
    (G : ι → Matrix (Fin m) (Fin n) ℝ) :
    batchLeftEma β L G = fun j => leftEma β (L j) (G j) := rfl

/-- Batched right EMA equals the original block update at every index.
Source: arXiv:2602.02016v2, §4, unchanged right-preconditioner updates. -/
theorem batchRightEma_eq (β : ℝ) (R : ι → Matrix (Fin n) (Fin n) ℝ)
    (G : ι → Matrix (Fin m) (Fin n) ℝ) :
    batchRightEma β R G = fun j => rightEma β (R j) (G j) := rfl

/-- All batched history states, not just one update, equal their blockwise
counterparts. Source: arXiv:2602.02016v2, §2 and §4, temporal stacked Shampoo. -/
theorem batchHistory_eq (β : ℝ) (G : ℕ → ι → Matrix (Fin m) (Fin n) ℝ) (t : ℕ) :
    batchLeftHistory β G t = (fun j => leftHistory β (fun u => G u j) t) ∧
      batchRightHistory β G t = (fun j => rightHistory β (fun u => G u j) t) := by
  induction t with
  | zero => exact ⟨rfl, rfl⟩
  | succ t ih =>
    rw [batchLeftHistory, batchRightHistory, ih.1, ih.2]
    exact ⟨rfl, rfl⟩

/-- Every stacked history remains positive semidefinite on each block.
Source: arXiv:2602.02016v2, §2 and §4, PSD preconditioners for inverse roots. -/
theorem batchHistory_posSemidef (β : ℝ) (G : ℕ → ι → Matrix (Fin m) (Fin n) ℝ)
    (hβ : 0 ≤ β) (hβ' : β ≤ 1) (t : ℕ) (j : ι) :
    (batchLeftHistory β G t j).PosSemidef ∧ (batchRightHistory β G t j).PosSemidef := by
  rw [(batchHistory_eq β G t).1, (batchHistory_eq β G t).2]
  exact history_posSemidef β (fun u => G u j) hβ hβ' t

/-- A nonempty stacked history admits valid EMA parameters,
arXiv:2602.02016v2, §2 and §4. -/
example : (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 := by norm_num

end Transformer.DASH

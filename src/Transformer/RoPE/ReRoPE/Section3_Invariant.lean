module

public import Transformer.RoPE.ReRoPE.Section3_Merge

/-!
# ReRoPE, 计算 (Computation): scores that a common rotation keeps

Su, *Transformer升级之路：12、无限外推的ReRoPE？*, kexue.fm/archives/9708, section 计算
(Computation), merges the two matrices of the dot product `q_iᵀ R^{−min(i−j, w)} k_j`
(`Section3_Merge`). The merge uses only that the score of a rotated query and a rotated key
depends on the difference of their positions, which every score `F` that a common rotation keeps,
`F (R^c x) (R^c y) = F x y`, has:

* `invariant_rotary_sub`: `F q (R^{−d} k) = F (R^a q) (R^{a−d} k)` for every `a`;
* `rerope_attend_eq_onlineMerge_of_invariant`, `leaky_attend_eq_onlineMerge_of_invariant`:
  ReRoPE's and Leaky ReRoPE's attention with the score `F` is the online merge of
  `F (R^i q_i) (R^j k_j)` over the window `i − j < w` and of `F (R^w q_i) k_j`, resp.
  `F (R^{(i−w)/k+w} q_i) (R^{j/k} k_j)`, over the keys beyond it. The dot product is such a
  score, and gives `Section3_Merge` back; the score of a complex head is another
  (`Section3_PhaseScore`).

So with such a score, too, each key is scored once, by one of the two rotations of the query and
of the key, and the attention is one pass.
-/

@[expose] public section

open Matrix
open Transformer.AttentionResiduals Transformer.Residual

namespace Transformer.RoPE.ReRoPE

open RoFormer

variable {p : ℕ} {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- A score `F` that a common rotation keeps depends on the difference of the positions only:
`F q (R^{−d} k) = F (R^a q) (R^{a−d} k)` for every `a`. -/
theorem invariant_rotary_sub (θ : Fin p → ℝ)
    {F : (Fin 2 × Fin p → ℝ) → (Fin 2 × Fin p → ℝ) → ℝ}
    (hF : ∀ c x y, F (rotary θ c *ᵥ x) (rotary θ c *ᵥ y) = F x y)
    (q key : Fin 2 × Fin p → ℝ) (a d : ℝ) :
    F q (rotary θ (-d) *ᵥ key) = F (rotary θ a *ᵥ q) (rotary θ (a - d) *ᵥ key) := by
  rw [← hF a q, mulVec_mulVec, ← rotary_add, sub_eq_add_neg]

/-- The dot product is a score that a common rotation keeps. -/
theorem dotProduct_rotary_rotary (θ : Fin p → ℝ) (c : ℝ) (x y : Fin 2 × Fin p → ℝ) :
    (rotary θ c *ᵥ x) ⬝ᵥ (rotary θ c *ᵥ y) = x ⬝ᵥ y := by
  rw [rotary_dotProduct, sub_self, rotary_zero, one_mulVec]

example (q key : Fin 2 × Fin 16 → ℝ) :
    q ⬝ᵥ (rotary (fun l : Fin 16 => frequency 10000 32 l) (-100) *ᵥ key) =
      (rotary (fun l : Fin 16 => frequency 10000 32 l) 16 *ᵥ q) ⬝ᵥ
        (rotary (fun l : Fin 16 => frequency 10000 32 l) (16 - 100) *ᵥ key) :=
  invariant_rotary_sub (F := fun x y => x ⬝ᵥ y) _ (dotProduct_rotary_rotary _) q key 16 100

/-- Computation: ReRoPE's attention at the query `i` over the keys `j ≤ i`, with a score `F` that
a common rotation keeps, is the online merge of `F (R^i q_i) (R^j k_j)` over the window
`i − j < w` and of `F (R^w q_i) k_j` over the keys beyond it. -/
theorem rerope_attend_eq_onlineMerge_of_invariant (θ : Fin p → ℝ) {w : ℕ} (hw : 0 < w) (i : ℕ)
    {F : (Fin 2 × Fin p → ℝ) → (Fin 2 × Fin p → ℝ) → ℝ}
    (hF : ∀ c x y, F (rotary θ c *ᵥ x) (rotary θ c *ᵥ y) = F x y)
    (q : Fin 2 × Fin p → ℝ) (key : ℕ → Fin 2 × Fin p → ℝ) (v : ℕ → E) (m₁ m₂ : ℝ) :
    attend (fun j : Fin (i + 1) =>
        F q (rotary θ (-rectifiedPosition w ((i - j : ℕ) : ℝ)) *ᵥ key j)) (fun j => v j) =
      onlineMerge
        (fun j : {j : Fin (i + 1) // i - (j : ℕ) < w} =>
          F (rotary θ i *ᵥ q) (rotary θ (j.1 : ℕ) *ᵥ key j.1))
        (fun j : {j : Fin (i + 1) // ¬i - (j : ℕ) < w} => F (rotary θ w *ᵥ q) (key j.1))
        (fun j => v j.1) (fun j => v j.1) m₁ m₂ := by
  have : Nonempty {j : Fin (i + 1) // i - (j : ℕ) < w} := ⟨⟨Fin.last i, by simpa using hw⟩⟩
  rw [attend_eq_onlineMerge (fun j : Fin (i + 1) => i - (j : ℕ) < w) _ _ m₁ m₂]
  congr 1
  · funext j
    have hj : (j.1 : ℕ) ≤ i := Nat.le_of_lt_succ j.1.2
    rw [Nat.cast_sub hj, rectifiedPosition, min_eq_left ((window_iff hj).mp j.2).le,
      invariant_rotary_sub θ hF q _ i, sub_sub_cancel]
  · funext j
    have hj : (j.1 : ℕ) ≤ i := Nat.le_of_lt_succ j.1.2
    rw [Nat.cast_sub hj, rectifiedPosition,
      min_eq_right (not_lt.mp (mt (window_iff hj).mpr j.2)), invariant_rotary_sub θ hF q _ w,
      sub_self, rotary_zero, one_mulVec]

example (q : Fin 2 × Fin 16 → ℝ) (key : ℕ → Fin 2 × Fin 16 → ℝ) (v : ℕ → ℝ) :
    attend (fun j : Fin (127 + 1) =>
        q ⬝ᵥ (rotary (fun l : Fin 16 => frequency 10000 32 l)
          (-rectifiedPosition (16 : ℕ) ((127 - j : ℕ) : ℝ)) *ᵥ key j)) (fun j => v j) =
      onlineMerge
        (fun j : {j : Fin (127 + 1) // 127 - (j : ℕ) < 16} =>
          (rotary (fun l : Fin 16 => frequency 10000 32 l) (127 : ℕ) *ᵥ q) ⬝ᵥ
            (rotary (fun l : Fin 16 => frequency 10000 32 l) (j.1 : ℕ) *ᵥ key j.1))
        (fun j : {j : Fin (127 + 1) // ¬127 - (j : ℕ) < 16} =>
          (rotary (fun l : Fin 16 => frequency 10000 32 l) (16 : ℕ) *ᵥ q) ⬝ᵥ key j.1)
        (fun j => v j.1) (fun j => v j.1) 0 0 :=
  rerope_attend_eq_onlineMerge_of_invariant (F := fun x y => x ⬝ᵥ y) _ (by norm_num) 127
    (dotProduct_rotary_rotary _) q key v 0 0

/-- Computation: Leaky ReRoPE's attention at the query `i` over the keys `j ≤ i`, with a score
`F` that a common rotation keeps, is the online merge of `F (R^i q_i) (R^j k_j)` over the window
`i − j < w` and of `F (R^{(i−w)/k+w} q_i) (R^{j/k} k_j)` over the keys beyond it. -/
theorem leaky_attend_eq_onlineMerge_of_invariant (θ : Fin p → ℝ) {w : ℕ} (hw : 0 < w) (k : ℝ)
    (i : ℕ) {F : (Fin 2 × Fin p → ℝ) → (Fin 2 × Fin p → ℝ) → ℝ}
    (hF : ∀ c x y, F (rotary θ c *ᵥ x) (rotary θ c *ᵥ y) = F x y)
    (q : Fin 2 × Fin p → ℝ) (key : ℕ → Fin 2 × Fin p → ℝ) (v : ℕ → E) (m₁ m₂ : ℝ) :
    attend (fun j : Fin (i + 1) =>
        F q (rotary θ (-leakyPosition w k ((i - j : ℕ) : ℝ)) *ᵥ key j)) (fun j => v j) =
      onlineMerge
        (fun j : {j : Fin (i + 1) // i - (j : ℕ) < w} =>
          F (rotary θ i *ᵥ q) (rotary θ (j.1 : ℕ) *ᵥ key j.1))
        (fun j : {j : Fin (i + 1) // ¬i - (j : ℕ) < w} =>
          F (rotary θ ((i - w) / k + w) *ᵥ q) (rotary θ ((j.1 : ℕ) / k) *ᵥ key j.1))
        (fun j => v j.1) (fun j => v j.1) m₁ m₂ := by
  have : Nonempty {j : Fin (i + 1) // i - (j : ℕ) < w} := ⟨⟨Fin.last i, by simpa using hw⟩⟩
  rw [attend_eq_onlineMerge (fun j : Fin (i + 1) => i - (j : ℕ) < w) _ _ m₁ m₂]
  congr 1
  · funext j
    have hj : (j.1 : ℕ) ≤ i := Nat.le_of_lt_succ j.1.2
    rw [Nat.cast_sub hj, leakyPosition, ite_eq_left ((window_iff hj).mp j.2),
      invariant_rotary_sub θ hF q _ i, sub_sub_cancel]
  · funext j
    have hj : (j.1 : ℕ) ≤ i := Nat.le_of_lt_succ j.1.2
    rw [Nat.cast_sub hj, leakyPosition, ite_eq_right (mt (window_iff hj).mpr j.2),
      invariant_rotary_sub θ hF q _ ((i - w) / k + w)]
    congr 3
    ring

example (q : Fin 2 × Fin 16 → ℝ) (key : ℕ → Fin 2 × Fin 16 → ℝ) (v : ℕ → ℝ) :
    attend (fun j : Fin (127 + 1) =>
        q ⬝ᵥ (rotary (fun l : Fin 16 => frequency 10000 32 l)
          (-leakyPosition (16 : ℕ) 16 ((127 - j : ℕ) : ℝ)) *ᵥ key j)) (fun j => v j) =
      onlineMerge
        (fun j : {j : Fin (127 + 1) // 127 - (j : ℕ) < 16} =>
          (rotary (fun l : Fin 16 => frequency 10000 32 l) (127 : ℕ) *ᵥ q) ⬝ᵥ
            (rotary (fun l : Fin 16 => frequency 10000 32 l) (j.1 : ℕ) *ᵥ key j.1))
        (fun j : {j : Fin (127 + 1) // ¬127 - (j : ℕ) < 16} =>
          (rotary (fun l : Fin 16 => frequency 10000 32 l)
              (((127 : ℕ) - (16 : ℕ)) / 16 + (16 : ℕ)) *ᵥ q) ⬝ᵥ
            (rotary (fun l : Fin 16 => frequency 10000 32 l) ((j.1 : ℕ) / 16) *ᵥ key j.1))
        (fun j => v j.1) (fun j => v j.1) 0 0 :=
  leaky_attend_eq_onlineMerge_of_invariant (F := fun x y => x ⬝ᵥ y) _ (by norm_num) 16 127
    (dotProduct_rotary_rotary _) q key v 0 0

end Transformer.RoPE.ReRoPE

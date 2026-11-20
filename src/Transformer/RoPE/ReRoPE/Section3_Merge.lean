module

public import Transformer.AttentionResiduals.Section4_OnlineMerge
public import Transformer.RoPE.ReRoPE.Section3_Computation

/-!
# ReRoPE, 计算 (Computation): one attention over two regions

Su, *Transformer升级之路：12、无限外推的ReRoPE？*, kexue.fm/archives/9708, section 计算
(Computation), computes both matrices `a⁽¹⁾`, `a⁽²⁾` in full and merges them by the condition
`i − j < w`: "whether ReRoPE or Leaky ReRoPE, computing the attention matrix twice is unavoidable
(if there is a more efficient implementation, please advise)", and the custom matrix rules out a
ready-made flash attention; in decoding "the cache of the Key sequence can only store the keys
before RoPE, and at each decoding step the corresponding RoPE is applied to the whole Key
sequence".

One pass of RoPE does not give the merged matrix (`Section3_SinglePass`), nor, on long sequences,
one pass of any fixed dimension (`Section3_Rank`, `Section3_RankLeaky`). But each of the two
matrices is needed only on its own region, and softmax over two disjoint sets of keys is the
online merge of their statistics, as between the tiles of FlashAttention (Dao et al.,
arXiv:2205.14135, §3.1, Tiling; `AttentionResiduals.onlineMerge_eq_attend`):

* `attend_eq_onlineMerge`: attention over all keys is the online merge of its restrictions to the
  keys with a property and to the others, for any reference shifts;
* `rerope_attend_eq_onlineMerge`: ReRoPE's attention at the query position `i` over the keys
  `j ≤ i` is the online merge of RoPE's attention `(R^i q_i)ᵀ(R^j k_j)` over the window
  `i − j < w` and of the attention `(R^w q_i)ᵀ k_j` over the keys beyond it;
* `leaky_attend_eq_onlineMerge`: the same for Leaky ReRoPE, whose keys beyond the window are
  attended with RoPE at the positions `(i − w)/k + w` and `j/k`;
* `card_window`, `card_window_add_card_far`: the window holds `min (i + 1) w` of the keys, and the
  two regions split the `i + 1` keys of the query, so each score is computed once, by one of the
  two matrices.

In decoding every key enters rotated by a position of its own, `j` in the window and `0`
(ReRoPE) or `j/k` (Leaky ReRoPE) beyond it, which does not change from step to step; only the
query is rotated anew, by `i` and by `w` or `(i − w)/k + w`. The source omits the scale of the
logit; a scale, like the `log n` factor of `Section4_Experiments`, multiplies the query `q`.
-/

@[expose] public section

open Matrix
open Transformer.AttentionResiduals Transformer.Residual

namespace Transformer.RoPE.ReRoPE

open RoFormer

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Attention does not depend on the order of the keys: reindexing them by an equivalence keeps
it. -/
theorem attend_comp_equiv {α β : Type*} [Fintype α] [Fintype β] (e : α ≃ β) (s : β → ℝ)
    (v : β → E) : attend (fun a => s (e a)) (fun a => v (e a)) = attend s v := by
  have hZ : ∑ a, Real.exp (s (e a)) = ∑ j, Real.exp (s j) :=
    Equiv.sum_comp e (fun j => Real.exp (s j))
  simp only [attend, weightedSum, depthWeight, depthPartition, hZ]
  exact Equiv.sum_comp e (fun j => (Real.exp (s j) / ∑ i, Real.exp (s i)) • v j)

/-- Attention over all keys is the online merge of its restrictions to the keys with `P` and to
the others, for any reference shifts `m₁`, `m₂`, when some key has `P`. -/
theorem attend_eq_onlineMerge {α : Type*} [Fintype α] (P : α → Prop) [DecidablePred P]
    [Nonempty {a // P a}] (s : α → ℝ) (v : α → E) (m₁ m₂ : ℝ) :
    attend s v = onlineMerge (fun a : {a // P a} => s a) (fun a : {a // ¬P a} => s a)
      (fun a => v a) (fun a => v a) m₁ m₂ := by
  rw [onlineMerge_eq_attend, ← attend_comp_equiv (Equiv.sumCompl P) s v]
  congr 1 <;> funext x <;> rcases x with a | a <;> rfl

example (s : Fin 3 → ℝ) (v : Fin 3 → ℝ) :
    attend s v = onlineMerge (fun a : {j : Fin 3 // (j : ℕ) < 2} => s a)
      (fun a : {j : Fin 3 // ¬(j : ℕ) < 2} => s a) (fun a => v a) (fun a => v a) 0 0 :=
  have : Nonempty {j : Fin 3 // (j : ℕ) < 2} := ⟨⟨0, by decide⟩⟩
  attend_eq_onlineMerge (fun j : Fin 3 => (j : ℕ) < 2) s v 0 0

/-- The key at `j` lies in the window of the query at `i` iff it does on the real line. -/
theorem window_iff {i j w : ℕ} (hj : j ≤ i) : i - j < w ↔ (i : ℝ) - j < w := by
  rw [← Nat.cast_sub hj, Nat.cast_lt]

example : 4096 - 4000 < 512 ↔ ((4096 : ℕ) : ℝ) - (4000 : ℕ) < (512 : ℕ) :=
  window_iff (by norm_num)

/-- Computation: ReRoPE's attention at the query position `i` over the keys `j ≤ i`, with the
scores `q_iᵀ R^{−min(i−j, w)} k_j`, is the online merge of two attentions over disjoint sets of
keys: the first matrix `(R^i q_i)ᵀ(R^j k_j)` over the window `i − j < w`, and the second matrix
`(R^w q_i)ᵀ k_j` over the keys beyond it, which are not rotated. -/
theorem rerope_attend_eq_onlineMerge {p : ℕ} (θ : Fin p → ℝ) {w : ℕ} (hw : 0 < w) (i : ℕ)
    (q : Fin 2 × Fin p → ℝ) (key : ℕ → Fin 2 × Fin p → ℝ) (v : ℕ → E) (m₁ m₂ : ℝ) :
    attend (fun j : Fin (i + 1) =>
        q ⬝ᵥ (rotary θ (-rectifiedPosition w ((i - j : ℕ) : ℝ)) *ᵥ key j)) (fun j => v j) =
      onlineMerge
        (fun j : {j : Fin (i + 1) // i - (j : ℕ) < w} =>
          (rotary θ i *ᵥ q) ⬝ᵥ (rotary θ (j.1 : ℕ) *ᵥ key j.1))
        (fun j : {j : Fin (i + 1) // ¬i - (j : ℕ) < w} => (rotary θ w *ᵥ q) ⬝ᵥ key j.1)
        (fun j => v j.1) (fun j => v j.1) m₁ m₂ := by
  have : Nonempty {j : Fin (i + 1) // i - (j : ℕ) < w} := ⟨⟨Fin.last i, by simpa using hw⟩⟩
  rw [attend_eq_onlineMerge (fun j : Fin (i + 1) => i - (j : ℕ) < w) _ _ m₁ m₂]
  congr 1
  · funext j
    have hj : (j.1 : ℕ) ≤ i := Nat.le_of_lt_succ j.1.2
    rw [Nat.cast_sub hj, ← merge_rectified, ite_eq_left ((window_iff hj).mp j.2)]
  · funext j
    have hj : (j.1 : ℕ) ≤ i := Nat.le_of_lt_succ j.1.2
    rw [Nat.cast_sub hj, ← merge_rectified, ite_eq_right (mt (window_iff hj).mpr j.2)]

example (q : Fin 2 × Fin 64 → ℝ) (key : ℕ → Fin 2 × Fin 64 → ℝ) (v : ℕ → ℝ) :
    attend (fun j : Fin (4096 + 1) =>
        q ⬝ᵥ (rotary (fun l : Fin 64 => frequency 10000 128 l)
          (-rectifiedPosition (512 : ℕ) ((4096 - j : ℕ) : ℝ)) *ᵥ key j)) (fun j => v j) =
      onlineMerge
        (fun j : {j : Fin (4096 + 1) // 4096 - (j : ℕ) < 512} =>
          (rotary (fun l : Fin 64 => frequency 10000 128 l) (4096 : ℕ) *ᵥ q) ⬝ᵥ
            (rotary (fun l : Fin 64 => frequency 10000 128 l) (j.1 : ℕ) *ᵥ key j.1))
        (fun j : {j : Fin (4096 + 1) // ¬4096 - (j : ℕ) < 512} =>
          (rotary (fun l : Fin 64 => frequency 10000 128 l) (512 : ℕ) *ᵥ q) ⬝ᵥ key j.1)
        (fun j => v j.1) (fun j => v j.1) 0 0 :=
  rerope_attend_eq_onlineMerge _ (by norm_num) 4096 q key v 0 0

/-- Computation: Leaky ReRoPE's attention at the query position `i` over the keys `j ≤ i` is the
online merge of the first matrix `(R^i q_i)ᵀ(R^j k_j)` over the window `i − j < w` and of the
second matrix `(R^{(i−w)/k+w} q_i)ᵀ(R^{j/k} k_j)`, RoPE at the positions of an interpolation, over
the keys beyond it. -/
theorem leaky_attend_eq_onlineMerge {p : ℕ} (θ : Fin p → ℝ) {w : ℕ} (hw : 0 < w) (k : ℝ)
    (i : ℕ) (q : Fin 2 × Fin p → ℝ) (key : ℕ → Fin 2 × Fin p → ℝ) (v : ℕ → E) (m₁ m₂ : ℝ) :
    attend (fun j : Fin (i + 1) =>
        q ⬝ᵥ (rotary θ (-leakyPosition w k ((i - j : ℕ) : ℝ)) *ᵥ key j)) (fun j => v j) =
      onlineMerge
        (fun j : {j : Fin (i + 1) // i - (j : ℕ) < w} =>
          (rotary θ i *ᵥ q) ⬝ᵥ (rotary θ (j.1 : ℕ) *ᵥ key j.1))
        (fun j : {j : Fin (i + 1) // ¬i - (j : ℕ) < w} =>
          (rotary θ ((i - w) / k + w) *ᵥ q) ⬝ᵥ (rotary θ ((j.1 : ℕ) / k) *ᵥ key j.1))
        (fun j => v j.1) (fun j => v j.1) m₁ m₂ := by
  have : Nonempty {j : Fin (i + 1) // i - (j : ℕ) < w} := ⟨⟨Fin.last i, by simpa using hw⟩⟩
  rw [attend_eq_onlineMerge (fun j : Fin (i + 1) => i - (j : ℕ) < w) _ _ m₁ m₂]
  congr 1
  · funext j
    have hj : (j.1 : ℕ) ≤ i := Nat.le_of_lt_succ j.1.2
    rw [Nat.cast_sub hj, ← merge_leaky, ite_eq_left ((window_iff hj).mp j.2)]
  · funext j
    have hj : (j.1 : ℕ) ≤ i := Nat.le_of_lt_succ j.1.2
    rw [Nat.cast_sub hj, ← merge_leaky, ite_eq_right (mt (window_iff hj).mpr j.2)]

example (q : Fin 2 × Fin 64 → ℝ) (key : ℕ → Fin 2 × Fin 64 → ℝ) (v : ℕ → ℝ) :
    attend (fun j : Fin (4096 + 1) =>
        q ⬝ᵥ (rotary (fun l : Fin 64 => frequency 10000 128 l)
          (-leakyPosition (128 : ℕ) 16 ((4096 - j : ℕ) : ℝ)) *ᵥ key j)) (fun j => v j) =
      onlineMerge
        (fun j : {j : Fin (4096 + 1) // 4096 - (j : ℕ) < 128} =>
          (rotary (fun l : Fin 64 => frequency 10000 128 l) (4096 : ℕ) *ᵥ q) ⬝ᵥ
            (rotary (fun l : Fin 64 => frequency 10000 128 l) (j.1 : ℕ) *ᵥ key j.1))
        (fun j : {j : Fin (4096 + 1) // ¬4096 - (j : ℕ) < 128} =>
          (rotary (fun l : Fin 64 => frequency 10000 128 l)
              (((4096 : ℕ) - (128 : ℕ)) / 16 + (128 : ℕ)) *ᵥ q) ⬝ᵥ
            (rotary (fun l : Fin 64 => frequency 10000 128 l) ((j.1 : ℕ) / 16) *ᵥ key j.1))
        (fun j => v j.1) (fun j => v j.1) 0 0 :=
  leaky_attend_eq_onlineMerge _ (by norm_num) 16 4096 q key v 0 0

/-- Computation, decoding: the window of the query at `i` holds `min (i + 1) w` of the keys
`j ≤ i`, so at most `w` keys are rotated by their own positions. -/
theorem card_window (i w : ℕ) :
    Fintype.card {j : Fin (i + 1) // i - (j : ℕ) < w} = min (i + 1) w := by
  rw [← Fintype.card_fin (min (i + 1) w)]
  refine Fintype.card_congr
    { toFun := fun j => ⟨i - j.1, Nat.lt_min.mpr ⟨by omega, j.2⟩⟩
      invFun := fun d => ⟨⟨i - d, by omega⟩, by have := Nat.lt_min.mp d.2; dsimp only; omega⟩
      left_inv := fun j => Subtype.ext (Fin.ext (by have := j.1.2; dsimp only; omega))
      right_inv := fun d => Fin.ext (by have := Nat.lt_min.mp d.2; dsimp only; omega) }

/-- Computation: the window and the keys beyond it split the `i + 1` keys of the query at `i`, so
the merged attention computes one score per key, by one of the two matrices, as RoPE does. -/
theorem card_window_add_card_far (i w : ℕ) :
    Fintype.card {j : Fin (i + 1) // i - (j : ℕ) < w} +
      Fintype.card {j : Fin (i + 1) // ¬i - (j : ℕ) < w} = i + 1 := by
  rw [← Fintype.card_sum, Fintype.card_congr (Equiv.sumCompl _), Fintype.card_fin]

end Transformer.RoPE.ReRoPE

module

public import Transformer.ReRoPEBlock.Defs
public import Transformer.ComplexFFN.Basic

/-!
# Inside the window the blocks agree; beyond it ReRoPE forgets the order of the far keys

Source: the blocks of `Transformer.ReRoPEBlock.Defs` (DiffGPT `5a37c51`, ReRoPE of
kexue.fm/archives/9708). The two blocks differ only in the position `ψ (i − j)` that a head
gives the distance of a key, `i − j` for LeRoPE and `min(i − j, w)` for ReRoPE.

* `block_rePosition_of_lt`: at a query `i < w` every distance is below the window, so the two
  blocks with the same `Φ`, scores, values and frequencies are one function. With learned
  frequencies the two classes agree there; with fixed ones ReRoPE's is LeRoPE's at `α = 0`.
* `block_rePosition_perm`: beyond the window every key has the position `w`, so a ReRoPE block
  at the query `i` is unchanged when the tokens at the positions `j` with `i − j ≥ w` are
  permuted among themselves. This holds for every `Φ` and every score and value, so for the
  whole block of the main model: its FFN, normalizations and mHC routing act per position.
* `norm_sub_le_of_rePosition`: hence a ReRoPE block within `δ` of any target on a set holding a
  sequence and its far-permuted copy has `‖T x − T (permute τ x)‖ ≤ 2δ`
  (`Transformer.ComplexFFN.gap_of_invariant`), at any number of heads, any width and any window
  `w` below the query.
-/

@[expose] public section

open Transformer.AttentionResiduals Transformer.RoPE.ReRoPE

namespace Transformer.ReRoPEBlock

variable {X E Y : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {H : ℕ}

/-- Positions that agree on the distances up to `i` give one block at the query `i`. -/
theorem block_congr_position {ψ ψ' : ℕ → ℝ} {i : ℕ} (hψ : ∀ δ ≤ i, ψ δ = ψ' δ)
    (Φ : X → (Fin H → E) → Y) (F : Fin H → ℕ → X → X → ℝ → ℝ) (v : Fin H → X → E)
    (x : ℕ → X) : block ψ Φ F v x i = block ψ' Φ F v x i := by
  simp only [block]
  congr 1
  funext h
  congr 1
  funext j
  rw [hψ _ (Nat.sub_le _ _)]

/-- Inside the window ReRoPE's position is the distance. -/
theorem rePosition_of_lt {w δ : ℕ} (h : δ < w) : rePosition w δ = lePosition δ := by
  rw [rePosition, rectifiedPosition, lePosition, min_eq_left (by exact_mod_cast h.le)]

/-- Beyond the window ReRoPE's position is `w`. -/
theorem rePosition_of_le {w δ : ℕ} (h : w ≤ δ) : rePosition w δ = w := by
  rw [rePosition, rectifiedPosition, min_eq_right (by exact_mod_cast h)]

/-- **Inside the window the blocks agree**: at a query `i < w` the ReRoPE block is the LeRoPE
block with the same `Φ`, scores and values. -/
theorem block_rePosition_of_lt {w i : ℕ} (hi : i < w) (Φ : X → (Fin H → E) → Y)
    (F : Fin H → ℕ → X → X → ℝ → ℝ) (v : Fin H → X → E) (x : ℕ → X) :
    block (rePosition w) Φ F v x i = block lePosition Φ F v x i :=
  block_congr_position (fun _ hδ => rePosition_of_lt (hδ.trans_lt hi)) Φ F v x

/-- One head scoring by the position, at the query `3` below the window `4`. -/
example (x : ℕ → ℝ) :
    block (rePosition 4) (fun _ a => a 0) (fun _ _ _ _ δ => δ) (fun (_ : Fin 1) y => y) x 3 =
      block lePosition (fun _ a => a 0) (fun _ _ _ _ δ => δ) (fun (_ : Fin 1) y => y) x 3 :=
  block_rePosition_of_lt (by norm_num) _ _ _ x

/-- A permutation of the keys `j ≤ i` that fixes every key of the window `i − j < w` keeps the
keys beyond the window beyond it. -/
theorem le_sub_perm {w i : ℕ} {τ : Equiv.Perm (Fin (i + 1))}
    (hτ : ∀ j : Fin (i + 1), i - (j : ℕ) < w → τ j = j) {j : Fin (i + 1)}
    (hj : w ≤ i - (j : ℕ)) : w ≤ i - (τ j : ℕ) := by
  by_contra h
  have hfix : τ j = j := τ.injective (hτ (τ j) (not_le.1 h))
  rw [hfix] at h
  exact h hj

/-- Such a permutation keeps ReRoPE's position of every key. -/
theorem rePosition_perm {w i : ℕ} {τ : Equiv.Perm (Fin (i + 1))}
    (hτ : ∀ j : Fin (i + 1), i - (j : ℕ) < w → τ j = j) (j : Fin (i + 1)) :
    rePosition w (i - (τ j : ℕ)) = rePosition w (i - (j : ℕ)) := by
  by_cases hj : i - (j : ℕ) < w
  · rw [hτ j hj]
  · rw [rePosition_of_le (not_lt.1 hj), rePosition_of_le (le_sub_perm hτ (not_lt.1 hj))]

/-- **ReRoPE does not see the order of the far keys**: if `x'` is `x` with the tokens at the
positions `j ≤ i` permuted by `τ`, and `τ` fixes every position of the window `i − j < w`,
`0 < w`, the ReRoPE block at the query `i` is the same on `x'` and on `x`, whatever `Φ`, scores
and values. -/
theorem block_rePosition_perm {w i : ℕ} (hw : 0 < w) {τ : Equiv.Perm (Fin (i + 1))}
    (hτ : ∀ j : Fin (i + 1), i - (j : ℕ) < w → τ j = j) (Φ : X → (Fin H → E) → Y)
    (F : Fin H → ℕ → X → X → ℝ → ℝ) (v : Fin H → X → E) {x x' : ℕ → X}
    (hx : ∀ j : Fin (i + 1), x' j = x (τ j)) :
    block (rePosition w) Φ F v x' i = block (rePosition w) Φ F v x i := by
  have hi : x' i = x i := by
    have h := hx (Fin.last i)
    rwa [hτ (Fin.last i) (by simpa using hw), Fin.val_last] at h
  simp only [block, hi]
  congr 1
  funext h
  refine Eq.trans ?_ (attend_comp_equiv τ
    (fun j : Fin (i + 1) => F h i (x i) (x j) (rePosition w (i - (j : ℕ))))
    (fun j : Fin (i + 1) => v h (x j)))
  congr 1
  · funext j
    rw [hx j, rePosition_perm hτ j]
  · funext j
    rw [hx j]

/-- The sequence `x` with its tokens at the positions `j ≤ i` permuted by `τ`: `x (τ j)` there. -/
def permute {i : ℕ} (τ : Equiv.Perm (Fin (i + 1))) (x : ℕ → X) : ℕ → X :=
  fun j => if h : j < i + 1 then x (τ ⟨j, h⟩) else x j

theorem permute_apply {i : ℕ} (τ : Equiv.Perm (Fin (i + 1))) (x : ℕ → X) (j : Fin (i + 1)) :
    permute τ x j = x (τ j) := by
  simp [permute, j.2]

/-- Swapping the first two tokens keeps the ReRoPE block at the query `3` with the window `2`. -/
example (x : ℕ → ℝ) :
    block (rePosition 2) (fun _ a => a 0) (fun _ _ _ _ δ => δ) (fun (_ : Fin 1) y => y)
        (permute (Equiv.swap (0 : Fin 4) 1) x) 3 =
      block (rePosition 2) (fun _ a => a 0) (fun _ _ _ _ δ => δ) (fun (_ : Fin 1) y => y) x 3 :=
  block_rePosition_perm (by norm_num) (fun j hj => by
      rw [Equiv.swap_apply_of_ne_of_ne] <;> (intro h; subst h; simp at hj)) _ _ _
    (permute_apply _ x)

/-- **Every ReRoPE block misses what the order of the far keys changes**: if a ReRoPE block at
the query `i` is within `δ` of a target `T`, any map of sequences, on a set holding `x` and
`permute τ x`, where `τ` fixes the window, then `‖T x − T (permute τ x)‖ ≤ 2δ`. -/
theorem norm_sub_le_of_rePosition [NormedAddCommGroup Y] {w i : ℕ} (hw : 0 < w)
    {τ : Equiv.Perm (Fin (i + 1))} (hτ : ∀ j : Fin (i + 1), i - (j : ℕ) < w → τ j = j)
    (Φ : X → (Fin H → E) → Y) (F : Fin H → ℕ → X → X → ℝ → ℝ) (v : Fin H → X → E)
    {T : (ℕ → X) → Y} {S : Set (ℕ → X)} {δ : ℝ}
    (hδ : ∀ x ∈ S, ‖block (rePosition w) Φ F v x i - T x‖ ≤ δ) {x : ℕ → X} (hx : x ∈ S)
    (hx' : permute τ x ∈ S) : ‖T x - T (permute τ x)‖ ≤ 2 * δ :=
  ComplexFFN.gap_of_invariant (f := fun x => block (rePosition w) Φ F v x i)
    (fun x => block_rePosition_perm hw hτ Φ F v (permute_apply τ x)) hδ hx hx'

/-- A ReRoPE block that is exactly the target `T` misses nothing. -/
example (x : ℕ → ℝ) :
    ‖block (rePosition 2) (fun _ a => a 0) (fun _ _ _ _ δ => δ) (fun (_ : Fin 1) y => y) x 3 -
        block (rePosition 2) (fun _ a => a 0) (fun _ _ _ _ δ => δ) (fun (_ : Fin 1) y => y)
          (permute (Equiv.swap (0 : Fin 4) 1) x) 3‖ ≤ 2 * 0 :=
  norm_sub_le_of_rePosition (Y := ℝ) (w := 2) (by norm_num) (fun j hj => by
      rw [Equiv.swap_apply_of_ne_of_ne] <;> (intro h; subst h; simp at hj))
    (fun _ a => a 0) (fun _ _ _ _ δ => δ) (fun (_ : Fin 1) y => y)
    (T := fun x => block (rePosition 2) (fun _ a => a 0) (fun _ _ _ _ δ => δ)
      (fun (_ : Fin 1) y => y) x 3) (S := Set.univ) (fun _ _ => by simp) (Set.mem_univ x)
    (Set.mem_univ _)

end Transformer.ReRoPEBlock

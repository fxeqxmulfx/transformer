module

public import Transformer.LeRoPEBlock.Defs

/-!
# A block with no positional encoding does not see the order of its keys

Source: the blocks of `Transformer.LeRoPEBlock.Defs` (DiffGPT `5a37c51`). A block with no
positional encoding scores a key from the position `i`, the query's token and the key's token,
and from nothing else: its scores ignore the distance `i − j`, as `plainLogit` does. Then the keys
enter the attention as a set.

* `attend_equiv`: renaming the keys does not change an attention;
* `block_noPos_perm`: so a permutation of the keys `j < i` leaves such a block at the query `i`
  unchanged, for every `Φ`, every scores and every values, hence for the whole main-model block:
  its FFN, normalizations and mHC routing act per position;
* `norm_sub_le_of_noPos`: hence a block with no positional encoding that is within `δ` of any
  target `T` on a set holding a sequence and its permuted copy has `‖T x − T x'‖ ≤ 2δ`.
-/

@[expose] public section

open Transformer.AttentionResiduals Transformer.Residual

namespace Transformer.LeRoPEBlock

variable {X E Y : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {H : ℕ}

/-- **Renaming the keys** does not change an attention. -/
theorem attend_equiv {ι κ : Type*} [Fintype ι] [Fintype κ] (σ : ι ≃ κ) (s : κ → ℝ) (u : κ → E) :
    attend (fun j => s (σ j)) (fun j => u (σ j)) = attend s u := by
  simp only [attend, weightedSum, depthWeight, depthPartition]
  rw [Equiv.sum_comp σ fun k => Real.exp (s k)]
  exact Equiv.sum_comp σ fun k => (Real.exp (s k) / ∑ k, Real.exp (s k)) • u k

/-- Swapping the scores and the values of two keys. -/
example (s : Fin 2 → ℝ) (u : Fin 2 → ℝ) :
    attend (fun j => s (Equiv.swap 0 1 j)) (fun j => u (Equiv.swap 0 1 j)) = attend s u :=
  attend_equiv (Equiv.swap 0 1) s u

/-- The sequence `x` with its tokens at the positions `j ≤ i` permuted by `τ`: `x (τ j)` there. -/
def permute {i : ℕ} (τ : Equiv.Perm (Fin (i + 1))) (x : ℕ → X) : ℕ → X :=
  fun j => if h : j < i + 1 then x (τ ⟨j, h⟩) else x j

theorem permute_apply {i : ℕ} (τ : Equiv.Perm (Fin (i + 1))) (x : ℕ → X) (j : Fin (i + 1)) :
    permute τ x j = x (τ j) := by
  simp [permute, j.2]

/-- **A block with no positional encoding does not see the order of the keys**: if `τ` fixes the
query's own position `i`, the block at the query `i` is the same on `x` with its tokens at the
positions `j ≤ i` permuted by `τ`, whatever `Φ`, the scores `G h i (x i) (x j)` and the values. -/
theorem block_noPos_perm {i : ℕ} {τ : Equiv.Perm (Fin (i + 1))} (hτ : τ (Fin.last i) = Fin.last i)
    (Φ : X → (Fin H → E) → Y) (G : Fin H → ℕ → X → X → ℝ) (v : Fin H → X → E) (x : ℕ → X) :
    block Φ (fun h i a b _ => G h i a b) v (permute τ x) i =
      block Φ (fun h i a b _ => G h i a b) v x i := by
  have hi : permute τ x i = x i := by
    have h := permute_apply τ x (Fin.last i)
    rwa [hτ, Fin.val_last] at h
  simp only [block, hi]
  congr 1
  funext h
  refine Eq.trans ?_ (attend_equiv τ (fun j : Fin (i + 1) => G h i (x i) (x j))
    (fun j : Fin (i + 1) => v h (x j)))
  congr 1 <;> funext j <;> rw [permute_apply]

/-- Swapping the first two tokens keeps the block of one head at the query `3`. -/
example (x : ℕ → ℝ) :
    block (fun _ a => a 0) (fun _ _ _ _ _ => 0) (fun (_ : Fin 1) y => y)
        (permute (Equiv.swap (0 : Fin 4) 1) x) 3 =
      block (fun _ a => a 0) (fun _ _ _ _ _ => 0) (fun (_ : Fin 1) y => y) x 3 :=
  block_noPos_perm (i := 3) (τ := Equiv.swap (0 : Fin 4) 1) (by decide) (fun _ a => a 0)
    (fun _ _ _ _ => 0) (fun (_ : Fin 1) y => y) x

/-- Two inputs with the same image are at least `‖T x − T x'‖ / 2` from a target: if `f x' = f x`
and `f` is within `δ` of `T` at both, then `‖T x − T x'‖ ≤ 2 δ`. -/
theorem norm_sub_le_of_eq {V W : Type*} [NormedAddCommGroup W] {f T : V → W} {x x' : V}
    (hf : f x' = f x) {δ : ℝ} (hx : ‖f x - T x‖ ≤ δ) (hx' : ‖f x' - T x'‖ ≤ δ) :
    ‖T x - T x'‖ ≤ 2 * δ := by
  have h : T x - T x' = (T x - f x) + (f x' - T x') := by
    rw [hf]
    abel
  rw [h, two_mul]
  exact (norm_add_le _ _).trans (add_le_add (by rw [norm_sub_rev]; exact hx) hx')

example : ‖(1 : ℝ) - 1‖ ≤ 2 * 0 :=
  norm_sub_le_of_eq (f := fun _ : ℝ => 1) (T := fun _ => 1) (x := 0) (x' := 1) rfl
    (by simp) (by simp)

/-- **A block with no positional encoding misses what the order of the keys changes**: if such
a block at the query `i` is within `δ` of a target `T` of the sequences on a set holding `x` and
`permute τ x`, where `τ` fixes the position `i`, then `‖T x − T (permute τ x)‖ ≤ 2δ`. -/
theorem norm_sub_le_of_noPos [NormedAddCommGroup Y] {i : ℕ} {τ : Equiv.Perm (Fin (i + 1))}
    (hτ : τ (Fin.last i) = Fin.last i) (Φ : X → (Fin H → E) → Y)
    (G : Fin H → ℕ → X → X → ℝ) (v : Fin H → X → E) {T : (ℕ → X) → Y} {S : Set (ℕ → X)}
    {δ : ℝ} (hδ : ∀ x ∈ S, ‖block Φ (fun h i a b _ => G h i a b) v x i - T x‖ ≤ δ)
    {x : ℕ → X} (hx : x ∈ S) (hx' : permute τ x ∈ S) : ‖T x - T (permute τ x)‖ ≤ 2 * δ :=
  norm_sub_le_of_eq (f := fun x => block Φ (fun h i a b _ => G h i a b) v x i)
    (block_noPos_perm hτ Φ G v x) (hδ x hx) (hδ _ hx')

/-- A block that is exactly the target misses nothing. -/
example (x : ℕ → ℝ) :
    ‖block (fun _ a => a 0) (fun _ _ _ _ _ => (0 : ℝ)) (fun (_ : Fin 1) y => y) x 3 -
        block (fun _ a => a 0) (fun _ _ _ _ _ => (0 : ℝ)) (fun (_ : Fin 1) y => y)
          (permute (Equiv.swap (0 : Fin 4) 1) x) 3‖ ≤ 2 * 0 :=
  norm_sub_le_of_noPos (Y := ℝ) (G := fun _ _ _ _ => 0) (by decide) (fun _ a => a 0)
    (fun (_ : Fin 1) y => y)
    (T := fun x => block (fun _ a => a 0) (fun _ _ _ _ _ => (0 : ℝ)) (fun (_ : Fin 1) y => y) x 3)
    (S := Set.univ) (fun _ _ => by simp) (Set.mem_univ x) (Set.mem_univ _)

end Transformer.LeRoPEBlock

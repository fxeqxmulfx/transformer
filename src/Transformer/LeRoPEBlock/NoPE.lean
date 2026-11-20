module

public import Transformer.LeRoPEBlock.PiSwap

/-!
# LeRoPE sees what a block with no positional encoding cannot

Source: the blocks of `Transformer.LeRoPEBlock.Defs` (DiffGPT `5a37c51`). A block whose scores
ignore the distance does not see the order of its keys (`block_noPos_perm`); the π head of the
main model, with one channel of frequency `π`, does (`norm_piHead_sub_swap`).

* `gap_noPos_piHead`: at the query `n + 2`, every block with no positional encoding, with any
  `Φ`, any number of heads, any scores and any values, misses the π head, on a set holding a
  sequence `x` and its copy with the first two tokens swapped, by
  `δ ≥ (1 − N^{−2κ}) / (2N) ‖u_a − u_b‖`, `N = n + 3`, `κ = qᵀq/d`. The π head is a head of
  the main model, so LeRoPE has a block that no block with no positional encoding approximates
  within that bound. The converse containment holds up to closure
  (`Transformer.LeRoPEBlock.Limit`).

The bound is small, about `0.004 ‖u_a − u_b‖` at `κ = 1` and `N = 128`, because the π head spreads
its weight over half of the keys: it is a proof that the order is seen, not that it is seen
well.
-/

@[expose] public section

open Matrix

namespace Transformer.LeRoPEBlock

variable {X E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {H p : ℕ}

/-- Swapping the first two positions fixes every later one, in particular the query's own. -/
theorem swap_zero_one_last (n : ℕ) :
    Equiv.swap (0 : Fin (n + 2 + 1)) 1 (Fin.last (n + 2)) = Fin.last (n + 2) :=
  Equiv.swap_apply_of_ne_of_ne (by simp [Fin.ext_iff]) (by simp [Fin.ext_iff])

/-- **Every block with no positional encoding misses the π head**: at the query `n + 2`, if such
a block is within `δ` of the π head on a set holding `x` and `x` with its first two tokens
swapped, then `(1 − N^{−2κ}) / (2N) ‖u_a − u_b‖ ≤ δ`, `N = n + 3`, `κ = qᵀq/d`. -/
theorem gap_noPos_piHead (θ : Fin p → ℝ) {q : Fin 2 × Fin p → ℝ}
    (hq : ∀ l, θ l ≠ Real.pi → q (0, l) = 0 ∧ q (1, l) = 0) {d : ℝ} (hd : 0 < d) (u : X → E)
    (n : ℕ) (Φ : X → (Fin H → E) → E) (G : Fin H → ℕ → X → X → ℝ) (v : Fin H → X → E)
    {S : Set (ℕ → X)} {δ : ℝ}
    (hδ : ∀ x ∈ S, ‖block Φ (fun h i a b _ => G h i a b) v x (n + 2) -
      piHead θ d q u x (n + 2)‖ ≤ δ) {x : ℕ → X} (hx : x ∈ S)
    (hx' : permute (Equiv.swap (0 : Fin (n + 2 + 1)) 1) x ∈ S) :
    (1 - ((n : ℝ) + 3) ^ (-2 * (q ⬝ᵥ q / d))) / (2 * ((n : ℝ) + 3)) * ‖u (x 0) - u (x 1)‖ ≤ δ := by
  have h := norm_sub_le_of_noPos (swap_zero_one_last n) Φ G v
    (T := fun y => piHead θ d q u y (n + 2)) hδ hx hx'
  have h2 := norm_piHead_sub_swap θ hq hd u x n
  have hN : ((n : ℝ) + 3) ≠ 0 := by positivity
  have h3 : (1 - ((n : ℝ) + 3) ^ (-2 * (q ⬝ᵥ q / d))) / (2 * ((n : ℝ) + 3)) * ‖u (x 0) - u (x 1)‖ =
      ((1 - ((n : ℝ) + 3) ^ (-2 * (q ⬝ᵥ q / d))) / ((n : ℝ) + 3) * ‖u (x 0) - u (x 1)‖) / 2 := by
    field_simp
  rw [h3]
  linarith

/-- The block of one head with a constant score and the π head of width `2` with `q = 1`,
`κ = 1`, at the query `3`, on the two sequences of reals that swap their first two tokens: the
bound holds with `δ` the larger of the two errors. -/
example (x : ℕ → ℝ) :
    (1 - ((1 : ℕ) + 3 : ℝ) ^ (-2 * ((fun _ : Fin 2 × Fin 1 => (1 : ℝ)) ⬝ᵥ
          (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) / 2))) / (2 * (((1 : ℕ) : ℝ) + 3)) *
        ‖(fun t : ℝ => t) (x 0) - (fun t : ℝ => t) (x 1)‖ ≤
      max ‖block (fun _ a => a 0) (fun (_ : Fin 1) _ _ _ _ => (0 : ℝ)) (fun _ y => y) x (1 + 2) -
          piHead (fun _ : Fin 1 => Real.pi) 2 (fun _ : Fin 2 × Fin 1 => (1 : ℝ))
            (fun t : ℝ => t) x (1 + 2)‖
        ‖block (fun _ a => a 0) (fun (_ : Fin 1) _ _ _ _ => (0 : ℝ)) (fun _ y => y)
            (permute (Equiv.swap (0 : Fin (1 + 2 + 1)) 1) x) (1 + 2) -
          piHead (fun _ : Fin 1 => Real.pi) 2 (fun _ : Fin 2 × Fin 1 => (1 : ℝ))
            (fun t : ℝ => t) (permute (Equiv.swap (0 : Fin (1 + 2 + 1)) 1) x) (1 + 2)‖ :=
  gap_noPos_piHead (fun _ : Fin 1 => Real.pi) (q := fun _ : Fin 2 × Fin 1 => (1 : ℝ))
    (fun _ h => absurd rfl h) (d := 2) two_pos (fun t : ℝ => t) 1 (fun _ a => a 0)
    (fun _ _ _ _ => 0) (fun _ y => y)
    (S := {x, permute (Equiv.swap (0 : Fin (1 + 2 + 1)) 1) x})
    (fun y hy => by
      rcases hy with rfl | rfl
      · exact le_max_left _ _
      · exact le_max_right _ _)
    (Set.mem_insert _ _) (Set.mem_insert_of_mem _ (Set.mem_singleton _))

end Transformer.LeRoPEBlock

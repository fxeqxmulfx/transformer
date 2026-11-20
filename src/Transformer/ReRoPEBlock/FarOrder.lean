module

public import Transformer.ReRoPEBlock.PiHead

/-!
# Beyond the window every ReRoPE block misses LeRoPE's π head

Source: the blocks of `Transformer.ReRoPEBlock.Defs` (DiffGPT `5a37c51`, ReRoPE of
kexue.fm/archives/9708). At the query `w + 1` the first two tokens stand at the distances `w + 1`
and `w`, both beyond ReRoPE's window, and ReRoPE gives both the position `w`.

* `gap_rePosition_of_cos`: every ReRoPE block, whatever its `Φ`, heads, scores, values and
  frequencies, within `δ` of a π head (`Transformer.ReRoPEBlock.PiHead`) whose positions of the
  two distances score `cos(π ψ(w + 1)) = −cos(π ψ w) = ±1`, on a set holding `x` and `x` with
  its first two tokens swapped, has `δ ≥ (1 − N^{−2κ}) / (2N) ‖u_a − u_b‖`, `N = w + 2`;
* `gap_rePosition_piHead`: LeRoPE's π head is such a target.

With scores free functions of the position the classes are nested: the ReRoPE block is the
LeRoPE block that reads its positions through the clamp (`rePosition_class_subset`), and
`gap_rePosition_piHead` makes the inclusion strict at every query beyond the window. With the
main model's rotary scores LeRoPE can only turn its keys by `δ θ`, not by `min(δ, w) θ`, and the
comparison is open beyond the window.
-/

@[expose] public section

open Matrix
open Transformer.RoPE.ReRoPE

namespace Transformer.ReRoPEBlock

variable {X E E' Y : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [NormedAddCommGroup E']
  [NormedSpace ℝ E'] {p H : ℕ}

/-- The first two keys of the query `w + 1` lie beyond the window `w`: swapping them fixes every
key `j` with `w + 1 − j < w`. -/
theorem swap_fixes_window {w : ℕ} (j : Fin (w + 1 + 1)) (hj : w + 1 - (j : ℕ) < w) :
    Equiv.swap (0 : Fin (w + 1 + 1)) 1 j = j := by
  rw [Equiv.swap_apply_of_ne_of_ne] <;> (intro h; subst h; simp at hj)

/-- **Every ReRoPE block misses a π head that sees the order of the far keys**: at the query
`w + 1`, `0 < w`, if the π head's positions score `cos(π ψ(w + 1)) = −cos(π ψ w) = ±1`, a ReRoPE
block with any `Φ`, heads, scores and values within `δ` of the π head on a set holding `x` and
`x` with its first two tokens swapped has `δ ≥ (1 − N^{−2κ}) / (2N) ‖u_a − u_b‖`, `N = w + 2`. -/
theorem gap_rePosition_of_cos {ψ : ℕ → ℝ} {w : ℕ} (hw : 0 < w)
    (hψ : Real.cos (ψ (w + 1) * Real.pi) = -Real.cos (ψ w * Real.pi))
    (hψ1 : |Real.cos (ψ w * Real.pi)| = 1) {d : ℝ} (hd : 0 < d) (q : Fin 2 × Fin p → ℝ)
    (u : X → E) (Φ : X → (Fin H → E') → E) (F : Fin H → ℕ → X → X → ℝ → ℝ) (v : Fin H → X → E')
    {S : Set (ℕ → X)} {δ : ℝ}
    (hδ : ∀ x ∈ S, ‖block (rePosition w) Φ F v x (w + 1) - piHead ψ d q u x (w + 1)‖ ≤ δ)
    {x : ℕ → X} (hx : x ∈ S) (hx' : permute (Equiv.swap (0 : Fin (w + 1 + 1)) 1) x ∈ S) :
    (1 - ((w : ℝ) + 2) ^ (-2 * (q ⬝ᵥ q / d))) / ((w : ℝ) + 2) * ‖u (x 0) - u (x 1)‖ ≤ 2 * δ :=
  (norm_piHead_sub_of_cos hψ hψ1 hd q u x).trans
    (norm_sub_le_of_rePosition hw swap_fixes_window Φ F v hδ hx hx')

/-- A ReRoPE head with the window `1` against LeRoPE's π head on a sequence and its swap, with
`δ` the larger of the two misses. -/
example (x : ℕ → ℝ) :=
  gap_rePosition_of_cos (w := 1) one_pos (cos_lePosition_succ 1).1 (cos_lePosition_succ 1).2
    (d := 2) two_pos (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) (fun t : ℝ => t) (fun _ a => a 0)
    (fun _ _ _ _ t => t) (fun (_ : Fin 1) (t : ℝ) => t)
    (S := {x, permute (Equiv.swap (0 : Fin (1 + 1 + 1)) 1) x})
    (δ := max
      ‖block (rePosition 1) (fun _ a => a 0) (fun _ _ _ _ t => t) (fun (_ : Fin 1) (t : ℝ) => t)
          x (1 + 1) - piHead lePosition 2 (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) (fun t : ℝ => t) x
          (1 + 1)‖
      ‖block (rePosition 1) (fun _ a => a 0) (fun _ _ _ _ t => t) (fun (_ : Fin 1) (t : ℝ) => t)
          (permute (Equiv.swap (0 : Fin (1 + 1 + 1)) 1) x) (1 + 1) -
        piHead lePosition 2 (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) (fun t : ℝ => t)
          (permute (Equiv.swap (0 : Fin (1 + 1 + 1)) 1) x) (1 + 1)‖)
    (by rintro y (rfl | rfl); exacts [le_max_left _ _, le_max_right _ _])
    (Set.mem_insert _ _) (Set.mem_insert_of_mem _ rfl)

/-- **Every ReRoPE block misses LeRoPE's π head**: at the query `w + 1`, `0 < w`, a ReRoPE block
within `δ` of LeRoPE's π head on a set holding `x` and `x` with its first two tokens swapped has
`δ ≥ (1 − N^{−2κ}) / (2N) ‖u_a − u_b‖`, `N = w + 2`. -/
theorem gap_rePosition_piHead {w : ℕ} (hw : 0 < w) {d : ℝ} (hd : 0 < d)
    (q : Fin 2 × Fin p → ℝ) (u : X → E) (Φ : X → (Fin H → E') → E)
    (F : Fin H → ℕ → X → X → ℝ → ℝ) (v : Fin H → X → E') {S : Set (ℕ → X)} {δ : ℝ}
    (hδ : ∀ x ∈ S,
      ‖block (rePosition w) Φ F v x (w + 1) - piHead lePosition d q u x (w + 1)‖ ≤ δ)
    {x : ℕ → X} (hx : x ∈ S) (hx' : permute (Equiv.swap (0 : Fin (w + 1 + 1)) 1) x ∈ S) :
    (1 - ((w : ℝ) + 2) ^ (-2 * (q ⬝ᵥ q / d))) / ((w : ℝ) + 2) * ‖u (x 0) - u (x 1)‖ ≤ 2 * δ :=
  gap_rePosition_of_cos hw (cos_lePosition_succ w).1 (cos_lePosition_succ w).2 hd q u Φ F v hδ
    hx hx'

/-- The window `2`, the query `3`: the ReRoPE π head against LeRoPE's, with `δ` the larger
miss. -/
example (x : ℕ → ℝ) :=
  gap_rePosition_piHead (w := 2) two_pos (d := 2) two_pos (fun _ : Fin 2 × Fin 1 => (1 : ℝ))
    (fun t : ℝ => t) (fun _ a => a 0)
    (fun (_ : Fin 1) => headScore (fun _ => Real.pi) 0 2 (fun _ => fun _ : Fin 2 × Fin 1 => 1)
      (fun _ => fun _ => 1)) (fun (_ : Fin 1) (t : ℝ) => t)
    (S := {x, permute (Equiv.swap (0 : Fin (2 + 1 + 1)) 1) x})
    (δ := max
      ‖piHead (rePosition 2) 2 (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) (fun t : ℝ => t) x (2 + 1) -
        piHead lePosition 2 (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) (fun t : ℝ => t) x (2 + 1)‖
      ‖piHead (rePosition 2) 2 (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) (fun t : ℝ => t)
          (permute (Equiv.swap (0 : Fin (2 + 1 + 1)) 1) x) (2 + 1) -
        piHead lePosition 2 (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) (fun t : ℝ => t)
          (permute (Equiv.swap (0 : Fin (2 + 1 + 1)) 1) x) (2 + 1)‖)
    (by rintro y (rfl | rfl); exacts [le_max_left _ _, le_max_right _ _])
    (Set.mem_insert _ _) (Set.mem_insert_of_mem _ rfl)

/-- **With free scores ReRoPE's blocks are LeRoPE's**: the ReRoPE block with the scores `F` is
the LeRoPE block whose scores read the position through the clamp `min(t, w)`. -/
theorem rePosition_class_subset (w : ℕ) :
    {f : (ℕ → X) → ℕ → Y | ∃ (Φ : X → (Fin H → E) → Y) (F : Fin H → ℕ → X → X → ℝ → ℝ)
        (v : Fin H → X → E), f = block (rePosition w) Φ F v} ⊆
      {f | ∃ (Φ : X → (Fin H → E) → Y) (F : Fin H → ℕ → X → X → ℝ → ℝ) (v : Fin H → X → E),
        f = block lePosition Φ F v} := by
  rintro f ⟨Φ, F, v, rfl⟩
  exact ⟨Φ, fun h i a b t => F h i a b (rectifiedPosition w t), v, rfl⟩

end Transformer.ReRoPEBlock

module

public import Transformer.ReRoPEBlock.FarOrder
public import Transformer.RoPE.InvLeakyReRoPE.Section2_Inversion

/-!
# Leaky ReRoPE in the block

Source: Su, kexue.fm/archives/9708, section 融合 (`papers/kexue-9708-rerope`), matrix
(leaky-rerope): Leaky ReRoPE gives the distance `δ` the position `δ` inside the window and
`w + (δ − w)/k` beyond it (`leakyPosition`); the post's runs take `w = 128, 256` and
`k = 8, 16, 32, 64`. `lkPosition w k` is that position in the block of
`Transformer.ReRoPEBlock.Defs`.

* `block_lkPosition_of_lt`: inside the window the Leaky ReRoPE block is the LeRoPE block.
* `lkPosition_class_eq`: for `k > 0` the position is invertible (`leakyPosition_inv_left`), so
  with scores free functions of the position the Leaky ReRoPE blocks are exactly the LeRoPE
  blocks. Leaky ReRoPE forgets nothing; ReRoPE forgets the order of the far keys.
* `headScore_lkPosition_of_le`: with the main model's rotary scores, beyond the window a Leaky
  ReRoPE head with the frequencies `θ` is a LeRoPE head with the frequencies `θ/k` whose keys
  are turned by the constant `(w − w/k) θ`: there Leaky ReRoPE is LeRoPE at `k` times lower
  frequencies (`headScore_mul_frequency`).
* `gap_rePosition_lkHead`: for a positive integer `k` the Leaky ReRoPE head with every frequency
  `kπ` is the π head at the positions `k (w + (δ − w)/k)` (`lkHead_eq_piHead`). The first two
  keys of the query `w + 1` stand at `kw + 1` and `kw`, so every ReRoPE block with the window `w`
  misses it as it misses LeRoPE's π head.
-/

@[expose] public section

open Matrix
open Transformer.RoPE.ReRoPE Transformer.RoPE.RoFormer Transformer.RoPE.InvLeakyReRoPE

namespace Transformer.ReRoPEBlock

variable {X E E' Y : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [NormedAddCommGroup E']
  [NormedSpace ℝ E'] {p H : ℕ}

/-- Leaky ReRoPE's position of the distance `δ`: `δ` inside the window `w`, `w + (δ − w)/k`
beyond it (`leakyPosition`). -/
noncomputable def lkPosition (w : ℕ) (k : ℝ) (δ : ℕ) : ℝ := leakyPosition w k δ

/-- Inside the window Leaky ReRoPE's position is the distance. -/
theorem lkPosition_of_lt {w δ : ℕ} (k : ℝ) (h : δ < w) : lkPosition w k δ = lePosition δ :=
  leakyPosition_of_lt k (by exact_mod_cast h)

example : lkPosition 128 16 100 = lePosition 100 := lkPosition_of_lt 16 (by norm_num)

/-- Beyond the window Leaky ReRoPE's positions step by `1/k`. -/
theorem lkPosition_of_le {w δ : ℕ} (k : ℝ) (h : w ≤ δ) : lkPosition w k δ = w + (δ - w) / k := by
  rw [lkPosition, leakyPosition, ite_eq_right (not_lt.2 (by exact_mod_cast h))]

example : lkPosition 128 16 144 = 128 + (144 - 128) / 16 := by
  exact_mod_cast lkPosition_of_le 16 (by norm_num : 128 ≤ 144)

/-- **Inside the window Leaky ReRoPE's block is LeRoPE's**: at a query `i < w`. -/
theorem block_lkPosition_of_lt {w i : ℕ} (hi : i < w) (k : ℝ) (Φ : X → (Fin H → E) → Y)
    (F : Fin H → ℕ → X → X → ℝ → ℝ) (v : Fin H → X → E) (x : ℕ → X) :
    block (lkPosition w k) Φ F v x i = block lePosition Φ F v x i :=
  block_congr_position (fun _ hδ => lkPosition_of_lt k (hδ.trans_lt hi)) Φ F v x

/-- One head scoring by the position, at the query `127` below the window `128`. -/
example (x : ℕ → ℝ) :
    block (lkPosition 128 16) (fun _ a => a 0) (fun _ _ _ _ δ => δ) (fun (_ : Fin 1) y => y) x 127 =
      block lePosition (fun _ a => a 0) (fun _ _ _ _ δ => δ) (fun (_ : Fin 1) y => y) x 127 :=
  block_lkPosition_of_lt (by norm_num) 16 _ _ _ x

/-- **With free scores Leaky ReRoPE's blocks are LeRoPE's**: for `k > 0` the Leaky ReRoPE block
with the scores `F` is the LeRoPE block whose scores read the position through
`leakyPosition w k`, and the LeRoPE block is the Leaky ReRoPE block whose scores read it through
the inverse `leakyPosition w k⁻¹`. -/
theorem lkPosition_class_eq (w : ℕ) {k : ℝ} (hk : 0 < k) :
    {f : (ℕ → X) → ℕ → Y | ∃ (Φ : X → (Fin H → E) → Y) (F : Fin H → ℕ → X → X → ℝ → ℝ)
        (v : Fin H → X → E), f = block (lkPosition w k) Φ F v} =
      {f | ∃ (Φ : X → (Fin H → E) → Y) (F : Fin H → ℕ → X → X → ℝ → ℝ) (v : Fin H → X → E),
        f = block lePosition Φ F v} := by
  ext f
  constructor
  · rintro ⟨Φ, F, v, rfl⟩
    exact ⟨Φ, fun h i a b t => F h i a b (leakyPosition w k t), v, rfl⟩
  · rintro ⟨Φ, F, v, rfl⟩
    refine ⟨Φ, fun h i a b t => F h i a b (leakyPosition w k⁻¹ t), v, ?_⟩
    funext x i
    simp only [block, lkPosition, lePosition, leakyPosition_inv_left _ hk]

example := lkPosition_class_eq (X := ℝ) (Y := ℝ) (E := ℝ) (H := 1) 128 (k := 16) (by norm_num)

/-- Scaling every frequency by `c` scales the position by `c`. -/
theorem headScore_mul_frequency (c : ℝ) (θ : Fin p → ℝ) (β d : ℝ) (Q K : X → Fin 2 × Fin p → ℝ)
    (i : ℕ) (x y : X) (t : ℝ) :
    headScore (fun l => c * θ l) β d Q K i x y t = headScore θ β d Q K i x y (c * t) := by
  have h : ∀ l, -t * (c * θ l) = -(c * t) * θ l := fun l => by ring
  simp only [headScore, rotary, h]

/-- **Beyond the window a Leaky ReRoPE head is a LeRoPE head at `k` times lower frequencies**:
at a distance `δ ≥ w` the head with the frequencies `θ` scores as the head with the frequencies
`θ/k` whose keys are turned by `(w − w/k) θ`. -/
theorem headScore_lkPosition_of_le {w δ : ℕ} (h : w ≤ δ) (k : ℝ) (θ : Fin p → ℝ) (β d : ℝ)
    (Q K : X → Fin 2 × Fin p → ℝ) (i : ℕ) (x y : X) :
    headScore θ β d Q K i x y (lkPosition w k δ) =
      headScore (fun l => k⁻¹ * θ l) β d Q (fun y => rotary θ (-(w - w / k)) *ᵥ K y) i x y δ := by
  rw [headScore_mul_frequency, headScore, headScore, lkPosition_of_le k h, mulVec_mulVec,
    ← rotary_add, show -((w : ℝ) + (δ - w) / k) = -(k⁻¹ * δ) + -(w - w / k) by ring]

/-- The window `128`, `k = 16`, at the distance `200`. -/
example (θ : Fin 16 → ℝ) (Q K : ℕ → Fin 2 × Fin 16 → ℝ) (x y : ℕ) :=
  headScore_lkPosition_of_le (by norm_num : 128 ≤ 200) 16 θ 0 32 Q K 300 x y

/-- **Leaky ReRoPE's `kπ` head**: one head with every frequency `kπ`, `β = 0` and the query and
the key `q` for every token, at Leaky ReRoPE's positions; the block returns its output. -/
noncomputable def lkHead (w : ℕ) (k d : ℝ) (q : Fin 2 × Fin p → ℝ) (u : X → E) :
    (ℕ → X) → ℕ → E :=
  block (lkPosition w k) (fun _ a => a 0)
    (fun (_ : Fin 1) => headScore (fun _ => k * Real.pi) 0 d (fun _ => q) (fun _ => q))
    (fun _ => u)

/-- Leaky ReRoPE's `kπ` head is the π head at the positions `k (w + (δ − w)/k)`. -/
theorem lkHead_eq_piHead (w : ℕ) (k d : ℝ) (q : Fin 2 × Fin p → ℝ) (u : X → E) :
    lkHead w k d q u = piHead (fun δ => k * lkPosition w k δ) d q u := by
  funext x i
  simp only [lkHead, piHead, block, headScore_mul_frequency]

/-- For a positive integer `k` the positions of the distances `w + 1` and `w` under the
frequency `kπ` are `kw + 1` and `kw`: they score `∓1`. -/
theorem cos_lkPosition {w k : ℕ} (hk : 0 < k) :
    Real.cos ((k : ℝ) * lkPosition w k (w + 1) * Real.pi) =
        -Real.cos ((k : ℝ) * lkPosition w k w * Real.pi) ∧
      |Real.cos ((k : ℝ) * lkPosition w k w * Real.pi)| = 1 := by
  have hk' : (k : ℝ) ≠ 0 := by positivity
  rw [lkPosition_of_le _ (Nat.le_succ w), lkPosition_of_le _ le_rfl]
  have e1 : (k : ℝ) * ((w : ℝ) + (((w + 1 : ℕ) : ℝ) - w) / k) * Real.pi =
      ((k * w : ℕ) : ℝ) * Real.pi + Real.pi := by
    push_cast
    field_simp
    ring
  have e0 : (k : ℝ) * ((w : ℝ) + ((w : ℝ) - w) / k) * Real.pi = ((k * w : ℕ) : ℝ) * Real.pi := by
    push_cast
    ring
  rw [e1, e0, Real.cos_add_pi, Real.cos_nat_mul_pi, abs_pow, abs_neg, abs_one, one_pow]
  exact ⟨rfl, rfl⟩

example : Real.cos ((16 : ℕ) * lkPosition 128 (16 : ℕ) (128 + 1) * Real.pi) =
    -Real.cos ((16 : ℕ) * lkPosition 128 (16 : ℕ) 128 * Real.pi) :=
  (cos_lkPosition (w := 128) (by norm_num : 0 < 16)).1

/-- **Every ReRoPE block misses Leaky ReRoPE's `kπ` head**: for a positive integer `k`, at the
query `w + 1`, `0 < w`, a ReRoPE block with the window `w` and any `Φ`, heads, scores and values
within `δ` of the head on a set holding `x` and `x` with its first two tokens swapped has
`δ ≥ (1 − N^{−2κ}) / (2N) ‖u_a − u_b‖`, `N = w + 2`. -/
theorem gap_rePosition_lkHead {w k : ℕ} (hw : 0 < w) (hk : 0 < k) {d : ℝ} (hd : 0 < d)
    (q : Fin 2 × Fin p → ℝ) (u : X → E) (Φ : X → (Fin H → E') → E)
    (F : Fin H → ℕ → X → X → ℝ → ℝ) (v : Fin H → X → E') {S : Set (ℕ → X)} {δ : ℝ}
    (hδ : ∀ x ∈ S, ‖block (rePosition w) Φ F v x (w + 1) - lkHead w k d q u x (w + 1)‖ ≤ δ)
    {x : ℕ → X} (hx : x ∈ S) (hx' : permute (Equiv.swap (0 : Fin (w + 1 + 1)) 1) x ∈ S) :
    (1 - ((w : ℝ) + 2) ^ (-2 * (q ⬝ᵥ q / d))) / ((w : ℝ) + 2) * ‖u (x 0) - u (x 1)‖ ≤ 2 * δ := by
  rw [lkHead_eq_piHead] at hδ
  exact gap_rePosition_of_cos hw (cos_lkPosition hk).1 (cos_lkPosition hk).2 hd q u Φ F v hδ hx
    hx'

/-- The ReRoPE π head with the window `1` against Leaky ReRoPE's `16π` head, `δ` the larger
miss. -/
example (x : ℕ → ℝ) :=
  gap_rePosition_lkHead (w := 1) (k := 16) one_pos (by norm_num) (d := 2) two_pos
    (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) (fun t : ℝ => t) (fun _ a => a 0)
    (fun (_ : Fin 1) => headScore (fun _ => Real.pi) 0 2 (fun _ => fun _ : Fin 2 × Fin 1 => 1)
      (fun _ => fun _ => 1)) (fun (_ : Fin 1) (t : ℝ) => t)
    (S := {x, permute (Equiv.swap (0 : Fin (1 + 1 + 1)) 1) x})
    (δ := max
      ‖piHead (rePosition 1) 2 (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) (fun t : ℝ => t) x (1 + 1) -
        lkHead 1 (16 : ℕ) 2 (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) (fun t : ℝ => t) x (1 + 1)‖
      ‖piHead (rePosition 1) 2 (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) (fun t : ℝ => t)
          (permute (Equiv.swap (0 : Fin (1 + 1 + 1)) 1) x) (1 + 1) -
        lkHead 1 (16 : ℕ) 2 (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) (fun t : ℝ => t)
          (permute (Equiv.swap (0 : Fin (1 + 1 + 1)) 1) x) (1 + 1)‖)
    (by rintro y (rfl | rfl); exacts [le_max_left _ _, le_max_right _ _])
    (Set.mem_insert _ _) (Set.mem_insert_of_mem _ rfl)

end Transformer.ReRoPEBlock

module

public import Transformer.AttentionResiduals.Section3_Attention
public import Transformer.RoPE.RoFormer.Section3_Rotary

/-!
# A block of the main model with LeRoPE: the definitions

Source: DiffGPT (`~/python_projects/gpt`, `5a37c51`), `research/main_model/README.txt`,
*Architecture*, and `research/main_model/model.json`. At the position `i` the block reads its four
streams into `z_i` and, per head of real width `d`,

    q_i = LeRoPE_i(R_d(W_Q z_i)),   k_i = LeRoPE_i(R_d(W_K z_i)),   v_i = W_V z_i,
    p_ij = softmax_{j ≤ i}((ln N_i / d)(Re Z_ij + β (Im Z_ij)² / d)),   Z_ij = ∑_c conj(q_ic) k_jc,

with `N_i = i + 1`. The heads' outputs `∑_j p_ij v_j` are concatenated and written into the
streams through `R_D(W_O ·)` by the phase-conjugated mHC, and the FFN branch follows with its own
routing. LeRoPE turns the channel `m` by the angle `i θ_m`, `θ_m = 10000^{−2m/d} e^{α_m}`, the
`α_m` learned and shared by Q, K, heads and layers (Karypis et al., arXiv:2607.10134v1, §3.1,
Eq. (5)): any positive frequency is reachable
(`Transformer.RoPE.LeRoPE.learned_frequency_surjective`).

Everything but the attention acts on one position at a time: the routers and their Sinkhorn maps,
the RMS normalizations, `W_O`, the FFN and the writes. So at the query `i` the block is

    block Φ F v x i = Φ (x i) (h ↦ attend_{j ≤ i} (F h i (x i) (x j) (i − j), v h (x j)))

(`block`): a map `Φ` of the query's own token and of `H` softmax attentions, where the head `h`
scores the key `j` from the position `i`, the query's token, the key's token and the distance
`i − j`. The token `x j` is everything the block reads at the position `j`, the four streams in
the main model. A positional scheme is a choice of the scores `F`:

* `phaseLogit`: SSMax's gain times `Re Z + β (Im Z)² / d`, with `Re Z = qᵀk` and
  `Im Z = −qᵀ J k` for the multiplication `J = rotateHalf` by `i`;
* `leropeLogit`: the main model's head, the logit of `Q x` and of `K y` turned back by the
  distance. `leropeLogit_sub` checks that this is LeRoPE's turn of the query by `i` and of the key
  by `j`;
* `plainLogit`: the same head without rotation, the head of a block with no positional encoding.

Every statement of `Transformer.LeRoPEBlock` leaves `Φ`, the scores' parameters and the values
arbitrary, so it covers the main model's `Φ`. `block_congr_of_le` is causality: the block at the
query `i` reads only the tokens `j ≤ i`.
-/

@[expose] public section

open Matrix
open Transformer.AttentionResiduals Transformer.RoPE.RoFormer

namespace Transformer.LeRoPEBlock

variable {X E Y : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {H p : ℕ}

/-- **One block at the query `i`**: the map `Φ` of the query's token `x i` and of `H` softmax
attentions over the keys `j ≤ i`. The head `h` scores the key `j` by
`F h i (x i) (x j) (i - j)` and reads the value `v h (x j)`. -/
noncomputable def block (Φ : X → (Fin H → E) → Y) (F : Fin H → ℕ → X → X → ℕ → ℝ)
    (v : Fin H → X → E) (x : ℕ → X) (i : ℕ) : Y :=
  Φ (x i) fun h =>
    attend (fun j : Fin (i + 1) => F h i (x i) (x j) (i - j)) fun j => v h (x j)

/-- **The block at the query `i` reads only the tokens `j ≤ i`**: sequences that agree up to `i`
give one output, whatever `Φ`, the scores and the values. -/
theorem block_congr_of_le (Φ : X → (Fin H → E) → Y) (F : Fin H → ℕ → X → X → ℕ → ℝ)
    (v : Fin H → X → E) {x x' : ℕ → X} (i : ℕ) (hx : ∀ j ≤ i, x j = x' j) :
    block Φ F v x i = block Φ F v x' i := by
  unfold block
  rw [hx i le_rfl]
  congr 1
  funext h
  congr 1
  · funext j
    rw [hx j (Nat.lt_succ_iff.mp j.2)]
  · funext j
    rw [hx j (Nat.lt_succ_iff.mp j.2)]

/-- One head; the tokens `0, 1, 2, 3` agree, the later ones do not: the block at `3` is the same. -/
example : block (H := 1) (fun _ u => u 0) (fun _ _ _ _ _ => (0 : ℝ)) (fun _ y => y)
      (fun j : ℕ => (j : ℝ)) 3 =
    block (H := 1) (fun _ u => u 0) (fun _ _ _ _ _ => (0 : ℝ)) (fun _ y => y)
      (fun j : ℕ => if j ≤ 3 then (j : ℝ) else 0) 3 :=
  block_congr_of_le _ _ _ 3 (fun j hj => by simp [hj])

/-- **The logit of a complex head**: the gain `g` times `Re Z + β (Im Z)² / d`, where
`Re Z = qᵀk` and `(Im Z)² = (qᵀ J k)²` for the pairs `q_l = q(0,l) + i q(1,l)` and `k_l` of the
query and of the key. -/
noncomputable def phaseLogit (g β d : ℝ) (q k : Fin 2 × Fin p → ℝ) : ℝ :=
  g * (q ⬝ᵥ k + β * (q ⬝ᵥ rotateHalf k) ^ 2 / d)

/-- **The logit of a head of the main model with LeRoPE** for the query token `x` at the position
`i` and the key token `y` at the distance `δ`: SSMax's gain `ln(i + 1)/d` times the phase logit
of `Q x` and of `K y` turned back by `δ`. -/
noncomputable def leropeLogit (θ : Fin p → ℝ) (β d : ℝ) (Q K : X → Fin 2 × Fin p → ℝ) (i : ℕ)
    (x y : X) (δ : ℕ) : ℝ :=
  phaseLogit (Real.log ((i : ℝ) + 1) / d) β d (Q x) (rotary θ (-(δ : ℝ)) *ᵥ K y)

/-- **The logit of the same head without positional encoding**: the query `Q x` and the key `K y`
as they are, at every distance. -/
noncomputable def plainLogit (β d : ℝ) (Q K : X → Fin 2 × Fin p → ℝ) (i : ℕ) (x y : X) : ℝ :=
  phaseLogit (Real.log ((i : ℝ) + 1) / d) β d (Q x) (K y)

/-- A common rotation keeps the dot product. -/
theorem dotProduct_rotary_rotary (θ : Fin p → ℝ) (c : ℝ) (x y : Fin 2 × Fin p → ℝ) :
    (rotary θ c *ᵥ x) ⬝ᵥ (rotary θ c *ᵥ y) = x ⬝ᵥ y := by
  rw [mulVec_dotProduct, mulVec_mulVec, rotary_transpose_mul, sub_self, rotary_zero, one_mulVec]

/-- The multiplication by `i` commutes with the rotations: `J (R^c x) = R^c (J x)`. -/
theorem rotateHalf_rotary (θ : Fin p → ℝ) (c : ℝ) (x : Fin 2 × Fin p → ℝ) :
    rotateHalf (rotary θ c *ᵥ x) = rotary θ c *ᵥ rotateHalf x := by
  rw [rotary_mulVec, rotary_mulVec]
  funext ⟨r, l⟩
  fin_cases r <;> simp [rotateHalf]
  ring

/-- **A common rotation keeps the phase logit.** -/
theorem phaseLogit_rotary (θ : Fin p → ℝ) (g β d c : ℝ) (x y : Fin 2 × Fin p → ℝ) :
    phaseLogit g β d (rotary θ c *ᵥ x) (rotary θ c *ᵥ y) = phaseLogit g β d x y := by
  rw [phaseLogit, phaseLogit, rotateHalf_rotary, dotProduct_rotary_rotary,
    dotProduct_rotary_rotary]

/-- **`leropeLogit` is LeRoPE's**: at the distance `i - j` it is the phase logit of the query
turned by `i` and of the key turned by `j`, the rotations LeRoPE applies at the positions `i` and
`j`. -/
theorem leropeLogit_sub (θ : Fin p → ℝ) (β d : ℝ) (Q K : X → Fin 2 × Fin p → ℝ) (i : ℕ) {j : ℕ}
    (hj : j ≤ i) (x y : X) :
    leropeLogit θ β d Q K i x y (i - j) =
      phaseLogit (Real.log ((i : ℝ) + 1) / d) β d (rotary θ i *ᵥ Q x) (rotary θ j *ᵥ K y) := by
  have h : rotary θ (j : ℝ) *ᵥ K y =
      rotary θ (i : ℝ) *ᵥ (rotary θ (-((i - j : ℕ) : ℝ)) *ᵥ K y) := by
    rw [mulVec_mulVec, ← rotary_add]
    congr 2
    rw [Nat.cast_sub hj]
    ring
  rw [leropeLogit, h, phaseLogit_rotary]

/-- Sixteen channels, the query at `127` and the key at `100`. -/
example (θ : Fin 16 → ℝ) (Q K : ℕ → Fin 2 × Fin 16 → ℝ) (x y : ℕ) :
    leropeLogit θ (1 / 2) 32 Q K 127 x y (127 - 100) =
      phaseLogit (Real.log ((127 : ℕ) + 1) / 32) (1 / 2) 32 (rotary θ (127 : ℕ) *ᵥ Q x)
        (rotary θ (100 : ℕ) *ᵥ K y) :=
  leropeLogit_sub θ (1 / 2) 32 Q K 127 (by norm_num) x y

/-- No frequency turns nothing. -/
theorem rotary_zero_frequencies (m : ℝ) : rotary (0 : Fin p → ℝ) m = 1 := by
  simp only [rotary, Pi.zero_apply, mul_zero, rotation_zero]
  exact blockDiagonal_one

/-- **Without frequencies the head has no positional encoding**: `leropeLogit` at `θ = 0` is
`plainLogit`, at every distance. -/
theorem leropeLogit_zero (β d : ℝ) (Q K : X → Fin 2 × Fin p → ℝ) (i : ℕ) (x y : X) (δ : ℕ) :
    leropeLogit 0 β d Q K i x y δ = plainLogit β d Q K i x y := by
  rw [leropeLogit, plainLogit, rotary_zero_frequencies, one_mulVec]

example (β d : ℝ) (Q K : ℕ → Fin 2 × Fin 4 → ℝ) (x y δ : ℕ) :
    leropeLogit 0 β d Q K 5 x y δ = plainLogit β d Q K 5 x y :=
  leropeLogit_zero β d Q K 5 x y δ

end Transformer.LeRoPEBlock

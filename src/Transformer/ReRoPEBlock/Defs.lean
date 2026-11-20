module

public import Transformer.RoPE.ReRoPE.Section3_PhaseScore

/-!
# A block of DiffGPT's main model, with LeRoPE or with ReRoPE

Source: DiffGPT (`~/python_projects/gpt`, `5a37c51`), `research/main_model/README.txt`,
*Architecture*, and `research/main_model/model.json`. At the position `i` the block reads its
four streams into `z_i` and, per head of real width `d`,

    q_i = LeRoPE_i(R_d(W_Q z_i)),   k_i = LeRoPE_i(R_d(W_K z_i)),   v_i = W_V z_i,
    p_ij = softmax_{j ≤ i}((ln N_i / d)(Re Z_ij + β (Im Z_ij)² / d)),   Z_ij = ∑_c conj(q_ic) k_jc,

with `N_i = i + 1`. The heads' outputs `∑_j p_ij v_j` are concatenated and written into the
streams through `R_D(W_O ·)` by the phase-conjugated mHC; the FFN branch
`R_D(W_F (W_g z' ⊙_ℂ W_u z')/√2)` follows with its own routing. LeRoPE turns the channel `m` by
the angle `i θ_m`, `θ_m = 10000^{−2m/d} e^{α_m}`, the `α_m` learned and shared by Q, K, heads and
layers. ReRoPE (Su, kexue.fm/archives/9708, `Transformer.RoPE.ReRoPE`) scores the key at the
distance `i − j` as if it stood at `min(i − j, w)` (`rectifiedPosition`).

Everything but the attention acts on one position at a time: the routers and their Sinkhorn
maps, the RMS normalizations, `W_O`, the FFN and the writes. So at the query `i` the block is

    block ψ Φ F v x i = Φ (x i) (h ↦ attend_{j ≤ i} (F h i (x i) (x j) (ψ (i − j)), v h (x j)))

(`block`): a map `Φ` of the query's own token and of `H` softmax attentions, where the head `h`
scores the key `j` from the query's token, the key's token and the position `ψ (i − j)` that it
gives their distance. The token `x j` is everything the block reads at the position `j`, the
four streams in the main model. The two blocks differ only in `ψ`: LeRoPE's `ψ δ = δ`
(`lePosition`), ReRoPE's `ψ δ = min(δ, w)` (`rePosition`).

`headScore` is the main model's score of a key: SSMax's gain `ln(i + 1)/d` times the phase
score of `Q x` and of `K y` turned back by `δ`. By `headScore_sub` it is the score of `q` turned
by `i` and `k` turned by `j`, as LeRoPE turns them. With ReRoPE the frequencies may stay learned
(LeReRoPE, `Transformer.RoPE.LeReRoPE`) or be fixed; the theorems leave them arbitrary.
-/

@[expose] public section

open Matrix
open Transformer.AttentionResiduals Transformer.RoPE.ReRoPE Transformer.RoPE.RoFormer
open Transformer.RoPE (frequency)

namespace Transformer.ReRoPEBlock

variable {X E Y : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {H : ℕ}

/-- **One block at the query `i`**: the map `Φ` of the query's token `x i` and of `H` softmax
attentions over the keys `j ≤ i`. The head `h` scores the key `j` by
`F h i (x i) (x j) (ψ (i − j))` and reads the value `v h (x j)`. -/
noncomputable def block (ψ : ℕ → ℝ) (Φ : X → (Fin H → E) → Y)
    (F : Fin H → ℕ → X → X → ℝ → ℝ) (v : Fin H → X → E) (x : ℕ → X) (i : ℕ) : Y :=
  Φ (x i) fun h =>
    attend (fun j : Fin (i + 1) => F h i (x i) (x j) (ψ (i - j))) fun j => v h (x j)

/-- LeRoPE's position of the distance `δ`: the distance itself. -/
def lePosition (δ : ℕ) : ℝ := δ

/-- ReRoPE's position of the distance `δ`: `min(δ, w)` (`rectifiedPosition`). -/
noncomputable def rePosition (w δ : ℕ) : ℝ := rectifiedPosition w δ

/-- **The score of a head of the main model** for the query token `x` at the position `i` and
the key token `y` at the position `δ` from it: SSMax's gain `ln(i + 1)/d` times the phase score
`Re Z + β (Im Z)²/d` (`phaseScore`) of `Q x` and of `K y` turned back by `δ`. -/
noncomputable def headScore {p : ℕ} (θ : Fin p → ℝ) (β d : ℝ) (Q K : X → Fin 2 × Fin p → ℝ)
    (i : ℕ) (x y : X) (δ : ℝ) : ℝ :=
  phaseScore (Real.log ((i : ℝ) + 1) / d) β d (Q x) (rotary θ (-δ) *ᵥ K y)

/-- `headScore` at the distance `i − j` is the phase score of the query turned by `i` and the key
turned by `j`, the rotations LeRoPE applies at the positions `i` and `j`. -/
theorem headScore_sub {p : ℕ} (θ : Fin p → ℝ) (β d : ℝ) (Q K : X → Fin 2 × Fin p → ℝ)
    (i : ℕ) (j : ℝ) (x y : X) :
    headScore θ β d Q K i x y (i - j) =
      phaseScore (Real.log ((i : ℝ) + 1) / d) β d (rotary θ i *ᵥ Q x) (rotary θ j *ᵥ K y) := by
  rw [headScore, invariant_rotary_sub θ (phaseScore_rotary θ _ β d) (Q x) (K y) i (i - j),
    sub_sub_cancel]

/-- Sixteen frequencies of LeRoPE at `α = 0`, the query at `127` and the key at `100`. -/
example (Q K : ℕ → Fin 2 × Fin 16 → ℝ) (x y : ℕ) :
    headScore (fun l : Fin 16 => frequency 10000 32 l) (1 / 2) 32 Q K 127 x y (127 - 100) =
      phaseScore (Real.log ((127 : ℕ) + 1) / 32) (1 / 2) 32
        (rotary (fun l : Fin 16 => frequency 10000 32 l) (127 : ℕ) *ᵥ Q x)
        (rotary (fun l : Fin 16 => frequency 10000 32 l) 100 *ᵥ K y) := by
  exact_mod_cast headScore_sub _ (1 / 2) 32 Q K 127 100 x y

end Transformer.ReRoPEBlock

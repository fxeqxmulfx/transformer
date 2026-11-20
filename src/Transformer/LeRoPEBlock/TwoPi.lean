module

public import Transformer.LeRoPEBlock.Defs
public import Transformer.RoPE.LeRoPE.Section3_LearnedFrequencies

/-!
# LeRoPE's blocks include every rotary block with frequencies `θ ≥ 0`, exactly

Source: the blocks of `Transformer.LeRoPEBlock.Defs` (DiffGPT `5a37c51`) and LeRoPE, Karypis et
al., arXiv:2607.10134v1, §3.1, Eq. (5): the learned frequency `θ̂_m = e^{α_m} ω_m` is positive and
reaches every positive target (`Transformer.RoPE.LeRoPE.learned_frequency_surjective`). A block
reads its keys at natural distances `δ`, and the frequency `2π` turns a natural distance by a
multiple of `2π`: it leaves its plane unrotated, as the frequency `0` does. Partial RoPE leaves
planes unrotated, and `Transformer.MRCR.RotaryPartialHead` uses `2π` for the same purpose. So
LeRoPE's positive frequencies lose nothing to the frequency `0`, nor to any other `θ ≥ 0`.

* `rotary_replace_zero`: replacing the frequencies `0` by `2π` keeps the rotation by every natural
  distance;
* `exists_pos_block_eq`: for every frequencies `θ ≥ 0` (RoPE's grid, partial RoPE with its
  unrotated planes, no rotation at all) and every `Φ`, projections, `β`, `d` and values, there are
  positive frequencies whose block is the block with `θ` on every sequence and every query;
* `exists_lerope_eq_noPos`: in particular **the block with no positional encoding is a LeRoPE
  block**, on every input, for every `Φ`;
* `exists_alpha_block_eq`: the positive frequencies are LeRoPE's own, `e^{α_m} ω_m` for any positive
  grid `ω`.

`Transformer.LeRoPEBlock.Limit` shows the same up to closure by small frequencies, which is the
reading that survives a perturbation of the frequencies.
-/

@[expose] public section

open Matrix Transformer.RoPE.RoFormer

namespace Transformer.LeRoPEBlock

variable {X E Y : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {H p : ℕ}

/-- **Blocks only read the scores**: equal scores at every position, token and distance give the
same block. -/
theorem block_congr_scores {Φ : X → (Fin H → E) → Y} {F F' : Fin H → ℕ → X → X → ℕ → ℝ}
    (hF : ∀ h i a b δ, F h i a b δ = F' h i a b δ) (v : Fin H → X → E) (x : ℕ → X) (i : ℕ) :
    block Φ F v x i = block Φ F' v x i := by
  unfold block
  congr 1
  funext h
  congr 1
  funext j
  exact hF _ _ _ _ _

/-- A head at no frequencies and the plain head: scores that agree at every distance
(`leropeLogit_zero`) give the same block. -/
example (Φ : ℝ → (Fin 1 → ℝ) → ℝ) (Q K : Fin 1 → ℝ → Fin 2 × Fin 1 → ℝ) (v : Fin 1 → ℝ → ℝ)
    (x : ℕ → ℝ) (i : ℕ) :
    block Φ (fun h => leropeLogit 0 (1 / 2) 2 (Q h) (K h)) v x i =
      block Φ (fun h _ a b _ => plainLogit (1 / 2) 2 (Q h) (K h) i a b) v x i :=
  block_congr_scores (fun _ _ _ _ _ => leropeLogit_zero _ _ _ _ _ _ _ _) v x i

/-- A plane at the frequency `2π` is unrotated at a natural distance. -/
theorem rotation_neg_nat_mul_two_pi (δ : ℕ) : rotation (-(δ : ℝ) * (2 * Real.pi)) = 1 := by
  have hc : Real.cos (-(δ : ℝ) * (2 * Real.pi)) = 1 := by
    rw [neg_mul, Real.cos_neg, Real.cos_nat_mul_two_pi]
  have hs : Real.sin (-(δ : ℝ) * (2 * Real.pi)) = 0 := by
    rw [neg_mul, Real.sin_neg, show (δ : ℝ) * (2 * Real.pi) = ((2 * δ : ℕ) : ℝ) * Real.pi by
      push_cast
      ring, Real.sin_nat_mul_pi, neg_zero]
  unfold rotation
  rw [hc, hs]
  ext i j
  fin_cases i <;> fin_cases j <;> simp

/-- **Frequency `2π` for frequency `0`**: replacing the frequencies `0` by `2π` keeps the rotation
by every natural distance. -/
theorem rotary_replace_zero (θ : Fin p → ℝ) (δ : ℕ) :
    rotary (fun l => if θ l = 0 then 2 * Real.pi else θ l) (-(δ : ℝ)) = rotary θ (-(δ : ℝ)) := by
  unfold rotary
  congr 1
  funext l
  show rotation (-(δ : ℝ) * (if θ l = 0 then 2 * Real.pi else θ l)) = rotation (-(δ : ℝ) * θ l)
  split_ifs with h
  · rw [h, mul_zero, rotation_zero, rotation_neg_nat_mul_two_pi]
  · rfl

/-- The grid `(0, 1/2)`: its second plane keeps its frequency, the first goes to `2π`. -/
example (δ : ℕ) :
    rotary (fun l => if (![0, 1 / 2] : Fin 2 → ℝ) l = 0 then 2 * Real.pi else ![0, 1 / 2] l)
        (-(δ : ℝ)) = rotary (![0, 1 / 2] : Fin 2 → ℝ) (-(δ : ℝ)) :=
  rotary_replace_zero _ δ

/-- **Positive frequencies with the rotations of any `θ ≥ 0`** at every natural distance. -/
theorem exists_pos_frequencies (θ : Fin p → ℝ) (hθ : ∀ l, 0 ≤ θ l) :
    ∃ θ' : Fin p → ℝ, (∀ l, 0 < θ' l) ∧ ∀ δ : ℕ, rotary θ' (-(δ : ℝ)) = rotary θ (-(δ : ℝ)) := by
  refine ⟨fun l => if θ l = 0 then 2 * Real.pi else θ l, fun l => ?_, rotary_replace_zero θ⟩
  by_cases h : θ l = 0
  · simp [h, Real.pi_pos]
  · simp [h, lt_of_le_of_ne (hθ l) (Ne.symm h)]

example : ∃ θ' : Fin 2 → ℝ, (∀ l, 0 < θ' l) ∧
    ∀ δ : ℕ, rotary θ' (-(δ : ℝ)) = rotary (![0, 1 / 2] : Fin 2 → ℝ) (-(δ : ℝ)) :=
  exists_pos_frequencies _ fun l => by fin_cases l <;> norm_num

/-- **Every rotary block with frequencies `θ ≥ 0` is a LeRoPE block**, on every sequence and every
query: for any `Φ`, projections `Q h`, `K h`, coordinates `β h`, width `d` and values `v`, positive
frequencies give the same block as `θ`. -/
theorem exists_pos_block_eq (Φ : X → (Fin H → E) → Y) (β : Fin H → ℝ) (d : ℝ)
    (Q K : Fin H → X → Fin 2 × Fin p → ℝ) (v : Fin H → X → E) {θ : Fin p → ℝ}
    (hθ : ∀ l, 0 ≤ θ l) :
    ∃ θ' : Fin p → ℝ, (∀ l, 0 < θ' l) ∧ ∀ (x : ℕ → X) (i : ℕ),
      block Φ (fun h => leropeLogit θ' (β h) d (Q h) (K h)) v x i =
        block Φ (fun h => leropeLogit θ (β h) d (Q h) (K h)) v x i := by
  obtain ⟨θ', hpos, hrot⟩ := exists_pos_frequencies θ hθ
  exact ⟨θ', hpos, fun x i =>
    block_congr_scores (fun h i a b δ => by simp only [leropeLogit, hrot δ]) v x i⟩

/-- Partial RoPE on two planes, the first unrotated and the second at `1/2`: its block is a
LeRoPE block, whatever `Φ`, the projections and the values are. -/
example (Φ : ℝ → (Fin 1 → ℝ) → ℝ) (Q K : Fin 1 → ℝ → Fin 2 × Fin 2 → ℝ) (v : Fin 1 → ℝ → ℝ) :=
  exists_pos_block_eq Φ (fun _ => 1 / 2) 4 Q K v (θ := ![0, 1 / 2]) fun l => by
    fin_cases l <;> norm_num

/-- **A block with no positional encoding is a LeRoPE block**, exactly: for any `Φ`, projections,
coordinates `β`, width and values there are positive frequencies whose LeRoPE block is the block
with the plain head, on every sequence and every query. -/
theorem exists_lerope_eq_noPos (Φ : X → (Fin H → E) → Y) (β : Fin H → ℝ) (d : ℝ)
    (Q K : Fin H → X → Fin 2 × Fin p → ℝ) (v : Fin H → X → E) :
    ∃ θ' : Fin p → ℝ, (∀ l, 0 < θ' l) ∧ ∀ (x : ℕ → X) (i : ℕ),
      block Φ (fun h => leropeLogit θ' (β h) d (Q h) (K h)) v x i =
        block Φ (fun h _ a b _ => plainLogit (β h) d (Q h) (K h) i a b) v x i := by
  obtain ⟨θ', hpos, h⟩ := exists_pos_block_eq Φ β d Q K v (θ := (0 : Fin p → ℝ)) fun _ => le_rfl
  exact ⟨θ', hpos, fun x i => (h x i).trans
    (block_congr_scores (fun h i a b δ => leropeLogit_zero _ _ _ _ _ _ _ _) v x i)⟩

/-- One head and one plane, on sequences of reals read as values. -/
example :=
  exists_lerope_eq_noPos (X := ℝ) (E := ℝ) (Y := ℝ) (H := 1) (p := 1) (fun _ a => a 0)
    (fun _ => 1 / 2) 2 (fun _ _ => fun _ => 1) (fun _ _ => fun _ => 1) (fun _ y => y)

/-- **LeRoPE's own parametrization reaches them**: for any positive grid `ω`, as RoPE's, there are
scales `α` with the block of the frequencies `e^{α_m} ω_m` equal to the block with any `θ ≥ 0`. -/
theorem exists_alpha_block_eq (Φ : X → (Fin H → E) → Y) (β : Fin H → ℝ) (d : ℝ)
    (Q K : Fin H → X → Fin 2 × Fin p → ℝ) (v : Fin H → X → E) {ω θ : Fin p → ℝ}
    (hω : ∀ l, 0 < ω l) (hθ : ∀ l, 0 ≤ θ l) :
    ∃ α : Fin p → ℝ, ∀ (x : ℕ → X) (i : ℕ),
      block Φ (fun h => leropeLogit (fun l => RoPE.LeRoPE.learnedFrequency (ω l) (α l)) (β h) d
          (Q h) (K h)) v x i =
        block Φ (fun h => leropeLogit θ (β h) d (Q h) (K h)) v x i := by
  obtain ⟨θ', hpos, h⟩ := exists_pos_block_eq Φ β d Q K v hθ
  choose α hα using fun l => RoPE.LeRoPE.learned_frequency_surjective (hω l) (hpos l)
  have hfun : (fun l => RoPE.LeRoPE.learnedFrequency (ω l) (α l)) = θ' := funext hα
  exact ⟨α, fun x i => by rw [hfun]; exact h x i⟩

/-- RoPE's grid `10000^{−2l/4}` on two planes, one plane unrotated, as in partial RoPE. -/
example :=
  exists_alpha_block_eq (X := ℝ) (E := ℝ) (Y := ℝ) (H := 1) (p := 2) (fun _ a => a 0)
    (fun _ => 1 / 2) 4 (fun _ _ => fun _ => 1) (fun _ _ => fun _ => 1) (fun _ y => y)
    (ω := fun l => (10000 : ℝ) ^ (-(2 * (l : ℝ)) / 4)) (θ := ![0, 1 / 2])
    (fun _ => Real.rpow_pos_of_pos (by norm_num) _) fun l => by fin_cases l <;> norm_num

end Transformer.LeRoPEBlock

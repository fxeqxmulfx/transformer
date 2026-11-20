module

public import Transformer.LeRoPEBlock.Defs
public import Transformer.MHC.PeriRMS.PhaseScoreHead

/-!
# The logits of the main model's heads are at most `(5/4) ln N`

`leropeLogit` is SSMax's gain `ln(i + 1)/d` times `Re Z + β (Im Z)² / d` of the query `Q x` and of
the key `K y` turned back by the distance. With `N = i + 1` visible keys:

* `leropeLogit_abs_le`: `|logit| ≤ (5/4) ln (i + 1)` for every distance, every frequency vector and
  every `|β| ≤ 1`, whenever `Q x` and `K y` have squared length at most the real width `d`, which
  the per-head parameter-free Q/K RMS gives. The rotation by the distance is orthogonal, so it
  keeps the length of the key; the bound is that of `Transformer.MHC.PeriRMS.phaseScore_abs_le`.
* `plainLogit_abs_le`: the same for the head without positional encoding.

The constant `5/4 = max_{x² + y² ≤ 1} |x| + y²` replaces the `1` of the real dot product; it
bounds the logit, not its derivative. Source: DiffGPT (`~/python_projects/gpt`, `fe1d528`),
`research/main_model/README.txt`, *Architecture* and the score paragraph.
-/

@[expose] public section

open Matrix
open Transformer.RoPE.RoFormer Transformer.MHC.PeriRMS

namespace Transformer.LeRoPEBlock

variable {X : Type*} {p : ℕ}

/-- `phaseLogit` is the phase score of the ReRoPE development, with the same formula. -/
theorem phaseLogit_eq_phaseScore (g β d : ℝ) (q k : Fin 2 × Fin p → ℝ) :
    phaseLogit g β d q k = Transformer.RoPE.ReRoPE.phaseScore g β d q k :=
  rfl

/-- **A head of the main model with LeRoPE has logit at most `(5/4) ln (i + 1)`** for the query at
the position `i`, any key at any distance and any frequencies, when `Q x` and `K y` have squared
length at most `d` and `|β| ≤ 1`. -/
theorem leropeLogit_abs_le (θ : Fin p → ℝ) (β d : ℝ) (hd : 0 < d) (hβ : |β| ≤ 1)
    (Q K : X → Fin 2 × Fin p → ℝ) (i : ℕ) (x y : X) (δ : ℕ)
    (hQ : Q x ⬝ᵥ Q x ≤ d) (hK : K y ⬝ᵥ K y ≤ d) :
    |leropeLogit θ β d Q K i x y δ| ≤ 5 / 4 * Real.log ((i : ℝ) + 1) := by
  have hN : (1 : ℝ) ≤ (i : ℝ) + 1 := le_add_of_nonneg_left (Nat.cast_nonneg i)
  exact phaseScore_abs_le ((i : ℝ) + 1) β d hN hd hβ (Q x) (rotary θ (-(δ : ℝ)) *ᵥ K y) hQ
    (by rwa [dotProduct_rotary_rotary])

/-- **Without positional encoding** the same head has logit at most `(5/4) ln (i + 1)`. -/
theorem plainLogit_abs_le (β d : ℝ) (hd : 0 < d) (hβ : |β| ≤ 1)
    (Q K : X → Fin 2 × Fin p → ℝ) (i : ℕ) (x y : X)
    (hQ : Q x ⬝ᵥ Q x ≤ d) (hK : K y ⬝ᵥ K y ≤ d) :
    |plainLogit β d Q K i x y| ≤ 5 / 4 * Real.log ((i : ℝ) + 1) := by
  rw [← leropeLogit_zero β d Q K i x y 0]
  exact leropeLogit_abs_le 0 β d hd hβ Q K i x y 0 hQ hK

/-- The main model's head, `d = 32`, sixteen channels, `β = 1/2`, the query at `127` and any key at
any distance, with the unit vectors `e₀` and `e₁` of the first pair. -/
example (θ : Fin 16 → ℝ) (δ : ℕ) :
    |leropeLogit θ (1 / 2) 32
      (fun _ : Unit => fun cl : Fin 2 × Fin 16 => if cl = (0, 0) then (1 : ℝ) else 0)
      (fun _ : Unit => fun cl : Fin 2 × Fin 16 => if cl = (1, 0) then (1 : ℝ) else 0)
      127 () () δ| ≤ 5 / 4 * Real.log (((127 : ℕ) : ℝ) + 1) :=
  leropeLogit_abs_le θ (1 / 2) 32 (by norm_num) (by norm_num) _ _ 127 () () δ
    (by simp [dotProduct])
    (by simp [dotProduct])

end Transformer.LeRoPEBlock

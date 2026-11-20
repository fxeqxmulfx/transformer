module

public import Transformer.MRCR.PhraseKeyReRope
public import Transformer.RoPE.ReRoPE.Section3_Computation

/-!
# ReRoPE's own formulas behind the clamped logit

Source: Su, *Transformer升级之路：12、无限外推的ReRoPE？*, kexue.fm/archives/9708
(`papers/kexue-9708-rerope`), formalized in `Transformer.RoPE.ReRoPE`.
`Transformer.MRCR.reropeLogit` adds to the dot product the RoPE score at the distance
`min (i − j) W`. This file derives that logit from the source's formulas.

* `reropeLogit_eq_rectifiedPosition`: the clamped distance is the position of Su's matrix
  (rerope), `rectifiedPosition W (i − j) = min(i − j, W)`.
* `dotProduct_rotary_pair`: the sine coefficient `B` of one frequency is the score of the
  constant pair `q = e_{(0,m)}`, `k = B e_{(1,m)}`: `qᵀ R^{−t} k = B sin(θ_m t)`.
* `reropeLogit_eq_merge`: the logit is the dot product plus Su's merged score of that pair,
  the first matrix `(R^i q)ᵀ(R^j k)` inside the window and the second `(R^W q)ᵀ k` outside it
  (`Transformer.RoPE.ReRoPE.merge_rectified`).

One deviation from the source remains. The dot product `x` is not rotated: the query and key
carry their content in coordinates that no rotation touches, as in partial RoPE. In Su's
ReRoPE every plane rotates.
-/

@[expose] public section

namespace Transformer.MRCR

open Matrix Transformer.RoPE.RoFormer
open Transformer.RoPE.ReRoPE (rectifiedPosition merge_rectified)

variable {K : ℕ}

/-- **The clamped distance is Su's**: `reropeLogit` takes the RoPE score at the position of the
matrix (rerope) of kexue.fm/archives/9708, `rectifiedPosition W (i − j) = min(i − j, W)`. -/
theorem reropeLogit_eq_rectifiedPosition (c : ℝ) (θ a b : Fin K → ℝ) (W i j : ℕ) (x : ℝ) :
    reropeLogit c θ a b W i j x =
      x / c + ropeScore θ a b (rectifiedPosition W ((i - j : ℕ) : ℝ)) := by
  rw [reropeLogit, rectifiedPosition, Nat.cast_min]

/-- **One sine is the score of a constant pair**: `e_{(0,m)}ᵀ R^{−t} (B e_{(1,m)}) = B sin(θ_m t)`,
with RoFormer's rotation `R^t` (arXiv:2104.09864v5, Eq. (15)). -/
theorem dotProduct_rotary_pair (θ : Fin K → ℝ) (m : Fin K) (B t : ℝ) :
    Pi.single (0, m) 1 ⬝ᵥ (rotary θ (-t) *ᵥ Pi.single (1, m) B) = B * Real.sin (θ m * t) := by
  simp [rotary, blockDiagonal_apply, Transformer.RoPE.RoFormer.rotation, mul_comm]

/-- **The logit is the dot product plus Su's merged score**: for `j ≤ i`, `reropeLogit` with the
sine coefficient `B` at the frequency `θ m` is `x / c` plus the merged attention score of
kexue.fm/archives/9708, 计算 (Computation), of the pair `q = e_{(0,m)}`, `k = B e_{(1,m)}`:
`(R^i q)ᵀ(R^j k)` for `i − j < W` and `(R^W q)ᵀ k` otherwise. -/
theorem reropeLogit_eq_merge (c : ℝ) (θ : Fin K → ℝ) (m : Fin K) (B : ℝ) (W : ℕ) {i j : ℕ}
    (hji : j ≤ i) (x : ℝ) :
    reropeLogit c θ 0 (Pi.single m B) W i j x = x / c +
      (if (i : ℝ) - j < W then
          (rotary θ i *ᵥ Pi.single (0, m) 1) ⬝ᵥ (rotary θ j *ᵥ Pi.single (1, m) B)
        else (rotary θ W *ᵥ Pi.single (0, m) 1) ⬝ᵥ Pi.single (1, m) B) := by
  rw [reropeLogit_eq_rectifiedPosition, merge_rectified,
    dotProduct_rotary_pair, ropeScore_single, Nat.cast_sub hji]

/-- The hypothesis of `reropeLogit_eq_merge` holds at the positions `7 ≥ 2`, the window `512`. -/
example := reropeLogit_eq_merge 1 (fun _ : Fin 1 => Real.pi / 1024) 0 (-3) 512
  (i := 7) (j := 2) (by norm_num) 0

end Transformer.MRCR

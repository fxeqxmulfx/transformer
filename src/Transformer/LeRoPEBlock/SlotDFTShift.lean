module

public import Transformer.LeRoPEBlock.SlotDFT

/-!
# The DFT head turned by `t`

Source: `review/lerope_block.md`, *Reading a slot* (added), RS3.

RS2's head turns its key by `R`, so it singles out the middle of `N = 2R + 1` keys. Turn it by
any `t ≤ 2R` instead. The distance `δ` then scores `ln N / (2R)` times `2 ∑_l cos((t − δ) θ_l)`,
and `t − δ` ranges over `[−2R, 2R]`, past the window `[−R, R]` of `sum_cos_dft`. The cosines of
the discrete Fourier transform repeat with the period `N`, so they still sum to `−1/2` at every
`0 < |t − δ| ≤ 2R` (`sum_cos_dft_of_abs_le`): the distance `t` scores `ln N` and the other `2R`
distances `−ln N / (2R)` (`leropeLogit_dft_shift`), and the key at the distance `t` keeps the
weight of RS2's middle key, `1 / (1 + 2R N^{−1−1/(2R)})` (`depthWeight_dft_shift`), next to the
query too.
-/

@[expose] public section

open Matrix Finset
open Transformer.AttentionResiduals Transformer.RoPE.RoFormer

namespace Transformer.LeRoPEBlock

/-- **The cosines of the discrete Fourier transform repeat with the period `2R + 1`**, so they sum
to `−1/2` at `x = 2π r / (2R + 1)` for every `0 < |r| ≤ 2R`. -/
theorem sum_cos_dft_of_abs_le {R : ℕ} {r : ℤ} (hr : r ≠ 0) (hrR : |r| ≤ 2 * R) :
    ∑ m ∈ range R, Real.cos ((m + 1) * (2 * Real.pi * r / (2 * R + 1))) = -1 / 2 := by
  rcases le_or_gt |r| R with h | h
  · exact sum_cos_dft hr h
  have hlo := (abs_le.1 hrR).1
  have hhi := (abs_le.1 hrR).2
  -- shift `r` by `c (2R + 1)` into `[−R, R]`
  obtain ⟨c, hc0, hcR⟩ : ∃ c : ℤ, r - c * (2 * R + 1) ≠ 0 ∧ |r - c * (2 * R + 1)| ≤ R := by
    rcases lt_abs.1 h with h | h
    · exact ⟨1, by omega, abs_le.2 ⟨by omega, by omega⟩⟩
    · exact ⟨-1, by omega, abs_le.2 ⟨by omega, by omega⟩⟩
  rw [← sum_cos_dft hc0 hcR]
  refine sum_congr rfl fun m _ => ?_
  have hN : (2 * (R : ℝ) + 1) ≠ 0 := by positivity
  rw [show ((m : ℝ) + 1) * (2 * Real.pi * r / (2 * R + 1)) =
      ((m : ℝ) + 1) * (2 * Real.pi * ((r - c * (2 * R + 1) : ℤ) : ℝ) / (2 * R + 1)) +
        ((((m + 1 : ℕ) : ℤ) * c : ℤ) : ℝ) * (2 * Real.pi) by push_cast; field_simp; ring,
    Real.cos_add_int_mul_two_pi]

/-- One band on three points: `cos(4π/3) = −1/2`. -/
example := sum_cos_dft_of_abs_le (R := 1) (r := 2) two_ne_zero (by norm_num)

/-- **The DFT head turned by `t`**: with the query `1`, the key `1` turned by `t ≤ 2R`, `β = 0` and
`d = 2R`, the head at the position `2R` scores the distance `t` by `ln N` and every other distance
by `−ln N / (2R)`, `N = 2R + 1`, whatever the tokens. -/
theorem leropeLogit_dft_shift {X : Type*} {R t : ℕ} (hR : 1 ≤ R) (ht : t ≤ 2 * R) (x y : X)
    {δ : ℕ} (hδ : δ ≤ 2 * R) :
    leropeLogit (dftFreq R) 0 (2 * R) (fun _ => 1) (fun _ => rotary (dftFreq R) t *ᵥ 1) (2 * R)
      x y δ = if δ = t then Real.log (2 * R + 1) else -(Real.log (2 * R + 1) / (2 * R)) := by
  have hR0 : (0 : ℝ) < R := Nat.cast_pos.2 hR
  have hm : -(δ : ℝ) + t = ((t - δ : ℤ) : ℝ) := by push_cast; ring
  simp only [leropeLogit, phaseLogit, zero_mul, zero_div, add_zero, mulVec_mulVec, ← rotary_add,
    hm, dotProduct_rotary_dft, Nat.cast_mul, Nat.cast_ofNat]
  split_ifs with h
  · simp only [h, sub_self, Int.cast_zero, mul_zero, zero_div, Real.cos_zero, sum_const,
      card_range, nsmul_eq_mul, mul_one]
    field_simp
  · rw [sum_cos_dft_of_abs_le (by omega) (abs_le.2 ⟨by omega, by omega⟩)]
    ring

/-- Three keys, the key turned by `1`: the head at the position `2` scores the distance `1` by
`ln 3`. -/
example := leropeLogit_dft_shift (X := Unit) (R := 1) (t := 1) le_rfl (by norm_num) () ()
  (δ := 1) (by norm_num)

/-- **The DFT head turned by `t` keeps `1 / (1 + 2R N^{−1−1/(2R)})` of its weight on the key at
the distance `t`**, `N = 2R + 1`, for every `t ≤ 2R` and whatever the tokens: as much as RS2's
head keeps on the middle key (`depthWeight_dft`). -/
theorem depthWeight_dft_shift {X : Type*} {R t : ℕ} (hR : 1 ≤ R) (ht : t ≤ 2 * R) (x : ℕ → X) :
    depthWeight (fun j : Fin (2 * R + 1) => leropeLogit (dftFreq R) 0 (2 * R) (fun _ => 1)
      (fun _ => rotary (dftFreq R) t *ᵥ 1) (2 * R) (x (2 * R)) (x j) (2 * R - j))
      ⟨2 * R - t, by omega⟩ =
        1 / (1 + 2 * R * (2 * (R : ℝ) + 1) ^ (-(1 + 1 / (2 * (R : ℝ))))) := by
  have hR0 : (0 : ℝ) < R := Nat.cast_pos.2 hR
  set L := Real.log (2 * (R : ℝ) + 1)
  set F : ℕ → ℝ := fun δ => if δ = t then L else -(L / (2 * R))
  have hFt : F t = L := ite_eq_left rfl
  have hFo (δ : ℕ) (h : δ ≠ t) : F δ = -(L / (2 * R)) := ite_eq_right h
  have hs (j : Fin (2 * R + 1)) : leropeLogit (dftFreq R) 0 (2 * R) (fun _ => 1)
      (fun _ => rotary (dftFreq R) t *ᵥ 1) (2 * R) (x (2 * R)) (x j) (2 * R - j) = F (2 * R - j) :=
    leropeLogit_dft_shift hR ht _ _ (Nat.sub_le _ _)
  have hZ : ∑ δ ∈ range (2 * R + 1), Real.exp (F δ) =
      Real.exp L + 2 * R * Real.exp (-(L / (2 * R))) := by
    rw [← add_sum_erase _ _ (mem_range.2 (by omega : t < 2 * R + 1)),
      sum_congr rfl fun δ hδ => congrArg Real.exp (hFo δ (ne_of_mem_erase hδ)), sum_const,
      card_erase_of_mem (mem_range.2 (by omega)), card_range, Nat.add_sub_cancel, nsmul_eq_mul,
      hFt]
    push_cast
    ring
  have hnum : leropeLogit (dftFreq R) 0 (2 * R) (fun _ => 1) (fun _ => rotary (dftFreq R) t *ᵥ 1)
      (2 * R) (x (2 * R)) (x (2 * R - t)) t = L := by
    rw [leropeLogit_dft_shift hR ht _ _ ht, ite_eq_left rfl]
  simp only [depthWeight, depthPartition, hs, sum_fin_exp_sub, hZ, Nat.sub_sub_self ht, hnum]
  rw [Real.rpow_def_of_pos (by positivity),
    show Real.log (2 * (R : ℝ) + 1) * -(1 + 1 / (2 * R)) = -(L / (2 * R)) - L by ring,
    Real.exp_sub]
  field_simp

/-- Three keys: the key next to the query keeps `1 / (1 + 2 · 3^{−3/2})`, as the middle one. -/
example := depthWeight_dft_shift (X := Unit) (R := 1) (t := 1) le_rfl (by norm_num) fun _ => ()

end Transformer.LeRoPEBlock

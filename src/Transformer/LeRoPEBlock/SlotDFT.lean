module

public import Transformer.LeRoPEBlock.PiHead
public import Transformer.LeRoPEBlock.SlotWeight

/-!
# A LeRoPE head that keeps more than half of its weight on the middle key

Source: `review/lerope_block.md`, *Reading a slot* (added), RS2.

`SlotWeight` caps the weight a head of the main model keeps on a key whose neighbours carry its
token. The cap is nearly reached. Give the head `R` bands at the frequencies of the discrete
Fourier transform on `2R + 1` points, `θ_l = 2π (l + 1) / (2R + 1)` (`dftFreq`), the query `1` on
every channel and the key `1` turned by `R`, with `β = 0` and `d = 2R`. At the distance `δ` the
score is `ln N / (2R)` times `2 ∑_l cos((R − δ) θ_l)`, `N = 2R + 1`. Dirichlet's kernel
(`two_mul_sin_mul_sum_cos`) sums these cosines to `−1/2` at every distance but `R`
(`sum_cos_dft`), so the middle key scores `ln N` and the other `2R` keys `−ln N / (2R)`
(`leropeLogit_dft`). The middle key keeps `1 / (1 + 2R N^{−1−1/(2R)})` of the weight
(`depthWeight_dft`), more than half for every `R` (`half_lt_depthWeight_dft`), while
`depthWeight_leropeLogit_le` allows at most `1 / (1 + R N^{−1−1/R})` to every head
(`depthWeight_dft_mem`). At `R = 16`, the main model's number of bands, the two are `0.535` and
`0.72`; as `R` grows they tend to `1/2` and `2/3`.
-/

@[expose] public section

open Matrix Finset
open Transformer.AttentionResiduals Transformer.RoPE.RoFormer

namespace Transformer.LeRoPEBlock

/-- **Dirichlet's kernel**: `2 sin(x/2) ∑_{m<R} cos((m + 1) x) = sin((R + 1/2) x) − sin(x/2)`. -/
theorem two_mul_sin_mul_sum_cos (x : ℝ) (R : ℕ) :
    2 * Real.sin (x / 2) * ∑ m ∈ range R, Real.cos ((m + 1) * x) =
      Real.sin ((R + 1 / 2) * x) - Real.sin (x / 2) := by
  induction R with
  | zero => simp [div_eq_inv_mul]
  | succ R ih =>
    have h1 : x / 2 - ((R : ℝ) + 1) * x = -(((R : ℝ) + 1 / 2) * x) := by ring
    have h2 : x / 2 + ((R : ℝ) + 1) * x = (((R + 1 : ℕ) : ℝ) + 1 / 2) * x := by push_cast; ring
    rw [sum_range_succ, mul_add, ih, Real.two_mul_sin_mul_cos, h1, h2, Real.sin_neg]
    ring

/-- **The cosines of the discrete Fourier transform sum to `−1/2`**: at `x = 2π r / (2R + 1)` with
`0 < |r| ≤ R`, `∑_{m<R} cos((m + 1) x) = −1/2`. -/
theorem sum_cos_dft {R : ℕ} {r : ℤ} (hr : r ≠ 0) (hrR : |r| ≤ R) :
    ∑ m ∈ range R, Real.cos ((m + 1) * (2 * Real.pi * r / (2 * R + 1))) = -1 / 2 := by
  have hN : (0 : ℝ) < 2 * R + 1 := by positivity
  have hR0 : (0 : ℝ) ≤ R := Nat.cast_nonneg R
  have hlo : -(R : ℝ) ≤ r := by exact_mod_cast (abs_le.1 hrR).1
  have hhi : (r : ℝ) ≤ R := by exact_mod_cast (abs_le.1 hrR).2
  have hD := two_mul_sin_mul_sum_cos (2 * Real.pi * r / (2 * R + 1)) R
  have h1 : ((R : ℝ) + 1 / 2) * (2 * Real.pi * r / (2 * R + 1)) = r * Real.pi := by
    rw [mul_div_assoc', div_eq_iff hN.ne']
    ring
  have h2 : 2 * Real.pi * r / (2 * R + 1) / 2 = Real.pi * r / (2 * R + 1) := by ring
  rw [h1, Real.sin_int_mul_pi, zero_sub, h2] at hD
  have hs : Real.sin (Real.pi * r / (2 * R + 1)) ≠ 0 := by
    rw [Ne, Real.sin_eq_zero_iff_of_lt_of_lt]
    · exact div_ne_zero (mul_ne_zero Real.pi_ne_zero (Int.cast_ne_zero.2 hr)) hN.ne'
    · rw [lt_div_iff₀ hN]
      nlinarith [mul_pos Real.pi_pos (show (0 : ℝ) < r + 2 * R + 1 by linarith)]
    · rw [div_lt_iff₀ hN]
      nlinarith [mul_pos Real.pi_pos (show (0 : ℝ) < 2 * R + 1 - r by linarith)]
  have h3 : Real.sin (Real.pi * r / (2 * R + 1)) *
      (2 * ∑ m ∈ range R, Real.cos ((m + 1) * (2 * Real.pi * r / (2 * R + 1))) + 1) = 0 := by
    linear_combination hD
  rcases mul_eq_zero.1 h3 with h | h
  · exact absurd h hs
  · linarith

/-- One band on three points: `cos(2π/3) = −1/2`. -/
example := sum_cos_dft (R := 1) (r := 1) one_ne_zero (by norm_num)

/-- **The frequencies of the discrete Fourier transform** on `2R + 1` points, one per band:
`θ_l = 2π (l + 1) / (2R + 1)`. -/
noncomputable def dftFreq (R : ℕ) (l : Fin R) : ℝ := 2 * Real.pi * ((l : ℝ) + 1) / (2 * R + 1)

/-- The query `1` against itself turned by `m`, at the DFT frequencies. -/
theorem dotProduct_rotary_dft (R : ℕ) (m : ℝ) :
    (1 : Fin 2 × Fin R → ℝ) ⬝ᵥ (rotary (dftFreq R) m *ᵥ 1) =
      2 * ∑ l ∈ range R, Real.cos ((l + 1) * (2 * Real.pi * m / (2 * R + 1))) := by
  rw [dotProduct_rotary_eq_sum_cos, mul_sum, ← Fin.sum_univ_eq_sum_range
    (fun l => 2 * Real.cos ((l + 1) * (2 * Real.pi * m / (2 * R + 1))))]
  refine sum_congr rfl fun l _ => ?_
  rw [dftFreq, show m * (2 * Real.pi * ((l : ℝ) + 1) / (2 * R + 1)) =
    ((l : ℝ) + 1) * (2 * Real.pi * m / (2 * R + 1)) by ring]
  norm_num

/-- **The DFT head's logit**: with the query `1`, the key `1` turned by `R`, `β = 0` and `d = 2R`,
the head at the position `2R` scores the distance `R` by `ln N` and every other distance by
`−ln N / (2R)`, `N = 2R + 1`, whatever the tokens. -/
theorem leropeLogit_dft {X : Type*} {R : ℕ} (hR : 1 ≤ R) (x y : X) {δ : ℕ} (hδ : δ ≤ 2 * R) :
    leropeLogit (dftFreq R) 0 (2 * R) (fun _ => 1) (fun _ => rotary (dftFreq R) R *ᵥ 1) (2 * R)
      x y δ = if δ = R then Real.log (2 * R + 1) else -(Real.log (2 * R + 1) / (2 * R)) := by
  have hR0 : (0 : ℝ) < R := Nat.cast_pos.2 hR
  have hm : -(δ : ℝ) + R = ((R - δ : ℤ) : ℝ) := by push_cast; ring
  simp only [leropeLogit, phaseLogit, zero_mul, zero_div, add_zero, mulVec_mulVec, ← rotary_add,
    hm, dotProduct_rotary_dft, Nat.cast_mul, Nat.cast_ofNat]
  split_ifs with h
  · simp only [h, sub_self, Int.cast_zero, mul_zero, zero_div, Real.cos_zero, sum_const,
      card_range, nsmul_eq_mul, mul_one]
    field_simp
  · rw [sum_cos_dft (by omega) (abs_le.2 ⟨by omega, by omega⟩)]
    ring

/-- Three keys: the head at the position `2` scores the distance `0` by `−ln 3 / 2`. -/
example := leropeLogit_dft (X := Unit) (R := 1) le_rfl () () (δ := 0) (by norm_num)

/-- **The DFT head keeps `1 / (1 + 2R N^{−1−1/(2R)})` of its weight on the middle key** of
`N = 2R + 1`, whatever the tokens (`review/lerope_block.md`, RS2). -/
theorem depthWeight_dft {X : Type*} {R : ℕ} (hR : 1 ≤ R) (x : ℕ → X) :
    depthWeight (fun j : Fin (2 * R + 1) => leropeLogit (dftFreq R) 0 (2 * R) (fun _ => 1)
      (fun _ => rotary (dftFreq R) R *ᵥ 1) (2 * R) (x (2 * R)) (x j) (2 * R - j))
      ⟨2 * R - R, by omega⟩ = 1 / (1 + 2 * R * (2 * (R : ℝ) + 1) ^ (-(1 + 1 / (2 * (R : ℝ))))) := by
  have hR0 : (0 : ℝ) < R := Nat.cast_pos.2 hR
  set L := Real.log (2 * (R : ℝ) + 1)
  set F : ℕ → ℝ := fun δ => if δ = R then L else -(L / (2 * R))
  have hFR : F R = L := ite_eq_left rfl
  have hFo (δ : ℕ) (h : δ ≠ R) : F δ = -(L / (2 * R)) := ite_eq_right h
  have hs (j : Fin (2 * R + 1)) : leropeLogit (dftFreq R) 0 (2 * R) (fun _ => 1)
      (fun _ => rotary (dftFreq R) R *ᵥ 1) (2 * R) (x (2 * R)) (x j) (2 * R - j) = F (2 * R - j) :=
    leropeLogit_dft hR _ _ (Nat.sub_le _ _)
  have hZ : ∑ δ ∈ range (2 * R + 1), Real.exp (F δ) =
      Real.exp L + 2 * R * Real.exp (-(L / (2 * R))) := by
    rw [← add_sum_erase _ _ (mem_range.2 (by omega : R < 2 * R + 1)),
      sum_congr rfl fun δ hδ => congrArg Real.exp (hFo δ (ne_of_mem_erase hδ)), sum_const,
      card_erase_of_mem (mem_range.2 (by omega)), card_range, Nat.add_sub_cancel, nsmul_eq_mul,
      hFR]
    push_cast
    ring
  have hnum : leropeLogit (dftFreq R) 0 (2 * R) (fun _ => 1) (fun _ => rotary (dftFreq R) R *ᵥ 1)
      (2 * R) (x (2 * R)) (x (2 * R - R)) R = L := by
    rw [leropeLogit_dft hR _ _ (by omega), ite_eq_left rfl]
  simp only [depthWeight, depthPartition, hs, sum_fin_exp_sub, hZ,
    Nat.sub_sub_self (by omega : R ≤ 2 * R), hnum]
  rw [Real.rpow_def_of_pos (by positivity),
    show Real.log (2 * (R : ℝ) + 1) * -(1 + 1 / (2 * R)) = -(L / (2 * R)) - L by ring,
    Real.exp_sub]
  field_simp

/-- The middle of three keys keeps `1 / (1 + 2 · 3^{−3/2})`. -/
example := depthWeight_dft (X := Unit) (R := 1) le_rfl fun _ => ()

/-- **The DFT head keeps more than half of its weight on the middle key**, for every `R`. -/
theorem half_lt_depthWeight_dft {X : Type*} {R : ℕ} (hR : 1 ≤ R) (x : ℕ → X) :
    1 / 2 < depthWeight (fun j : Fin (2 * R + 1) => leropeLogit (dftFreq R) 0 (2 * R)
      (fun _ => 1) (fun _ => rotary (dftFreq R) R *ᵥ 1) (2 * R) (x (2 * R)) (x j) (2 * R - j))
      ⟨2 * R - R, by omega⟩ := by
  have hR0 : (0 : ℝ) < R := Nat.cast_pos.2 hR
  have h1 : (2 * (R : ℝ) + 1) ^ (-(1 + 1 / (2 * (R : ℝ)))) ≤ (2 * (R : ℝ) + 1)⁻¹ := by
    rw [← Real.rpow_neg_one]
    have : 0 < 1 / (2 * (R : ℝ)) := by positivity
    exact Real.rpow_le_rpow_of_exponent_le (by linarith) (by linarith)
  have h2 : 2 * (R : ℝ) * (2 * (R : ℝ) + 1)⁻¹ < 1 := by
    rw [← div_eq_mul_inv, div_lt_one (by positivity)]
    linarith
  rw [depthWeight_dft hR x, div_lt_div_iff₀ (by norm_num) (by positivity)]
  linarith [mul_le_mul_of_nonneg_left h1 (by positivity : (0 : ℝ) ≤ 2 * R)]

example := half_lt_depthWeight_dft (X := Unit) (R := 16) (by norm_num) fun _ => ()

/-- **Both sides** (`review/lerope_block.md`, RS2): the DFT head meets the hypotheses of
`depthWeight_leropeLogit_le`, and its weight on the middle of `N = 2R + 1` keys lies above `1/2`
and at most at the cap `1 / (1 + R N^{−(1 + 1/R)})` that every head obeys. -/
theorem depthWeight_dft_mem {R : ℕ} (hR : 1 ≤ R) :
    depthWeight (fun j : Fin (2 * R + 1) => leropeLogit (dftFreq R) 0 (2 * R) (fun _ : Unit => 1)
      (fun _ => rotary (dftFreq R) R *ᵥ 1) (2 * R) () () (2 * R - j)) ⟨2 * R - R, by omega⟩ ∈
      Set.Ioc (1 / 2) (1 / (1 + R * (2 * (R : ℝ) + 1) ^ (-(1 + 1 / (R : ℝ))))) := by
  have hq : (1 : Fin 2 × Fin R → ℝ) ⬝ᵥ 1 = 2 * R := by simp [dotProduct]
  have h := depthWeight_leropeLogit_le (dftFreq R) (β := 0) (d := 2 * R) (by positivity)
    (fun _ : Unit => 1) (fun _ => rotary (dftFreq R) R *ᵥ 1) (fun _ => ()) () (i := 2 * R)
    (t := R) hR le_rfl (by omega) (fun _ _ _ => rfl) hq.le
    (by rw [dotProduct_rotary_self]; exact hq.le)
  rw [abs_zero, add_zero, one_mul] at h
  push_cast at h
  exact ⟨half_lt_depthWeight_dft hR fun _ => (), h⟩

/-- The main model's `16` bands on `33` keys. -/
example := depthWeight_dft_mem (R := 16) (by norm_num)

end Transformer.LeRoPEBlock

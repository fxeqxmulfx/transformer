module

public import Transformer.LeRoPEBlock.ReadSeveral
public import Transformer.LeRoPEBlock.SlotDFTShift

/-!
# The DFT heads read every key

Source: `review/lerope_block.md`, *Reading a slot* (added), RS4.

Take RS2's `R` bands at the frequencies of the discrete Fourier transform on `2R + 1` points. At a
query position `i ≤ 2R` the head turned by `t ≤ 2R` scores the distance `t` by `ln n` and every
other distance by `−ln n / (2R)`, `n = i + 1` (`leropeLogit_dft_shift_at`). So the head turned by
`i − s` puts `p = P / (P + i)` on the key `s` and `q = 1 / (P + i)` on each of the other `i` keys,
`P = n^{1 + 1/(2R)}` (`depthWeight_dftScores`).

These heads are a family of `ReadSeveral`. The head of `s` reads the key `s` as
`v_s = (o_s − q ∑_j v_j) / (p − q)` (`dft_read`), the sum `∑_j v_j` taken from a head with the
zero query, which is uniform (`leropeLogit_zero_query`, `dft_read_uniform`), and the reading is
stable with `p − q = (P − 1)/(P + i)` (`ReadPrice`). At `i = 2R` that is more than `1/2`
(`half_lt_dftGap`): the `2R + 1` heads read all `2R + 1` keys and at most double an error.
-/

@[expose] public section

open Matrix Finset
open Transformer.AttentionResiduals Transformer.Residual Transformer.RoPE.RoFormer

namespace Transformer.LeRoPEBlock

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- **The DFT head turned by `t` at any query position `i`**: the distance `t` scores `ln (i + 1)`
and every other distance `δ ≤ 2R` scores `−ln (i + 1) / (2R)`. -/
theorem leropeLogit_dft_shift_at {X : Type*} {R t : ℕ} (hR : 1 ≤ R) (ht : t ≤ 2 * R) (i : ℕ)
    (x y : X) {δ : ℕ} (hδ : δ ≤ 2 * R) :
    leropeLogit (dftFreq R) 0 (2 * R) (fun _ => 1) (fun _ => rotary (dftFreq R) t *ᵥ 1) i x y δ =
      if δ = t then Real.log ((i : ℝ) + 1) else -(Real.log ((i : ℝ) + 1) / (2 * R)) := by
  have hR0 : (0 : ℝ) < R := Nat.cast_pos.2 hR
  have hm : -(δ : ℝ) + t = ((t - δ : ℤ) : ℝ) := by push_cast; ring
  simp only [leropeLogit, phaseLogit, zero_mul, zero_div, add_zero, mulVec_mulVec, ← rotary_add,
    hm, dotProduct_rotary_dft]
  split_ifs with h
  · simp only [h, sub_self, Int.cast_zero, mul_zero, zero_div, Real.cos_zero, sum_const,
      card_range, nsmul_eq_mul, mul_one]
    field_simp
  · rw [sum_cos_dft_of_abs_le (by omega) (abs_le.2 ⟨by omega, by omega⟩)]
    ring

/-- At the position `5` of a window of three bands the head turned by `2` scores the distance `2`
by `ln 6`. -/
example := leropeLogit_dft_shift_at (X := Unit) (R := 3) (t := 2) (by norm_num) (by norm_num) 5
  () () (δ := 2) (by norm_num)

/-- **A head with the zero query is uniform**: it scores every key `0`. -/
theorem leropeLogit_zero_query {X : Type*} {p : ℕ} (θ : Fin p → ℝ) (β d : ℝ)
    (K : X → Fin 2 × Fin p → ℝ) (i : ℕ) (x y : X) (δ : ℕ) :
    leropeLogit θ β d (fun _ => 0) K i x y δ = 0 := by
  simp [leropeLogit, phaseLogit]

/-- `P = (i + 1)^{1 + 1/(2R)}`: at the query position `i` the DFT head keeps `P / (P + i)` on its
key and `1 / (P + i)` on each other key. -/
noncomputable def dftPow (R i : ℕ) : ℝ := ((i : ℝ) + 1) ^ (1 + 1 / (2 * (R : ℝ)))

/-- `p − q = (P − 1)/(P + i)`, the smallest singular value of the DFT heads' weights. -/
noncomputable def dftGap (R i : ℕ) : ℝ := (dftPow R i - 1) / (dftPow R i + i)

/-- The scores of the DFT head turned by `t` at the query position `i`: the query `1`, the key `1`
turned by `t`, `β = 0`, `d = 2R`. -/
noncomputable def dftScores {X : Type*} (R t i : ℕ) (x : ℕ → X) : Fin (i + 1) → ℝ :=
  fun j => leropeLogit (dftFreq R) 0 (2 * R) (fun _ => 1) (fun _ => rotary (dftFreq R) t *ᵥ 1) i
    (x i) (x j) (i - j)

/-- **The weights of the DFT head turned by `i − s`** at the query position `i ≤ 2R`: `P / (P + i)`
on the key `s` and `1 / (P + i)` on every other key. -/
theorem depthWeight_dftScores {X : Type*} {R i : ℕ} (hR : 1 ≤ R) (hi : i ≤ 2 * R) (x : ℕ → X)
    (s j : Fin (i + 1)) :
    depthWeight (dftScores R (i - s) i x) j =
      if j = s then dftPow R i / (dftPow R i + i) else 1 / (dftPow R i + i) := by
  have hR0 : (0 : ℝ) < R := Nat.cast_pos.2 hR
  have hs := s.2
  have hj := j.2
  set L := Real.log ((i : ℝ) + 1) with hL
  set F : ℕ → ℝ := fun δ => if δ = i - s then L else -(L / (2 * R))
  have hFt : F (i - s) = L := ite_eq_left rfl
  have hFo (δ : ℕ) (h : δ ≠ i - s) : F δ = -(L / (2 * R)) := ite_eq_right h
  have hsc (j : Fin (i + 1)) : dftScores R (i - s) i x j = F (i - j) :=
    leropeLogit_dft_shift_at hR (by omega) i _ _ (by omega)
  have hZ : ∑ δ ∈ range (i + 1), Real.exp (F δ) = Real.exp L + i * Real.exp (-(L / (2 * R))) := by
    rw [← add_sum_erase _ _ (mem_range.2 (by omega : i - s < i + 1)),
      sum_congr rfl fun δ hδ => congrArg Real.exp (hFo δ (ne_of_mem_erase hδ)), sum_const,
      card_erase_of_mem (mem_range.2 (by omega)), card_range, Nat.add_sub_cancel, nsmul_eq_mul,
      hFt]
  have hP : Real.exp L = dftPow R i * Real.exp (-(L / (2 * R))) := by
    rw [dftPow, Real.rpow_def_of_pos (by positivity), ← Real.exp_add, ← hL]
    congr 1
    field_simp
    ring
  have he := Real.exp_pos (-(L / (2 * R)))
  have hpos : 0 < dftPow R i + i := by unfold dftPow; positivity
  have hD : dftPow R i * Real.exp (-(L / (2 * R))) + i * Real.exp (-(L / (2 * R))) ≠ 0 := by
    rw [← add_mul]; exact (mul_pos hpos he).ne'
  simp only [depthWeight, depthPartition, hsc, sum_fin_exp_sub, hZ]
  split_ifs with hjs
  · rw [hjs, hFt, hP, div_eq_div_iff hD hpos.ne']
    ring
  · rw [hFo _ fun h => hjs (Fin.ext (by omega)), hP, div_eq_div_iff hD hpos.ne']
    ring

/-- Three keys, one band: the head turned by `2` puts `P / (P + 2)` on the first key. -/
example := depthWeight_dftScores (X := Unit) (R := 1) (i := 2) le_rfl le_rfl (fun _ => ()) 0 0

theorem dftGap_eq (R i : ℕ) :
    dftGap R i = dftPow R i / (dftPow R i + i) - 1 / (dftPow R i + i) := by
  rw [dftGap, div_sub_div_same]

theorem one_lt_dftPow {R i : ℕ} (hi : 1 ≤ i) : 1 < dftPow R i := by
  have : (1 : ℝ) ≤ i := by exact_mod_cast hi
  exact Real.one_lt_rpow (by linarith) (by positivity)

/-- **The DFT heads read every key** at the query position `i`, `1 ≤ i ≤ 2R`: the average `o_s` of
the head turned by `i − s` gives the value at `s` as `(o_s − q ∑_j v_j) / (p − q)`,
`q = 1 / (P + i)`, whatever the values. -/
theorem dft_read {X : Type*} {R i : ℕ} (hR : 1 ≤ R) (hi1 : 1 ≤ i) (hi : i ≤ 2 * R) (x : ℕ → X)
    (v : Fin (i + 1) → E) (s : Fin (i + 1)) :
    (dftGap R i)⁻¹ • (attend (dftScores R (i - s) i x) v - (1 / (dftPow R i + i)) • ∑ j, v j) =
      v s := by
  have hP := one_lt_dftPow (R := R) hi1
  have hpos : 0 < dftPow R i + i := by positivity
  rw [attend, funext (depthWeight_dftScores hR hi x s), dftGap_eq]
  exact decode_weightedSum_ite (fun h => by
    rw [div_eq_div_iff hpos.ne' hpos.ne'] at h
    nlinarith) v s

/-- The main model's `16` bands at the position `32`: the head turned by `32 − s` reads the key
`s` of `33`. -/
example (v : Fin 33 → ℝ) (s : Fin 33) :=
  dft_read (X := Unit) (R := 16) (i := 32) (by norm_num) (by norm_num) le_rfl (fun _ => ()) v s

/-- **With a uniform head**: `∑_j v_j` is `n` times the average of a head that scores every key `0`
(as one with the zero query does), so `v_s = (o_s − (n q) o_0) / (p − q)`. -/
theorem dft_read_uniform {X : Type*} {R i : ℕ} (hR : 1 ≤ R) (hi1 : 1 ≤ i) (hi : i ≤ 2 * R)
    (x : ℕ → X) (v : Fin (i + 1) → E) (s : Fin (i + 1)) :
    (dftGap R i)⁻¹ • (attend (dftScores R (i - s) i x) v -
      (((i : ℝ) + 1) / (dftPow R i + i)) • attend (fun _ : Fin (i + 1) => (0 : ℝ)) v) = v s := by
  have hpos : 0 < dftPow R i + i := by unfold dftPow; positivity
  have hu : attend (fun _ : Fin (i + 1) => (0 : ℝ)) v = (1 / ((i : ℝ) + 1)) • ∑ j, v j := by
    simp [attend, weightedSum, depthWeight_zero, smul_sum]
  rw [hu, smul_smul, show ((i : ℝ) + 1) / (dftPow R i + i) * (1 / ((i : ℝ) + 1)) =
    1 / (dftPow R i + i) by field_simp]
  exact dft_read hR hi1 hi x v s

/-- Main has four heads: one uniform and three turned read three keys, here the key `31`. -/
example (v : Fin 33 → ℝ) := dft_read_uniform (X := Unit) (R := 16) (i := 32) (by norm_num)
  (by norm_num) le_rfl (fun _ => ()) v 31

end Transformer.LeRoPEBlock

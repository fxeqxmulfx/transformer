module

public import Transformer.LeRoPEBlock.SlotDFTShift
public import Transformer.LeRoPEBlock.SlotGeometric

/-!
# The keys at the ends of the context, from both sides

Source: `review/lerope_block.md`, *Reading a slot* (added), RS3.

From above: geometric weights (`SlotGeometric`) on `T + 1` keys of a slot's token on one side of
it cap every head of the main model there at
`1 / (1 + e^{(1 − r^T)(r − ln(1 − r))} N^{−(2(1 + r^{T+1})/(1 + r) + |β|)})` for every `r ∈ (0, 1)`:
older keys for the key next to the query (`depthWeight_leropeLogit_le_geom`), newer keys for the
first key of the context (`depthWeight_leropeLogit_le_geom_newer`). From below: the DFT head
turned by `t` keeps more than half of its weight on the key at the distance `t`
(`depthWeight_dft_shift`). On `N = 2R + 1` keys of one token the two meet in
`depthWeight_adjacent_mem` and `depthWeight_first_mem`: at `β = 0` and the main model's `16`
bands on `33` keys, more than `1/2`, and at most `0.676` next to the query and `0.668` at the
start (at `r = 0.91`).
-/

@[expose] public section

open Matrix Finset
open Transformer.AttentionResiduals Transformer.RoPE.RoFormer

namespace Transformer.LeRoPEBlock

variable {p : ℕ}

/-- **A head of the main model keeps at most
`1 / (1 + e^{(1 − r^T)(r − ln(1 − r))} N^{−(2(1 + r^{T+1})/(1 + r) + |β|)})` of its weight on a key
whose `T + 1` older keys carry its token**, `N = i + 1`, for every `r ∈ (0, 1)`, whatever the
frequencies and the query's token, when the query and the key have squared length at most `d`
(`review/lerope_block.md`, RS3). -/
theorem depthWeight_leropeLogit_le_geom {X : Type*} (θ : Fin p → ℝ) {β d : ℝ} (hd : 0 < d)
    (Q K : X → Fin 2 × Fin p → ℝ) (x : ℕ → X) (y : X) {i t T : ℕ} (hti : t + T + 1 ≤ i)
    (hx : ∀ δ, t ≤ δ → δ ≤ t + T + 1 → x (i - δ) = y) {r : ℝ} (hr0 : 0 < r) (hr1 : r < 1)
    (hQ : Q (x i) ⬝ᵥ Q (x i) ≤ d) (hK : K y ⬝ᵥ K y ≤ d) :
    depthWeight (fun j : Fin (i + 1) => leropeLogit θ β d Q K i (x i) (x j) (i - j))
      ⟨i - t, by omega⟩ ≤ 1 / (1 + Real.exp ((1 - r ^ T) * (r - Real.log (1 - r))) *
        ((i : ℝ) + 1) ^ (-(2 * (1 + r ^ (T + 1)) / (1 + r) + |β|))) := by
  have hV0 : 0 ≤ 2 * (1 + r ^ (T + 1)) / (1 + r) :=
    div_nonneg (by nlinarith [pow_pos hr0 (T + 1)]) (by linarith)
  have h := depthWeight_leropeLogit_le_of_weights θ (β := β) hd Q K x y (by omega : t ≤ i)
    (range (T + 1)) (fun s => t + (s + 1)) (fun a _ b _ h => Nat.add_right_cancel
      (Nat.add_left_cancel h)) (fun s _ => show t + (s + 1) ≠ t by omega)
    (fun s hs => show t + (s + 1) ≤ i by have := mem_range.1 hs; omega)
    (hx t le_rfl (by omega))
    (fun s hs => show x (i - (t + (s + 1))) = y from
      hx _ (by omega) (by have := mem_range.1 hs; omega))
    (fun s => ((s + 1 : ℕ) : ℝ)) (fun s _ => show ((t + (s + 1) : ℕ) : ℝ) = t + ((s + 1 : ℕ) : ℝ)
      by push_cast; ring)
    (geomWeight r T) (fun j _ => geomWeight_nonneg hr0.le hr1.le T j) (sum_geomWeight r T) hV0
    (fun φ => defect_geomWeight_le hr0.le T φ) hQ hK
  refine h.trans (one_div_le_one_div_of_le (by positivity) ?_)
  gcongr
  exact le_entropy_geomWeight hr0 hr1 T

/-- The main model's `33` keys, the key next to the query, `r = 0.91`. -/
example (θ : Fin 16 → ℝ) (β : ℝ) (Q K : Unit → Fin 2 × Fin 16 → ℝ) (hQ : Q () ⬝ᵥ Q () ≤ 32)
    (hK : K () ⬝ᵥ K () ≤ 32) :=
  depthWeight_leropeLogit_le_geom θ (β := β) (d := 32) (by norm_num) Q K (fun _ => ()) ()
    (i := 32) (t := 1) (T := 30) (by norm_num) (fun _ _ _ => rfl) (r := 91 / 100) (by norm_num)
    (by norm_num) hQ hK

/-- **The mirror image**: the same cap holds on a key whose `T + 1` newer keys carry its token,
such as the first key of the context. -/
theorem depthWeight_leropeLogit_le_geom_newer {X : Type*} (θ : Fin p → ℝ) {β d : ℝ} (hd : 0 < d)
    (Q K : X → Fin 2 × Fin p → ℝ) (x : ℕ → X) (y : X) {i t T : ℕ} (hti : t ≤ i)
    (hTt : T + 1 ≤ t) (hx : ∀ δ, t ≤ δ + T + 1 → δ ≤ t → x (i - δ) = y) {r : ℝ} (hr0 : 0 < r)
    (hr1 : r < 1) (hQ : Q (x i) ⬝ᵥ Q (x i) ≤ d) (hK : K y ⬝ᵥ K y ≤ d) :
    depthWeight (fun j : Fin (i + 1) => leropeLogit θ β d Q K i (x i) (x j) (i - j))
      ⟨i - t, by omega⟩ ≤ 1 / (1 + Real.exp ((1 - r ^ T) * (r - Real.log (1 - r))) *
        ((i : ℝ) + 1) ^ (-(2 * (1 + r ^ (T + 1)) / (1 + r) + |β|))) := by
  have hV0 : 0 ≤ 2 * (1 + r ^ (T + 1)) / (1 + r) :=
    div_nonneg (by nlinarith [pow_pos hr0 (T + 1)]) (by linarith)
  have h := depthWeight_leropeLogit_le_of_weights θ (β := β) hd Q K x y hti
    (range (T + 1)) (fun s => t - (s + 1)) (fun a ha b hb h => by
      rw [coe_range, Set.mem_Iio] at ha hb
      simp only at h
      omega)
    (fun s hs => show t - (s + 1) ≠ t by have := mem_range.1 hs; omega)
    (fun s _ => show t - (s + 1) ≤ i by omega) (hx t (by omega) le_rfl)
    (fun s hs => show x (i - (t - (s + 1))) = y from
      hx _ (by have := mem_range.1 hs; omega) (by omega))
    (fun s => -((s + 1 : ℕ) : ℝ)) (fun s hs => show ((t - (s + 1) : ℕ) : ℝ) =
      t + -((s + 1 : ℕ) : ℝ) by rw [Nat.cast_sub (by have := mem_range.1 hs; omega)]; ring)
    (geomWeight r T) (fun j _ => geomWeight_nonneg hr0.le hr1.le T j) (sum_geomWeight r T) hV0
    (fun φ => by simpa only [neg_mul, Real.cos_neg, Real.sin_neg, mul_neg, sum_neg_distrib,
      neg_sq] using defect_geomWeight_le hr0.le T φ) hQ hK
  refine h.trans (one_div_le_one_div_of_le (by positivity) ?_)
  gcongr
  exact le_entropy_geomWeight hr0 hr1 T

/-- The main model's `33` keys, the first key of the context, `r = 0.91`. -/
example (θ : Fin 16 → ℝ) (β : ℝ) (Q K : Unit → Fin 2 × Fin 16 → ℝ) (hQ : Q () ⬝ᵥ Q () ≤ 32)
    (hK : K () ⬝ᵥ K () ≤ 32) :=
  depthWeight_leropeLogit_le_geom_newer θ (β := β) (d := 32) (by norm_num) Q K (fun _ => ()) ()
    (i := 32) (t := 32) (T := 31) le_rfl (by norm_num) (fun _ _ _ => rfl) (r := 91 / 100)
    (by norm_num) (by norm_num) hQ hK

/-- **Both sides for the key next to the query** (`review/lerope_block.md`, RS3): on `N = 2R + 1`
keys the DFT head turned by `1` keeps more than half of its weight on the key next to the query,
and at most the geometric cap that every head with `β = 0` obeys there, for every `r ∈ (0, 1)`. -/
theorem depthWeight_adjacent_mem {R : ℕ} (hR : 1 ≤ R) {r : ℝ} (hr0 : 0 < r) (hr1 : r < 1) :
    depthWeight (fun j : Fin (2 * R + 1) => leropeLogit (dftFreq R) 0 (2 * R) (fun _ : Unit => 1)
      (fun _ => rotary (dftFreq R) 1 *ᵥ 1) (2 * R) () () (2 * R - j)) ⟨2 * R - 1, by omega⟩ ∈
      Set.Ioc (1 / 2) (1 / (1 + Real.exp ((1 - r ^ (2 * R - 2)) * (r - Real.log (1 - r))) *
        (2 * (R : ℝ) + 1) ^ (-(2 * (1 + r ^ (2 * R - 1)) / (1 + r))))) := by
  have hq : (1 : Fin 2 × Fin R → ℝ) ⬝ᵥ 1 = 2 * R := by simp [dotProduct]
  have h := depthWeight_leropeLogit_le_geom (dftFreq R) (β := 0) (d := 2 * R) (by positivity)
    (fun _ : Unit => 1) (fun _ => rotary (dftFreq R) 1 *ᵥ 1) (fun _ => ()) () (i := 2 * R)
    (t := 1) (T := 2 * R - 2) (by omega) (fun _ _ _ => rfl) hr0 hr1 hq.le
    (by rw [dotProduct_rotary_self]; exact hq.le)
  rw [show 2 * R - 2 + 1 = 2 * R - 1 by omega, abs_zero, add_zero] at h
  push_cast at h
  refine ⟨?_, h⟩
  have hs := depthWeight_dft_shift (X := Unit) (t := 1) hR (by omega) fun _ => ()
  rw [Nat.cast_one] at hs
  rw [hs, ← depthWeight_dft hR fun _ => ()]
  exact half_lt_depthWeight_dft hR fun _ => ()

/-- The main model's `16` bands on `33` keys, `r = 0.91`: the cap is `0.676`. -/
example := depthWeight_adjacent_mem (R := 16) (by norm_num) (r := 91 / 100) (by norm_num)
  (by norm_num)

/-- **Both sides for the first key** (`review/lerope_block.md`, RS3): on `N = 2R + 1` keys the DFT
head turned by `2R` keeps more than half of its weight on the first key of the context, and at
most the geometric cap that every head with `β = 0` obeys there, for every `r ∈ (0, 1)`. -/
theorem depthWeight_first_mem {R : ℕ} (hR : 1 ≤ R) {r : ℝ} (hr0 : 0 < r) (hr1 : r < 1) :
    depthWeight (fun j : Fin (2 * R + 1) => leropeLogit (dftFreq R) 0 (2 * R) (fun _ : Unit => 1)
      (fun _ => rotary (dftFreq R) (2 * (R : ℝ)) *ᵥ 1) (2 * R) () () (2 * R - j))
      ⟨2 * R - 2 * R, by omega⟩ ∈
      Set.Ioc (1 / 2) (1 / (1 + Real.exp ((1 - r ^ (2 * R - 1)) * (r - Real.log (1 - r))) *
        (2 * (R : ℝ) + 1) ^ (-(2 * (1 + r ^ (2 * R)) / (1 + r))))) := by
  have hq : (1 : Fin 2 × Fin R → ℝ) ⬝ᵥ 1 = 2 * R := by simp [dotProduct]
  have h := depthWeight_leropeLogit_le_geom_newer (dftFreq R) (β := 0) (d := 2 * R)
    (by positivity) (fun _ : Unit => 1) (fun _ => rotary (dftFreq R) (2 * (R : ℝ)) *ᵥ 1)
    (fun _ => ()) () (i := 2 * R) (t := 2 * R) (T := 2 * R - 1) le_rfl (by omega)
    (fun _ _ _ => rfl) hr0 hr1 hq.le (by rw [dotProduct_rotary_self]; exact hq.le)
  rw [show 2 * R - 1 + 1 = 2 * R by omega, abs_zero, add_zero] at h
  push_cast at h
  refine ⟨?_, h⟩
  have hs := depthWeight_dft_shift (X := Unit) (t := 2 * R) hR le_rfl fun _ => ()
  push_cast at hs
  rw [hs, ← depthWeight_dft hR fun _ => ()]
  exact half_lt_depthWeight_dft hR fun _ => ()

/-- The main model's `16` bands on `33` keys, `r = 0.91`: the cap is `0.668`. -/
example := depthWeight_first_mem (R := 16) (by norm_num) (r := 91 / 100) (by norm_num)
  (by norm_num)

end Transformer.LeRoPEBlock

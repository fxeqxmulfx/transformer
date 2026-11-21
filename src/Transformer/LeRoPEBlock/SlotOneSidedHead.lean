module

public import Transformer.LeRoPEBlock.SlotOneSided

/-!
# A head of the main model on a key with keys of its token on one side

Source: `review/lerope_block.md`, *Reading a slot* (added), RS3.

A head of the main model scores the key at the distance `t` by
`ℓ(t) = g (q ⬝ᵥ R(−t) k + β (q ⬝ᵥ J R(−t) k)² / d)`. Under weights `w ≥ 0` summing to one at the
offsets `e x`, with `|1 − ∑ w e^{i e θ}| ≤ V` at every frequency, each band of the first term has
weighted margins at most `V` times half the squared lengths of its pairs
(`sum_weight_band_sub_le`, `band_coeff_sq_le`), and the second term moves by at most
`|β| |q|² |k|²/d` (`sum_weight_phaseLogit_sub_le`). With Q/K RMS and SSMax's gain `ln N / d` the
weighted margins are at most `(V + |β|) ln N`, so a key whose weighted keys carry its token keeps
at most `1 / (1 + e^H N^{−(V + |β|)})` of the weight, `H = −∑ w log w`, whatever the frequencies
and the query's token (`depthWeight_leropeLogit_le_of_weights`).
-/

@[expose] public section

open Matrix Finset
open Transformer.AttentionResiduals Transformer.RoPE.RoFormer

namespace Transformer.LeRoPEBlock

variable {p : ℕ}

/-- **The weighted margins of a head of the main model** are at most
`g (V (|q|² + |k|²)/2 + |β| |q|² |k|²/d)` when `|1 − ∑ w e^{i e θ}| ≤ V` at every frequency. -/
theorem sum_weight_phaseLogit_sub_le (θ : Fin p → ℝ) {g β d : ℝ} (hg : 0 ≤ g) (hd : 0 < d)
    (c : ℝ) {ι : Type*} (S : Finset ι) (w e : ι → ℝ) (hw : ∀ x ∈ S, 0 ≤ w x)
    (hw1 : ∑ x ∈ S, w x = 1) {V : ℝ} (hV0 : 0 ≤ V)
    (hV : ∀ φ, (1 - ∑ x ∈ S, w x * Real.cos (e x * φ)) ^ 2 +
      (∑ x ∈ S, w x * Real.sin (e x * φ)) ^ 2 ≤ V ^ 2) (q k : Fin 2 × Fin p → ℝ) :
    ∑ x ∈ S, w x * (phaseLogit g β d q (rotary θ (-c) *ᵥ k) -
      phaseLogit g β d q (rotary θ (-(c + e x)) *ᵥ k)) ≤
        g * (V * ((q ⬝ᵥ q + k ⬝ᵥ k) / 2) + |β| * ((q ⬝ᵥ q) * (k ⬝ᵥ k)) / d) := by
  set P := (q ⬝ᵥ q) * (k ⬝ᵥ k)
  set y : ℝ → ℝ := fun t => q ⬝ᵥ rotateHalf (rotary θ (-t) *ᵥ k)
  have hy (t : ℝ) : y t ^ 2 ≤ P := sq_dotProduct_rotateHalf_le θ t q k
  have hterm (t : ℝ) : β * y c ^ 2 / d - β * y t ^ 2 / d ≤ |β| * P / d := by
    rw [← sub_div, ← mul_sub]
    refine div_le_div_of_nonneg_right ?_ hd.le
    have h : |y c ^ 2 - y t ^ 2| ≤ P := by
      rw [abs_le]
      constructor <;> nlinarith [sq_nonneg (y c), sq_nonneg (y t), hy c, hy t]
    exact (le_abs_self _).trans ((abs_mul _ _).trans_le
      (mul_le_mul_of_nonneg_left h (abs_nonneg _)))
  have hband : ∑ x ∈ S, w x * (q ⬝ᵥ (rotary θ (-c) *ᵥ k) -
      q ⬝ᵥ (rotary θ (-(c + e x)) *ᵥ k)) ≤ V * ((q ⬝ᵥ q + k ⬝ᵥ k) / 2) := by
    simp only [dotProduct_rotary_neg, ← sum_sub_distrib, mul_sum]
    rw [sum_comm]
    refine (sum_le_sum fun l _ => sum_weight_band_sub_le S w e hw1 _ _ (θ l) c hV0
      (A := (q (0, l) ^ 2 + q (1, l) ^ 2 + k (0, l) ^ 2 + k (1, l) ^ 2) / 2) (by positivity)
      (band_coeff_sq_le _ _ _ _) (hV (θ l))).trans_eq ?_
    rw [← mul_sum]
    simp only [dotProduct, Fintype.sum_prod_type, Fin.sum_univ_two, ← sum_div, sum_add_distrib,
      sq]
    ring
  have key : ∀ x ∈ S, w x * (phaseLogit g β d q (rotary θ (-c) *ᵥ k) -
      phaseLogit g β d q (rotary θ (-(c + e x)) *ᵥ k)) ≤
        g * (w x * (q ⬝ᵥ (rotary θ (-c) *ᵥ k) - q ⬝ᵥ (rotary θ (-(c + e x)) *ᵥ k)) +
          w x * (|β| * P / d)) := fun x hx => by
    have h := mul_le_mul_of_nonneg_left (hterm (c + e x)) (mul_nonneg hg (hw x hx))
    simp only [phaseLogit, y] at h ⊢
    linarith
  calc _ ≤ ∑ x ∈ S, g * (w x * (q ⬝ᵥ (rotary θ (-c) *ᵥ k) -
        q ⬝ᵥ (rotary θ (-(c + e x)) *ᵥ k)) + w x * (|β| * P / d)) := sum_le_sum key
    _ = g * (∑ x ∈ S, w x * (q ⬝ᵥ (rotary θ (-c) *ᵥ k) -
        q ⬝ᵥ (rotary θ (-(c + e x)) *ᵥ k)) + |β| * P / d) := by
        rw [← mul_sum, sum_add_distrib, ← sum_mul, hw1, one_mul]
    _ ≤ g * (V * ((q ⬝ᵥ q + k ⬝ᵥ k) / 2) + |β| * P / d) := by gcongr

/-- **A head of the main model keeps at most `1 / (1 + e^H N^{−(V + |β|)})` of its weight on a key
whose weighted keys carry its token**, `N = i + 1`, whatever the frequencies and the query's
token: for weights `w ≥ 0` summing to one, of entropy `H = −∑ w log w`, on keys at the distances
`k s = t + e s` on either side of the slot, with `|1 − ∑ w e^{i e θ}| ≤ V` at every frequency,
when the query and the key have squared length at most `d` (`review/lerope_block.md`, RS3). -/
theorem depthWeight_leropeLogit_le_of_weights {X ι : Type*} (θ : Fin p → ℝ) {β d : ℝ}
    (hd : 0 < d) (Q K : X → Fin 2 × Fin p → ℝ) (x : ℕ → X) (y : X) {i t : ℕ} (hti : t ≤ i)
    (S : Finset ι) (k : ι → ℕ) (hk : Set.InjOn k S) (hkt : ∀ s ∈ S, k s ≠ t)
    (hki : ∀ s ∈ S, k s ≤ i) (hxt : x (i - t) = y) (hxs : ∀ s ∈ S, x (i - k s) = y)
    (e : ι → ℝ) (hke : ∀ s ∈ S, (k s : ℝ) = t + e s)
    (w : ι → ℝ) (hw : ∀ s ∈ S, 0 ≤ w s) (hw1 : ∑ s ∈ S, w s = 1) {V : ℝ} (hV0 : 0 ≤ V)
    (hV : ∀ φ, (1 - ∑ s ∈ S, w s * Real.cos (e s * φ)) ^ 2 +
      (∑ s ∈ S, w s * Real.sin (e s * φ)) ^ 2 ≤ V ^ 2)
    (hQ : Q (x i) ⬝ᵥ Q (x i) ≤ d) (hK : K y ⬝ᵥ K y ≤ d) :
    depthWeight (fun j : Fin (i + 1) => leropeLogit θ β d Q K i (x i) (x j) (i - j))
      ⟨i - t, by omega⟩ ≤ 1 / (1 + Real.exp (-∑ s ∈ S, w s * Real.log (w s)) *
        ((i : ℝ) + 1) ^ (-(V + |β|))) := by
  set L := Real.log ((i : ℝ) + 1) with hLdef
  set H := -∑ s ∈ S, w s * Real.log (w s)
  have hL : 0 ≤ L := Real.log_nonneg (by linarith [(Nat.cast_nonneg i : (0 : ℝ) ≤ i)])
  have hfun : (fun j : Fin (i + 1) => leropeLogit θ β d Q K i (x i) (x j) (i - j)) =
      fun j : Fin (i + 1) => leropeLogit θ β d Q K i (x i) (x (i - (i - j))) (i - j) := by
    funext j
    rw [Nat.sub_sub_self j.is_le]
  set f : ℕ → ℝ := fun δ => leropeLogit θ β d Q K i (x i) (x (i - δ)) δ
  have hrw : ∀ s ∈ S, w s * (f t - f (k s)) =
      w s * (phaseLogit (L / d) β d (Q (x i)) (rotary θ (-(t : ℝ)) *ᵥ K y) -
        phaseLogit (L / d) β d (Q (x i)) (rotary θ (-((t : ℝ) + e s)) *ᵥ K y)) :=
    fun s hs => by simp only [f, leropeLogit, hxt, hxs s hs, ← hke s hs, L]
  have hP : (Q (x i) ⬝ᵥ Q (x i)) * (K y ⬝ᵥ K y) / d ≤ d :=
    (div_le_iff₀ hd).2 <| mul_le_mul hQ hK (sum_nonneg fun _ _ => mul_self_nonneg _) hd.le
  have hin : V * ((Q (x i) ⬝ᵥ Q (x i) + K y ⬝ᵥ K y) / 2) +
      |β| * ((Q (x i) ⬝ᵥ Q (x i)) * (K y ⬝ᵥ K y)) / d ≤ d * (V + |β|) := by
    rw [mul_div_assoc]
    nlinarith [abs_nonneg β, mul_le_mul_of_nonneg_left hP (abs_nonneg β)]
  have hM : ∑ s ∈ S, w s * (f t - f (k s)) ≤ (V + |β|) * L := by
    rw [sum_congr rfl hrw]
    refine (sum_weight_phaseLogit_sub_le θ (div_nonneg hL hd.le) hd t S w e hw hw1 hV0 hV
      (Q (x i)) (K y)).trans ?_
    calc _ ≤ L / d * (d * (V + |β|)) := mul_le_mul_of_nonneg_left hin (div_nonneg hL hd.le)
      _ = (V + |β|) * L := by field_simp
  have h := depthWeight_le_of_weighted hti S k hk hkt hki f w hw hw1 hM
  have he : Real.exp (H - (V + |β|) * L) = Real.exp H * ((i : ℝ) + 1) ^ (-(V + |β|)) := by
    rw [Real.rpow_def_of_pos (by positivity), ← Real.exp_add, hLdef]
    congr 1
    ring
  rw [hfun]
  exact h.trans_eq (by rw [he])

/-- Three keys of one token, the slot next to the query: every head keeps at most
`1 / (1 + 3^{−(2 + |β|)})` there. -/
example (θ : Fin 1 → ℝ) (β : ℝ) (Q K : Unit → Fin 2 × Fin 1 → ℝ) (hQ : Q () ⬝ᵥ Q () ≤ 2)
    (hK : K () ⬝ᵥ K () ≤ 2) :
    depthWeight (fun j : Fin (2 + 1) => leropeLogit θ β 2 Q K 2 () () (2 - j)) ⟨2 - 1, by omega⟩ ≤
      1 / (1 + Real.exp (-∑ s ∈ range 1, (fun _ => (1 : ℝ)) s *
        Real.log ((fun _ => (1 : ℝ)) s)) * ((2 : ℕ) + 1 : ℝ) ^ (-(2 + |β|))) :=
  depthWeight_leropeLogit_le_of_weights θ two_pos Q K (fun _ => ()) () (i := 2) (t := 1)
    (by norm_num) (range 1) (fun _ => 2)
    (fun x hx y hy _ => by rw [coe_range, Set.mem_Iio] at hx hy; omega) (fun _ _ => by norm_num)
    (fun _ _ => le_rfl) rfl (fun _ _ => rfl) (fun _ => 1) (fun _ _ => by norm_num) _
    (fun _ _ => zero_le_one) (by simp) (by norm_num) (fun φ => by
      simp only [sum_range_one, one_mul]
      nlinarith [Real.cos_sq_add_sin_sq φ, Real.neg_one_le_cos φ, Real.cos_le_one φ]) hQ hK

end Transformer.LeRoPEBlock

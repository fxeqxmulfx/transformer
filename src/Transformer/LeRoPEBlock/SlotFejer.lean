module

public import Transformer.LeRoPEBlock.Bound

/-!
# The Fejér sum of a LeRoPE head's margins

Source: `review/lerope_block.md`, *Reading a slot* (added), RS1.

A head of the main model scores the key at the distance `t` by
`ℓ(t) = g (q ⬝ᵥ R(−t) k + β (q ⬝ᵥ J R(−t) k)² / d)`, with LeRoPE's rotation `R` and
`J = rotateHalf`. To single out the key at the distance `c` it needs every margin `ℓ(c) − ℓ(t)`
large. They cannot all be, whatever the frequencies: summed over the pairs `s, s' < n` at the
distances `c + s − s'`, which weighs the distance `c + r` by Fejér's `n − |r|`, they are at most
`n² g ((|q|² + |k|²)/2 + |β| |q|² |k|² / d)` (`sum_sum_phaseLogit_sub_le`).

The reason is one identity per band. The band `l` contributes `X cos(tθ) + Y sin(tθ)` to
`q ⬝ᵥ R(−t) k` (`dotProduct_rotary_neg`). Its double sum at the distances `c + s − s'` is its
value at `c` times `F = |∑_{s < n} e^{isθ}|²` (`sum_sum_band`), and `0 ≤ F ≤ n²`
(`sq_add_sq_sum_le`). So the double sum of the band's margins is its value at `c` times `n² − F`,
at most `n²` times its amplitude (`sum_sum_band_sub_le`).
-/

@[expose] public section

open Matrix Finset
open Transformer.RoPE.RoFormer

namespace Transformer.LeRoPEBlock

variable {p : ℕ}

/-- **One band of a rotary score** at the distance `t`: `X cos(tθ) + Y sin(tθ)`. -/
noncomputable def band (X Y θ t : ℝ) : ℝ := X * Real.cos (t * θ) + Y * Real.sin (t * θ)

/-- **The band identity**: `∑_{s,s' < n} band(c + s − s') = band(c) |∑_{s < n} e^{isθ}|²`. -/
theorem sum_sum_band (X Y θ c : ℝ) (n : ℕ) :
    ∑ s ∈ range n, ∑ s' ∈ range n, band X Y θ (c + s - s') =
      band X Y θ c * ((∑ s ∈ range n, Real.cos (s * θ)) ^ 2 +
        (∑ s ∈ range n, Real.sin (s * θ)) ^ 2) := by
  set b' := X * Real.sin (c * θ) - Y * Real.cos (c * θ)
  have h (s s' : ℕ) : band X Y θ (c + s - s') =
      (band X Y θ c * Real.cos (s * θ) - b' * Real.sin (s * θ)) * Real.cos (s' * θ) +
        (b' * Real.cos (s * θ) + band X Y θ c * Real.sin (s * θ)) * Real.sin (s' * θ) := by
    simp only [band, b']
    rw [show (c + s - s') * θ = c * θ + s * θ - s' * θ by ring, Real.cos_sub, Real.sin_sub,
      Real.cos_add, Real.sin_add]
    ring
  simp only [h, sum_add_distrib, ← mul_sum, ← sum_mul, sum_sub_distrib]
  ring

/-- `|∑_{s < n} e^{isθ}|² ≤ n²`. -/
theorem sq_add_sq_sum_le (θ : ℝ) (n : ℕ) :
    (∑ s ∈ range n, Real.cos (s * θ)) ^ 2 + (∑ s ∈ range n, Real.sin (s * θ)) ^ 2 ≤
      (n : ℝ) ^ 2 := by
  have hc := sum_mul_sq_le_sq_mul_sq (range n) (fun s : ℕ => Real.cos (s * θ)) fun _ => 1
  have hs := sum_mul_sq_le_sq_mul_sq (range n) (fun s : ℕ => Real.sin (s * θ)) fun _ => 1
  simp only [mul_one, one_pow, sum_const, card_range, nsmul_eq_mul] at hc hs
  have h1 : ∑ s ∈ range n, Real.cos (s * θ) ^ 2 + ∑ s ∈ range n, Real.sin (s * θ) ^ 2 = n := by
    rw [← sum_add_distrib]
    simp [Real.cos_sq_add_sin_sq]
  nlinarith

/-- **The double sum of a band's margins** is at most `n²` times any bound `a` on its value
at `c`. -/
theorem sum_sum_band_sub_le (X Y θ c a : ℝ) (n : ℕ) (ha : |band X Y θ c| ≤ a) :
    ∑ s ∈ range n, ∑ s' ∈ range n, (band X Y θ c - band X Y θ (c + s - s')) ≤ (n : ℝ) ^ 2 * a := by
  simp only [sum_sub_distrib, sum_const, card_range, nsmul_eq_mul, sum_sum_band]
  have hF := sq_add_sq_sum_le θ n
  have hF0 : 0 ≤ (∑ s ∈ range n, Real.cos (s * θ)) ^ 2 + (∑ s ∈ range n, Real.sin (s * θ)) ^ 2 :=
    by positivity
  have hb := abs_le.mp ha
  nlinarith

/-- A band of amplitude one, three distances. -/
example (θ c : ℝ) : ∑ s ∈ range 3, ∑ s' ∈ range 3, (band 1 0 θ c - band 1 0 θ (c + s - s')) ≤
    ((3 : ℕ) : ℝ) ^ 2 * 1 :=
  sum_sum_band_sub_le 1 0 θ c 1 3 (by simpa [band] using Real.abs_cos_le_one (c * θ))

/-- **LeRoPE's score is a sum of bands**: the band `l` contributes `X cos(tθ_l) + Y sin(tθ_l)` with
`X = q₀k₀ + q₁k₁` and `Y = q₀k₁ − q₁k₀` for its pairs `q₀ + i q₁` and `k₀ + i k₁`. -/
theorem dotProduct_rotary_neg (θ : Fin p → ℝ) (t : ℝ) (q k : Fin 2 × Fin p → ℝ) :
    q ⬝ᵥ (rotary θ (-t) *ᵥ k) = ∑ l, band (q (0, l) * k (0, l) + q (1, l) * k (1, l))
      (q (0, l) * k (1, l) - q (1, l) * k (0, l)) (θ l) t := by
  rw [rotary_mulVec]
  simp only [dotProduct, Fintype.sum_prod_type, Fin.sum_univ_two, rotateHalf, band,
    ← sum_add_distrib]
  refine sum_congr rfl fun l _ => ?_
  simp only [Fin.isValue, ↓reduceIte, one_ne_zero, neg_mul, Real.cos_neg, Real.sin_neg]
  ring

/-- A band of LeRoPE's score is at most half the squared lengths of its pairs. -/
theorem abs_band_le (q₀ q₁ k₀ k₁ θ t : ℝ) :
    |band (q₀ * k₀ + q₁ * k₁) (q₀ * k₁ - q₁ * k₀) θ t| ≤
      (q₀ ^ 2 + q₁ ^ 2 + k₀ ^ 2 + k₁ ^ 2) / 2 := by
  have h := Real.sin_sq_add_cos_sq (t * θ)
  set u₀ := k₀ * Real.cos (t * θ) + k₁ * Real.sin (t * θ)
  set u₁ := k₁ * Real.cos (t * θ) - k₀ * Real.sin (t * θ)
  have hb : band (q₀ * k₀ + q₁ * k₁) (q₀ * k₁ - q₁ * k₀) θ t = q₀ * u₀ + q₁ * u₁ := by
    simp only [band, u₀, u₁]
    ring
  have hu : u₀ ^ 2 + u₁ ^ 2 = k₀ ^ 2 + k₁ ^ 2 := by
    simp only [u₀, u₁]
    linear_combination (k₀ ^ 2 + k₁ ^ 2) * h
  rw [hb, abs_le]
  constructor <;> nlinarith [sq_nonneg (q₀ + u₀), sq_nonneg (q₁ + u₁), sq_nonneg (q₀ - u₀),
    sq_nonneg (q₁ - u₁)]

/-- **The Fejér sum of the margins of LeRoPE's score** is at most `n² (|q|² + |k|²)/2`, whatever
the frequencies. -/
theorem sum_sum_dotProduct_rotary_sub_le (θ : Fin p → ℝ) (c : ℝ) (n : ℕ)
    (q k : Fin 2 × Fin p → ℝ) :
    ∑ s ∈ range n, ∑ s' ∈ range n, (q ⬝ᵥ (rotary θ (-c) *ᵥ k) -
      q ⬝ᵥ (rotary θ (-(c + s - s')) *ᵥ k)) ≤ (n : ℝ) ^ 2 * ((q ⬝ᵥ q + k ⬝ᵥ k) / 2) := by
  have hswap (f : ℕ → ℕ → Fin p → ℝ) : ∑ s ∈ range n, ∑ s' ∈ range n, ∑ l, f s s' l =
      ∑ l, ∑ s ∈ range n, ∑ s' ∈ range n, f s s' l :=
    (sum_congr rfl fun _ _ => sum_comm).trans sum_comm
  simp only [dotProduct_rotary_neg, ← sum_sub_distrib]
  rw [hswap]
  refine (sum_le_sum fun l _ => sum_sum_band_sub_le _ _ _ _ _ n
    (abs_band_le (q (0, l)) (q (1, l)) (k (0, l)) (k (1, l)) _ _)).trans_eq ?_
  rw [← mul_sum]
  simp only [dotProduct, Fintype.sum_prod_type, Fin.sum_univ_two, ← sum_div, sum_add_distrib, sq]
  ring

/-- `(q ⬝ᵥ J R(−t) k)² ≤ |q|² |k|²`: the imaginary part of the score is at most the product of
the lengths. -/
theorem sq_dotProduct_rotateHalf_le (θ : Fin p → ℝ) (t : ℝ) (q k : Fin 2 × Fin p → ℝ) :
    (q ⬝ᵥ rotateHalf (rotary θ (-t) *ᵥ k)) ^ 2 ≤ (q ⬝ᵥ q) * (k ⬝ᵥ k) := by
  have h := sum_mul_sq_le_sq_mul_sq univ q (rotateHalf (rotary θ (-t) *ᵥ k))
  have hJ (v : Fin 2 × Fin p → ℝ) : ∑ i, rotateHalf v i ^ 2 = v ⬝ᵥ v := by
    simp only [rotateHalf, dotProduct, Fintype.sum_prod_type, Fin.sum_univ_two, Fin.isValue,
      ↓reduceIte, one_ne_zero, sq, ← sum_add_distrib]
    exact sum_congr rfl fun _ _ => by ring
  rw [hJ, dotProduct_rotary_self] at h
  simpa only [dotProduct, sq] using h

/-- **The Fejér sum of the margins of a head of the main model** is at most
`n² g ((|q|² + |k|²)/2 + |β| |q|² |k|² / d)` for a gain `g ≥ 0`, whatever the frequencies. -/
theorem sum_sum_phaseLogit_sub_le (θ : Fin p → ℝ) {g β d : ℝ} (hg : 0 ≤ g) (hd : 0 < d) (c : ℝ)
    (n : ℕ) (q k : Fin 2 × Fin p → ℝ) :
    ∑ s ∈ range n, ∑ s' ∈ range n, (phaseLogit g β d q (rotary θ (-c) *ᵥ k) -
      phaseLogit g β d q (rotary θ (-(c + s - s')) *ᵥ k)) ≤
        (n : ℝ) ^ 2 * (g * ((q ⬝ᵥ q + k ⬝ᵥ k) / 2 + |β| * ((q ⬝ᵥ q) * (k ⬝ᵥ k)) / d)) := by
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
  have key (s s' : ℕ) : phaseLogit g β d q (rotary θ (-c) *ᵥ k) -
      phaseLogit g β d q (rotary θ (-(c + s - s')) *ᵥ k) ≤
        g * ((q ⬝ᵥ (rotary θ (-c) *ᵥ k) - q ⬝ᵥ (rotary θ (-(c + s - s')) *ᵥ k)) +
          |β| * P / d) := by
    have h := mul_le_mul_of_nonneg_left (hterm (c + s - s')) hg
    simp only [phaseLogit, y] at h ⊢
    linarith
  calc _ ≤ ∑ s ∈ range n, ∑ s' ∈ range n, g * ((q ⬝ᵥ (rotary θ (-c) *ᵥ k) -
        q ⬝ᵥ (rotary θ (-(c + s - s')) *ᵥ k)) + |β| * P / d) :=
        sum_le_sum fun s _ => sum_le_sum fun s' _ => key s s'
    _ = g * (∑ s ∈ range n, ∑ s' ∈ range n, (q ⬝ᵥ (rotary θ (-c) *ᵥ k) -
        q ⬝ᵥ (rotary θ (-(c + s - s')) *ᵥ k)) + (n : ℝ) ^ 2 * (|β| * P / d)) := by
        simp only [mul_add, sum_add_distrib, ← mul_sum, sum_const, card_range, nsmul_eq_mul]
        ring
    _ ≤ g * ((n : ℝ) ^ 2 * ((q ⬝ᵥ q + k ⬝ᵥ k) / 2) + (n : ℝ) ^ 2 * (|β| * P / d)) := by
        gcongr
        exact sum_sum_dotProduct_rotary_sub_le θ c n q k
    _ = _ := by ring

/-- Two bands, the slot at the distance `5`, three distances on each side. -/
example (θ : Fin 2 → ℝ) (q k : Fin 2 × Fin 2 → ℝ) :
    ∑ s ∈ range 4, ∑ s' ∈ range 4, (phaseLogit 1 (1 / 2) 4 q (rotary θ (-5) *ᵥ k) -
      phaseLogit 1 (1 / 2) 4 q (rotary θ (-(5 + s - s')) *ᵥ k)) ≤
        ((4 : ℕ) : ℝ) ^ 2 * (1 * ((q ⬝ᵥ q + k ⬝ᵥ k) / 2 + |1 / 2| * ((q ⬝ᵥ q) * (k ⬝ᵥ k)) / 4)) :=
  sum_sum_phaseLogit_sub_le θ zero_le_one (by norm_num) 5 4 q k

end Transformer.LeRoPEBlock

module

public import Transformer.Precision.Float

/-!
# Local error bounds from actual binary32 grid points

Source: `review/flat_spectrum.md`, *Primitive execution certificates*, C1;
derived from `Precision.ieee` (IEEE 754 binary32 decoding) and `IsNearest`.
The tie rule is immaterial to the bounds. Nearest finite-grid rounding
models finite, non-overflowing results; it does not model NaNs, infinity,
flush-to-zero, or a particular CUDA transcendental implementation.
-/

@[expose] public section
noncomputable section

namespace Transformer.Precision

/-- C1: two representable endpoints bound nearest-rounding error by half
their distance. The endpoints need not be adjacent for this sufficient bound. -/
theorem IsNearest.error_of_bracket {G : Set ℝ} {Q : ℝ → ℝ} (hQ : IsNearest G Q)
    {a b x : ℝ} (ha : a ∈ G) (hb : b ∈ G) (hax : a ≤ x) (hxb : x ≤ b) :
    |Q x - x| ≤ (b - a) / 2 := by
  have h1 := (hQ x).2 a ha
  have h2 := (hQ x).2 b hb
  rw [abs_of_nonpos (sub_nonpos.mpr hax)] at h1
  rw [abs_of_nonneg (sub_nonneg.mpr hxb)] at h2
  linarith

/-- Decoded finite binary32 magnitude, with an explicit sign bit, exponent
field and fraction field. The exponent zero case includes gradual underflow. -/
def binary32Value (negative : Bool) (e m : ℕ) : ℝ :=
  (if negative then -1 else 1) *
    (if e = 0 then (m : ℝ) * (2 : ℝ) ^ (-149 : ℤ)
      else (8388608 + (m : ℝ)) * (2 : ℝ) ^ ((e : ℤ) - 150))

/-- C1: the error analysis uses actual finite binary32 patterns, not an
unbounded idealized exponent grid. All normal and subnormal fields are included. -/
theorem binary32_value_mem (negative : Bool) (e m : ℕ) (he : e < 255) (hm : m < 8388608) :
    binary32Value negative e m ∈ grid 8 23 := by
  let bits := (if negative then 2147483648 else 0) + 8388608 * e + m
  have hbits : bits < 2 ^ (1 + 8 + 23) := by
    norm_num
    dsimp [bits]
    split <;> omega
  have hmantissa : bits % 8388608 = m := by dsimp [bits]; split <;> omega
  have hexponent : bits / 8388608 % 256 = e := by dsimp [bits]; split <;> omega
  have hsign : bits / 2147483648 % 2 = if negative then 1 else 0 := by
    dsimp [bits]
    split <;> omega
  refine ⟨bits, hbits, ?_⟩
  norm_num only [ieee, bias]
  rw [hmantissa, hexponent, hsign]
  have he' : e ≠ 255 := by omega
  have hexp : (e : ℤ) - 127 - 23 = (e : ℤ) - 150 := by ring
  cases negative <;> by_cases he0 : e = 0 <;> norm_num [binary32Value, he', he0, hexp]
  ring

/-- C1: bracketing by any two decoded finite binary32 values certifies a
single operation's absolute error under round-to-nearest. -/
theorem binary32_bracket_error {Q : ℝ → ℝ} (hQ : IsNearest (grid 8 23) Q)
    (na nb : Bool) (ea ma eb mb : ℕ) (hea : ea < 255) (hma : ma < 8388608)
    (heb : eb < 255) (hmb : mb < 8388608) (x : ℝ)
    (ha : binary32Value na ea ma ≤ x) (hb : x ≤ binary32Value nb eb mb) :
    |Q x - x| ≤ (binary32Value nb eb mb - binary32Value na ea ma) / 2 :=
  hQ.error_of_bracket (binary32_value_mem na ea ma hea hma)
    (binary32_value_mem nb eb mb heb hmb) ha hb

/-- C1: within one normal binade, two consecutive fraction fields are
separated by 2^(e-150), so nearest rounding loses at most half that spacing. -/
theorem binary32_normal_error {Q : ℝ → ℝ} (hQ : IsNearest (grid 8 23) Q)
    (e m : ℕ) (he0 : 0 < e) (he : e < 255) (hm : m + 1 < 8388608) (x : ℝ)
    (ha : binary32Value false e m ≤ x) (hb : x ≤ binary32Value false e (m + 1)) :
    |Q x - x| ≤ (2 : ℝ) ^ ((e : ℤ) - 150) / 2 := by
  have h := binary32_bracket_error hQ false false e m e (m + 1) he (by omega) he hm x ha hb
  have he' : e ≠ 0 := by omega
  exact h.trans_eq (by simp [binary32Value, he']; ring)

/-- C1: on the certified normal interval the relative error is at most
2^-24, half the float32 RMSNorm epsilon. Binade boundaries can instead
use the general two-endpoint certificate. -/
theorem binary32_normal_relative_error {Q : ℝ → ℝ} (hQ : IsNearest (grid 8 23) Q)
    (e m : ℕ) (he0 : 0 < e) (he : e < 255) (hm : m + 1 < 8388608) (x : ℝ)
    (ha : binary32Value false e m ≤ x) (hb : x ≤ binary32Value false e (m + 1)) :
    |Q x - x| ≤ (1 / (2 : ℝ) ^ 24) * |x| := by
  have hs : 0 < (2 : ℝ) ^ ((e : ℤ) - 150) := zpow_pos (by norm_num) _
  have he' : e ≠ 0 := by omega
  have hx : 8388608 * (2 : ℝ) ^ ((e : ℤ) - 150) ≤ x := by
    simp only [binary32Value, Bool.false_eq_true, ↓reduceIte, he', one_mul] at ha
    nlinarith [Nat.cast_nonneg (α := ℝ) m]
  have hx0 : 0 ≤ x := (by positivity : 0 ≤ 8388608 * (2 : ℝ) ^ ((e : ℤ) - 150)).trans hx
  rw [abs_of_nonneg hx0]
  exact (binary32_normal_error hQ e m he0 he hm x ha hb).trans (by norm_num; linarith)

/-- C1: gradual-underflow intervals have a fixed absolute error bound,
including results close to zero; a purely relative bound would be invalid. -/
theorem binary32_subnormal_error {Q : ℝ → ℝ} (hQ : IsNearest (grid 8 23) Q)
    (m : ℕ) (hm : m + 1 < 8388608) (x : ℝ)
    (ha : binary32Value false 0 m ≤ x) (hb : x ≤ binary32Value false 0 (m + 1)) :
    |Q x - x| ≤ (2 : ℝ) ^ (-149 : ℤ) / 2 := by
  have h := binary32_bracket_error hQ false false 0 m 0 (m + 1)
    (by norm_num) (by omega) (by norm_num) hm x ha hb
  exact h.trans_eq (by simp [binary32Value]; ring)

example : binary32Value false 127 0 ∈ grid 8 23 := binary32_value_mem false 127 0 (by decide) (by decide)
example : binary32Value true 1 0 ∈ grid 8 23 := binary32_value_mem true 1 0 (by decide) (by decide)
example : binary32Value false 0 1 ∈ grid 8 23 := binary32_value_mem false 0 1 (by decide) (by decide)

/-- A real nearest binary32 map exists and certifies a nonrepresentable
point between 1 and the next float, with a strictly positive error budget. -/
example : ∃ Q : ℝ → ℝ, IsNearest (grid 8 23) Q ∧
    |Q (1 + (1 / (2 : ℝ) ^ 25)) - (1 + 1 / (2 : ℝ) ^ 25)| ≤
      (1 / (2 : ℝ) ^ 24) * |1 + 1 / (2 : ℝ) ^ 25| := by
  obtain ⟨Q, hQ⟩ := exists_isNearest_grid (E := 8) (by decide) 23
  exact ⟨Q, hQ, binary32_normal_relative_error hQ 127 0 (by decide) (by decide) (by decide) _
    (by norm_num [binary32Value]) (by norm_num [binary32Value])⟩

example : ∃ Q : ℝ → ℝ, IsNearest (grid 8 23) Q ∧
    |Q ((2 : ℝ) ^ (-151 : ℤ)) - (2 : ℝ) ^ (-151 : ℤ)| ≤ (2 : ℝ) ^ (-149 : ℤ) / 2 := by
  obtain ⟨Q, hQ⟩ := exists_isNearest_grid (E := 8) (by decide) 23
  exact ⟨Q, hQ, binary32_subnormal_error hQ 0 (by decide) _
    (by norm_num [binary32Value]) (by norm_num [binary32Value])⟩

example : ∃ Q : ℝ → ℝ, IsNearest (grid 8 23) Q ∧
    |Q (1 + 1 / (2 : ℝ) ^ 25) - (1 + 1 / (2 : ℝ) ^ 25)| ≤ (2 : ℝ) ^ (-23 : ℤ) / 2 := by
  obtain ⟨Q, hQ⟩ := exists_isNearest_grid (E := 8) (by decide) 23
  exact ⟨Q, hQ, binary32_normal_error hQ 127 0 (by decide) (by decide) (by decide) _
    (by norm_num [binary32Value]) (by norm_num [binary32Value])⟩

example : ∃ Q : ℝ → ℝ, IsNearest (grid 8 23) Q ∧
    |Q (1 + 1 / (2 : ℝ) ^ 25) - (1 + 1 / (2 : ℝ) ^ 25)| ≤
      (binary32Value false 127 1 - binary32Value false 127 0) / 2 := by
  obtain ⟨Q, hQ⟩ := exists_isNearest_grid (E := 8) (by decide) 23
  exact ⟨Q, hQ, binary32_bracket_error hQ false false 127 0 127 1
    (by decide) (by decide) (by decide) (by decide) _
    (by norm_num [binary32Value]) (by norm_num [binary32Value])⟩

example : ∃ Q : ℝ → ℝ, IsNearest (grid 8 23) Q ∧
    |Q (1 + 1 / (2 : ℝ) ^ 25) - (1 + 1 / (2 : ℝ) ^ 25)| ≤
      (binary32Value false 127 1 - binary32Value false 127 0) / 2 := by
  obtain ⟨Q, hQ⟩ := exists_isNearest_grid (E := 8) (by decide) 23
  exact ⟨Q, hQ, hQ.error_of_bracket
    (binary32_value_mem false 127 0 (by decide) (by decide))
    (binary32_value_mem false 127 1 (by decide) (by decide))
    (by norm_num [binary32Value]) (by norm_num [binary32Value])⟩

end Transformer.Precision

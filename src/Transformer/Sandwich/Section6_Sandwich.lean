module

public import Mathlib.Analysis.Real.Pi.Irrational
public import Transformer.Sandwich.Section5_Sinusoidal

/-!
# §6.1 Sandwich

Chi, Fan, Rudnicky, Ramadge, arXiv:2212.10356v2, §6.1: "we drop the cross terms and keep only the
inner product of two positional embeddings at `m` and `n`. Now `p_m^⊤ p_n` with `m, n ∈ [1, L]`
become the temporal bias terms of Sandwich:
`p_m^⊤ p_n = Σ_{i=1}^{d̄/2} sin(m/10000^{2i/d̄}) sin(n/10000^{2i/d̄}) + cos(m/10000^{2i/d̄})
cos(n/10000^{2i/d̄}) = Σ_{i=1}^{d̄/2} cos((m − n)/10000^{2i/d̄})`." "The largest value of
`p_m^⊤ p_n` happens at the point where `m − n = 0`, which gives the maximum value of `d̄/2`."
Eq. (6) is `(p_m^⊤ p_n − d̄/2)/h` with ALiBi's compression ratios `h = n · 8/H`, "added after the
scaled softmax is done in Eq. (1)".

* `freq D i`: the frequencies `1/10000^{2i/d̄}` for `d̄ = 2D` and the paper's `i = 1, …, d̄/2`,
  here `i + 1` for `i : Fin D`.
* `sinusoid_dot`: the display, `p_m^⊤ p_n = Σ_i cos((m − n) θ_i)`, for any frequencies.
* `sinusoid_dot_le`, `sinusoid_dot_self`: the maximum `d̄/2`, at `m = n`; `sinusoid_dot_eq_iff`:
  for integer positions only there, since `θ_{d̄/2} = 1/10000` and `π` is irrational.
* `sandwich D h t`: Eq. (6) at the distance `t = m − n` (`sandwich_eq`), a relative positional
  embedding. It is `≤ 0` (`sandwich_nonpos`), `0` exactly at `t = 0` for integers
  (`sandwich_eq_zero_iff`), and `≥ −2D/h` (`neg_le_sandwich`): bounded, unlike ALiBi's bias.
* `sandwichHeads D H`: head `j` has the compression ratio `h = (j + 1) · 8/H`, ALiBi's exponent;
  `score` adds the bias after the scaling, as §6.1 says.
-/

@[expose] public section

namespace Transformer.Sandwich

open Finset Matrix

variable {D : ℕ}

/-- Sandwich's frequencies `1/10000^{2i/d̄}` for `d̄ = 2D` and the paper's `i = 1, …, d̄/2`
(§6.1), here `i + 1`. -/
noncomputable def freq (D : ℕ) (i : Fin D) : ℝ := 1 / 10000 ^ (((i : ℝ) + 1) / D)

/-- **§6.1's display**: `p_m^⊤ p_n = Σ_i sin(m θ_i) sin(n θ_i) + cos(m θ_i) cos(n θ_i)
= Σ_i cos((m − n) θ_i)`. -/
theorem sinusoid_dot (θ : Fin D → ℝ) (m n : ℝ) :
    sinusoid θ m ⬝ᵥ sinusoid θ n = ∑ i, Real.cos ((m - n) * θ i) := by
  simp only [dotProduct, sinusoid, Fintype.sum_sum_type, Sum.elim_inl, Sum.elim_inr,
    ← sum_add_distrib]
  refine sum_congr rfl fun i _ => ?_
  rw [sub_mul, Real.cos_sub]
  ring

/-- **The largest value of `p_m^⊤ p_n` is `d̄/2`** (§6.1). -/
theorem sinusoid_dot_le (θ : Fin D → ℝ) (m n : ℝ) : sinusoid θ m ⬝ᵥ sinusoid θ n ≤ D := by
  rw [sinusoid_dot]
  calc ∑ i, Real.cos ((m - n) * θ i) ≤ ∑ _i : Fin D, (1 : ℝ) :=
        sum_le_sum fun i _ => Real.cos_le_one _
    _ = D := by simp

/-- **`p_m^⊤ p_n = d̄/2` at `m − n = 0`** (§6.1). -/
theorem sinusoid_dot_self (θ : Fin D → ℝ) (m : ℝ) : sinusoid θ m ⬝ᵥ sinusoid θ m = D := by
  simp [sinusoid_dot]

/-- `cos(t/10000) = 1` for an integer `t` only at `t = 0`: `π` is irrational. -/
theorem cos_div_eq_one_iff (t : ℤ) : Real.cos (t * (1 / 10000)) = 1 ↔ t = 0 := by
  refine ⟨fun h => ?_, fun h => by simp [h]⟩
  obtain ⟨k, hk⟩ := (Real.cos_eq_one_iff _).1 h
  by_contra ht
  have hk0 : (k : ℝ) ≠ 0 := fun hk0 => by
    rw [hk0, zero_mul] at hk
    exact ht (by exact_mod_cast (by linarith : (t : ℝ) = 0))
  apply irrational_pi.ne_rat ((t : ℚ) / (20000 * k))
  push_cast
  rw [eq_div_iff (mul_ne_zero (by norm_num) hk0)]
  linear_combination 10000 * hk

/-- The last of Sandwich's frequencies is `1/10000`. -/
theorem freq_last (hD : 1 ≤ D) : freq D ⟨D - 1, by omega⟩ = 1 / 10000 := by
  have h1 : ((D - 1 : ℕ) : ℝ) + 1 = D := by
    rw [Nat.cast_sub hD, Nat.cast_one, sub_add_cancel]
  have h2 : (D : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (by omega)
  simp only [freq, h1, div_self h2, Real.rpow_one]

/-- The hypothesis of `freq_last` holds for `D = 64`, `d̄ = 128`. -/
example : freq 64 ⟨63, by norm_num⟩ = 1 / 10000 := freq_last (by norm_num)

/-- **`p_m^⊤ p_n = d̄/2` only at `m = n`** for integer positions and `d̄ ≥ 2`. -/
theorem sinusoid_dot_eq_iff (hD : 1 ≤ D) (m n : ℤ) :
    sinusoid (freq D) m ⬝ᵥ sinusoid (freq D) n = D ↔ m = n := by
  refine ⟨fun h => ?_, fun h => by rw [h, sinusoid_dot_self]⟩
  rw [sinusoid_dot] at h
  have hsum : ∑ i : Fin D, (1 - Real.cos ((m - n : ℝ) * freq D i)) = 0 := by
    rw [sum_sub_distrib, h]
    simp
  have hlast := (sum_eq_zero_iff_of_nonneg fun i _ => sub_nonneg.2 (Real.cos_le_one _)).1 hsum
    ⟨D - 1, by omega⟩ (mem_univ _)
  rw [freq_last hD, sub_eq_zero, eq_comm, show (m - n : ℝ) = ((m - n : ℤ) : ℝ) by push_cast; ring,
    cos_div_eq_one_iff, sub_eq_zero] at hlast
  exact hlast

/-- The hypothesis of `sinusoid_dot_eq_iff` holds for `d̄ = 128`. -/
example (m n : ℤ) : sinusoid (freq 64) m ⬝ᵥ sinusoid (freq 64) n = 64 ↔ m = n := by
  exact_mod_cast sinusoid_dot_eq_iff (D := 64) (by norm_num) m n

/-- **Eq. (6)**, Sandwich's bias at the distance `t = m − n` with the compression ratio `h`:
`(Σ_i cos(t θ_i) − d̄/2)/h`. -/
noncomputable def sandwich (D : ℕ) (h t : ℝ) : ℝ := (∑ i, Real.cos (t * freq D i) - D) / h

/-- **Eq. (6) is `(p_m^⊤ p_n − d̄/2)/h`**, a function of `m − n`: a relative positional
embedding. -/
theorem sandwich_eq (D : ℕ) (h m n : ℝ) :
    sandwich D h (m - n) = (sinusoid (freq D) m ⬝ᵥ sinusoid (freq D) n - D) / h := by
  rw [sinusoid_dot, sandwich]

/-- Sandwich's bias is at most `0`. -/
theorem sandwich_nonpos {h : ℝ} (hh : 0 < h) (t : ℝ) : sandwich D h t ≤ 0 := by
  have := sinusoid_dot_le (freq D) t 0
  rw [sinusoid_dot, sub_zero] at this
  rw [sandwich]
  exact div_nonpos_of_nonpos_of_nonneg (by linarith) hh.le

/-- The hypothesis of `sandwich_nonpos` holds for `h = 8`. -/
example (t : ℝ) : sandwich 64 8 t ≤ 0 := sandwich_nonpos (by norm_num) t

/-- **Sandwich's bias is `0` exactly at the distance `0`**, for integer distances and `d̄ ≥ 2`. -/
theorem sandwich_eq_zero_iff (hD : 1 ≤ D) {h : ℝ} (hh : 0 < h) (t : ℤ) :
    sandwich D h t = 0 ↔ t = 0 := by
  rw [show (t : ℝ) = ((t : ℤ) : ℝ) - ((0 : ℤ) : ℝ) by simp, sandwich_eq, div_eq_zero_iff,
    or_iff_left hh.ne', sub_eq_zero, sinusoid_dot_eq_iff hD]

/-- The hypotheses of `sandwich_eq_zero_iff` hold for `d̄ = 128` and `h = 8`. -/
example (t : ℤ) : sandwich 64 8 t = 0 ↔ t = 0 := sandwich_eq_zero_iff (by norm_num) (by norm_num) t

/-- **Sandwich's bias is bounded**: it is at least `−2D/h = −d̄/h`. -/
theorem neg_le_sandwich {h : ℝ} (hh : 0 < h) (t : ℝ) : -(2 * D) / h ≤ sandwich D h t := by
  rw [sandwich, div_le_div_iff_of_pos_right hh]
  have : -(D : ℝ) ≤ ∑ i, Real.cos (t * freq D i) := by
    calc -(D : ℝ) = ∑ _i : Fin D, (-1 : ℝ) := by simp
      _ ≤ _ := sum_le_sum fun i _ => Real.neg_one_le_cos _
  linarith

/-- The hypothesis of `neg_le_sandwich` holds for `h = 8`. -/
example (t : ℝ) : -(2 * ((64 : ℕ) : ℝ)) / 8 ≤ sandwich 64 8 t :=
  neg_le_sandwich (by norm_num) t

/-- Sandwich with `H` heads and `d̄ = 2D` (§6.1): head `j` has the compression ratio
`h = (j + 1) · 8/H`, ALiBi's exponent. -/
noncomputable def sandwichHeads (D H : ℕ) : Fin H → ℕ → EReal :=
  fun j t => ((sandwich D (exponent H j) t : ℝ) : EReal)

/-- The compression ratios are positive. -/
theorem exponent_pos {H : ℕ} (j : Fin H) : 0 < exponent H j := by
  have : (0 : ℝ) < H := by exact_mod_cast j.pos
  unfold exponent
  positivity

end Transformer.Sandwich

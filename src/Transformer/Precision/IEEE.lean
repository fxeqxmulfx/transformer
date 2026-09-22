/-
# IEEE binary floating point

The binary interchange formats of IEEE 754-2008, §3.4: a pattern of `1 + E + M`
bits (sign, biased exponent, mantissa) decodes to a subnormal, a normal number,
or an infinity or NaN, which have no value in `ℝ` and decode to `none`.
binary16 is `E = 5, M = 10`, bfloat16 `E = 8, M = 7`, binary32 `E = 8, M = 23`.

What the limits of `Transformer.Precision` need of a format is here: its
numbers form a finite set containing `0` (`grid_finite`, `zero_mem_grid`), no
nonzero number is closer to `0` than the smallest subnormal (`minSub_le_abs`),
and every number has `M + 1` significant bits (`grid_hasSignificand`).
-/

import Transformer.Precision.Accumulate
import Mathlib.Data.Set.Finite.Lemmas

namespace Transformer
namespace Precision

/-- The exponent bias `2^{E-1} - 1` of an IEEE binary format. -/
def bias (E : ℕ) : ℤ := 2 ^ (E - 1) - 1

/-- **IEEE binary decoding.**  A pattern of `1 + E + M` bits: sign, biased
exponent `e`, mantissa `m`.  `e = 0` is subnormal, `m · 2^{1 - bias - M}`;
`0 < e < 2^E - 1` is normal, `(2^M + m) · 2^{e - bias - M}`; `e = 2^E - 1` is an
infinity or a NaN and has no real value (IEEE 754-2008, §3.4). -/
noncomputable def ieee (E M bits : ℕ) : Option ℝ :=
  let m := bits % 2 ^ M
  let e := bits / 2 ^ M % 2 ^ E
  let sgn : ℝ := if bits / 2 ^ (E + M) % 2 = 1 then -1 else 1
  if e = 2 ^ E - 1 then none
  else if e = 0 then some (sgn * m * (2 : ℝ) ^ (1 - bias E - M))
  else some (sgn * (2 ^ M + m) * (2 : ℝ) ^ ((e : ℤ) - bias E - M))

/-- IEEE binary16 (IEEE 754-2008, §3.6, Table 3.5). -/
noncomputable def f16 (bits : ℕ) : Option ℝ := ieee 5 10 bits

/-- bfloat16: the upper half of a binary32. -/
noncomputable def bf16 (bits : ℕ) : Option ℝ := ieee 8 7 bits

/-- IEEE binary32, `float`. -/
noncomputable def f32 (bits : ℕ) : Option ℝ := ieee 8 23 bits

/-- The smallest positive number of the format: the subnormal `2^{1 - bias - M}`. -/
noncomputable def minSub (E M : ℕ) : ℝ := (2 : ℝ) ^ (1 - bias E - M)

theorem minSub_pos (E M : ℕ) : 0 < minSub E M := zpow_pos (by norm_num) _

/-- **No number of the format is closer to `0` than the smallest subnormal.** -/
theorem minSub_le_abs {E M bits : ℕ} {v : ℝ} (h : ieee E M bits = some v) (hv : v ≠ 0) :
    minSub E M ≤ |v| := by
  unfold ieee at h
  simp only at h
  have hsgn : |(if bits / 2 ^ (E + M) % 2 = 1 then (-1 : ℝ) else 1)| = 1 := by
    split_ifs <;> simp
  generalize (if bits / 2 ^ (E + M) % 2 = 1 then (-1 : ℝ) else 1) = sgn at h hsgn
  split_ifs at h with h1 h2
  · obtain rfl := Option.some.inj h
    have hm : bits % 2 ^ M ≠ 0 := by rintro hm; apply hv; simp [hm]
    rw [abs_mul, abs_mul, hsgn, one_mul, abs_of_nonneg (by positivity),
      abs_of_pos (zpow_pos (by norm_num) _), minSub]
    have : (1 : ℝ) ≤ (bits % 2 ^ M : ℕ) := by exact_mod_cast Nat.one_le_iff_ne_zero.2 hm
    nlinarith [zpow_pos (show (0 : ℝ) < 2 by norm_num) (1 - bias E - M)]
  · obtain rfl := Option.some.inj h
    rw [abs_mul, abs_mul, hsgn, one_mul, abs_of_nonneg (by positivity),
      abs_of_pos (zpow_pos (by norm_num) _), minSub]
    have he : (1 : ℤ) ≤ (bits / 2 ^ M % 2 ^ E : ℕ) := by exact_mod_cast Nat.one_le_iff_ne_zero.2 h2
    have hz : (2 : ℝ) ^ (1 - bias E - M) ≤ 2 ^ (((bits / 2 ^ M % 2 ^ E : ℕ) : ℤ) - bias E - M) :=
      zpow_le_zpow_right₀ (by norm_num) (by linarith)
    have hM : (1 : ℝ) ≤ 2 ^ M + (bits % 2 ^ M : ℕ) := by
      have := one_le_pow₀ (n := M) (by norm_num : (1 : ℝ) ≤ 2)
      have : (0 : ℝ) ≤ (bits % 2 ^ M : ℕ) := by positivity
      linarith
    nlinarith [zpow_pos (show (0 : ℝ) < 2 by norm_num) (1 - bias E - M)]

/-- The hypotheses of `minSub_le_abs` are satisfiable: the binary16 pattern
`0x3C00` is `1`. -/
example : f16 0x3C00 = some 1 := by
  norm_num [f16, ieee, bias]

/-- The smallest positive binary16 number is `2^{-24}`. -/
theorem minSub_f16 : minSub 5 10 = 2 ^ (-24 : ℤ) := by norm_num [minSub, bias]

/-- The smallest positive bfloat16 number is `2^{-133}`. -/
theorem minSub_bf16 : minSub 8 7 = 2 ^ (-133 : ℤ) := by norm_num [minSub, bias]

/-- The smallest positive binary32 number is `2^{-149}`. -/
theorem minSub_f32 : minSub 8 23 = 2 ^ (-149 : ℤ) := by norm_num [minSub, bias]

/-- The smallest positive FP8 E4M3 number is `2^{-9}`. -/
theorem minSub_e4m3 : minSub 4 3 = 2 ^ (-9 : ℤ) := by norm_num [minSub, bias]

/-- The smallest positive FP8 E5M2 number is `2^{-16}`. -/
theorem minSub_e5m2 : minSub 5 2 = 2 ^ (-16 : ℤ) := by norm_num [minSub, bias]

/-- The smallest positive FP4 E2M1 number is `2^{-1}` (OCP Microscaling (MX)
Formats Specification v1.0, the E2M1 element type). -/
theorem minSub_e2m1 : minSub 2 1 = 2 ^ (-1 : ℤ) := by norm_num [minSub, bias]

/-- The pattern `0x0001` is the smallest positive binary16 number. -/
theorem f16_one : f16 1 = some (minSub 5 10) := by
  norm_num [f16, ieee, minSub, bias]

/-- The largest finite binary16 number, `0x7BFF`, is `65504`. -/
theorem f16_max : f16 0x7BFF = some 65504 := by
  norm_num [f16, ieee, bias]

/-- The numbers of the IEEE format with `E` exponent and `M` mantissa bits. -/
def grid (E M : ℕ) : Set ℝ := {v | ∃ b < 2 ^ (1 + E + M), ieee E M b = some v}

theorem zero_mem_grid {E : ℕ} (hE : 1 ≤ E) (M : ℕ) : (0 : ℝ) ∈ grid E M := by
  refine ⟨0, by positivity, ?_⟩
  have : (2 : ℕ) ^ E - 1 ≠ 0 := by
    have := Nat.one_lt_two_pow_iff.2 (by omega : E ≠ 0); omega
  simp [ieee, this.symm]

theorem grid_finite (E M : ℕ) : (grid E M).Finite := by
  refine ((Finset.range (2 ^ (1 + E + M))).image fun b => (ieee E M b).getD 0).finite_toSet.subset ?_
  rintro v ⟨b, hb, h⟩
  exact Finset.mem_coe.2 (Finset.mem_image.2 ⟨b, Finset.mem_range.2 hb, by simp [h]⟩)

/-- Every number of the format is `± n · 2^k` with `n < 2^{M+1}`. -/
theorem ieee_abs {E M b : ℕ} {z : ℝ} (h : ieee E M b = some z) :
    ∃ n : ℕ, n < 2 ^ (M + 1) ∧ ∃ k : ℤ, |z| = n * (2 : ℝ) ^ k := by
  have hsgn : |(if b / 2 ^ (E + M) % 2 = 1 then (-1 : ℝ) else 1)| = 1 := by
    split_ifs <;> simp
  have hm : b % 2 ^ M < 2 ^ M := Nat.mod_lt _ (by positivity)
  unfold ieee at h
  simp only at h
  generalize (if b / 2 ^ (E + M) % 2 = 1 then (-1 : ℝ) else 1) = sgn at h hsgn
  split_ifs at h <;> obtain rfl := Option.some.inj h
  · refine ⟨b % 2 ^ M, by rw [pow_succ]; omega, 1 - bias E - M, ?_⟩
    rw [abs_mul, abs_mul, hsgn, one_mul, abs_of_nonneg (by positivity),
      abs_of_pos (zpow_pos (by norm_num) _)]
  · refine ⟨2 ^ M + b % 2 ^ M, by rw [pow_succ]; omega, ((b / 2 ^ M % 2 ^ E : ℕ) : ℤ) - bias E - M, ?_⟩
    rw [abs_mul, abs_mul, hsgn, one_mul, abs_of_nonneg (by positivity),
      abs_of_pos (zpow_pos (by norm_num) _)]
    push_cast; rfl

/-- **An IEEE format with `M` mantissa bits has `M + 1` significant bits**, so
`accum_stall`, `accum_stall_abs` and `accum_le` apply to it. -/
theorem grid_hasSignificand (E M : ℕ) : HasSignificand (M + 1) (grid E M) :=
  fun _ ⟨_, _, h⟩ => ieee_abs h

end Precision
end Transformer

module

public import Mathlib.Analysis.Complex.ExponentialBounds
public import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# §4: T5's logarithmic buckets

Chi, Fan, Ramadge, Rudnicky, arXiv:2205.09921v2, §4, Connection to Prior Work: T5 assigns the
relative position `m − n` the bias `bucket[0]` if `m − n < 0`, `bucket[m − n]` if
`0 ≤ m − n < 16`, and `bucket[min(31, ⌊log((m − n)/16) / log(128/16) · 16⌋)]` if `m − n ≥ 16`;
"the log factor is approximately `7.7 log((m − n)/16)`. Therefore, T5 is using a logarithmic
bucket assignment".

* The factor is `(16 / log 8) log((m − n)/16)`, and `7.69 < 16 / log 8 < 7.7` (`t5_log_factor`).
* As printed, the formula sends the distance `16` to `bucket[0]`, the bucket of the distance `0`,
  right after sending `15` to `bucket[15]`: the printed assignment is not monotone in the distance
  (`t5BucketPrinted_sixteen`, `not_monotone_t5BucketPrinted`), and the distances `16, …, 127`
  share the buckets of the exact distances `0, …, 15`.
* The formula presumably omits the offset of the 16 exact buckets: with it,
  `min(31, 16 + ⌊…⌋)` (`t5Bucket`), the assignment is monotone (`monotone_t5Bucket`) and uses the
  32 buckets `0, …, 31` (`t5Bucket_lt`).
-/

@[expose] public section

namespace Transformer.KERPLE

/-- **The log factor of T5's buckets** (§4):
`log(x/16) / log(128/16) · 16 = (16 / log 8) log(x/16)`,
"approximately `7.7 log(x/16)`": `7.69 < 16 / log 8 < 7.7`. -/
theorem t5_log_factor (x : ℝ) :
    Real.log (x / 16) / Real.log (128 / 16) * 16 = 16 / Real.log 8 * Real.log (x / 16) ∧
      7.69 < 16 / Real.log 8 ∧ 16 / Real.log 8 < 7.7 := by
  have h8 : Real.log 8 = 3 * Real.log 2 := by
    rw [show (8 : ℝ) = 2 ^ 3 by norm_num, Real.log_pow]
    norm_num
  have hl := Real.log_two_gt_d9
  have hu := Real.log_two_lt_d9
  refine ⟨by norm_num; ring, ?_, ?_⟩
  · rw [h8, lt_div_iff₀ (by linarith)]
    linarith
  · rw [h8, div_lt_iff₀ (by linarith)]
    linarith

/-- T5's bucket of the relative position `r = m − n`, as printed in §4. -/
noncomputable def t5BucketPrinted (r : ℤ) : ℕ :=
  if r < 0 then 0 else if r < 16 then r.toNat
  else min 31 ⌊Real.log ((r : ℝ) / 16) / Real.log (128 / 16) * 16⌋₊

/-- **The printed formula sends the distance `16` to `bucket[0]`**, the bucket of the distance
`0`. -/
theorem t5BucketPrinted_sixteen :
    t5BucketPrinted 16 = t5BucketPrinted 0 ∧ t5BucketPrinted 0 = 0 := by
  norm_num [t5BucketPrinted]

/-- **The printed formula is not monotone** in the distance: `15 ↦ bucket[15]`, `16 ↦ bucket[0]`;
§4 calls it "a logarithmic bucket assignment". -/
theorem not_monotone_t5BucketPrinted : ¬Monotone t5BucketPrinted := fun h => by
  have h1516 := h (show (15 : ℤ) ≤ 16 by norm_num)
  norm_num [t5BucketPrinted] at h1516

/-- T5's bucket of the relative position `r = m − n` with the offset `16` of the exact buckets:
`min(31, 16 + ⌊log(r/16) / log(128/16) · 16⌋)` for `r ≥ 16`. -/
noncomputable def t5Bucket (r : ℤ) : ℕ :=
  if r < 0 then 0 else if r < 16 then r.toNat
  else min 31 (16 + ⌊Real.log ((r : ℝ) / 16) / Real.log (128 / 16) * 16⌋₊)

/-- **With the offset, the buckets are monotone** in the relative position. -/
theorem monotone_t5Bucket : Monotone t5Bucket := by
  intro r s hrs
  have hlog : 0 < Real.log (128 / 16) := Real.log_pos (by norm_num)
  simp only [t5Bucket]
  split_ifs <;> try omega
  have hr : (16 : ℝ) ≤ r := by exact_mod_cast not_lt.mp ‹¬r < 16›
  have hrs' : (r : ℝ) / 16 ≤ s / 16 := by
    have : (r : ℝ) ≤ s := by exact_mod_cast hrs
    linarith
  have hf : Real.log ((r : ℝ) / 16) / Real.log (128 / 16) * 16 ≤
      Real.log ((s : ℝ) / 16) / Real.log (128 / 16) * 16 :=
    mul_le_mul_of_nonneg_right (div_le_div_of_nonneg_right
      (Real.log_le_log (by linarith) hrs') hlog.le) (by norm_num)
  have := Nat.floor_mono hf
  omega

/-- With the offset, the buckets are the 32 indices `0, …, 31`. -/
theorem t5Bucket_lt (r : ℤ) : t5Bucket r < 32 := by
  simp only [t5Bucket]
  split_ifs <;> omega

end Transformer.KERPLE
